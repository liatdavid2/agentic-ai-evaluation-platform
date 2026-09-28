import time, uuid, math, statistics, json
from datetime import datetime, timezone
from app.dataset import task_id, reference_actions, reference_tool_names
from app.providers.heuristic import HeuristicProvider
from app.providers.openai_provider import OpenAIProvider

def get_provider(name):
    return OpenAIProvider() if name == "openai" else HeuristicProvider()

def percentile(values, q=.95):
    if not values:
        return 0.0
    s = sorted(values)
    i = max(0, min(len(s)-1, math.ceil(q*len(s))-1))
    return float(s[i])

def f1(p, r):
    return 0.0 if p+r == 0 else 2*p*r/(p+r)

def lcs_length(a, b):
    if not a or not b:
        return 0
    dp = [0]*(len(b)+1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, start=1):
            old = dp[j]
            if x == y:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = old
    return dp[-1]

def normalize_args(d):
    if not isinstance(d, dict):
        return {}
    return {str(k): str(v) for k,v in d.items()}

def argument_score(expected_actions, actual_calls):
    """
    Soft argument match: for calls with the same tool name, measure how many
    expected key/value pairs are reproduced. Does not require exact JSON order.
    """
    expected_by_name = {}
    for a in expected_actions:
        name = a.get("name") or a.get("tool") or a.get("tool_name")
        if not name:
            continue
        expected_by_name.setdefault(str(name), []).append(normalize_args(a.get("arguments", {})))

    scores = []
    for c in actual_calls:
        name = str(c.get("name",""))
        if name not in expected_by_name:
            continue
        actual = normalize_args(c.get("arguments", {}))
        best = 0.0
        for exp in expected_by_name[name]:
            if not exp:
                best = max(best, 1.0)
                continue
            hits = sum(1 for k,v in exp.items() if actual.get(k) == v)
            best = max(best, hits / max(1, len(exp)))
        scores.append(best)
    return sum(scores)/len(scores) if scores else (1.0 if not expected_by_name else 0.0)

def evaluate_run(task, result, latency_ms):
    expected_actions = reference_actions(task)
    expected_tools = reference_tool_names(task)
    calls = result.get("tool_calls", []) or []
    actual_tools = [str(x.get("name","")) for x in calls]

    exp_set, act_set = set(expected_tools), set(actual_tools)
    tool_precision = len(exp_set & act_set)/len(act_set) if act_set else (1.0 if not exp_set else 0.0)
    tool_recall = len(exp_set & act_set)/len(exp_set) if exp_set else 1.0
    tool_f1 = f1(tool_precision, tool_recall)

    invalid_calls = sum(1 for x in calls if x.get("status") == "invalid" or x.get("name") == "unknown_tool")
    redundant_calls = max(0, len(actual_tools)-len(set(actual_tools)))

    seq_lcs = lcs_length(expected_tools, actual_tools)
    trajectory_order_score = (
        seq_lcs / len(expected_tools)
        if expected_tools else 1.0
    )
    exact_tool_sequence = (expected_tools == actual_tools)
    extra_tool_ratio = (
        max(0, len(actual_tools)-len(expected_tools))/max(1,len(actual_tools))
        if actual_tools else 0.0
    )
    arg_score = argument_score(expected_actions, calls)

    trace = result.get("trace", []) or []
    step_count = len(trace)
    tool_call_count = len(calls)
    useful_calls = sum(1 for c in calls if c.get("useful", True))
    useful_tool_ratio = useful_calls/max(1, tool_call_count) if tool_call_count else 1.0

    success = bool(result.get("success", False))
    policy = bool(result.get("policy_compliant", False))
    tokens = int(result.get("tokens", 0) or 0)

    # Failure taxonomy is deterministic from observable run signals.
    failure_reasons = []
    if not success:
        failure_reasons.append("task_outcome_failure")
    if not policy:
        failure_reasons.append("policy_violation")
    if tool_recall < 1:
        failure_reasons.append("missing_reference_tool")
    if tool_precision < 1:
        failure_reasons.append("unexpected_tool")
    if invalid_calls:
        failure_reasons.append("invalid_tool_call")
    if redundant_calls:
        failure_reasons.append("redundant_tool_call")
    if arg_score < 1 and expected_actions:
        failure_reasons.append("argument_mismatch")
    if trajectory_order_score < 1 and expected_tools:
        failure_reasons.append("trajectory_order_mismatch")

    return {
        "success": success,
        "policy_compliant": policy,
        "tool_precision": round(tool_precision,4),
        "tool_recall": round(tool_recall,4),
        "tool_f1": round(tool_f1,4),
        "argument_accuracy": round(arg_score,4),
        "trajectory_order_score": round(trajectory_order_score,4),
        "exact_tool_sequence": exact_tool_sequence,
        "useful_tool_ratio": round(useful_tool_ratio,4),
        "invalid_tool_calls": invalid_calls,
        "redundant_tool_calls": redundant_calls,
        "extra_tool_ratio": round(extra_tool_ratio,4),
        "tool_call_count": tool_call_count,
        "step_count": step_count,
        "latency_ms": round(latency_ms,2),
        "tokens": tokens,
        "estimated_cost_usd": round(tokens/1_000_000,6),
        "final_answer": str(result.get("final_answer","")),
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
        "trace": trace,
        "failure_reasons": failure_reasons,
    }

def aggregate(task_success, runs, configuration, provider, task_count, repeats, bid):
    n=max(1,len(runs)); tn=max(1,len(task_success))
    lat=[r["latency_ms"] for r in runs]
    tok=[r["tokens"] for r in runs]

    success_count=sum(r["success"] for r in runs)
    successful_cost=sum(r["estimated_cost_usd"] for r in runs if r["success"])
    total_cost=sum(r["estimated_cost_usd"] for r in runs)

    failure_counts={}
    for r in runs:
        for reason in r["failure_reasons"]:
            failure_counts[reason]=failure_counts.get(reason,0)+1

    # Per-task consistency: fraction of tasks whose repeat outcomes are identical.
    consistent_tasks=sum(1 for xs in task_success.values() if len(set(xs)) <= 1)

    return {
        "benchmark_id":bid,
        "created_at":datetime.now(timezone.utc).isoformat(),
        "task_count":task_count,
        "repeats":repeats,
        "total_runs":len(runs),
        "configuration":configuration,
        "provider":provider,

        "task_success_rate":round(success_count/n,4),
        "pass_at_k":round(sum(any(v) for v in task_success.values())/tn,4),
        "pass_power_k":round(sum(all(v) for v in task_success.values())/tn,4),
        "repeat_consistency":round(consistent_tasks/tn,4),

        "policy_compliance_rate":round(sum(r["policy_compliant"] for r in runs)/n,4),

        "mean_tool_precision":round(sum(r["tool_precision"] for r in runs)/n,4),
        "mean_tool_recall":round(sum(r["tool_recall"] for r in runs)/n,4),
        "mean_tool_f1":round(sum(r["tool_f1"] for r in runs)/n,4),
        "mean_argument_accuracy":round(sum(r["argument_accuracy"] for r in runs)/n,4),
        "mean_trajectory_order_score":round(sum(r["trajectory_order_score"] for r in runs)/n,4),
        "exact_tool_sequence_rate":round(sum(r["exact_tool_sequence"] for r in runs)/n,4),
        "mean_useful_tool_ratio":round(sum(r["useful_tool_ratio"] for r in runs)/n,4),
        "invalid_calls_per_run":round(sum(r["invalid_tool_calls"] for r in runs)/n,4),
        "redundant_calls_per_run":round(sum(r["redundant_tool_calls"] for r in runs)/n,4),
        "mean_extra_tool_ratio":round(sum(r["extra_tool_ratio"] for r in runs)/n,4),
        "mean_tool_calls":round(sum(r["tool_call_count"] for r in runs)/n,2),
        "mean_steps":round(sum(r["step_count"] for r in runs)/n,2),

        "mean_latency_ms":round(statistics.mean(lat),2) if lat else 0,
        "median_latency_ms":round(statistics.median(lat),2) if lat else 0,
        "p95_latency_ms":round(percentile(lat),2),
        "mean_tokens":round(statistics.mean(tok),2) if tok else 0,
        "median_tokens":round(statistics.median(tok),2) if tok else 0,
        "p95_tokens":round(percentile(tok),2),
        "total_cost_usd":round(total_cost,6),
        "cost_per_success_usd":round(total_cost/max(1,success_count),6),

        "failure_taxonomy":failure_counts,
    }

async def run_benchmark(tasks, policy, req):
    selected=tasks[:req.task_count]
    provider=get_provider(req.provider)
    bid=str(uuid.uuid4())
    runs=[]
    task_success={}

    for i,task in enumerate(selected):
        tid=task_id(task,i)
        task_success[tid]=[]
        for rep in range(req.repeats):
            t0=time.perf_counter()
            result=await provider.run(task,req.configuration,policy)
            latency=(time.perf_counter()-t0)*1000
            ev=evaluate_run(task,result,latency)
            ev.update({
                "run_id":str(uuid.uuid4()),
                "task_id":tid,
                "repeat":rep+1,
                "configuration":req.configuration,
                "provider":req.provider,
            })
            runs.append(ev)
            task_success[tid].append(ev["success"])

    summary=aggregate(task_success,runs,req.configuration,req.provider,len(selected),req.repeats,bid)
    return summary,runs
