import time
import uuid
import math
import statistics
import os
import asyncio
import copy
from datetime import datetime, timezone

from app.dataset import task_id, instruction, reference_actions, communicate_info
from app.providers.heuristic import HeuristicProvider
from app.providers.openai_provider import OpenAIProvider
from app.tooling import MUTATING_TOOLS, RetailToolExecutor


def get_provider(name):
    return OpenAIProvider() if name == "openai" else HeuristicProvider()


def percentile(values, q=.95):
    if not values:
        return 0.0
    s = sorted(values)
    return float(s[max(0, min(len(s)-1, math.ceil(q*len(s))-1))])


def f1(p, r):
    return 0.0 if p+r == 0 else 2*p*r/(p+r)


def norm(v):
    if isinstance(v, dict):
        return {str(k): norm(x) for k, x in sorted(v.items())}
    if isinstance(v, list):
        return [norm(x) for x in v]
    return str(v)


def action_match(e, a):
    return (
        e.get("name") == a.get("name")
        and norm(e.get("arguments", {})) == norm(a.get("arguments", {}))
    )


def soft_argument_score(expected, actual):
    scores = []
    for e in expected:
        same = [a for a in actual if a.get("name") == e.get("name")]
        if not same:
            scores.append(0.0)
            continue
        ea = e.get("arguments", {}) or {}
        if not ea:
            scores.append(1.0)
            continue
        best = 0.0
        for a in same:
            aa = a.get("arguments", {}) or {}
            hits = sum(
                1 for k, v in ea.items()
                if norm(aa.get(k)) == norm(v)
            )
            best = max(best, hits/max(1, len(ea)))
        scores.append(best)
    return sum(scores)/len(scores) if scores else 1.0


def communication_score(task, answer):
    required = communicate_info(task)
    if not required:
        return 1.0, True
    text = str(answer or "").lower()
    hits = sum(1 for x in required if str(x).lower() in text)
    score = hits/len(required)
    return score, score == 1.0


def expected_final_state(task, initial_db):
    """
    Build the expected final environment by replaying only the benchmark's
    gold mutating actions on a clean copy of the initial DB.

    Lookup/authentication actions are intentionally ignored because they
    should not affect final-state success.
    """
    ex = RetailToolExecutor(initial_db)
    affected = {"orders": set(), "users": set()}

    for action in reference_actions(task):
        name = action.get("name")
        args = action.get("arguments", {}) or {}

        if name not in MUTATING_TOOLS:
            continue

        result = ex.execute(name, args)

        if result.get("ok"):
            if args.get("order_id"):
                affected["orders"].add(str(args["order_id"]))
            if args.get("user_id"):
                affected["users"].add(str(args["user_id"]))

    return ex.db, affected


def project_state(db, affected):
    """
    Compare only entities that the gold task is supposed to mutate.
    This avoids requiring unrelated parts of the database to match.
    """
    out = {"orders": {}, "users": {}}

    for oid in sorted(affected["orders"]):
        out["orders"][oid] = copy.deepcopy(
            db.get("orders", {}).get(oid)
        )

    for uid in sorted(affected["users"]):
        out["users"][uid] = copy.deepcopy(
            db.get("users", {}).get(uid)
        )

    return norm(out)


def evaluate_run(task, result, latency_ms, initial_db):
    expected = reference_actions(task)
    expected_tools = [a.get("name") for a in expected if a.get("name")]
    actual = result.get("tool_calls", []) or []
    actual_tools = [c.get("name") for c in actual if c.get("name")]

    # Diagnostic trajectory metrics
    es, aset = set(expected_tools), set(actual_tools)
    precision = len(es & aset)/len(aset) if aset else (1.0 if not es else 0.0)
    recall = len(es & aset)/len(es) if es else 1.0
    arg = soft_argument_score(expected, actual)
    exact_action_recall = (
        sum(any(action_match(e, a) for a in actual) for e in expected)/len(expected)
        if expected else 1.0
    )

    # Official-style end-state comparison
    gold_db, affected = expected_final_state(task, initial_db)
    actual_db = result.get("environment", initial_db)

    has_mutation = bool(affected["orders"] or affected["users"])

    if has_mutation:
        expected_projection = project_state(gold_db, affected)
        actual_projection = project_state(actual_db, affected)
        final_state_match = expected_projection == actual_projection
        final_state_score = 1.0 if final_state_match else 0.0
    else:
        expected_projection = {"orders": {}, "users": {}}
        actual_projection = {"orders": {}, "users": {}}
        final_state_match = True
        final_state_score = 1.0

    comm_score, comm_ok = communication_score(
        task,
        result.get("final_answer", "")
    )

    term = result.get("termination_reason", "completed")

    # Task success:
    # - action tasks: final DB state + required communication
    # - informational tasks: required communication
    if has_mutation:
        success = bool(final_state_match and comm_ok and term == "completed")
    else:
        success = bool(comm_ok and term == "completed")

    inp = int(result.get("input_tokens", 0) or 0)
    out = int(result.get("output_tokens", 0) or 0)
    cached = int(result.get("cached_input_tokens", 0) or 0)

    sim_in = int(result.get("simulator_input_tokens", 0) or 0)
    sim_out = int(result.get("simulator_output_tokens", 0) or 0)
    sim_reasoning = int(result.get("simulator_reasoning_tokens", 0) or 0)

    ir = float(os.getenv("INPUT_COST_PER_1M", "0"))
    cr = float(os.getenv("CACHED_INPUT_COST_PER_1M", "0"))
    orr = float(os.getenv("OUTPUT_COST_PER_1M", "0"))

    cost = (
        max(0, inp-cached)*ir
        + cached*cr
        + out*orr
        + sim_in*ir
        + sim_out*orr
    )/1_000_000

    failures = []
    if not success:
        failures.append("task_outcome_failure")
    if has_mutation and not final_state_match:
        failures.append("final_state_mismatch")
    if comm_score < 1:
        failures.append("missing_required_communication")
    if recall < 1:
        failures.append("missing_reference_tool")
    if precision < 1:
        failures.append("unexpected_tool")
    if arg < 1:
        failures.append("argument_mismatch")
    if any(c.get("status") == "error" for c in actual):
        failures.append("tool_execution_error")
    if term != "completed":
        failures.append(term)

    budget = result.get("token_budget", {}) or {}
    max_run = int(budget.get("max_output_per_run", 0) or 0)

    return {
        "success": success,

        # Primary outcome
        "final_state_match": final_state_match,
        "final_state_score": final_state_score,
        "communication_recall": round(comm_score, 4),

        # Diagnostics
        "tool_precision": round(precision, 4),
        "tool_recall": round(recall, 4),
        "tool_f1": round(f1(precision, recall), 4),
        "argument_accuracy": round(arg, 4),
        "exact_action_recall": round(exact_action_recall, 4),

        "invalid_tool_calls": sum(1 for c in actual if c.get("status") == "invalid"),
        "tool_execution_errors": sum(1 for c in actual if c.get("status") == "error"),
        "useful_tool_ratio": round(
            sum(1 for c in actual if c.get("status") == "ok")/max(1, len(actual)),
            4
        ),
        "tool_call_count": len(actual),
        "step_count": len(result.get("trace", []) or []),
        "llm_turns": int(result.get("llm_turns", 0) or 0),
        "user_simulator_turns": int(result.get("user_simulator_turns", 0) or 0),
        "reasoning_tokens": int(result.get("reasoning_tokens", 0) or 0),

        # Token accounting
        "input_tokens": inp,
        "cached_input_tokens": cached,
        "output_tokens": out,
        "simulator_input_tokens": sim_in,
        "simulator_output_tokens": sim_out,
        "simulator_reasoning_tokens": sim_reasoning,
        "tokens": inp + out + sim_in + sim_out,

        "latency_ms": round(latency_ms, 2),
        "estimated_cost_usd": round(cost, 6),
        "termination_reason": term,

        "output_budget_max": max_run,
        "output_budget_used_pct": round(out/max_run, 4) if max_run else 0.0,

        "final_answer": str(result.get("final_answer", "")),
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
        "required_communicate_info": communicate_info(task),
        "trace": result.get("trace", []) or [],
        "failure_reasons": failures,

        # Helpful inspector snapshots
        "expected_state_projection": expected_projection,
        "actual_state_projection": actual_projection,
    }


def aggregate(ts, runs, configuration, provider, task_count, repeats, bid):
    n = max(1, len(runs))
    tn = max(1, len(ts))
    succ = sum(r["success"] for r in runs)

    lat = [r["latency_ms"] for r in runs]
    inp = [r["input_tokens"] for r in runs]
    out = [r["output_tokens"] for r in runs]
    sim_in = [r["simulator_input_tokens"] for r in runs]
    sim_out = [r["simulator_output_tokens"] for r in runs]

    fail = {}
    term = {}
    for r in runs:
        term[r["termination_reason"]] = term.get(r["termination_reason"], 0) + 1
        for x in r["failure_reasons"]:
            fail[x] = fail.get(x, 0) + 1

    return {
        "benchmark_id": bid,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_count": task_count,
        "repeats": repeats,
        "total_runs": len(runs),
        "configuration": configuration,
        "provider": provider,

        "task_success_rate": round(succ/n, 4),
        "final_state_match_rate": round(
            sum(r["final_state_match"] for r in runs)/n, 4
        ),
        "pass_at_k": round(sum(any(v) for v in ts.values())/tn, 4),
        "pass_power_k": round(sum(all(v) for v in ts.values())/tn, 4),
        "repeat_consistency": round(
            sum(1 for v in ts.values() if len(set(v)) <= 1)/tn, 4
        ),

        "mean_tool_precision": round(sum(r["tool_precision"] for r in runs)/n, 4),
        "mean_tool_recall": round(sum(r["tool_recall"] for r in runs)/n, 4),
        "mean_tool_f1": round(sum(r["tool_f1"] for r in runs)/n, 4),
        "mean_argument_accuracy": round(sum(r["argument_accuracy"] for r in runs)/n, 4),
        "mean_exact_action_recall": round(sum(r["exact_action_recall"] for r in runs)/n, 4),
        "mean_communication_recall": round(sum(r["communication_recall"] for r in runs)/n, 4),
        "mean_useful_tool_ratio": round(sum(r["useful_tool_ratio"] for r in runs)/n, 4),
        "tool_errors_per_run": round(sum(r["tool_execution_errors"] for r in runs)/n, 3),
        "mean_tool_calls": round(sum(r["tool_call_count"] for r in runs)/n, 2),

        "mean_llm_turns": round(sum(r["llm_turns"] for r in runs)/n, 2),
        "mean_user_simulator_turns": round(
            sum(r["user_simulator_turns"] for r in runs)/n, 2
        ),
        "mean_reasoning_tokens": round(
            sum(r["reasoning_tokens"] for r in runs)/n, 1
        ),

        "mean_latency_ms": round(statistics.mean(lat), 2),
        "median_latency_ms": round(statistics.median(lat), 2),
        "p95_latency_ms": round(percentile(lat), 2),

        "mean_input_tokens": round(statistics.mean(inp), 1),
        "p95_input_tokens": round(percentile(inp), 1),
        "mean_output_tokens": round(statistics.mean(out), 1),
        "p95_output_tokens": round(percentile(out), 1),
        "mean_simulator_input_tokens": round(statistics.mean(sim_in), 1),
        "mean_simulator_output_tokens": round(statistics.mean(sim_out), 1),

        "total_input_tokens": sum(inp),
        "total_output_tokens": sum(out),
        "total_simulator_input_tokens": sum(sim_in),
        "total_simulator_output_tokens": sum(sim_out),
        "total_cached_input_tokens": sum(r["cached_input_tokens"] for r in runs),

        "total_cost_usd": round(sum(r["estimated_cost_usd"] for r in runs), 6),
        "cost_per_success_usd": round(
            sum(r["estimated_cost_usd"] for r in runs)/max(1, succ), 6
        ),

        "failure_taxonomy": fail,
        "termination_reasons": term,
    }


async def run_benchmark(tasks, policy, db, req, progress_cb=None):
    selected = tasks[:req.task_count]
    provider = get_provider(req.provider)
    bid = str(uuid.uuid4())
    runs = []
    ts = {task_id(t, i): [] for i, t in enumerate(selected)}

    sem = asyncio.Semaphore(
        max(1, int(os.getenv("MAX_CONCURRENT_TASKS", "3")))
    )
    lock = asyncio.Lock()

    async def one(i, t, rep):
        tid = task_id(t, i)

        if progress_cb:
            progress_cb({
                "current_task_index": i+1,
                "current_task_id": tid,
                "current_task_text": instruction(t)[:500],
                "current_repeat": rep+1,
                "current_stage": "waiting_for_slot",
            })

        async with sem:
            if progress_cb:
                progress_cb({
                    "current_task_index": i+1,
                    "current_task_id": tid,
                    "current_task_text": instruction(t)[:500],
                    "current_repeat": rep+1,
                    "current_stage": "agent_user_dialogue",
                })

            t0 = time.perf_counter()
            result = await provider.run(
                t, req.configuration, policy, db=db
            )
            latency = (time.perf_counter()-t0)*1000

            if progress_cb:
                progress_cb({"current_stage": "final_state_evaluation"})

            ev = evaluate_run(t, result, latency, db)
            ev.update({
                "run_id": str(uuid.uuid4()),
                "task_id": tid,
                "repeat": rep+1,
                "configuration": req.configuration,
                "provider": req.provider,
            })

            async with lock:
                runs.append(ev)
                ts[tid].append(ev["success"])

            if progress_cb:
                progress_cb({
                    "completed_one": True,
                    "last_run_success": ev["success"],
                    "current_stage": "run_completed",
                    "last_run_latency_ms": ev["latency_ms"],
                    "last_input_tokens": (
                        ev["input_tokens"] + ev["simulator_input_tokens"]
                    ),
                    "last_output_tokens": (
                        ev["output_tokens"] + ev["simulator_output_tokens"]
                    ),
                    "last_tool_calls": ev["tool_call_count"],
                })

    await asyncio.gather(*(
        one(i, t, r)
        for i, t in enumerate(selected)
        for r in range(req.repeats)
    ))

    runs.sort(
        key=lambda x: (
            int(x["task_id"]) if str(x["task_id"]).isdigit() else 999999,
            x["repeat"],
        )
    )

    return (
        aggregate(
            ts, runs, req.configuration, req.provider,
            len(selected), req.repeats, bid
        ),
        runs,
    )
