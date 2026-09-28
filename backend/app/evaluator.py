import time,uuid,math,statistics,os,asyncio
from datetime import datetime,timezone
from app.dataset import task_id,instruction,reference_actions,communicate_info
from app.providers.heuristic import HeuristicProvider
from app.providers.openai_provider import OpenAIProvider
from app.tooling import MUTATING_TOOLS

def get_provider(name):return OpenAIProvider() if name=="openai" else HeuristicProvider()
def percentile(values,q=.95):
    if not values:return 0.0
    s=sorted(values);return float(s[max(0,min(len(s)-1,math.ceil(q*len(s))-1))])
def f1(p,r):return 0.0 if p+r==0 else 2*p*r/(p+r)
def norm(v):
    if isinstance(v,dict):return {str(k):norm(x) for k,x in sorted(v.items())}
    if isinstance(v,list):return [norm(x) for x in v]
    return str(v)
def action_match(e,a):return e.get("name")==a.get("name") and norm(e.get("arguments",{}))==norm(a.get("arguments",{}))

def soft_argument_score(expected,actual):
    scores=[]
    for e in expected:
        same=[a for a in actual if a.get("name")==e.get("name")]
        if not same:scores.append(0.0);continue
        ea=e.get("arguments",{}) or {}
        if not ea:scores.append(1.0);continue
        best=0.0
        for a in same:
            aa=a.get("arguments",{}) or {}
            hits=sum(1 for k,v in ea.items() if norm(aa.get(k))==norm(v))
            best=max(best,hits/max(1,len(ea)))
        scores.append(best)
    return sum(scores)/len(scores) if scores else 1.0

def policy_checks(actual):
    violations=[];authenticated=False;seen={}
    for c in actual:
        n=c.get("name");args=c.get("arguments",{}) or {}
        if n in {"find_user_id_by_name_zip","find_user_id_by_email"} and c.get("status")=="ok":authenticated=True
        if (n in {"get_user_details","get_order_details"} or n in MUTATING_TOOLS) and not authenticated:
            violations.append("personal_action_before_authentication")
        oid=args.get("order_id")
        if oid and n in {"modify_pending_order_items","exchange_delivered_order_items"}:
            key=(n,oid);seen[key]=seen.get(key,0)+1
            if seen[key]>1:violations.append("multiple_modify_or_exchange_calls_same_order")
    return len(violations)==0,sorted(set(violations))

def evaluate_run(task,result,latency_ms):
    expected=reference_actions(task)
    expected_tools=[a.get("name") for a in expected if a.get("name")]
    actual=result.get("tool_calls",[]) or []
    actual_tools=[c.get("name") for c in actual if c.get("name")]
    es,as_=set(expected_tools),set(actual_tools)
    precision=len(es&as_)/len(as_) if as_ else (1.0 if not es else 0.0)
    recall=len(es&as_)/len(es) if es else 1.0
    arg=soft_argument_score(expected,actual)
    exact=sum(any(action_match(e,a) for a in actual) for e in expected)/len(expected) if expected else 1.0
    expected_mut=[e for e in expected if e.get("name") in MUTATING_TOOLS]
    mut_ok=all(any(action_match(e,a) and a.get("status")=="ok" for a in actual) for e in expected_mut)
    req=communicate_info(task);answer=str(result.get("final_answer",""))
    comm=sum(1 for x in req if x.lower() in answer.lower())/len(req) if req else 1.0
    policy_ok,violations=policy_checks(actual)
    term=result.get("termination_reason","completed")
    success=bool(mut_ok and comm==1.0 and policy_ok and term=="completed")
    inp=int(result.get("input_tokens",0) or 0);out=int(result.get("output_tokens",0) or 0);cached=int(result.get("cached_input_tokens",0) or 0)
    ir=float(os.getenv("INPUT_COST_PER_1M","0"));cr=float(os.getenv("CACHED_INPUT_COST_PER_1M","0"));orr=float(os.getenv("OUTPUT_COST_PER_1M","0"))
    cost=((max(0,inp-cached)*ir)+(cached*cr)+(out*orr))/1_000_000
    failures=[]
    if not success:failures.append("task_outcome_failure")
    if recall<1:failures.append("missing_reference_tool")
    if precision<1:failures.append("unexpected_tool")
    if arg<1:failures.append("argument_mismatch")
    if comm<1:failures.append("missing_required_communication")
    if not policy_ok:failures.append("observable_policy_violation")
    if any(c.get("status")=="error" for c in actual):failures.append("tool_execution_error")
    if term!="completed":failures.append(term)
    budget=result.get("token_budget",{}) or {};routing=result.get("routing",{}) or {}
    max_run=int(budget.get("max_output_per_run",0) or 0)
    return {
        "success":success,"observable_policy_compliant":policy_ok,"policy_violations":violations,"confirmation_check":"not_evaluated",
        "tool_precision":round(precision,4),"tool_recall":round(recall,4),"tool_f1":round(f1(precision,recall),4),
        "argument_accuracy":round(arg,4),"exact_action_recall":round(exact,4),"communication_recall":round(comm,4),
        "invalid_tool_calls":sum(1 for c in actual if c.get("status")=="invalid"),
        "tool_execution_errors":sum(1 for c in actual if c.get("status")=="error"),
        "redundant_tool_calls":max(0,len(actual_tools)-len(set((c.get("name"),str(norm(c.get("arguments",{})))) for c in actual))),
        "useful_tool_ratio":round(sum(1 for c in actual if c.get("status")=="ok")/max(1,len(actual)),4),
        "tool_call_count":len(actual),"step_count":len(result.get("trace",[]) or []),"llm_turns":int(result.get("llm_turns",0) or 0),
        "latency_ms":round(latency_ms,2),"input_tokens":inp,"cached_input_tokens":cached,"output_tokens":out,"tokens":inp+out,
        "output_budget_max":max_run,"output_budget_used_pct":round(out/max_run,4) if max_run else 0.0,
        "routed_tool_count":int(routing.get("tool_count",0) or 0),"routed_policy_chars":int(routing.get("policy_chars",0) or 0),
        "estimated_cost_usd":round(cost,6),"termination_reason":term,"final_answer":answer,
        "expected_tools":expected_tools,"actual_tools":actual_tools,"required_communicate_info":req,
        "trace":result.get("trace",[]) or [],"failure_reasons":failures
    }

def aggregate(ts,runs,configuration,provider,task_count,repeats,bid):
    n=max(1,len(runs));tn=max(1,len(ts));succ=sum(r["success"] for r in runs)
    lat=[r["latency_ms"] for r in runs];inp=[r["input_tokens"] for r in runs];out=[r["output_tokens"] for r in runs];tok=[r["tokens"] for r in runs]
    fail={};term={}
    for r in runs:
        term[r["termination_reason"]]=term.get(r["termination_reason"],0)+1
        for x in r["failure_reasons"]:fail[x]=fail.get(x,0)+1
    return {
        "benchmark_id":bid,"created_at":datetime.now(timezone.utc).isoformat(),"task_count":task_count,"repeats":repeats,"total_runs":len(runs),
        "configuration":configuration,"provider":provider,
        "task_success_rate":round(succ/n,4),"pass_at_k":round(sum(any(v) for v in ts.values())/tn,4),"pass_power_k":round(sum(all(v) for v in ts.values())/tn,4),
        "repeat_consistency":round(sum(1 for v in ts.values() if len(set(v))<=1)/tn,4),
        "observable_policy_compliance_rate":round(sum(r["observable_policy_compliant"] for r in runs)/n,4),
        "mean_tool_precision":round(sum(r["tool_precision"] for r in runs)/n,4),"mean_tool_recall":round(sum(r["tool_recall"] for r in runs)/n,4),"mean_tool_f1":round(sum(r["tool_f1"] for r in runs)/n,4),
        "mean_argument_accuracy":round(sum(r["argument_accuracy"] for r in runs)/n,4),"mean_exact_action_recall":round(sum(r["exact_action_recall"] for r in runs)/n,4),
        "mean_communication_recall":round(sum(r["communication_recall"] for r in runs)/n,4),"mean_useful_tool_ratio":round(sum(r["useful_tool_ratio"] for r in runs)/n,4),
        "invalid_calls_per_run":round(sum(r["invalid_tool_calls"] for r in runs)/n,3),"tool_errors_per_run":round(sum(r["tool_execution_errors"] for r in runs)/n,3),
        "redundant_calls_per_run":round(sum(r["redundant_tool_calls"] for r in runs)/n,3),"mean_tool_calls":round(sum(r["tool_call_count"] for r in runs)/n,2),
        "mean_steps":round(sum(r["step_count"] for r in runs)/n,2),"mean_llm_turns":round(sum(r["llm_turns"] for r in runs)/n,2),
        "mean_routed_tool_count":round(sum(r["routed_tool_count"] for r in runs)/n,2),"mean_routed_policy_chars":round(sum(r["routed_policy_chars"] for r in runs)/n,1),
        "mean_latency_ms":round(statistics.mean(lat),2),"median_latency_ms":round(statistics.median(lat),2),"p95_latency_ms":round(percentile(lat),2),
        "mean_input_tokens":round(statistics.mean(inp),1),"p95_input_tokens":round(percentile(inp),1),"mean_output_tokens":round(statistics.mean(out),1),"p95_output_tokens":round(percentile(out),1),
        "mean_tokens":round(statistics.mean(tok),1),"p95_tokens":round(percentile(tok),1),"total_input_tokens":sum(inp),"total_output_tokens":sum(out),
        "total_cached_input_tokens":sum(r["cached_input_tokens"] for r in runs),"mean_output_budget_used_pct":round(sum(r["output_budget_used_pct"] for r in runs)/n,4),
        "total_cost_usd":round(sum(r["estimated_cost_usd"] for r in runs),6),"cost_per_success_usd":round(sum(r["estimated_cost_usd"] for r in runs)/max(1,succ),6),
        "failure_taxonomy":fail,"termination_reasons":term
    }

async def run_benchmark(tasks,policy,db,req,progress_cb=None):
    selected=tasks[:req.task_count];provider=get_provider(req.provider);bid=str(uuid.uuid4())
    runs=[];ts={task_id(t,i):[] for i,t in enumerate(selected)}
    sem=asyncio.Semaphore(max(1,int(os.getenv("MAX_CONCURRENT_TASKS","3"))));lock=asyncio.Lock()
    async def one(i,t,rep):
        tid=task_id(t,i)
        if progress_cb:progress_cb({"current_task_index":i+1,"current_task_id":tid,"current_task_text":instruction(t)[:500],"current_repeat":rep+1,"current_stage":"waiting_for_slot"})
        async with sem:
            if progress_cb:progress_cb({"current_task_index":i+1,"current_task_id":tid,"current_task_text":instruction(t)[:500],"current_repeat":rep+1,"current_stage":"calling_agent"})
            t0=time.perf_counter();result=await provider.run(t,req.configuration,policy,db=db);lat=(time.perf_counter()-t0)*1000
            if progress_cb:progress_cb({"current_stage":"evaluating_run"})
            ev=evaluate_run(t,result,lat);ev.update({"run_id":str(uuid.uuid4()),"task_id":tid,"repeat":rep+1,"configuration":req.configuration,"provider":req.provider})
            async with lock:runs.append(ev);ts[tid].append(ev["success"])
            if progress_cb:progress_cb({"completed_one":True,"last_run_success":ev["success"],"current_stage":"run_completed","last_run_latency_ms":ev["latency_ms"],"last_input_tokens":ev["input_tokens"],"last_output_tokens":ev["output_tokens"],"last_tool_calls":ev["tool_call_count"],"last_output_budget_pct":ev["output_budget_used_pct"]})
    await asyncio.gather(*(one(i,t,r) for i,t in enumerate(selected) for r in range(req.repeats)))
    runs.sort(key=lambda x:(int(x["task_id"]) if str(x["task_id"]).isdigit() else 999999,x["repeat"]))
    return aggregate(ts,runs,req.configuration,req.provider,len(selected),req.repeats,bid),runs
