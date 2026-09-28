import csv, io, asyncio, traceback, time
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from app.models import BenchmarkRequest
from app.dataset import dataset_status,load_tasks,load_db,load_policy
from app.evaluator import run_benchmark
from app.storage import init_db,save_benchmark,history,get_benchmark
from app.jobs import create_job,get_job,add_log,update_progress

app=FastAPI(title="Agentic AI Evaluation Platform",version="4.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

@app.on_event("startup")
def startup():init_db()

@app.get("/health")
def health():return {"status":"ok","version":"4.0.0"}

@app.get("/api/dataset/status")
def status():return dataset_status()

@app.get("/api/history")
def benchmark_history():return history()

@app.get("/api/benchmarks/{benchmark_id}")
def benchmark_details(benchmark_id:str):
    b=get_benchmark(benchmark_id)
    if not b:raise HTTPException(404,"Benchmark not found")
    return b

async def _run_job(job_id,req):
    job=get_job(job_id);tasks=load_tasks();db=load_db();policy=load_policy()
    if not tasks or not db:
        job["status"]="failed";job["error"]="Dataset incomplete";return
    job["status"]="running";job["started_at_epoch"]=time.time();job["current_stage"]="starting"
    add_log(job,f"Benchmark started: {job['task_count']} tasks × {req.repeats} repeats = {job['total_runs']} runs")
    add_log(job,f"Concurrency is controlled by MAX_CONCURRENT_TASKS; provider={req.provider}")

    def progress(ev):
        if ev.get("completed_one"):
            job["completed_runs"]+=1
            job["successes_so_far"]+=1 if ev.get("last_run_success") else 0
            job["failures_so_far"]+=0 if ev.get("last_run_success") else 1
            job["input_tokens_so_far"]+=int(ev.get("last_input_tokens",0))
            job["output_tokens_so_far"]+=int(ev.get("last_output_tokens",0))
            job["tool_calls_so_far"]+=int(ev.get("last_tool_calls",0));job["last_output_budget_pct"]=float(ev.get("last_output_budget_pct",0) or 0)
            update_progress(job)
            add_log(job,f"Completed {job['completed_runs']}/{job['total_runs']} | task {job['current_task_index']}/{job['task_count']} | "
                        f"repeat {job['current_repeat']}/{job['repeats']} | {'SUCCESS' if ev.get('last_run_success') else 'FAILED'} | "
                        f"{ev.get('last_run_latency_ms',0):.0f} ms | in {ev.get('last_input_tokens',0)} / out {ev.get('last_output_tokens',0)} tokens | output budget {ev.get('last_output_budget_pct',0)*100:.0f}%")
        else:
            update_progress(job,**ev)

    try:
        summary,runs=await run_benchmark(tasks,policy,db,req,progress_cb=progress)
        save_benchmark(summary["benchmark_id"],summary["created_at"],summary,runs)
        job["benchmark_id"]=summary["benchmark_id"];job["result"]={"summary":summary,"runs":runs}
        job["status"]="completed";job["current_stage"]="completed";job["finished_at_epoch"]=time.time()
        update_progress(job,completed_runs=job["total_runs"]);add_log(job,"Benchmark completed.")
    except Exception as e:
        job["status"]="failed";job["current_stage"]="failed";job["error"]=f"{type(e).__name__}: {e}";job["finished_at_epoch"]=time.time()
        add_log(job,f"Benchmark failed: {job['error']}");traceback.print_exc()

@app.post("/api/benchmark/start")
async def start(req:BenchmarkRequest):
    tasks=load_tasks()
    if not tasks:raise HTTPException(400,"No tasks found")
    count=min(req.task_count,len(tasks));job=create_job(count*req.repeats,count,req.repeats,req.configuration,req.provider)
    asyncio.create_task(_run_job(job["job_id"],req));return {"job_id":job["job_id"]}

@app.get("/api/benchmark/jobs/{job_id}")
def progress(job_id:str):
    j=get_job(job_id)
    if not j:raise HTTPException(404,"Job not found")
    return {k:v for k,v in j.items() if k!="result"}

@app.get("/api/benchmark/jobs/{job_id}/result")
def result(job_id:str):
    j=get_job(job_id)
    if not j:raise HTTPException(404,"Job not found")
    if j["status"]=="failed":raise HTTPException(500,j["error"] or "Benchmark failed")
    if j["status"]!="completed":raise HTTPException(409,"Benchmark still running")
    return j["result"]

@app.get("/api/benchmarks/{benchmark_id}/runs.csv")
def export_csv(benchmark_id:str):
    b=get_benchmark(benchmark_id)
    if not b:raise HTTPException(404,"Benchmark not found")
    rows=b["runs"];fields=[
        "run_id","task_id","repeat","configuration","provider","success","observable_policy_compliant",
        "tool_precision","tool_recall","tool_f1","argument_accuracy","exact_action_recall","communication_recall",
        "invalid_tool_calls","tool_execution_errors","redundant_tool_calls","tool_call_count","step_count","llm_turns","reasoning_tokens",
        "latency_ms","input_tokens","cached_input_tokens","output_tokens","tokens","output_budget_max","output_budget_used_pct","routed_tool_count","routed_policy_chars","estimated_cost_usd",
        "termination_reason","failure_reasons"]
    sio=io.StringIO();w=csv.DictWriter(sio,fieldnames=fields,extrasaction="ignore");w.writeheader()
    for r in rows:
        rr=dict(r);rr["failure_reasons"]="|".join(rr.get("failure_reasons",[]));w.writerow(rr)
    return StreamingResponse(iter([sio.getvalue()]),media_type="text/csv",
        headers={"Content-Disposition":f'attachment; filename="{benchmark_id}-runs.csv"'})
