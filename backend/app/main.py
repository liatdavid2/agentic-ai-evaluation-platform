import csv, io, json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from app.models import BenchmarkRequest
from app.dataset import dataset_status, load_tasks, load_policy
from app.evaluator import run_benchmark
from app.storage import init_db, save_benchmark, history, get_benchmark

app=FastAPI(title="Agentic AI Evaluation Platform",version="2.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

@app.on_event("startup")
def startup():
    init_db()

@app.get("/health")
def health():
    return {"status":"ok","version":"2.0.0"}

@app.get("/api/dataset/status")
def status():
    return dataset_status()

@app.get("/api/history")
def benchmark_history():
    return history()

@app.get("/api/benchmarks/{benchmark_id}")
def benchmark_details(benchmark_id:str):
    b=get_benchmark(benchmark_id)
    if not b:
        raise HTTPException(404,"Benchmark not found")
    return b

@app.post("/api/benchmark")
async def benchmark(req:BenchmarkRequest):
    tasks=load_tasks()
    if not tasks:
        raise HTTPException(400,"No tasks found. Add data/retail/tasks.json.")
    summary,runs=await run_benchmark(tasks,load_policy(),req)
    save_benchmark(summary["benchmark_id"],summary["created_at"],summary,runs)
    return {"summary":summary,"runs":runs}

@app.get("/api/benchmarks/{benchmark_id}/runs.csv")
def export_runs_csv(benchmark_id:str):
    b=get_benchmark(benchmark_id)
    if not b:
        raise HTTPException(404,"Benchmark not found")
    rows=b["runs"]
    fields=[
        "run_id","task_id","repeat","configuration","provider","success","policy_compliant",
        "tool_precision","tool_recall","tool_f1","argument_accuracy","trajectory_order_score",
        "exact_tool_sequence","useful_tool_ratio","invalid_tool_calls","redundant_tool_calls",
        "extra_tool_ratio","tool_call_count","step_count","latency_ms","tokens",
        "estimated_cost_usd","failure_reasons"
    ]
    sio=io.StringIO()
    w=csv.DictWriter(sio,fieldnames=fields,extrasaction="ignore")
    w.writeheader()
    for r in rows:
        rr=dict(r)
        rr["failure_reasons"]="|".join(rr.get("failure_reasons",[]))
        w.writerow(rr)
    return StreamingResponse(
        iter([sio.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition":f'attachment; filename="{benchmark_id}-runs.csv"'}
    )
