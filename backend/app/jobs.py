import time, uuid
JOBS={}

def create_job(total_runs,task_count,repeats,configuration,provider):
    j={"job_id":str(uuid.uuid4()),"status":"queued","created_at_epoch":time.time(),"started_at_epoch":None,"finished_at_epoch":None,
       "elapsed_seconds":0,"task_count":task_count,"repeats":repeats,"total_runs":total_runs,"completed_runs":0,"progress_pct":0.0,
       "current_task_index":0,"current_task_id":None,"current_task_text":"","current_repeat":0,"current_stage":"queued",
       "successes_so_far":0,"failures_so_far":0,"input_tokens_so_far":0,"output_tokens_so_far":0,"tool_calls_so_far":0,"last_output_budget_pct":0.0,
       "configuration":configuration,"provider":provider,"logs":[],"benchmark_id":None,"error":None,"result":None}
    JOBS[j["job_id"]]=j;return j

def get_job(job_id):
    j=JOBS.get(job_id)
    if not j:return None
    if j["started_at_epoch"]:
        j["elapsed_seconds"]=round((j["finished_at_epoch"] or time.time())-j["started_at_epoch"],1)
    return j

def add_log(j,msg):
    stamp=time.strftime("%H:%M:%S");j["logs"].append(f"[{stamp}] {msg}")
    j["logs"]=j["logs"][-250:]

def update_progress(j,**kwargs):
    j.update(kwargs)
    if j["total_runs"]:j["progress_pct"]=round(j["completed_runs"]/j["total_runs"]*100,1)
