import React,{useEffect,useMemo,useRef,useState}from"react";
import{createRoot}from"react-dom/client";
import"./styles.css";

const API=import.meta.env.VITE_API_BASE_URL||"http://localhost:8007";
const pct=v=>`${Math.round((v||0)*100)}%`;
const num=(v,d=1)=>Number(v||0).toFixed(d);
const fmt=s=>{
  s=Math.floor(Number(s||0));
  return `${String(Math.floor(s/60)).padStart(2,"0")}:${String(s%60).padStart(2,"0")}`;
};

function errorMessage(detail,status){
 const d=detail?.detail ?? detail;
 if(Array.isArray(d)){
   return d.map(x=>{
     if(typeof x==="string")return x;
     if(x&&typeof x==="object")return x.msg||JSON.stringify(x);
     return String(x);
   }).join(" | ");
 }
 if(d&&typeof d==="object")return JSON.stringify(d);
 return String(d||`Request failed${status?` (${status})`:""}`);
}

async function readJson(r){
 const body=await r.json().catch(()=>null);
 if(!r.ok)throw new Error(errorMessage(body,r.status));
 return body;
}

function Pill({ok,children}){
 return <span className={`pill ${ok?"ok":"bad"}`}>{children}</span>
}

function Metric({label,value,hint}){
 return <div className="metric">
  <div className="metric-label">{label}</div>
  <div className="metric-value">{value}</div>
  <div className="metric-hint">{hint}</div>
 </div>
}

function App(){
 const[dataset,setDataset]=useState(null);
 const[history,setHistory]=useState([]);
 const[result,setResult]=useState(null);
 const[selectedRun,setSelectedRun]=useState(null);

 const[taskCount,setTaskCount]=useState(5);
 const[repeats,setRepeats]=useState(1);
 const[provider,setProvider]=useState("openai");

 const[loading,setLoading]=useState(false);
 const[job,setJob]=useState(null);
 const[error,setError]=useState("");
 const poll=useRef(null);

 const refresh=async()=>{
  try{
   const[d,h]=await Promise.all([
    fetch(`${API}/api/dataset/status`).then(readJson),
    fetch(`${API}/api/history`).then(readJson)
   ]);
   setDataset(d);
   setHistory(Array.isArray(h)?h:[]);
  }catch(e){
   setError(e?.message||String(e));
  }
 };

 useEffect(()=>{
  refresh();
  return()=>poll.current&&clearInterval(poll.current);
 },[]);

 const pollJob=id=>{
  poll.current=setInterval(async()=>{
   try{
    const j=await fetch(`${API}/api/benchmark/jobs/${id}`).then(readJson);
    setJob(j);

    if(j.status==="completed"){
     clearInterval(poll.current);
     const b=await fetch(`${API}/api/benchmark/jobs/${id}/result`).then(readJson);
     setResult(b);
     setSelectedRun(b.runs?.[0]||null);
     setLoading(false);
     refresh();
    }

    if(j.status==="failed"){
     clearInterval(poll.current);
     setError(errorMessage(j.error||"Benchmark failed"));
     setLoading(false);
    }
   }catch(e){
    clearInterval(poll.current);
    setError(e?.message||String(e));
    setLoading(false);
   }
  },1000);
 };

 const run=async()=>{
  setLoading(true);
  setResult(null);
  setSelectedRun(null);
  setError("");

  try{
   const b=await fetch(`${API}/api/benchmark/start`,{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({
     task_count:+taskCount,
     repeats:+repeats,
     configuration:"tools",
     provider
    })
   }).then(readJson);

   setJob({
    job_id:b.job_id,
    status:"queued",
    completed_runs:0,
    total_runs:+taskCount*+repeats,
    progress_pct:0,
    logs:[]
   });

   pollJob(b.job_id);
  }catch(e){
   setError(e?.message||String(e));
   setLoading(false);
  }
 };

 const s=result?.summary;
 const failures=useMemo(
  ()=>result?.runs?.filter(r=>r.failure_reasons?.length)||[],
  [result]
 );

 return <div className="app">
  <aside className="sidebar">
   <div className="brand">
    <div className="logo">AE</div>
    <div><strong>Agent Eval</strong><span>Clean Baseline</span></div>
   </div>

   <nav>
    <a className="active">Evaluation Lab</a>
    <a href={`${API}/docs`} target="_blank">API Docs ↗</a>
   </nav>

   <div className="side-card">
    <div className="side-title">Dataset</div>
    {dataset&&<>
     <div className="status-row"><span>Retail</span><Pill ok={dataset.ready}>{dataset.ready?"Ready":"Missing"}</Pill></div>
     <div className="status-row"><span>Tasks</span><b>{dataset.task_count}</b></div>
     <div className="status-row"><span>Products</span><b>{dataset.product_count}</b></div>
     <div className="status-row"><span>Users</span><b>{dataset.user_count}</b></div>
     <div className="status-row"><span>Orders</span><b>{dataset.order_count}</b></div>
    </>}
   </div>
  </aside>

  <main>
   <header>
    <div>
     <div className="eyebrow">GENAI / AGENTIC AI</div>
     <h1>Clean Baseline Agent</h1>
     <p>LLM + real tools + Retail DB + evaluator. No simulator, no guardrail engine, no hard-coded trajectory.</p>
    </div>
    <div className="header-badge">τ²-bench Retail</div>
   </header>

   <section className="budget-card">
    <div>
     <strong>Baseline architecture</strong>
     <span>Task → LLM → real tools → db.json → LLM → final answer → evaluator</span>
    </div>
    <div className="budget-tip">Backend: localhost:8007</div>
   </section>

   <section className="panel">
    <div className="panel-head">
     <div>
      <h2>Benchmark configuration</h2>
      <p>Start with 5 tasks × 1 repeat and verify actual tool trajectories.</p>
     </div>
    </div>

    <div className="controls">
     <label>Tasks
      <input type="number" min="1" max="114" value={taskCount} onChange={e=>setTaskCount(e.target.value)}/>
     </label>
     <label>Repeats / task
      <input type="number" min="1" max="10" value={repeats} onChange={e=>setRepeats(e.target.value)}/>
     </label>
     <label>Agent setup
      <select value="tools" disabled>
       <option value="tools">Clean LLM + Tools Baseline</option>
      </select>
     </label>
     <label>Provider
      <select value={provider} onChange={e=>setProvider(e.target.value)}>
       <option value="openai">OpenAI</option>
       <option value="heuristic">Heuristic smoke test</option>
      </select>
     </label>
     <button disabled={loading||!dataset?.ready} onClick={run}>
      {loading?"Running…":"Run benchmark"}
     </button>
    </div>
   </section>

   {error&&<div className="error">{error}</div>}

   {loading&&job&&<section className="panel progress-panel">
    <div className="panel-head">
     <div><h2>Live progress</h2><p>Actual model/tool execution progress.</p></div>
     <div className="progress-badge">{job.status}</div>
    </div>

    <div className="progress-top">
     <div>
      <div className="big-progress">{num(job.progress_pct,1)}%</div>
      <div className="subtle">{job.completed_runs||0}/{job.total_runs||0} runs · elapsed {fmt(job.elapsed_seconds)}</div>
     </div>
     <div className="progress-stats">
      <div><small>SUCCESS</small><b>{job.successes_so_far||0}</b></div>
      <div><small>FAILED</small><b>{job.failures_so_far||0}</b></div>
      <div><small>INPUT TOKENS</small><b>{job.input_tokens_so_far||0}</b></div>
      <div><small>OUTPUT TOKENS</small><b>{job.output_tokens_so_far||0}</b></div>
      <div><small>TOOL CALLS</small><b>{job.tool_calls_so_far||0}</b></div>
     </div>
    </div>

    <div className="live-progress"><i style={{width:`${job.progress_pct||0}%`}}/></div>

    <div className="current-work">
     <div><small>STAGE</small><b>{String(job.current_stage||"").replaceAll("_"," ")}</b></div>
     <div><small>CURRENT TASK</small><span>{job.current_task_text||"Waiting..."}</span></div>
    </div>

    <div className="live-log">
     <div className="live-log-head"><b>Live run log</b><span>updates every second</span></div>
     <div className="log-body">{(job.logs||[]).slice(-15).map((x,i)=><div key={i}>{x}</div>)}</div>
    </div>
   </section>}

   {s&&<>
    <div className="section-title"><h2>1. Outcome & Reliability</h2></div>
    <section className="metrics-grid">
     <Metric label="Task success" value={pct(s.task_success_rate)} hint="Final action outcome + required communication"/>
     <Metric label={`pass@${s.repeats}`} value={pct(s.pass_at_k)} hint="At least one success"/>
     <Metric label={`pass^${s.repeats}`} value={pct(s.pass_power_k)} hint="Every repeat succeeds"/>
     <Metric label="Repeat consistency" value={pct(s.repeat_consistency)} hint={s.repeats===1?"Not meaningful with one repeat":"Stable outcomes"}/>
     <Metric label="Observable policy" value={pct(s.observable_policy_compliance_rate)} hint="Trace-checkable rules"/>
    </section>

    <div className="section-title"><h2>2. Tool & Action Quality</h2><p>Final Action measures outcome. Tool/argument metrics diagnose how the agent got there.</p></div>
    <section className="metrics-grid">
     <Metric label="Final action score" value={pct(s.mean_final_action_score)} hint="Expected side effect matched, independent of lookup path"/>
     <Metric label="Tool precision" value={pct(s.mean_tool_precision)} hint="Expected among actual"/>
     <Metric label="Tool recall" value={pct(s.mean_tool_recall)} hint="Reference tools recovered"/>
     <Metric label="Tool F1" value={pct(s.mean_tool_f1)} hint="Balanced tool score"/>
     <Metric label="Argument accuracy" value={pct(s.mean_argument_accuracy)} hint="Gold argument match"/>
     <Metric label="Exact action recall" value={pct(s.mean_exact_action_recall)} hint="Tool + arguments"/>
     <Metric label="Communication recall" value={pct(s.mean_communication_recall)} hint="Required info in final answer"/>
     <Metric label="Useful tool ratio" value={pct(s.mean_useful_tool_ratio)} hint="Successful executions"/>
     <Metric label="Tool errors / run" value={num(s.tool_errors_per_run,2)} hint="Rejected by environment"/>
    </section>

    <div className="section-title">
     <h2>3. Tokens & Efficiency</h2>
     <p>Output caps are headroom; reasoning tokens count toward output usage.</p>
    </div>
    <section className="metrics-grid">
     <Metric label="Input / run" value={num(s.mean_input_tokens,0)} hint={`P95 ${num(s.p95_input_tokens,0)}`}/>
     <Metric label="Output / run" value={num(s.mean_output_tokens,0)} hint={`P95 ${num(s.p95_output_tokens,0)}`}/>
     <Metric label="Reasoning / run" value={num(s.mean_reasoning_tokens,0)} hint="When reported by model API"/>
     <Metric label="Total input" value={s.total_input_tokens} hint={`Cached ${s.total_cached_input_tokens}`}/>
     <Metric label="Total output" value={s.total_output_tokens} hint="All runs combined"/>
     <Metric label="LLM turns / run" value={num(s.mean_llm_turns,1)} hint="Max 9"/>
     <Metric label="Tool calls / run" value={num(s.mean_tool_calls,1)} hint="Actual calls"/>
     <Metric label="Mean latency" value={`${num(s.mean_latency_ms,0)} ms`} hint={`P95 ${num(s.p95_latency_ms,0)} ms`}/>
    </section>

    <section className="two-col">
     <div className="panel">
      <div className="panel-head">
       <div><h2>Runs</h2><p>Click a run to inspect the exact trajectory.</p></div>
       <a className="download" href={`${API}/api/benchmarks/${s.benchmark_id}/runs.csv`}>Export CSV</a>
      </div>
      <div className="table-wrap">
       <table>
        <thead><tr><th>Task</th><th>Outcome</th><th>Final Action</th><th>Tool F1</th><th>Args</th><th>Tokens In/Out</th><th>Termination</th></tr></thead>
        <tbody>{result.runs.map(r=>
         <tr key={r.run_id} onClick={()=>setSelectedRun(r)} className={selectedRun?.run_id===r.run_id?"selected":""}>
          <td>{r.task_id}</td>
          <td><Pill ok={r.success}>{r.success?"Success":"Failed"}</Pill></td>
          <td>{pct(r.final_action_score)}</td>
          <td>{pct(r.tool_f1)}</td>
          <td>{pct(r.argument_accuracy)}</td>
          <td>{r.input_tokens}/{r.output_tokens}</td>
          <td>{r.termination_reason}</td>
         </tr>
        )}</tbody>
       </table>
      </div>
     </div>

     <div className="panel">
      <div className="panel-head"><div><h2>Run inspector</h2><p>Reference vs actual tool path.</p></div></div>

      {selectedRun&&<>
       <div className="trace-meta">
        <Pill ok={selectedRun.success}>{selectedRun.success?"SUCCESS":"FAILED"}</Pill>
        <span>Task {selectedRun.task_id}</span>
        <span>{selectedRun.llm_turns} LLM turns</span>
       </div>

       <div className="run-budget-grid">
        <div><small>INPUT</small><b>{selectedRun.input_tokens}</b></div>
        <div><small>OUTPUT</small><b>{selectedRun.output_tokens}</b></div>
        <div><small>REASONING</small><b>{selectedRun.reasoning_tokens||0}</b></div>
        <div><small>OUTPUT CAP</small><b>{selectedRun.output_budget_max}</b></div>
        <div><small>TOOLS</small><b>{selectedRun.tool_call_count}</b></div>
       </div>

       <div className="compare">
        <div>
         <small>REFERENCE TOOLS</small>
         {selectedRun.expected_tools.map((x,i)=><code key={i}>{x}</code>)}
        </div>
        <div>
         <small>ACTUAL TOOLS</small>
         {selectedRun.actual_tools.length
          ? selectedRun.actual_tools.map((x,i)=><code key={i}>{x}</code>)
          : <span>None</span>}
        </div>
       </div>

       {selectedRun.failure_reasons?.length>0&&
        <div className="reason-box">
         {selectedRun.failure_reasons.map(x=><span key={x}>{x.replaceAll("_"," ")}</span>)}
        </div>
       }

       <div className="timeline">
        {selectedRun.trace.map((t,i)=>
         <div className="trace-step" key={i}>
          <div className="dot"></div>
          <div>
           <small>{String(t.type).toUpperCase()}</small>
           <b>{t.name||t.message||t.reason||`Step ${i+1}`}</b>
           {t.arguments&&<pre>{JSON.stringify(t.arguments,null,2)}</pre>}
           {t.result&&<pre>{JSON.stringify(t.result,null,2)}</pre>}
          </div>
         </div>
        )}
       </div>
      </>}
     </div>
    </section>

    <section className="panel">
     <div className="panel-head">
      <div><h2>Failure analysis</h2><p>{failures.length} failed/diagnostic runs.</p></div>
     </div>
     <div className="table-wrap">
      <table>
       <thead><tr><th>Task</th><th>Reasons</th><th>Termination</th><th>Tool calls</th><th>Output tokens</th></tr></thead>
       <tbody>{failures.map(r=>
        <tr key={r.run_id} onClick={()=>setSelectedRun(r)}>
         <td>{r.task_id}</td>
         <td>{r.failure_reasons.join(", ")}</td>
         <td>{r.termination_reason}</td>
         <td>{r.tool_call_count}</td>
         <td>{r.output_tokens}</td>
        </tr>
       )}</tbody>
      </table>
     </div>
    </section>
   </>}

   <section className="panel">
    <div className="panel-head"><div><h2>Experiment history</h2><p>Saved benchmark summaries.</p></div></div>
    <div className="history-grid">
     {history.slice(0,12).map(h=>
      <div className="history-card" key={h.benchmark_id}>
       <div><b>{h.configuration}</b><span>{h.provider}</span></div>
       <strong>{pct(h.task_success_rate)}</strong>
       <small>{h.task_count} tasks × {h.repeats}</small>
      </div>
     )}
    </div>
   </section>
  </main>
 </div>
}

createRoot(document.getElementById("root")).render(<App/>);
