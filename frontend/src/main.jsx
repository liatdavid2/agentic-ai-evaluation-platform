import React,{useEffect,useMemo,useRef,useState}from"react";
import{createRoot}from"react-dom/client";
import"./styles.css";
const API=import.meta.env.VITE_API_BASE_URL||"http://localhost:8000";
const pct=v=>`${Math.round((v||0)*100)}%`,num=(v,d=1)=>Number(v||0).toFixed(d);
const fmt=s=>{s=Math.floor(Number(s||0));return `${String(Math.floor(s/60)).padStart(2,"0")}:${String(s%60).padStart(2,"0")}`};
function Pill({ok,children}){return <span className={`pill ${ok?"ok":"bad"}`}>{children}</span>}
function Metric({label,value,hint}){return <div className="metric"><div className="metric-label">{label}</div><div className="metric-value">{value}</div><div className="metric-hint">{hint}</div></div>}

function App(){
 const[dataset,setDataset]=useState(null),[history,setHistory]=useState([]),[result,setResult]=useState(null),[selectedRun,setSelectedRun]=useState(null);
 const[taskCount,setTaskCount]=useState(5),[repeats,setRepeats]=useState(1),[configuration,setConfiguration]=useState("tools"),[provider,setProvider]=useState("openai");
 const[loading,setLoading]=useState(false),[job,setJob]=useState(null),[error,setError]=useState("");const poll=useRef(null);

 const refresh=async()=>{try{const[d,h]=await Promise.all([fetch(`${API}/api/dataset/status`).then(r=>r.json()),fetch(`${API}/api/history`).then(r=>r.json())]);setDataset(d);setHistory(Array.isArray(h)?h:[])}catch(e){setError(String(e))}};
 useEffect(()=>{refresh();return()=>poll.current&&clearInterval(poll.current)},[]);

 const pollJob=id=>{poll.current=setInterval(async()=>{try{const r=await fetch(`${API}/api/benchmark/jobs/${id}`),j=await r.json();if(!r.ok)throw new Error(j.detail);setJob(j);
  if(j.status==="completed"){clearInterval(poll.current);const rr=await fetch(`${API}/api/benchmark/jobs/${id}/result`),b=await rr.json();if(!rr.ok)throw new Error(b.detail);setResult(b);setSelectedRun(b.runs?.[0]||null);setLoading(false);refresh()}
  if(j.status==="failed"){clearInterval(poll.current);setError(j.error||"Failed");setLoading(false)}}catch(e){clearInterval(poll.current);setError(e.message);setLoading(false)}},1000)};

 const run=async()=>{setLoading(true);setResult(null);setError("");const r=await fetch(`${API}/api/benchmark/start`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({task_count:+taskCount,repeats:+repeats,configuration,provider})});const b=await r.json();if(!r.ok){setError(b.detail);setLoading(false);return}setJob({job_id:b.job_id,status:"queued",completed_runs:0,total_runs:+taskCount*+repeats,progress_pct:0,logs:[]});pollJob(b.job_id)};
 const s=result?.summary,failures=useMemo(()=>result?.runs?.filter(r=>r.failure_reasons?.length)||[],[result]);

 return <div className="app"><aside className="sidebar">
  <div className="brand"><div className="logo">AE</div><div><strong>Agent Eval</strong><span>Token-Efficient Evaluation</span></div></div>
  <nav><a className="active">Evaluation Lab</a><a href={`${API}/docs`} target="_blank">API Docs ↗</a></nav>
  <div className="side-card"><div className="side-title">Dataset</div>{dataset&&<><div className="status-row"><span>Retail</span><Pill ok={dataset.ready}>{dataset.ready?"Ready":"Missing"}</Pill></div><div className="status-row"><span>Tasks</span><b>{dataset.task_count}</b></div><div className="status-row"><span>Products</span><b>{dataset.product_count}</b></div><div className="status-row"><span>Users</span><b>{dataset.user_count}</b></div><div className="status-row"><span>Orders</span><b>{dataset.order_count}</b></div></>}</div>
 </aside>

 <main><header><div><div className="eyebrow">GENAI / AGENTIC AI</div><h1>Deep Evaluation Lab</h1><p>Real tool calling with strict output budgets, routed policies and routed tool sets.</p></div><div className="header-badge">τ²-bench Retail</div></header>

 <section className="budget-card">
  <div><strong>Default token controls</strong><span>Per-turn output ≤ 80 · Per-run output ≤ 180 · Max 7 agent turns · Routed policy/tools</span></div>
  <div className="budget-tip">Goal: keep output small while preserving tool-use quality.</div>
 </section>

 <section className="panel"><div className="panel-head"><div><h2>Benchmark configuration</h2><p>Start small, then scale after token use looks reasonable.</p></div></div><div className="controls">
  <label>Tasks<input type="number" min="1" max="114" value={taskCount} onChange={e=>setTaskCount(e.target.value)}/></label>
  <label>Repeats / task<input type="number" min="1" max="10" value={repeats} onChange={e=>setRepeats(e.target.value)}/></label>
  <label>Agent setup<select value={configuration} onChange={e=>setConfiguration(e.target.value)}><option value="baseline">Baseline</option><option value="tools">LLM + Tools</option><option value="reflection">Tools + Reflection</option><option value="multi_agent">Multi-Agent</option></select></label>
  <label>Provider<select value={provider} onChange={e=>setProvider(e.target.value)}><option value="openai">OpenAI</option><option value="heuristic">Heuristic smoke test</option></select></label>
  <button disabled={loading||!dataset?.ready} onClick={run}>{loading?"Running…":"Run benchmark"}</button>
 </div></section>{error&&<div className="error">{error}</div>}

 {loading&&job&&<section className="panel progress-panel"><div className="panel-head"><div><h2>Live progress</h2><p>Runtime, tokens and tools update after every completed run.</p></div><div className="progress-badge">{job.status}</div></div>
  <div className="progress-top"><div><div className="big-progress">{num(job.progress_pct,1)}%</div><div className="subtle">{job.completed_runs||0}/{job.total_runs||0} runs · elapsed {fmt(job.elapsed_seconds)}</div></div>
   <div className="progress-stats"><div><small>SUCCESS</small><b>{job.successes_so_far||0}</b></div><div><small>FAILED</small><b>{job.failures_so_far||0}</b></div><div><small>INPUT TOKENS</small><b>{job.input_tokens_so_far||0}</b></div><div><small>OUTPUT TOKENS</small><b>{job.output_tokens_so_far||0}</b></div><div><small>TOOL CALLS</small><b>{job.tool_calls_so_far||0}</b></div></div></div>
  <div className="live-progress"><i style={{width:`${job.progress_pct||0}%`}}/></div>
  <div className="current-work"><div><small>STAGE</small><b>{String(job.current_stage||"").replaceAll("_"," ")}</b></div><div><small>CURRENT TASK</small><span>{job.current_task_text||"Waiting..."}</span></div></div>
  <div className="budget-usage"><span>Last run output budget used</span><b>{pct(job.last_output_budget_pct||0)}</b></div>
  <div className="live-log"><div className="live-log-head"><b>Live run log</b><span>updates every second</span></div><div className="log-body">{(job.logs||[]).slice(-15).map((x,i)=><div key={i}>{x}</div>)}</div></div>
 </section>}

 {s&&<>
  <div className="section-title"><h2>1. Outcome & Reliability</h2></div><section className="metrics-grid">
   <Metric label="Task success" value={pct(s.task_success_rate)} hint="Benchmark-grounded"/>
   <Metric label={`pass@${s.repeats}`} value={pct(s.pass_at_k)} hint="At least one success"/>
   <Metric label={`pass^${s.repeats}`} value={pct(s.pass_power_k)} hint="Every repeat succeeds"/>
   <Metric label="Repeat consistency" value={pct(s.repeat_consistency)} hint="Stable result"/>
   <Metric label="Observable policy" value={pct(s.observable_policy_compliance_rate)} hint="Trace-checkable rules"/>
  </section>

  <div className="section-title"><h2>2. Tool & Action Quality</h2></div><section className="metrics-grid">
   <Metric label="Tool precision" value={pct(s.mean_tool_precision)} hint="Expected among actual"/>
   <Metric label="Tool recall" value={pct(s.mean_tool_recall)} hint="Gold tools recovered"/>
   <Metric label="Tool F1" value={pct(s.mean_tool_f1)} hint="Balanced tool score"/>
   <Metric label="Argument accuracy" value={pct(s.mean_argument_accuracy)} hint="Gold argument match"/>
   <Metric label="Exact action recall" value={pct(s.mean_exact_action_recall)} hint="Tool + arguments"/>
   <Metric label="Communication recall" value={pct(s.mean_communication_recall)} hint="Required info in answer"/>
   <Metric label="Useful tool ratio" value={pct(s.mean_useful_tool_ratio)} hint="Successful tool executions"/>
   <Metric label="Tool errors / run" value={num(s.tool_errors_per_run,2)} hint="Rejected by environment"/>
  </section>

  <div className="section-title"><h2>3. Token Budget & Efficiency</h2><p>This section makes the cost controls explicit.</p></div><section className="metrics-grid">
   <Metric label="Input / run" value={num(s.mean_input_tokens,0)} hint={`P95 ${num(s.p95_input_tokens,0)}`}/>
   <Metric label="Output / run" value={num(s.mean_output_tokens,0)} hint={`P95 ${num(s.p95_output_tokens,0)} · hard cap 180`}/>
   <Metric label="Output budget used" value={pct(s.mean_output_budget_used_pct)} hint="Average fraction of the 180-token run cap"/>
   <Metric label="Total input" value={s.total_input_tokens} hint={`Cached ${s.total_cached_input_tokens}`}/>
   <Metric label="Total output" value={s.total_output_tokens} hint="All runs combined"/>
   <Metric label="LLM turns / run" value={num(s.mean_llm_turns,1)} hint="Max 7"/>
   <Metric label="Tools exposed / run" value={num(s.mean_routed_tool_count,1)} hint="Routed subset, not all tools"/>
   <Metric label="Policy chars / run" value={num(s.mean_routed_policy_chars,0)} hint="Routed policy only"/>
   <Metric label="Mean latency" value={`${num(s.mean_latency_ms,0)} ms`} hint={`P95 ${num(s.p95_latency_ms,0)} ms`}/>
   <Metric label="Cost / success" value={`$${num(s.cost_per_success_usd,4)}`} hint="0 until pricing env vars set"/>
  </section>

  <section className="two-col"><div className="panel"><div className="panel-head"><div><h2>Runs</h2><p>Click a run to inspect exact token use and tool calls.</p></div><a className="download" href={`${API}/api/benchmarks/${s.benchmark_id}/runs.csv`}>Export CSV</a></div>
   <div className="table-wrap"><table><thead><tr><th>Task</th><th>Outcome</th><th>Tool F1</th><th>Args</th><th>Tokens In/Out</th><th>Budget</th></tr></thead><tbody>{result.runs.map(r=><tr key={r.run_id} onClick={()=>setSelectedRun(r)} className={selectedRun?.run_id===r.run_id?"selected":""}><td>{r.task_id}</td><td><Pill ok={r.success}>{r.success?"Success":"Failed"}</Pill></td><td>{pct(r.tool_f1)}</td><td>{pct(r.argument_accuracy)}</td><td>{r.input_tokens}/{r.output_tokens}</td><td>{pct(r.output_budget_used_pct)}</td></tr>)}</tbody></table></div></div>

   <div className="panel"><div className="panel-head"><div><h2>Run inspector</h2><p>Tool path, token budget and final result.</p></div></div>{selectedRun&&<>
    <div className="trace-meta"><Pill ok={selectedRun.success}>{selectedRun.success?"SUCCESS":"FAILED"}</Pill><span>Task {selectedRun.task_id}</span><span>{selectedRun.llm_turns} LLM turns</span></div>
    <div className="run-budget-grid"><div><small>INPUT</small><b>{selectedRun.input_tokens}</b></div><div><small>OUTPUT</small><b>{selectedRun.output_tokens}</b></div><div><small>OUTPUT CAP</small><b>{selectedRun.output_budget_max}</b></div><div><small>USED</small><b>{pct(selectedRun.output_budget_used_pct)}</b></div><div><small>TOOLS EXPOSED</small><b>{selectedRun.routed_tool_count}</b></div></div>
    <div className="compare"><div><small>REFERENCE TOOLS</small>{selectedRun.expected_tools.map((x,i)=><code key={i}>{x}</code>)}</div><div><small>ACTUAL TOOLS</small>{selectedRun.actual_tools.length?selectedRun.actual_tools.map((x,i)=><code key={i}>{x}</code>):<span>None</span>}</div></div>
    {selectedRun.failure_reasons?.length>0&&<div className="reason-box">{selectedRun.failure_reasons.map(x=><span key={x}>{x.replaceAll("_"," ")}</span>)}</div>}
    <div className="timeline">{selectedRun.trace.map((t,i)=><div className="trace-step" key={i}><div className="dot"></div><div><small>{String(t.type).toUpperCase()}</small><b>{t.name||t.message||`Step ${i+1}`}</b>{t.arguments&&<pre>{JSON.stringify(t.arguments,null,2)}</pre>}{t.result&&<pre>{JSON.stringify(t.result,null,2)}</pre>}</div></div>)}</div>
   </>}</div></section>

   <section className="panel"><div className="panel-head"><div><h2>Failure analysis</h2><p>{failures.length} runs contain one or more failures.</p></div></div>
    <div className="table-wrap"><table><thead><tr><th>Task</th><th>Reasons</th><th>Termination</th><th>Tool calls</th><th>Output tokens</th></tr></thead><tbody>{failures.map(r=><tr key={r.run_id} onClick={()=>setSelectedRun(r)}><td>{r.task_id}</td><td>{r.failure_reasons.join(", ")}</td><td>{r.termination_reason}</td><td>{r.tool_call_count}</td><td>{r.output_tokens}/{r.output_budget_max}</td></tr>)}</tbody></table></div>
   </section>
 </>}

 <section className="panel"><div className="panel-head"><div><h2>Experiment history</h2><p>Saved benchmark summaries.</p></div></div><div className="history-grid">{history.slice(0,12).map(h=><div className="history-card" key={h.benchmark_id}><div><b>{h.configuration}</b><span>{h.provider}</span></div><strong>{pct(h.task_success_rate)}</strong><small>{h.task_count} tasks × {h.repeats}</small></div>)}</div></section>
 </main></div>
}
createRoot(document.getElementById("root")).render(<App/>);
