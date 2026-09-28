import React,{useEffect,useMemo,useState}from"react";
import{createRoot}from"react-dom/client";
import"./styles.css";
const API=import.meta.env.VITE_API_BASE_URL||"http://localhost:8000";
const pct=v=>`${Math.round((v||0)*100)}%`;
const num=(v,d=1)=>Number(v||0).toFixed(d);

function Pill({ok,children}){return <span className={`pill ${ok?"ok":"bad"}`}>{children}</span>}
function Metric({label,value,hint}){return <div className="metric"><div className="metric-label">{label}</div><div className="metric-value">{value}</div><div className="metric-hint">{hint}</div></div>}
function Bar({label,value}){
  return <div className="bar-row"><span>{label}</span><div className="bar"><i style={{width:`${Math.min(100,(value||0)*100)}%`}}/></div><b>{pct(value)}</b></div>
}

function App(){
 const[dataset,setDataset]=useState(null),[history,setHistory]=useState([]),[result,setResult]=useState(null),[selectedRun,setSelectedRun]=useState(null);
 const[taskCount,setTaskCount]=useState(20),[repeats,setRepeats]=useState(5),[configuration,setConfiguration]=useState("tools"),[provider,setProvider]=useState("heuristic");
 const[loading,setLoading]=useState(false),[error,setError]=useState("");

 const refresh=async()=>{try{
   const[d,h]=await Promise.all([fetch(`${API}/api/dataset/status`).then(r=>r.json()),fetch(`${API}/api/history`).then(r=>r.json())]);
   setDataset(d);setHistory(h);
 }catch(e){setError(String(e))}};
 useEffect(()=>{refresh()},[]);

 const run=async()=>{
   setLoading(true);setError("");
   try{
     const r=await fetch(`${API}/api/benchmark`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
       task_count:Number(taskCount),repeats:Number(repeats),configuration,provider
     })});
     const b=await r.json();if(!r.ok)throw new Error(b.detail||"Benchmark failed");
     setResult(b);setSelectedRun(b.runs?.[0]||null);await refresh();
   }catch(e){setError(e.message||String(e))}finally{setLoading(false)}
 };
 const s=result?.summary;
 const failures=useMemo(()=>result?.runs?.filter(x=>x.failure_reasons?.length).slice(0,100)||[],[result]);
 const failureEntries=s?Object.entries(s.failure_taxonomy||{}).sort((a,b)=>b[1]-a[1]):[];

 return <div className="app">
  <aside className="sidebar">
   <div className="brand"><div className="logo">AE</div><div><strong>Agent Eval</strong><span>Deep Evaluation Lab</span></div></div>
   <nav><a className="active">Evaluation Lab</a><a href="http://localhost:8000/docs" target="_blank">API Docs ↗</a></nav>
   <div className="side-card"><div className="side-title">Dataset</div>{dataset?<>
    <div className="status-row"><span>Retail tasks</span><Pill ok={dataset.tasks_json}>{dataset.tasks_json?"Ready":"Missing"}</Pill></div>
    <div className="status-row"><span>Tasks</span><b>{dataset.task_count}</b></div>
    <div className="status-row"><span>DB</span><b>{dataset.db_json?"✓":"—"}</b></div>
    <div className="status-row"><span>Policy</span><b>{dataset.policy_md?"✓":"—"}</b></div>
   </>:<span>Loading…</span>}</div>
  </aside>

  <main>
   <header><div><div className="eyebrow">GENAI / AGENTIC AI</div><h1>Deep Evaluation Lab</h1>
   <p>Measure outcome, reliability, policy adherence, tool behavior, trajectory quality and efficiency.</p></div><div className="header-badge">τ²-bench Retail</div></header>

   {!dataset?.ready&&<div className="notice"><strong>Dataset not added yet.</strong><span>Add the Retail files under <code>data/retail/</code>.</span></div>}

   <section className="panel">
    <div className="panel-head"><div><h2>Benchmark configuration</h2><p>Repeat each task to expose instability, not just one-shot capability.</p></div></div>
    <div className="controls">
     <label>Tasks<input type="number" min="1" max="500" value={taskCount} onChange={e=>setTaskCount(e.target.value)}/></label>
     <label>Repeats / task<input type="number" min="1" max="10" value={repeats} onChange={e=>setRepeats(e.target.value)}/></label>
     <label>Agent setup<select value={configuration} onChange={e=>setConfiguration(e.target.value)}><option value="baseline">Baseline</option><option value="tools">LLM + Tools</option><option value="reflection">Tools + Reflection</option><option value="multi_agent">Multi-Agent</option></select></label>
     <label>Provider<select value={provider} onChange={e=>setProvider(e.target.value)}><option value="heuristic">Heuristic / free smoke test</option><option value="openai">OpenAI</option></select></label>
     <button disabled={loading||!dataset?.ready} onClick={run}>{loading?"Running…":"Run benchmark"}</button>
    </div>
   </section>

   {error&&<div className="error">{error}</div>}

   {s&&<>
    <div className="section-title"><h2>1. Outcome & Reliability</h2><p>Can the agent solve the task, and can it do so consistently?</p></div>
    <section className="metrics-grid">
     <Metric label="Task success" value={pct(s.task_success_rate)} hint={`${s.total_runs} runs`}/>
     <Metric label={`pass@${s.repeats}`} value={pct(s.pass_at_k)} hint="At least one successful repeat"/>
     <Metric label={`pass^${s.repeats}`} value={pct(s.pass_power_k)} hint="All repeats successful"/>
     <Metric label="Repeat consistency" value={pct(s.repeat_consistency)} hint="Same success/fail outcome across repeats"/>
     <Metric label="Policy compliance" value={pct(s.policy_compliance_rate)} hint="Policy-safe runs"/>
    </section>

    <div className="section-title"><h2>2. Tool-use Quality</h2><p>Reference coverage plus argument correctness and wasted calls.</p></div>
    <section className="metrics-grid">
     <Metric label="Tool precision" value={pct(s.mean_tool_precision)} hint="Expected tools among called tools"/>
     <Metric label="Tool recall" value={pct(s.mean_tool_recall)} hint="Reference tools recovered"/>
     <Metric label="Tool F1" value={pct(s.mean_tool_f1)} hint="Balanced tool-use score"/>
     <Metric label="Argument accuracy" value={pct(s.mean_argument_accuracy)} hint="Expected argument key/value match"/>
     <Metric label="Useful tool ratio" value={pct(s.mean_useful_tool_ratio)} hint="Useful calls / all calls"/>
     <Metric label="Invalid / run" value={num(s.invalid_calls_per_run,2)} hint="Malformed or unknown tools"/>
     <Metric label="Redundant / run" value={num(s.redundant_calls_per_run,2)} hint="Repeated calls"/>
     <Metric label="Avg tool calls" value={num(s.mean_tool_calls,1)} hint="Efficiency"/>
    </section>

    <div className="section-title"><h2>3. Trajectory Evaluation</h2><p>How closely did the agent's path align with reference actions, without making outcome depend on one exact path?</p></div>
    <section className="metrics-grid">
     <Metric label="Order score" value={pct(s.mean_trajectory_order_score)} hint="LCS-based reference order coverage"/>
     <Metric label="Exact sequence" value={pct(s.exact_tool_sequence_rate)} hint="Strict diagnostic only"/>
     <Metric label="Extra tool ratio" value={pct(s.mean_extra_tool_ratio)} hint="Additional calls beyond reference"/>
     <Metric label="Avg steps" value={num(s.mean_steps,1)} hint="Trace length"/>
    </section>

    <div className="section-title"><h2>4. Production Efficiency</h2><p>Operational cost of getting the answer.</p></div>
    <section className="metrics-grid">
     <Metric label="Mean latency" value={`${num(s.mean_latency_ms,0)} ms`} hint={`Median ${num(s.median_latency_ms,0)} ms`}/>
     <Metric label="P95 latency" value={`${num(s.p95_latency_ms,0)} ms`} hint="Tail latency"/>
     <Metric label="Mean tokens" value={num(s.mean_tokens,0)} hint={`P95 ${num(s.p95_tokens,0)}`}/>
     <Metric label="Total cost" value={`$${num(s.total_cost_usd,4)}`} hint={`$${num(s.cost_per_success_usd,4)} / success`}/>
    </section>

    <section className="two-col">
     <div className="panel">
      <div className="panel-head"><div><h2>Failure taxonomy</h2><p>Where agent runs break down.</p></div></div>
      {failureEntries.length===0?<div className="success-box">No observed failure signals.</div>:<div className="failure-bars">
       {failureEntries.map(([k,v])=><div className="failure-row" key={k}><span>{k.replaceAll("_"," ")}</span><b>{v}</b></div>)}
      </div>}
     </div>
     <div className="panel">
      <div className="panel-head"><div><h2>Quality profile</h2><p>One view of the main normalized scores.</p></div></div>
      <Bar label="Outcome" value={s.task_success_rate}/><Bar label="Policy" value={s.policy_compliance_rate}/><Bar label="Tool F1" value={s.mean_tool_f1}/><Bar label="Arguments" value={s.mean_argument_accuracy}/><Bar label="Trajectory" value={s.mean_trajectory_order_score}/><Bar label="Reliability" value={s.pass_power_k}/>
     </div>
    </section>

    <section className="two-col">
     <div className="panel"><div className="panel-head"><div><h2>Runs</h2><p>Click a run for evidence-level inspection.</p></div>
      <a className="download" href={`${API}/api/benchmarks/${s.benchmark_id}/runs.csv`}>Export CSV</a></div>
      <div className="table-wrap"><table><thead><tr><th>Task</th><th>Run</th><th>Outcome</th><th>Tool F1</th><th>Args</th><th>Order</th></tr></thead>
      <tbody>{result.runs.slice(0,150).map(r=><tr key={r.run_id} onClick={()=>setSelectedRun(r)} className={selectedRun?.run_id===r.run_id?"selected":""}>
       <td>{r.task_id}</td><td>#{r.repeat}</td><td><Pill ok={r.success}>{r.success?"Success":"Failed"}</Pill></td><td>{pct(r.tool_f1)}</td><td>{pct(r.argument_accuracy)}</td><td>{pct(r.trajectory_order_score)}</td>
      </tr>)}</tbody></table></div>
     </div>

     <div className="panel"><div className="panel-head"><div><h2>Run inspector</h2><p>Reference vs actual trajectory and failure reasons.</p></div></div>
      {!selectedRun?<div className="empty">Select a run.</div>:<>
       <div className="trace-meta"><Pill ok={selectedRun.success}>{selectedRun.success?"SUCCESS":"FAILED"}</Pill><span>{selectedRun.task_id}</span><span>{selectedRun.latency_ms} ms</span></div>
       {selectedRun.failure_reasons?.length>0&&<div className="reason-box">{selectedRun.failure_reasons.map(x=><span key={x}>{x.replaceAll("_"," ")}</span>)}</div>}
       <div className="compare"><div><small>REFERENCE TOOLS</small>{selectedRun.expected_tools?.length?selectedRun.expected_tools.map((x,i)=><code key={i}>{x}</code>):<span>None</span>}</div>
       <div><small>ACTUAL TOOLS</small>{selectedRun.actual_tools?.length?selectedRun.actual_tools.map((x,i)=><code key={i}>{x}</code>):<span>None</span>}</div></div>
       <div className="timeline">{(selectedRun.trace||[]).map((t,i)=><div className="trace-step" key={i}><div className="dot"></div><div><small>{String(t.type||"step").toUpperCase()}</small><b>{t.name||t.message||`Step ${i+1}`}</b>{t.arguments&&<pre>{JSON.stringify(t.arguments,null,2)}</pre>}</div></div>)}</div>
      </>}
     </div>
    </section>

    <section className="panel"><div className="panel-head"><div><h2>Failure analysis table</h2><p>Detailed diagnostic signals for problematic runs.</p></div></div>
     {failures.length===0?<div className="success-box">No failures or diagnostic mismatches.</div>:<div className="table-wrap"><table><thead><tr><th>Task</th><th>Reasons</th><th>Precision</th><th>Recall</th><th>Args</th><th>Order</th><th>Invalid</th><th>Redundant</th></tr></thead>
     <tbody>{failures.map(r=><tr key={r.run_id} onClick={()=>setSelectedRun(r)}><td>{r.task_id}</td><td>{r.failure_reasons.join(", ")}</td><td>{pct(r.tool_precision)}</td><td>{pct(r.tool_recall)}</td><td>{pct(r.argument_accuracy)}</td><td>{pct(r.trajectory_order_score)}</td><td>{r.invalid_tool_calls}</td><td>{r.redundant_tool_calls}</td></tr>)}</tbody></table></div>}
    </section>
   </>}

   <section className="panel"><div className="panel-head"><div><h2>Experiment history</h2><p>Saved locally in SQLite for reproducible comparisons.</p></div></div>
    {history.length===0?<div className="empty">No experiments yet.</div>:<div className="history-grid">{history.slice(0,12).map(h=><div className="history-card" key={h.benchmark_id}><div><b>{h.configuration}</b><span>{h.provider}</span></div><strong>{pct(h.task_success_rate)}</strong><small>{h.task_count} tasks × {h.repeats} repeats · tool F1 {pct(h.mean_tool_f1)}</small></div>)}</div>}
   </section>
  </main>
 </div>
}
createRoot(document.getElementById("root")).render(<App/>);
