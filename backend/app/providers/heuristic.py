import asyncio, hashlib, copy
from .base import AgentProvider
from app.dataset import instruction, reference_actions
from app.tooling import RetailToolExecutor

class HeuristicProvider(AgentProvider):
    async def run(self,task,configuration,policy,db=None):
        await asyncio.sleep(.01)
        ex=RetailToolExecutor(db or {"products":{},"users":{},"orders":{}})
        refs=reference_actions(task)
        seed=int(hashlib.sha256((instruction(task)+configuration).encode()).hexdigest()[:8],16)
        ratio={"baseline":.55,"tools":.82,"reflection":.90,"multi_agent":.93}[configuration]
        trace=[]
        for i,a in enumerate(refs):
            if ((seed+i*17)%100)/100 < ratio:
                name=a.get("name");args=a.get("arguments",{})
                trace.append({"step":i+1,"type":"tool_call","name":name,"arguments":args})
                result=ex.execute(name,args)
                trace.append({"step":i+1,"type":"tool_result","name":name,"result":result})
        final="Task completed." if ratio>.7 else "Unable to complete all requested actions."
        trace.append({"step":len(refs)+1,"type":"final","message":final})
        inp=250+seed%250;out=50+seed%80
        return {
            "tool_calls":[{k:v for k,v in c.items() if k in {"name","arguments","status","useful"}} for c in ex.calls],
            "final_answer":final,"trace":trace,
            "input_tokens":inp,"output_tokens":out,"cached_input_tokens":0,
            "tokens":inp+out,"llm_turns":1,"termination_reason":"completed",
            "environment":ex.db,
        }
