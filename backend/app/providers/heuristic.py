import asyncio
import hashlib
from .base import AgentProvider
from app.dataset import instruction, reference_actions

class HeuristicProvider(AgentProvider):
    async def run(self, task: dict, configuration: str, policy: str) -> dict:
        await asyncio.sleep(0.01)
        refs = reference_actions(task)
        seed = int(hashlib.sha256((instruction(task) + configuration).encode()).hexdigest()[:8], 16)
        keep_ratio = {"baseline":0.55, "tools":0.82, "reflection":0.90, "multi_agent":0.93}[configuration]

        actual = []
        for i, a in enumerate(refs):
            if ((seed + i * 17) % 100) / 100 < keep_ratio:
                actual.append({
                    "name": a.get("name") or a.get("tool") or a.get("tool_name") or "unknown_tool",
                    "arguments": a.get("arguments", {}),
                    "status": "ok",
                    "useful": True,
                })

        if configuration == "baseline" and seed % 4 == 0:
            actual.append({"name":"unknown_tool","arguments":{},"status":"invalid","useful":False})
        if configuration in {"tools","multi_agent"} and seed % 5 == 0 and actual:
            actual.append({**actual[-1], "useful":False})

        trace = [{"step":1,"type":"plan","message":f"Analyze task using {configuration}."}]
        for j, call in enumerate(actual, start=2):
            trace.append({"step":j,"type":"tool_call",**call})
        success = len(refs) == 0 or len([x for x in actual if x["name"] != "unknown_tool"]) >= max(1, int(len(refs)*0.75))
        trace.append({"step":len(trace)+1,"type":"final","message":"Task completed." if success else "Task failed."})

        return {
            "success": success,
            "policy_compliant": seed % 19 != 0,
            "tool_calls": actual,
            "final_answer": "Task completed." if success else "Task could not be completed reliably.",
            "trace": trace,
            "tokens": 350 + seed % 800,
        }
