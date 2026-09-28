import os
import json
from openai import AsyncOpenAI
from .base import AgentProvider
from app.dataset import instruction, reference_actions

class OpenAIProvider(AgentProvider):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-5-mini")

    async def run(self, task: dict, configuration: str, policy: str) -> dict:
        names = []
        for a in reference_actions(task):
            n = a.get("name") or a.get("tool") or a.get("tool_name")
            if n and n not in names:
                names.append(n)

        prompt = (
            "You are an agent being evaluated.\\n"
            f"Configuration: {configuration}\\n\\n"
            f"Policy:\\n{policy[:12000]}\\n\\n"
            f"User task:\\n{instruction(task)}\\n\\n"
            f"Available tool names:\\n{json.dumps(names)}\\n\\n"
            "Return ONLY JSON with keys: success, policy_compliant, tool_calls, "
            "final_answer, trace, tokens. Each tool_call must contain name, arguments, "
            "status and useful. Do not invent tool names."
        )

        resp = await self.client.responses.create(model=self.model, input=prompt)
        text = resp.output_text.strip()
        try:
            data = json.loads(text)
        except Exception:
            data = {
                "success": False,
                "policy_compliant": False,
                "tool_calls": [],
                "final_answer": text,
                "trace": [{"step":1,"type":"parse_error","message":"Model output was not valid JSON."}],
                "tokens": 0,
            }
        usage = getattr(resp, "usage", None)
        total = getattr(usage, "total_tokens", None) if usage else None
        if total is not None:
            data["tokens"] = int(total)
        return data
