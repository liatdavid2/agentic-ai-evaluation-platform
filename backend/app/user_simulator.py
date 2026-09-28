import os
from openai import AsyncOpenAI


class LLMUserSimulator:
    """
    LLM-based user simulator for multi-turn benchmark execution.

    The simulator receives the task's user instructions as its private script.
    It plays only the user, follows conditional instructions in the scenario,
    and returns __END__ when the conversation should end.
    """

    def __init__(self, task_text: str):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        configured = os.getenv("USER_SIM_MODEL", "").strip()
        if configured in {"", "${OPENAI_MODEL}"}:
            configured = os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.model = configured
        self.max_turns = int(os.getenv("USER_SIM_MAX_TURNS", "6"))
        self.max_output_tokens = int(os.getenv("USER_SIM_MAX_OUTPUT_TOKENS", "180"))
        self.task_text = task_text
        self.turns = 0
        self.previous_response_id = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0

    def _usage(self, resp):
        u = getattr(resp, "usage", None)
        inp = int(getattr(u, "input_tokens", 0) or 0) if u else 0
        out = int(getattr(u, "output_tokens", 0) or 0) if u else 0
        od = getattr(u, "output_tokens_details", None) if u else None
        reasoning = int(getattr(od, "reasoning_tokens", 0) or 0) if od else 0
        return inp, out, reasoning

    async def reply(self, agent_message: str) -> str:
        if self.turns >= self.max_turns:
            return "__END__"

        system = (
            "You are simulating the USER in a retail customer-support benchmark. "
            "Follow the private scenario instructions exactly, including any conditional "
            "behavior such as changing your mind after a confirmation question. "
            "Do not act like the support agent. Do not use tools. Do not invent facts "
            "that the scenario does not give you. Keep replies concise and natural. "
            "If the user's goal is satisfied, the agent has clearly finished, or there "
            "is nothing further the scripted user should say, reply with exactly __END__.\n\n"
            "PRIVATE USER SCENARIO:\n"
            + self.task_text
        )

        kwargs = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Support agent said:\n{agent_message}\n\nReply as the user."},
            ],
            "max_output_tokens": self.max_output_tokens,
            "reasoning": {"effort": "low"},
        }

        if self.previous_response_id:
            kwargs["previous_response_id"] = self.previous_response_id
            kwargs["input"] = [
                {"role": "user", "content": f"Support agent said:\n{agent_message}\n\nReply as the user."}
            ]

        resp = await self.client.responses.create(**kwargs)
        self.previous_response_id = resp.id
        self.turns += 1

        i, o, r = self._usage(resp)
        self.input_tokens += i
        self.output_tokens += o
        self.reasoning_tokens += r

        text = (getattr(resp, "output_text", "") or "").strip()
        return text or "__END__"
