import os
import json

from openai import AsyncOpenAI

from .base import AgentProvider
from app.dataset import instruction
from app.tooling import RetailToolExecutor, TOOL_SCHEMAS
from app.user_simulator import LLMUserSimulator


class OpenAIProvider(AgentProvider):
    """
    Paper-style baseline:
      user simulator <-> agent <-> real tools <-> DB
    """

    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-5-mini")

        self.max_steps = int(os.getenv("MAX_AGENT_STEPS", "12"))
        self.max_output_per_turn = int(os.getenv("MAX_OUTPUT_TOKENS_PER_TURN", "1200"))
        self.max_output_per_run = int(os.getenv("MAX_OUTPUT_TOKENS_PER_RUN", "6000"))
        self.max_task_chars = int(os.getenv("MAX_TASK_INPUT_CHARS", "4000"))
        self.max_policy_chars = int(os.getenv("MAX_POLICY_CHARS", "8000"))
        self.max_tool_result_chars = int(os.getenv("MAX_TOOL_RESULT_CHARS", "3500"))
        self.reasoning_effort = os.getenv("REASONING_EFFORT", "low")

    def _usage(self, resp):
        u = getattr(resp, "usage", None)
        inp = int(getattr(u, "input_tokens", 0) or 0) if u else 0
        out = int(getattr(u, "output_tokens", 0) or 0) if u else 0
        idet = getattr(u, "input_tokens_details", None) if u else None
        cached = int(getattr(idet, "cached_tokens", 0) or 0) if idet else 0
        odet = getattr(u, "output_tokens_details", None) if u else None
        reasoning = int(getattr(odet, "reasoning_tokens", 0) or 0) if odet else 0
        return inp, out, cached, reasoning

    def _response_diagnostics(self, resp):
        status = getattr(resp, "status", None)
        incomplete = getattr(resp, "incomplete_details", None)
        reason = getattr(incomplete, "reason", None) if incomplete is not None else None
        return status, reason

    async def run(self, task, configuration, policy, db=None):
        executor = RetailToolExecutor(
            db or {"products": {}, "users": {}, "orders": {}}
        )

        task_text = instruction(task)[:self.max_task_chars]
        policy_text = (policy or "")[:self.max_policy_chars]
        simulator = LLMUserSimulator(task_text)

        system = (
            "You are a retail support agent under evaluation. "
            "Solve the customer's request using the available tools and retail policy. "
            "Use tools for all database facts and actions. Do not invent IDs or state. "
            "You may call tools over multiple turns. If policy requires confirmation, "
            "ask the customer and wait for their reply. When the user's request is done, "
            "give a concise final answer. Do not narrate hidden reasoning.\n\n"
            "RETAIL POLICY:\n" + policy_text
        )

        input_items = [
            {"role": "system", "content": system},
            {"role": "user", "content": task_text},
        ]
        previous_response_id = None
        trace = []

        input_tokens = output_tokens = cached_tokens = reasoning_tokens = 0
        final_answer = ""
        termination_reason = "completed"
        llm_turns = 0
        response_statuses = []
        incomplete_reasons = []

        for step in range(1, self.max_steps + 1):
            remaining = self.max_output_per_run - output_tokens
            if remaining <= 0:
                termination_reason = "output_token_budget_exceeded"
                break

            kwargs = {
                "model": self.model,
                "input": input_items,
                "tools": TOOL_SCHEMAS,
                "max_output_tokens": min(self.max_output_per_turn, remaining),
                "parallel_tool_calls": False,
                "reasoning": {"effort": self.reasoning_effort},
            }
            if previous_response_id:
                kwargs["previous_response_id"] = previous_response_id

            resp = await self.client.responses.create(**kwargs)
            previous_response_id = resp.id
            llm_turns += 1

            i, o, c, r = self._usage(resp)
            input_tokens += i
            output_tokens += o
            cached_tokens += c
            reasoning_tokens += r

            status, incomplete_reason = self._response_diagnostics(resp)
            response_statuses.append(status)
            if incomplete_reason:
                incomplete_reasons.append(incomplete_reason)

            if status == "incomplete" and incomplete_reason == "max_output_tokens":
                trace.append({
                    "step": step,
                    "type": "incomplete",
                    "reason": "max_output_tokens",
                })
                termination_reason = "max_output_tokens"
                break

            calls = [
                x for x in getattr(resp, "output", [])
                if getattr(x, "type", None) == "function_call"
            ]

            if calls:
                next_input = []
                for call in calls:
                    try:
                        args = json.loads(call.arguments or "{}")
                    except Exception:
                        args = {}

                    trace.append({
                        "step": step,
                        "type": "tool_call",
                        "name": call.name,
                        "arguments": args,
                    })

                    result = executor.execute(call.name, args)

                    trace.append({
                        "step": step,
                        "type": "tool_result",
                        "name": call.name,
                        "result": result,
                    })

                    compact = json.dumps(
                        result, ensure_ascii=False, separators=(",", ":")
                    )[:self.max_tool_result_chars]

                    next_input.append({
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": compact,
                    })

                input_items = next_input
                continue

            agent_text = (getattr(resp, "output_text", "") or "").strip()
            trace.append({
                "step": step,
                "type": "assistant_message",
                "message": agent_text[:1200],
            })

            user_reply = await simulator.reply(agent_text)

            if user_reply == "__END__":
                final_answer = agent_text
                break

            trace.append({
                "step": step,
                "type": "user_simulator",
                "message": user_reply[:800],
            })

            input_items = [
                {"role": "user", "content": user_reply}
            ]
        else:
            termination_reason = "max_steps_exceeded"

        return {
            "tool_calls": [
                {k: v for k, v in call.items()
                 if k in {"name", "arguments", "status", "useful"}}
                for call in executor.calls
            ],
            "final_answer": final_answer,
            "trace": trace,

            # Agent tokens
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_tokens,
            "reasoning_tokens": reasoning_tokens,

            # User-simulator tokens
            "simulator_input_tokens": simulator.input_tokens,
            "simulator_output_tokens": simulator.output_tokens,
            "simulator_reasoning_tokens": simulator.reasoning_tokens,

            "tokens": (
                input_tokens + output_tokens
                + simulator.input_tokens + simulator.output_tokens
            ),

            "llm_turns": llm_turns,
            "user_simulator_turns": simulator.turns,
            "termination_reason": termination_reason,
            "response_statuses": response_statuses,
            "incomplete_reasons": incomplete_reasons,

            # Critical for final-state evaluation
            "environment": executor.db,

            "token_budget": {
                "max_output_per_turn": self.max_output_per_turn,
                "max_output_per_run": self.max_output_per_run,
                "output_used": output_tokens,
                "output_remaining": max(0, self.max_output_per_run-output_tokens),
            },

            "routing": {
                "tool_count": len(TOOL_SCHEMAS),
                "tool_names": [x["name"] for x in TOOL_SCHEMAS],
                "policy_chars": len(policy_text),
                "task_chars": len(task_text),
            },
        }
