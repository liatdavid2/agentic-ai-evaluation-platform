import os
import json

from openai import AsyncOpenAI

from .base import AgentProvider
from app.dataset import instruction
from app.tooling import RetailToolExecutor, filtered_schemas
from app.routing import policy_for_task, tool_names_for_task, detect_intents


class OpenAIProvider(AgentProvider):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-5-mini")

        self.max_steps = int(os.getenv("MAX_AGENT_STEPS", "9"))
        self.max_output_per_turn = int(
            os.getenv("MAX_OUTPUT_TOKENS_PER_TURN", "220")
        )
        self.max_output_per_run = int(
            os.getenv("MAX_OUTPUT_TOKENS_PER_RUN", "650")
        )

        self.max_task_chars = int(
            os.getenv("MAX_TASK_INPUT_CHARS", "1800")
        )
        self.max_policy_chars = int(
            os.getenv("MAX_POLICY_CHARS", "2200")
        )
        self.max_tool_result_chars = int(
            os.getenv("MAX_TOOL_RESULT_CHARS", "1800")
        )

        self.reasoning_effort = os.getenv(
            "REASONING_EFFORT",
            "low"
        )

    def _usage(self, resp):
        usage = getattr(resp, "usage", None)

        input_tokens = (
            int(getattr(usage, "input_tokens", 0) or 0)
            if usage else 0
        )
        output_tokens = (
            int(getattr(usage, "output_tokens", 0) or 0)
            if usage else 0
        )

        details = (
            getattr(usage, "input_tokens_details", None)
            if usage else None
        )
        cached_tokens = (
            int(getattr(details, "cached_tokens", 0) or 0)
            if details else 0
        )

        return input_tokens, output_tokens, cached_tokens

    async def run(self, task, configuration, policy, db=None):
        executor = RetailToolExecutor(
            db or {
                "products": {},
                "users": {},
                "orders": {},
            }
        )

        task_text = instruction(task)[:self.max_task_chars]

        routed_policy = policy_for_task(
            task_text,
            self.max_policy_chars
        )

        tool_names = tool_names_for_task(task_text)
        tools = filtered_schemas(tool_names)

        intents = detect_intents(task_text)

        action_intents = {
            "cancel",
            "modify_items",
            "modify_address",
            "modify_payment",
            "return",
            "exchange",
        }

        requires_action = any(
            intent in action_intents
            for intent in intents
        )

        system = (
            "You are a concise retail support agent under evaluation. "
            "Use tools for all database facts. "
            "Do not narrate reasoning. "
            "Make at most one tool call per turn. "
            "Continue using tools until the user's request is actually completed. "
            "Authentication alone is never task completion. "
            "Do not stop after identifying the user. "
            "Use tool results to resolve order ids, product variants, prices, "
            "availability, and payment details whenever possible. "
            "Do not ask the user for information that can be obtained from tools. "
            "Keep final answers very short. "
            "Follow the policy below.\n\n"
            + routed_policy
        )

        input_items = [
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": task_text,
            },
        ]

        previous_response_id = None
        trace = []

        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0

        final_answer = ""
        termination_reason = "completed"
        llm_turns = 0

        for step in range(1, self.max_steps + 1):

            remaining = (
                self.max_output_per_run
                - output_tokens
            )

            if remaining <= 0:
                termination_reason = (
                    "output_token_budget_exceeded"
                )
                break

            turn_budget = min(
                self.max_output_per_turn,
                remaining,
            )

            successful_mutation = any(
                call.get("name") in {
                    "cancel_pending_order",
                    "modify_pending_order_items",
                    "modify_pending_order_address",
                    "modify_pending_order_payment",
                    "return_delivered_order_items",
                    "exchange_delivered_order_items",
                    "modify_user_address",
                }
                and call.get("status") == "ok"
                for call in executor.calls
            )

            kwargs = {
                "model": self.model,
                "input": input_items,
                "tools": tools,
                "max_output_tokens": turn_budget,
                "reasoning": {
                    "effort": self.reasoning_effort
                },

                # Keep orchestration sequential. This reduces the chance that
                # multiple pending function calls must be satisfied at once.
                "parallel_tool_calls": False,
            }

            # Prevent premature final answers on action tasks.
            if requires_action and not successful_mutation:
                kwargs["tool_choice"] = "required"
            else:
                kwargs["tool_choice"] = "auto"

            if previous_response_id:
                kwargs["previous_response_id"] = (
                    previous_response_id
                )

            resp = await self.client.responses.create(
                **kwargs
            )

            previous_response_id = resp.id
            llm_turns += 1

            i, o, c = self._usage(resp)

            input_tokens += i
            output_tokens += o
            cached_tokens += c

            calls = [
                x
                for x in getattr(resp, "output", [])
                if getattr(x, "type", None)
                == "function_call"
            ]

            if not calls:
                final_answer = (
                    getattr(resp, "output_text", "")
                    or ""
                ).strip()

                trace.append(
                    {
                        "step": step,
                        "type": "final",
                        "message": final_answer[:600],
                    }
                )

                break

            # Satisfy every function call returned by the API.
            # Even with parallel_tool_calls=False, this is defensive and prevents:
            # "No tool output found for function call ..."
            next_input_items = []

            for call in calls:
                try:
                    args = json.loads(
                        call.arguments or "{}"
                    )
                except Exception:
                    args = {}

                trace.append(
                    {
                        "step": step,
                        "type": "tool_call",
                        "name": call.name,
                        "arguments": args,
                    }
                )

                result = executor.execute(
                    call.name,
                    args
                )

                trace.append(
                    {
                        "step": step,
                        "type": "tool_result",
                        "name": call.name,
                        "result": result,
                    }
                )

                compact = json.dumps(
                    result,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )[:self.max_tool_result_chars]

                next_input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": compact,
                    }
                )

            input_items = next_input_items

        else:
            termination_reason = (
                "max_steps_exceeded"
            )

        return {
            "tool_calls": [
                {
                    k: v
                    for k, v in call.items()
                    if k in {
                        "name",
                        "arguments",
                        "status",
                        "useful",
                    }
                }
                for call in executor.calls
            ],

            "final_answer": final_answer,
            "trace": trace,

            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_tokens,
            "tokens": input_tokens + output_tokens,

            "llm_turns": llm_turns,
            "termination_reason": termination_reason,

            "token_budget": {
                "max_output_per_turn":
                    self.max_output_per_turn,
                "max_output_per_run":
                    self.max_output_per_run,
                "output_used":
                    output_tokens,
                "output_remaining":
                    max(
                        0,
                        self.max_output_per_run
                        - output_tokens,
                    ),
            },

            "routing": {
                "tool_count": len(tools),
                "tool_names": tool_names,
                "policy_chars":
                    len(routed_policy),
                "task_chars":
                    len(task_text),
            },
        }
