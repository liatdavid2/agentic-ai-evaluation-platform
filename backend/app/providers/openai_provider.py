import os
import json

from openai import AsyncOpenAI

from .base import AgentProvider
from app.dataset import instruction
from app.tooling import RetailToolExecutor, filtered_schemas
from app.routing import policy_for_task, tool_names_for_task, detect_intents
from app.user_simulator import ScriptedUserSimulator
from app.guardrails import GuardrailEngine, ALL_MUTATION_TOOLS


class OpenAIProvider(AgentProvider):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-5-mini")

        self.max_steps = int(os.getenv("MAX_AGENT_STEPS", "9"))
        self.max_output_per_turn = int(
            os.getenv("MAX_OUTPUT_TOKENS_PER_TURN", "1200")
        )
        self.max_output_per_run = int(
            os.getenv("MAX_OUTPUT_TOKENS_PER_RUN", "4000")
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

        self.reasoning_effort = os.getenv("REASONING_EFFORT", "low")
        self.max_simulator_turns = int(
            os.getenv("MAX_SIMULATOR_TURNS", "3")
        )

    def _usage(self, resp):
        usage = getattr(resp, "usage", None)

        input_tokens = int(
            getattr(usage, "input_tokens", 0) or 0
        ) if usage else 0

        output_tokens = int(
            getattr(usage, "output_tokens", 0) or 0
        ) if usage else 0

        input_details = (
            getattr(usage, "input_tokens_details", None)
            if usage else None
        )
        cached_tokens = int(
            getattr(input_details, "cached_tokens", 0) or 0
        ) if input_details else 0

        output_details = (
            getattr(usage, "output_tokens_details", None)
            if usage else None
        )
        reasoning_tokens = int(
            getattr(output_details, "reasoning_tokens", 0) or 0
        ) if output_details else 0

        return input_tokens, output_tokens, cached_tokens, reasoning_tokens

    def _response_diagnostics(self, resp):
        status = getattr(resp, "status", None)
        incomplete = getattr(resp, "incomplete_details", None)

        reason = None
        if incomplete is not None:
            reason = getattr(incomplete, "reason", None)
            if reason is None:
                reason = str(incomplete)

        return status, reason

    async def run(self, task, configuration, policy, db=None):
        executor = RetailToolExecutor(
            db or {
                "products": {},
                "users": {},
                "orders": {},
            }
        )

        guardrails_enabled = configuration == "guardrailed"
        guardrails = GuardrailEngine()

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

        simulator = ScriptedUserSimulator(
            task_text,
            max_turns=self.max_simulator_turns,
        )

        system = (
            "You are a concise retail support agent under evaluation. "
            "Use tools for all database facts. "
            "Never invent IDs, orders, products, variants, prices or payment methods. "
            "Do not narrate reasoning. "
            "Make at most one tool call per turn. "
            "Continue until the user's request is actually completed. "
            "Authentication alone is never task completion. "
            "Do not ask the user for information that can be obtained from tools. "
            "Before a database-changing action, ask for explicit confirmation when required. "
            "If a tool call is rejected by a guardrail, follow the repair_hint and retry. "
            "Keep final answers very short. "
            "Follow the policy below.\n\n"
            + routed_policy
        )

        input_items = [
            {"role": "system", "content": system},
            {"role": "user", "content": task_text},
        ]

        previous_response_id = None
        trace = []

        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        reasoning_tokens = 0

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

            turn_budget = min(
                self.max_output_per_turn,
                remaining,
            )

            authenticated = bool(executor.authenticated_user_id)

            kwargs = {
                "model": self.model,
                "input": input_items,
                "tools": tools,
                "max_output_tokens": turn_budget,
                "reasoning": {
                    "effort": self.reasoning_effort
                },
                "parallel_tool_calls": False,
            }

            # Free agent still needs auth first for action tasks.
            if requires_action and not authenticated:
                kwargs["tool_choice"] = "required"
            else:
                kwargs["tool_choice"] = "auto"

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

            # Important: max_output_tokens includes hidden reasoning tokens.
            # If the model exhausted the turn budget before emitting a tool call
            # or visible answer, do not misclassify it as a normal final answer.
            if status == "incomplete" and incomplete_reason == "max_output_tokens":
                trace.append({
                    "step": step,
                    "type": "incomplete",
                    "reason": "max_output_tokens",
                    "output_tokens_so_far": output_tokens,
                    "reasoning_tokens_so_far": reasoning_tokens,
                })
                termination_reason = "max_output_tokens"
                break

            calls = [
                x for x in getattr(resp, "output", [])
                if getattr(x, "type", None) == "function_call"
            ]

            if calls:
                next_input_items = []

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

                    if guardrails_enabled:
                        allowed, rejection = guardrails.validate(
                            call.name,
                            args,
                        )
                    else:
                        allowed, rejection = True, None

                    if allowed:
                        result = executor.execute(
                            call.name,
                            args
                        )
                        if guardrails_enabled:
                            guardrails.observe_tool_result(
                                call.name,
                                result,
                            )
                            # Mirror executor auth into guardrail state.
                            if executor.authenticated_user_id:
                                guardrails.state.authenticated_user_id = executor.authenticated_user_id

                            if call.name in ALL_MUTATION_TOOLS and result.get("ok"):
                                guardrails.clear_confirmation_after_mutation()
                    else:
                        result = rejection

                    trace.append({
                        "step": step,
                        "type": (
                            "guardrail_rejection"
                            if result.get("guardrail_rejected")
                            else "tool_result"
                        ),
                        "name": call.name,
                        "result": result,
                    })

                    compact = json.dumps(
                        result,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )[:self.max_tool_result_chars]

                    next_input_items.append({
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": compact,
                    })

                input_items = next_input_items
                continue

            agent_text = (
                getattr(resp, "output_text", "")
                or ""
            ).strip()

            trace.append({
                "step": step,
                "type": "assistant_message",
                "message": agent_text[:800],
            })

            successful_mutation = any(
                call.get("name") in ALL_MUTATION_TOOLS
                and call.get("status") == "ok"
                for call in executor.calls
            )

            if requires_action and not successful_mutation:
                simulated_reply = simulator.respond(agent_text)

                if simulated_reply:
                    if guardrails_enabled:
                        guardrails.mark_confirmation()

                    trace.append({
                        "step": step,
                        "type": "user_simulator",
                        "message": simulated_reply[:500],
                    })

                    input_items = [{
                        "role": "user",
                        "content": simulated_reply,
                    }]
                    continue

                final_answer = agent_text
                termination_reason = "premature_final_before_action"
                break

            final_answer = agent_text
            break

        else:
            termination_reason = "max_steps_exceeded"

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
            "reasoning_tokens": reasoning_tokens,
            "tokens": input_tokens + output_tokens,

            "llm_turns": llm_turns,
            "user_simulator_turns": simulator.turns,
            "termination_reason": termination_reason,

            "response_statuses": response_statuses,
            "incomplete_reasons": incomplete_reasons,

            "guardrails_enabled": guardrails_enabled,
            "guardrail_rejections": (
                guardrails.state.violations
                if guardrails_enabled
                else []
            ),
            "guardrail_rejection_count": (
                len(guardrails.state.violations)
                if guardrails_enabled
                else 0
            ),

            "token_budget": {
                "max_output_per_turn": self.max_output_per_turn,
                "max_output_per_run": self.max_output_per_run,
                "output_used": output_tokens,
                "output_remaining": max(
                    0,
                    self.max_output_per_run - output_tokens,
                ),
            },

            "routing": {
                "tool_count": len(tools),
                "tool_names": tool_names,
                "policy_chars": len(routed_policy),
                "task_chars": len(task_text),
            },
        }
