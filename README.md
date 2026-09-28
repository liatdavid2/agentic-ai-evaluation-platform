# Agentic AI Evaluation Platform — v5 Token Budget Edition

This version is optimized for **much lower token usage** while keeping the evaluation visible and auditable.

The uploaded Retail dataset is already included:
- 114 tasks
- 50 products
- 500 users
- 1000 orders

## Key token-saving changes

1. **Hard output budget per turn**
   - `MAX_OUTPUT_TOKENS_PER_TURN=80`

2. **Hard cumulative output budget per run**
   - `MAX_OUTPUT_TOKENS_PER_RUN=180`
   - once exhausted, the run ends with:
     `output_token_budget_exceeded`

3. **Fewer agent turns**
   - `MAX_AGENT_STEPS=7`

4. **Policy routing**
   - only the policy sections relevant to the task are sent

5. **Tool routing**
   - only a relevant subset of tools is exposed to the model

6. **Compact tool results**
   - tool responses contain only fields the agent needs
   - results are capped with `MAX_TOOL_RESULT_CHARS`

7. **Parallel independent runs**
   - `MAX_CONCURRENT_TASKS=3`

## Default limits

```env
MAX_AGENT_STEPS=7
MAX_OUTPUT_TOKENS_PER_TURN=80
MAX_OUTPUT_TOKENS_PER_RUN=180
MAX_TASK_INPUT_CHARS=1800
MAX_POLICY_CHARS=2200
MAX_TOOL_RESULT_CHARS=2500
MAX_CONCURRENT_TASKS=3
```

## What the UI now shows

### Live
- completed / total runs
- input tokens so far
- output tokens so far
- tool calls so far
- last run output-budget utilization
- current task / stage / elapsed time
- live log

### After the run
- average input tokens / run
- average output tokens / run
- P95 input / output tokens
- total input / output tokens
- average output-budget utilization
- LLM turns / run
- tools exposed / run
- routed policy size
- tool precision / recall / F1
- argument accuracy
- exact action recall
- communication recall
- task success / pass@k / pass^k
- per-run inspector with token budget and trace

## Run

```bat
copy .env.example .env
```

Add:

```env
OPENAI_API_KEY=YOUR_KEY
```

If port 8000 is already occupied, change:

```yaml
# docker-compose.yml
ports:
  - "8001:8000"
```

and:

```env
VITE_API_BASE_URL=http://localhost:8001
```

Then:

```bat
docker compose down
docker compose up --build
```

UI:

```text
http://localhost:8080
```

## Recommended first experiment

```text
Tasks = 5
Repeats = 1
Provider = OpenAI
Agent setup = LLM + Tools
```

Check `Output / run` and `Output budget used`. If these stay comfortably below the 180-token cap and quality remains acceptable, scale to 20 tasks.

## Cost

Token counts are real. Dollar cost is only calculated if you explicitly set current model pricing:

```env
INPUT_COST_PER_1M=...
CACHED_INPUT_COST_PER_1M=...
OUTPUT_COST_PER_1M=...
```

When unset, dollar metrics remain zero instead of pretending an outdated price is correct.


## v6 trajectory fix

This update targets premature agent termination after authentication.

Changes:
- `MAX_AGENT_STEPS`: 7 -> 9
- `MAX_OUTPUT_TOKENS_PER_TURN`: 80 -> 220
- `MAX_OUTPUT_TOKENS_PER_RUN`: 180 -> 650
- `MAX_TOOL_RESULT_CHARS`: 2500 -> 1800
- Added `REASONING_EFFORT=low`
- Action tasks use `tool_choice="required"` until a successful mutating action occurs
- Authentication is explicitly not treated as task completion
- The system prompt tells the agent to keep resolving DB information through tools instead of asking the user for information tools can provide
- UI banner now reflects the real limits


## v7 function-call orchestration fix

This version fixes the OpenAI Responses API error:

```text
No tool output found for function call ...
```

Changes:
- `parallel_tool_calls=False`
- every returned `function_call` now receives a matching `function_call_output`
- keeps the trajectory sequential when possible
- preserves v6 behavior that forces action tasks to continue using tools until a successful action is performed

This fix removes orchestration-level 400 errors. It does **not** guarantee 100% benchmark success; remaining failures can still come from wrong tool selection, wrong arguments, insufficient routing, policy requirements, or tasks that need a user-simulator/confirmation turn.


## v8 multi-turn user simulator

This version addresses the main validity gap in earlier runs: many Retail tasks require an explicit user confirmation before a mutating action.

New flow:

```text
User task
  -> Agent
  -> Tool(s)
  -> Agent asks for confirmation
  -> Deterministic user simulator replies
  -> Agent performs mutating tool action
  -> Final response
```

Key changes:
- Added `ScriptedUserSimulator` with no extra LLM/API cost.
- The simulator only answers confirmation-style questions; it does not invent database facts.
- Before authentication, action tasks require a tool call.
- After authentication, `tool_choice=auto` so the agent can ask for policy-required confirmation.
- After simulator confirmation, the agent can continue to the mutating action.
- Added `premature_final_before_action` termination reason.
- Added response status and incomplete-reason tracking.
- Added reasoning-token tracking when reported by the API.
- UI now shows user-simulator turns and reasoning-token usage.
- Existing v7 fix for matching every function call with a function-call output remains intact.

Important:
The simulator is intentionally deterministic and lightweight. It improves benchmark fidelity for confirmation flows without adding another model call, but it is not a full conversational user model.


## v9 — industry-style guardrails

This version adds a production-style runtime guardrail layer without hard-coding the benchmark gold trajectory.

### Two comparable agent setups

**Free Agent**
- LLM + tools + policy
- model chooses tools and arguments directly

**Guardrailed Agent**
- same LLM, tools and policy
- deterministic validation before tool execution
- invalid calls are rejected with a structured repair hint
- agent may retry

### Guardrails implemented

- authentication required before personal-state tools
- `user_id` must have been observed previously
- `order_id` must have been observed previously
- `product_id` must have been observed previously
- item ids must have been observed previously
- payment method ids must come from observed user details
- mutating tools require explicit confirmation
- rejected tool calls never reach the environment
- repair messages are returned to the model

The guardrails do **not** know the reference/gold trajectory. They enforce generic invariants only.

### New evaluation signal

The UI now reports:

```text
Guardrail rejects / run
```

and each run records the rejected call, reason and repair hint. This enables a direct comparison:

```text
Free Agent
vs
Guardrailed Agent
```

on task success, tool F1, arguments, policy, tokens and latency.


## v10 — output headroom + incomplete-response handling

This version fixes a failure mode where the model used its entire small
`max_output_tokens` allowance on hidden reasoning and emitted no tool call.

Changes:
- `MAX_OUTPUT_TOKENS_PER_TURN`: 220 -> 1200
- `MAX_OUTPUT_TOKENS_PER_RUN`: 650 -> 4000
- keeps `REASONING_EFFORT=low`
- explicitly detects `response.status == "incomplete"` with
  `incomplete_details.reason == "max_output_tokens"`
- such runs terminate as `max_output_tokens` instead of being misclassified as
  `premature_final_before_action`
- UI text now explains that the cap is headroom and that reasoning tokens count
  toward the output-token budget

The larger caps are ceilings only; they do not force the model to consume all of
those tokens.


## v11 — port 8007 + request/React error fix

This project now consistently uses:

```text
Backend host URL: http://localhost:8007
Container backend port: 8000
Frontend: http://localhost:8080
VITE_API_BASE_URL=http://localhost:8007
```

`docker-compose.yml` contains:

```yaml
ports:
  - "8007:8000"
```

Also fixed:
- backend `BenchmarkRequest` now accepts `configuration="guardrailed"`
- frontend API errors are normalized to strings before rendering
- FastAPI 422 validation arrays no longer crash React with error #31
- all benchmark start/progress/result fetches use shared safe JSON/error handling
