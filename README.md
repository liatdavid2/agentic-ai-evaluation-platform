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
