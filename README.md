# Agentic AI Evaluation Platform — v12 Clean Baseline

This version intentionally removes the orchestration layers that were added in
later experiments and returns to the simplest meaningful agent baseline:

```text
Task
  -> OpenAI LLM
  -> real Retail tools
  -> db.json
  -> LLM
  -> ...
  -> final answer
  -> evaluator
```

## Removed

- User Simulator
- Guardrail Engine
- state machine / phase orchestration
- forced mutation loops
- confirmation gate in code
- grounded-ID blocking
- repair loops
- task-specific trajectory forcing

## Kept

- real OpenAI function calling
- real Retail tool execution against a fresh copy of `db.json`
- all 14 Retail tools
- full Retail policy in the model context
- `parallel_tool_calls=False`
- matching `function_call_output` for every returned function call
- input/output/cached/reasoning token tracking
- high enough output headroom for reasoning models
- explicit incomplete/max-output detection
- benchmark metrics and run inspector
- live progress
- SQLite history
- CSV export

## Network / ports

This project always uses:

```text
Backend host: http://localhost:8007
Backend container port: 8000
Frontend: http://localhost:8080
VITE_API_BASE_URL=http://localhost:8007
```

Docker mapping:

```yaml
ports:
  - "8007:8000"
```

## Recommended first run

```text
Tasks = 5
Repeats = 1
Provider = OpenAI
Agent setup = Clean LLM + Tools Baseline
```

The goal of this version is to establish a trustworthy baseline before adding
any guardrails or user simulation as separate ablations.
