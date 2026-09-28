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


## v13 — outcome-based evaluator

The agent itself is unchanged from v12.

The evaluator now separates **task outcome** from **trajectory similarity**.

### Task Success

For action tasks:

```text
correct final mutating action/effect
+ all required communicate_info
+ completed run
```

For informational tasks:

```text
required communicate_info
+ completed run
```

Task Success no longer requires the entire sequence of lookup tools to match the
reference trajectory.

### Still reported separately

- Tool Precision
- Tool Recall
- Tool F1
- Argument Accuracy
- Exact Action Recall
- Tool errors
- Policy diagnostics

### New metric

`Final Action Score`

This measures how closely the successful mutating action matches the expected
side effect. It ignores lookup/authentication trajectory differences.

This makes it possible for two valid trajectories to receive the same successful
task outcome while still exposing their different tool-use behavior.
