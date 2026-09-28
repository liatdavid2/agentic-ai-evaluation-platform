# Agentic AI Evaluation Platform — v14 Paper-Style

This version changes the evaluation architecture to match the core benchmark
idea more closely:

```text
LLM User Simulator
        ↕
     LLM Agent
        ↕
     Real Tools
        ↕
      db.json
        ↓
Final DB State Evaluator
```

## What changed

### 1. LLM user simulator
The user is now simulated by a second LLM conversation.

It receives the task scenario as a private script and:
- answers clarification / confirmation questions
- follows conditional task instructions
- can change its mind when the task script says to
- returns `__END__` when the user conversation is over

This is separate from the evaluated agent.

### 2. Final-state evaluator
Task Success is now based on the final environment state rather than requiring
the agent's lookup trajectory to match the reference.

The evaluator:
1. clones the initial `db.json`
2. replays the gold **mutating actions** on one copy
3. uses the agent's actual final DB as the second copy
4. compares only the user/order entities that the task is supposed to mutate
5. combines final-state match with required `communicate_info`

Lookup/authentication calls remain diagnostic only.

### 3. Trajectory diagnostics remain
The UI still reports:
- Tool Precision / Recall / F1
- Argument Accuracy
- Exact Action Recall
- Communication Recall
- Tool errors
- Tokens / latency

These explain how the agent behaved but do not define the final DB outcome.

### 4. Better state mutation
Retail mutating tools now leave explicit state changes that can be compared:
- item modification updates order items
- exchange stores an exchange request
- return stores a return request
- cancellation/address/payment/user-address mutations update their state

### 5. Separate token accounting
The UI reports both:
- evaluated Agent tokens
- User Simulator tokens

## Ports

```text
Backend host: http://localhost:8007
Backend container: 8000
Frontend: http://localhost:8080
VITE_API_BASE_URL=http://localhost:8007
```

## Suggested first run

```text
Tasks = 5
Repeats = 1
Provider = OpenAI
```

Then inspect:
- Task Success
- Final State Match
- User-sim turns
- Tool F1 / Argument Accuracy
- Expected Final State vs Actual Final State
