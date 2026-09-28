# Agentic AI Evaluation Platform — Deep Eval Edition

A focused Agentic / GenAI evaluation platform. No Grafana and no Prometheus: the project is centered on **evaluation depth**.

## Add the tau2-bench Retail data

```text
data/retail/
├── tasks.json
├── db.json
├── policy.md
└── split_tasks.json   # optional
```

## Local run

```bat
copy .env.example .env
docker compose up --build
```

Open:

- UI: http://localhost:8080
- API docs: http://localhost:8000/docs

## Evaluation coverage

### 1. Outcome
- Task Success Rate
- `pass@k`
- `pass^k`
- Repeat Consistency

### 2. Policy / guardrails
- Policy Compliance Rate

### 3. Tool-use evaluation
- Tool Precision
- Tool Recall
- Tool F1
- Argument Accuracy
- Useful Tool Ratio
- Invalid Calls / Run
- Redundant Calls / Run
- Extra Tool Ratio
- Mean Tool Calls

### 4. Trajectory evaluation
- LCS-based Tool Order Score
- Exact Tool Sequence Rate (diagnostic only)
- Mean Trace Steps
- Reference vs Actual trace inspector

### 5. Efficiency
- Mean / Median / P95 Latency
- Mean / Median / P95 Tokens
- Total Estimated Cost
- Cost per Successful Run

### 6. Failure analysis
Automatic taxonomy:
- task outcome failure
- policy violation
- missing reference tool
- unexpected tool
- invalid tool call
- redundant tool call
- argument mismatch
- trajectory order mismatch

### 7. Reproducibility
- SQLite experiment history
- CSV export of every run
- Repeat count is configurable

## Important evaluation principle

Reference actions are **not** treated as the only valid path to success.
Outcome metrics are kept separate from trajectory diagnostics so an alternative valid path can still receive a successful outcome.

## Experiment configurations

- Baseline
- LLM + Tools
- Tools + Reflection
- Multi-Agent

The default `heuristic` mode is a free smoke test for the pipeline.
Use a real LLM provider for research results.

## AWS / Terraform

```bat
cd infra\\terraform
terraform init
terraform apply -var="key_name=YOUR_KEY_NAME" -var="private_key_path=C:/path/key.pem"
```

Destroy after use:

```bat
terraform destroy -var="key_name=YOUR_KEY_NAME" -var="private_key_path=C:/path/key.pem"
```
