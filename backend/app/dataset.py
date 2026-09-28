import json
import os
from pathlib import Path
from typing import Any

DATASET_DIR = Path(os.getenv("DATASET_DIR", "/app/data/retail"))

def _read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def dataset_status() -> dict[str, Any]:
    tasks = _read_json(DATASET_DIR / "tasks.json", [])
    if isinstance(tasks, dict) and isinstance(tasks.get("tasks"), list):
        count = len(tasks["tasks"])
    elif isinstance(tasks, list):
        count = len(tasks)
    elif isinstance(tasks, dict):
        count = len(tasks)
    else:
        count = 0
    return {
        "tasks_json": (DATASET_DIR / "tasks.json").exists(),
        "db_json": (DATASET_DIR / "db.json").exists(),
        "policy_md": (DATASET_DIR / "policy.md").exists(),
        "split_tasks_json": (DATASET_DIR / "split_tasks.json").exists(),
        "task_count": count,
        "ready": (DATASET_DIR / "tasks.json").exists(),
    }

def load_tasks() -> list[dict[str, Any]]:
    raw = _read_json(DATASET_DIR / "tasks.json", [])
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("tasks"), list):
        return raw["tasks"]
    if isinstance(raw, dict):
        vals = list(raw.values())
        if vals and all(isinstance(x, dict) for x in vals):
            return vals
    return []

def load_policy() -> str:
    p = DATASET_DIR / "policy.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""

def task_id(task: dict, index: int) -> str:
    return str(task.get("id") or task.get("task_id") or task.get("ticket_id") or f"task-{index:04d}")

def instruction(task: dict) -> str:
    return str(task.get("instruction") or task.get("user_instruction") or task.get("prompt") or task.get("request") or "")

def reference_actions(task: dict) -> list[dict]:
    ec = task.get("evaluation_criteria")
    candidates = [
        task.get("actions"),
        task.get("reference_actions"),
        ec.get("actions") if isinstance(ec, dict) else None,
        ec.get("reference_actions") if isinstance(ec, dict) else None,
    ]
    for c in candidates:
        if isinstance(c, list):
            return [x for x in c if isinstance(x, dict)]
    return []

def reference_tool_names(task: dict) -> list[str]:
    out = []
    for a in reference_actions(task):
        n = a.get("name") or a.get("tool") or a.get("tool_name")
        if n:
            out.append(str(n))
    return out
