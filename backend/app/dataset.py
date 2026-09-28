import json, os
from pathlib import Path
from typing import Any

DATASET_DIR=Path(os.getenv("DATASET_DIR","/app/data/retail"))

def _read_json(path,default):
    if not path.exists(): return default
    with path.open("r",encoding="utf-8") as f:return json.load(f)

def dataset_status():
    tasks=_read_json(DATASET_DIR/"tasks.json",[])
    db=_read_json(DATASET_DIR/"db.json",{})
    return {
        "tasks_json":(DATASET_DIR/"tasks.json").exists(),
        "db_json":(DATASET_DIR/"db.json").exists(),
        "policy_md":(DATASET_DIR/"policy.md").exists(),
        "task_count":len(tasks) if isinstance(tasks,list) else 0,
        "product_count":len(db.get("products",{})),
        "user_count":len(db.get("users",{})),
        "order_count":len(db.get("orders",{})),
        "ready":all((DATASET_DIR/x).exists() for x in ["tasks.json","db.json","policy.md"])
    }

def load_tasks(): return _read_json(DATASET_DIR/"tasks.json",[])
def load_db(): return _read_json(DATASET_DIR/"db.json",{})
def load_policy():
    p=DATASET_DIR/"policy.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""

def task_id(task,index): return str(task.get("id") or f"task-{index:04d}")

def instruction(task):
    x=task.get("user_scenario",{}).get("instructions",{})
    parts=[
        x.get("task_instructions") or "",
        "Reason for call: "+str(x.get("reason_for_call") or ""),
        "Known information: "+str(x.get("known_info") or ""),
        "Unknown information: "+str(x.get("unknown_info") or ""),
    ]
    return "\n".join(p for p in parts if p and not p.endswith(": None"))

def reference_actions(task):
    x=task.get("evaluation_criteria",{}).get("actions",[])
    return x if isinstance(x,list) else []

def reference_tool_names(task):
    return [str(a.get("name")) for a in reference_actions(task) if a.get("name")]

def communicate_info(task):
    x=task.get("evaluation_criteria",{}).get("communicate_info",[])
    return [str(v) for v in x] if isinstance(x,list) else []
