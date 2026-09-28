import sqlite3, os, json
from pathlib import Path

DB = Path(os.getenv("DATABASE_PATH", "/app/state/evals.db"))

def init_db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS benchmarks (
          benchmark_id TEXT PRIMARY KEY,
          created_at TEXT NOT NULL,
          summary_json TEXT NOT NULL,
          runs_json TEXT NOT NULL
        )
        """)

def save_benchmark(benchmark_id, created_at, summary, runs):
    with sqlite3.connect(DB) as c:
        c.execute("INSERT OR REPLACE INTO benchmarks VALUES (?,?,?,?)",
                  (benchmark_id, created_at, json.dumps(summary), json.dumps(runs)))

def history(limit=30):
    with sqlite3.connect(DB) as c:
        rows = c.execute(
            "SELECT benchmark_id, created_at, summary_json FROM benchmarks ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [{"benchmark_id":r[0],"created_at":r[1],**json.loads(r[2])} for r in rows]

def get_benchmark(benchmark_id):
    with sqlite3.connect(DB) as c:
        r = c.execute(
            "SELECT created_at, summary_json, runs_json FROM benchmarks WHERE benchmark_id=?",
            (benchmark_id,)
        ).fetchone()
    if not r:
        return None
    return {"benchmark_id":benchmark_id,"created_at":r[0],"summary":json.loads(r[1]),"runs":json.loads(r[2])}
