"""FastAPI serving layer: takes a natural-language question, generates SQL
with the fine-tuned model (the strategy the eval harness showed actually
works -- see README), executes it against the toy DB, and returns both the
SQL and the result so a caller can audit what ran, not just trust a raw
answer string.
"""
import sqlite3
import time

from fastapi import FastAPI

from db import DB_PATH, build_db
from generation import STRATEGIES

if not DB_PATH.exists():
    build_db()

app = FastAPI(title="llm-app-mvp: NL to SQL")

DEFAULT_STRATEGY = "fine_tuned"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/query")
def query(question: str, strategy: str = DEFAULT_STRATEGY):
    if strategy not in STRATEGIES:
        return {"error": f"unknown strategy '{strategy}', choose from {list(STRATEGIES)}"}

    start = time.perf_counter()
    sql = STRATEGIES[strategy](question)

    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(sql).fetchall()
        columns = [d[0] for d in conn.execute(sql).description] if rows or True else []
        error = None
    except Exception as e:
        rows, columns, error = None, None, str(e)
    finally:
        conn.close()

    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "question": question,
        "strategy": strategy,
        "generated_sql": sql,
        "columns": columns,
        "rows": rows,
        "error": error,
        "latency_ms": round(latency_ms, 2),
    }
