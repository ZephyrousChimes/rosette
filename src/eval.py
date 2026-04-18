"""Execution-accuracy eval harness: run the generated SQL and the gold SQL
against the same database and compare result sets, instead of comparing SQL
strings token-for-token. Two syntactically different queries
(`SELECT name FROM ...` vs `SELECT customers.name FROM ...`) can be
semantically identical, and string-match would score that as wrong.
"""
import sqlite3
from pathlib import Path

from db import DB_PATH, build_db
from examples import EVAL_SET
from generation import STRATEGIES

def run_sql(conn, sql: str):
    try:
        rows = conn.execute(sql).fetchall()
        return set(rows), None
    except Exception as e:
        return None, str(e)


def execution_accuracy(strategy_fn, conn) -> dict:
    correct = 0
    exec_errors = 0
    results = []
    for question, gold_sql in EVAL_SET:
        gold_rows, gold_err = run_sql(conn, gold_sql)
        assert gold_err is None, f"gold SQL itself failed: {gold_sql} -> {gold_err}"

        pred_sql = strategy_fn(question)
        pred_rows, pred_err = run_sql(conn, pred_sql)

        is_correct = pred_err is None and pred_rows == gold_rows
        correct += int(is_correct)
        exec_errors += int(pred_err is not None)
        results.append({
            "question": question,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "correct": is_correct,
            "exec_error": pred_err,
        })

    n = len(EVAL_SET)
    return {
        "execution_accuracy": correct / n,
        "exec_error_rate": exec_errors / n,
        "n": n,
        "results": results,
    }


def main():
    if not DB_PATH.exists():
        build_db()
    conn = sqlite3.connect(DB_PATH)

    summary = {}
    for name, fn in STRATEGIES.items():
        print(f"\n=== {name} ===")
        report = execution_accuracy(fn, conn)
        summary[name] = {
            "execution_accuracy": report["execution_accuracy"],
            "exec_error_rate": report["exec_error_rate"],
        }
        for r in report["results"]:
            mark = "OK " if r["correct"] else "FAIL"
            print(f"[{mark}] {r['question']}")
            print(f"        gold: {r['gold_sql']}")
            print(f"        pred: {r['pred_sql']}" + (f"  (error: {r['exec_error']})" if r["exec_error"] else ""))

    print("\n=== summary ===")
    for name, s in summary.items():
        print(f"{name:12s} execution_accuracy={s['execution_accuracy']:.2f} exec_error_rate={s['exec_error_rate']:.2f}")

    conn.close()
    return summary


if __name__ == "__main__":
    main()
