# Rosette

Natural-language-to-SQL assistant over a toy e-commerce database, with an
execution-accuracy eval harness comparing three strategies head to head --
mirroring the "in-house vs. 3rd-party vs. fine-tuned" comparison framework
real ML platform teams use before committing to one approach, rather than
picking one by default.

## The three strategies compared

| Strategy      | Model                                      | What it gets, beyond the schema |
|---------------|---------------------------------------------|----------------------------------|
| `zero_shot`   | `google/flan-t5-small` (generic instruction-tuned) | nothing |
| `rag_fewshot` | same generic model                          | top-3 similar (question, SQL) examples, retrieved via TF-IDF from a 20-example bank |
| `fine_tuned`  | `cssupport/t5-small-awesome-text-to-sql` (t5-small fine-tuned on ~262K text-to-SQL pairs) | nothing extra -- the model itself has been adapted |

The retrieval-vs-fine-tuning choice is treated as an empirical question, not
a default. RAG only helps if the base model already understands the *task
shape* and just needs task-specific priors at inference time; fine-tuning
is required when the base model's *output distribution itself* needs to
change (e.g. it doesn't reliably produce valid SQL syntax at all). Running
both against the same held-out eval set is how you find out which is true,
rather than assuming it.

## Eval methodology

Execution accuracy, not string match: the generated SQL and the gold SQL
are both run against the same SQLite database and their **result sets**
are compared. Two syntactically different queries can be semantically
identical (`SELECT name FROM x` vs `SELECT x.name FROM x`), so string
comparison would misscore correct answers as wrong.

The 10 held-out eval questions are disjoint from the 20-example few-shot
bank used for RAG retrieval -- if an eval question's own gold example were
retrievable, "RAG" would just be doing lookup, not generation, and the
comparison would be meaningless.

## Results (10 held-out questions)

| Strategy      | Execution accuracy | SQL syntax-error rate |
|---------------|--------------------:|------------------------:|
| zero_shot     | 0%                  | 100%                    |
| rag_fewshot   | 0%                  | 60%                     |
| fine_tuned    | 40%                 | 0%                      |

**Read on the result**: few-shot examples cut the syntax-error rate
(100% -> 60%) but didn't produce a single execution-correct query --
the generic model's failure mode is that it doesn't reliably produce
*valid SQL grammar* at all, and no amount of in-context examples fixes
that within a small model's capacity. The fine-tuned model never produces
invalid SQL and gets the semantics right on 4/10 held-out questions.
This is the concrete evidence for choosing fine-tuning over RAG for this
task -- not an assumption going in.

Remaining failures in the fine-tuned model are semantic, not syntactic
(e.g. `SUM(quantity)` without the right `WHERE`/`JOIN` condition, or
`GROUP BY` added where none was needed) -- the kind of gap that more
fine-tuning data or a larger base model would close, not what RAG or
prompt engineering would.

## Run it

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/db.py               # builds the toy SQLite DB
python src/eval.py             # runs all 3 strategies against the held-out set
uvicorn serve:app --app-dir src --port 8002
curl "http://127.0.0.1:8002/query?question=How%20many%20products%20cost%20more%20than%20500%3F"
```

## Files

- `src/db.py` -- toy SQLite schema + seed data (customers/products/orders)
- `src/examples.py` -- few-shot corpus (RAG retrieval source) and the
  disjoint held-out eval set
- `src/retrieval.py` -- TF-IDF retrieval over the few-shot corpus
- `src/generation.py` -- the three NL->SQL strategies
- `src/eval.py` -- execution-accuracy harness, run against all 3 strategies
- `src/serve.py` -- FastAPI `/query` endpoint (defaults to the winning
  strategy from the eval), executes the generated SQL and returns both the
  query and the result so a caller can audit what actually ran
