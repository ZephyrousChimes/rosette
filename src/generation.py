"""Three NL->SQL strategies, compared head to head -- mirroring the
in-house-vs-3rd-party-vs-fine-tuned comparison framework real ML platforms
(Uber's Michelangelo Gen AI layer among them) use to decide which approach
to actually ship, instead of picking one by default.

  1. zero-shot   : generic instruction-tuned model (flan-t5-small), given
                    only the schema + question. No SQL specialization.
  2. rag_fewshot : same generic model, but the prompt is augmented with the
                    k most similar (question, SQL) examples retrieved from a
                    small example bank via TF-IDF.
  3. fine_tuned  : a model already fine-tuned specifically for text-to-SQL
                    (cssupport/t5-small-awesome-text-to-sql), given only the
                    schema + question, no few-shot examples.

The retrieval-vs-fine-tuning choice is an empirical question, not a default:
RAG helps when the base model already understands the task shape but lacks
task-specific priors; fine-tuning helps when the base model's *output
distribution* itself needs to change. Running all three against the same
eval set is how you find out which is true here instead of assuming it.
"""
from functools import lru_cache

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from db import get_schema_text
from retrieval import retrieve_examples

BASE_MODEL_NAME = "google/flan-t5-small"
FINETUNED_MODEL_NAME = "cssupport/t5-small-awesome-text-to-sql"
FINETUNED_TOKENIZER_SOURCE = "t5-small"


@lru_cache(maxsize=None)
def _load(model_name: str):
    tokenizer_source = FINETUNED_TOKENIZER_SOURCE if model_name == FINETUNED_MODEL_NAME else model_name
    tok = T5Tokenizer.from_pretrained(tokenizer_source)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    return tok, model


def _generate(model_name: str, prompt: str, max_length: int = 128) -> str:
    tok, model = _load(model_name)
    inputs = tok(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=max_length)
    return tok.decode(out[0], skip_special_tokens=True)


def zero_shot(question: str) -> str:
    prompt = (
        "Translate the question to a SQL query given this schema.\n"
        f"Schema:\n{get_schema_text()}\n"
        f"Question: {question}\nSQL:"
    )
    return _generate(BASE_MODEL_NAME, prompt)


def rag_fewshot(question: str, k: int = 3) -> str:
    examples = retrieve_examples(question, k=k)
    example_block = "\n".join(f"Q: {q}\nSQL: {sql}" for q, sql in examples)
    prompt = (
        "Translate the question to a SQL query given this schema. "
        "Here are similar examples.\n"
        f"Schema:\n{get_schema_text()}\n"
        f"{example_block}\n"
        f"Q: {question}\nSQL:"
    )
    return _generate(BASE_MODEL_NAME, prompt)


def fine_tuned(question: str) -> str:
    prompt = "tables:\n" + get_schema_text() + "\nquery for: " + question
    return _generate(FINETUNED_MODEL_NAME, prompt)


STRATEGIES = {
    "zero_shot": zero_shot,
    "rag_fewshot": rag_fewshot,
    "fine_tuned": fine_tuned,
}
