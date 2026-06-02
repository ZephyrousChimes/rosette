"""TF-IDF retrieval over the few-shot example bank.

TF-IDF instead of a dense embedding model on purpose: the example bank is a
few dozen short, template-like NL questions, not free-form prose, so a
sparse lexical match on the question text is enough to find structurally
similar examples (same aggregation, same filter shape) -- and it needs no
model download or GPU, which matters for a project meant to run anywhere
without setup friction.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from examples import FEWSHOT_CORPUS

_questions = [q for q, _ in FEWSHOT_CORPUS]
_vectorizer = TfidfVectorizer().fit(_questions)
_corpus_matrix = _vectorizer.transform(_questions)


def retrieve_examples(question: str, k: int = 3) -> list[tuple[str, str]]:
    q_vec = _vectorizer.transform([question])
    sims = cosine_similarity(q_vec, _corpus_matrix)[0]
    top_idx = sims.argsort()[::-1][:k]
    return [FEWSHOT_CORPUS[i] for i in top_idx]
