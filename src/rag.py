from .config import POLICY_FILE


def _sections() -> list[dict]:
    sections = []
    current = None
    for line in POLICY_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("[POLICY "):
            if current:
                sections.append(current)
            current = {"title": line.strip("[]"), "text": line}
        elif current:
            current["text"] += "\n" + line
    if current:
        sections.append(current)
    return sections


def retrieve_policies(query: str, k: int = 2) -> list[dict]:
    sections = _sections()
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        matrix = TfidfVectorizer(stop_words="english").fit_transform([s["text"] for s in sections] + [query])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        order = scores.argsort()[::-1][:k]
        return [{**sections[i], "score": float(scores[i])} for i in order if scores[i] > 0]
    except Exception:
        return sections[:k]


def format_policy_context(results: list[dict]) -> str:
    return "\n\n".join(f"{item['title']}\n{item['text']}" for item in results)

