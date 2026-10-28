from .config import POLICY_FILE, ROOT

_chroma_collection = None


def _sections() -> list[dict]:
    sections, current = [], None
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


def _chroma_results(query: str, k: int) -> list[dict]:
    global _chroma_collection
    import chromadb
    client = chromadb.PersistentClient(path=str(ROOT / "reports" / "chroma"))
    _chroma_collection = client.get_or_create_collection("aeromitra_policies")
    sections = _sections()
    if _chroma_collection.count() == 0:
        _chroma_collection.add(
            ids=[f"policy-{i}" for i in range(len(sections))],
            documents=[s["text"] for s in sections],
            metadatas=[{"title": s["title"], "source_file": POLICY_FILE.name} for s in sections],
        )
    result = _chroma_collection.query(query_texts=[query], n_results=k)
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return [{"title": meta.get("title", "Policy"), "text": doc, "score": round(1 - float(distance), 4)} for doc, meta, distance in zip(documents, metadatas, distances)]


def _tfidf_results(query: str, k: int) -> list[dict]:
    sections = _sections()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    matrix = TfidfVectorizer(stop_words="english").fit_transform([s["text"] for s in sections] + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    order = scores.argsort()[::-1][:k]
    return [{**sections[i], "score": float(scores[i])} for i in order if scores[i] > 0]


def retrieve_policies(query: str, k: int = 2) -> list[dict]:
    try:
        return _chroma_results(query, k)
    except Exception:
        return _tfidf_results(query, k)


def format_policy_context(results: list[dict]) -> str:
    return "\n\n".join(f"{item['title']}\n{item['text']}" for item in results)

