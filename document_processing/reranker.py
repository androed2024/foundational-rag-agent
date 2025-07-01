from typing import List, Dict, Any

try:
    from sentence_transformers import CrossEncoder
except Exception:  # noqa: S110
    CrossEncoder = None


class CrossEncoderReranker:
    """Optional cross-encoder reranking using MiniLM."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if CrossEncoder is None:
            self.model = None
        else:
            self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.model or not results:
            return results
        pairs = [(query, r["content"]) for r in results]
        scores = self.model.predict(pairs)

        # for r, s in zip(results, scores):
        #    r["rerank_score"] = float(s)

        # gewichtete Kombination für rerank
        for r, s in zip(results, scores):
            sim = r.get("similarity", 0)
            r["rerank_score"] = float(0.5 * sim + 0.5 * s)

        return sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
