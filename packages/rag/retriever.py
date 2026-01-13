from __future__ import annotations

from .retrieval.retriever import Retriever as CoreRetriever


class Retriever(CoreRetriever):
    """Compatibility wrapper that returns chunks in a dict for the UI layer."""

    def orchestrate_retrieval(self, query: str, chunk_dir: str):
        return super().orchestrate_retrieval(query, chunk_dir)
