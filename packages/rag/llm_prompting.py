from __future__ import annotations

from typing import List

from .prompting.llm_prompting import LLMPrompting as CoreLLM


class LLMPrompting(CoreLLM):
    """Compatibility wrapper that threads optional settings into the core generator."""

    def __init__(self, settings=None):
        # The local summarizer no longer requires a remote model name, but we
        # preserve the hook so future configs can tweak behavior via settings.
        super().__init__()

    def generate_answer(
        self,
        query: str,
        chunks: List[str],
        citations=None,
        retrieval_stats=None,
    ):
        return super().generate_answer(
            query=query,
            chunks=chunks,
            citations=citations,
            retrieval_stats=retrieval_stats,
        )
