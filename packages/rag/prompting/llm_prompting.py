from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Sequence

from utils.RAGLogger import RAGLogger


class LLMPrompting:
    """Deterministic local summarizer that never calls external LLMs."""

    def __init__(self, max_highlights: int = 4, min_chunk_chars: int = 40):
        self.logger = RAGLogger("LLMPrompting")
        self.max_highlights = max(1, max_highlights)
        self.min_chunk_chars = max(1, min_chunk_chars)

        self.logger.info(
            "LLMPrompting initialized in local mode",
            max_highlights=self.max_highlights,
            min_chunk_chars=self.min_chunk_chars,
        )

    def generate_answer(
        self,
        query: str,
        chunks: Sequence[str],
        citations: List[Dict[str, Any]] | None = None,
        retrieval_stats: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        normalized_chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

        if not normalized_chunks:
            self.logger.warning(
                "No chunks available for local summarization",
                query_preview=query[:80] + "..." if len(query) > 80 else query,
            )
            fallback = "I could not find any supporting passages for that question in the local index."
            return {
                "answer": fallback,
                "citations": [],
                "highlights": [],
                "stats": {
                    "answer_time_seconds": round(time.time() - start_time, 3),
                    "chunks_available": 0,
                    "chunks_used": 0,
                },
            }

        resolved_citations = self._ensure_citations(normalized_chunks, citations)
        scored_chunks = self._score_chunks(query, normalized_chunks)
        highlights = self._select_highlights(scored_chunks, resolved_citations)

        answer_markdown = self._compose_answer(query, highlights, resolved_citations, retrieval_stats)
        total_time = time.time() - start_time

        self.logger.info(
            "Local summary generated",
            query_preview=query[:80] + "..." if len(query) > 80 else query,
            chunks_available=len(normalized_chunks),
            highlights=len(highlights),
            answer_time_seconds=round(total_time, 3),
        )

        return {
            "answer": answer_markdown,
            "citations": resolved_citations,
            "highlights": highlights,
            "stats": {
                "answer_time_seconds": round(total_time, 3),
                "chunks_available": len(normalized_chunks),
                "chunks_used": len(highlights),
            }
        }

    def _score_chunks(self, query: str, chunks: Sequence[str]) -> List[Dict[str, Any]]:
        query_terms = set(self._tokenize(query)) or set()
        scored: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            chunk_terms = set(self._tokenize(chunk))
            overlap = len(chunk_terms & query_terms)
            coverage = min(len(chunk), 800) / 800
            recency_bonus = 1 / (idx + 1)
            score = overlap * 4 + coverage + recency_bonus

            if len(chunk) < self.min_chunk_chars:
                score *= 0.5

            scored.append({
                "index": idx,
                "text": chunk,
                "score": round(score, 4),
            })

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored

    def _select_highlights(
        self,
        scored_chunks: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not scored_chunks:
            return []

        top_entries = scored_chunks[: self.max_highlights] or scored_chunks
        highlights: List[Dict[str, Any]] = []

        for entry in top_entries:
            citation = citations[entry["index"]]
            highlights.append(
                {
                    "text": self._summarize_chunk(entry["text"]),
                    "citation_id": citation["id"],
                    "citation_label": citation["label"],
                    "doc_name": citation["doc_name"],
                    "score": entry["score"],
                }
            )

        return highlights

    def _compose_answer(
        self,
        query: str,
        highlights: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        retrieval_stats: Dict[str, Any] | None,
    ) -> str:
        if not highlights:
            return "No grounded answer is available for this question."

        lead = highlights[0]["text"]
        intro = (
            f"{lead} This directly addresses the question \"{query.strip()}\" using {len(highlights)} "
            "evidence-backed passages."
        )

        answer_lines = ["### Executive Answer", intro, "", "### Key Evidence"]

        for highlight in highlights:
            answer_lines.append(
                f"- {highlight['text']} [{highlight['citation_id']}]"
            )

        answer_lines.append("")
        answer_lines.append("### Sources")
        for citation in citations:
            link_target = citation.get("source_path") or "#"
            preview = citation.get("preview", "")
            score = citation.get("score")
            score_text = f" · score {score:.3f}" if isinstance(score, (int, float)) else ""
            answer_lines.append(
                f"- [{citation['id']} — {citation['label']}]({link_target}){score_text} — {preview}"
            )

        if retrieval_stats:
            stats_line = (
                f"\n> Retrieval: {retrieval_stats.get('chunks_retrieved', '-') } chunks | "
                f"Embed {retrieval_stats.get('embedding_time_seconds', '-') }s | "
                f"Search {retrieval_stats.get('search_time_seconds', '-') }s | "
                f"Fetch {retrieval_stats.get('fetch_time_seconds', '-') }s"
            )
            answer_lines.append(stats_line)

        return "\n".join(answer_lines)

    def _ensure_citations(
        self,
        chunks: Sequence[str],
        citations: List[Dict[str, Any]] | None,
    ) -> List[Dict[str, Any]]:
        if citations and len(citations) == len(chunks):
            return citations

        fallback: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks, start=1):
            fallback.append(
                {
                    "id": f"S{idx}",
                    "doc_name": f"chunk-{idx}",
                    "chunk_index": idx,
                    "source_path": "",
                    "preview": self._build_preview(chunk),
                    "label": f"chunk-{idx}",
                    "score": None,
                }
            )

        return fallback

    def _summarize_chunk(self, chunk: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", chunk.strip())
        for sentence in sentences:
            cleaned = sentence.strip()
            if cleaned:
                return cleaned if len(cleaned) <= 220 else cleaned[:220].rstrip() + "..."
        return self._build_preview(chunk)

    def _build_preview(self, text: str, limit: int = 180) -> str:
        snippet = " ".join(text.split())
        if len(snippet) <= limit:
            return snippet
        return snippet[:limit].rstrip() + "..."

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9]+", text.lower())


if __name__ == "__main__":
    demo_query = "Summarize the governance model."
    demo_chunks = [
        "Governance briefing slide outlines the approval stages and RACI.",
        "Overview document highlights that IT owns the ingestion process.",
    ]

    generator = LLMPrompting()
    result = generator.generate_answer(demo_query, demo_chunks)
    print(result["answer"])
