import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import gradio as gr

ROOT_DIR = Path(__file__).resolve().parents[2]
PACKAGES_DIR = ROOT_DIR / "packages"
for path in (ROOT_DIR, PACKAGES_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from configs.settings import settings
from rag.retriever import Retriever
from rag.llm_prompting import LLMPrompting


retriever = Retriever(settings.OUTPUT_DIR)
llm = LLMPrompting(settings)


def build_local_answer(chunks: List[str]) -> str:
    combined = "\n\n".join(chunks) if chunks else "[No retrieved content]"
    excerpt = combined[:1000]
    return (
        "I could not reach the local summarizer, so here is a direct excerpt from the retrieved documents:\n\n"
        + excerpt
    )


def render_sidebar(history_entries: List[Dict[str, str]]) -> str:
    if not history_entries:
        return "**Session History**\n\n_Start a conversation to see it here._"

    blocks = ["**Session History**"]
    for item in history_entries[-12:][::-1]:
        blocks.append(
            f"- **{item['timestamp']}** · {item['question']}\n  - {item['answer_preview']}"
        )
    return "\n\n".join(blocks)


def format_citations(citations: List[Dict[str, str]]) -> str:
    if not citations:
        return "_No citations available for this response._"

    lines = ["**Citations**"]
    for citation in citations:
        link_target = citation.get("source_path") or "#"
        lines.append(
            f"- [{citation['id']} — {citation['label']}]({link_target}) — {citation.get('preview', '')}"
        )
    return "\n".join(lines)


def normalize_chat_messages(history: List[Any] | None) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    if not history:
        return normalized

    for item in history:
        if isinstance(item, dict) and {"role", "content"}.issubset(item):
            normalized.append({
                "role": str(item.get("role", "assistant")),
                "content": str(item.get("content", "")),
            })
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user, bot = item
            normalized.append({"role": "user", "content": str(user)})
            normalized.append({"role": "assistant", "content": str(bot)})

    return normalized


def decorate_answer(answer_markdown: str, elapsed: float, retrieval_stats: Dict[str, Any] | None, stats: Dict[str, Any]) -> str:
    timing = f"⏱️ {elapsed:.2f}s end-to-end | chunks used: {stats.get('chunks_used', '-')}/{stats.get('chunks_available', '-')}"
    if retrieval_stats:
        timing += (
            f" · embed {retrieval_stats.get('embedding_time_seconds', '-') }s"
            f" · search {retrieval_stats.get('search_time_seconds', '-') }s"
            f" · fetch {retrieval_stats.get('fetch_time_seconds', '-') }s"
        )
    return f"{answer_markdown}\n\n---\n{timing}\n(Local-only summarization)"


def update_sidebar(history_entries: List[Dict[str, str]], question: str, payload: Dict[str, Any], elapsed: float) -> List[Dict[str, str]]:
    first_highlight = payload.get("highlights", [{}])[0]
    preview = first_highlight.get("text") or payload.get("answer", "")[:120]
    history_entries.append(
        {
            "timestamp": datetime.now().strftime("%H:%M"),
            "question": question,
            "answer_preview": preview,
            "elapsed": f"{elapsed:.2f}s",
        }
    )
    return history_entries[-20:]


def answer_question(message: str, chat_history: List[List[str]], sidebar_history: List[Dict[str, str]]):
    start_time = time.time()
    chat_history = normalize_chat_messages(chat_history)
    sidebar_history = list(sidebar_history or [])

    try:
        retrieval = retriever.orchestrate_retrieval(message, settings.OUTPUT_DIR)
        chunks = retrieval.get("chunks", [])
        citations = retrieval.get("citations", [])
        retrieval_stats = retrieval.get("stats")

        if not chunks:
            raise ValueError("No retrieval results available")

        payload = llm.generate_answer(
            query=message,
            chunks=chunks,
            citations=citations,
            retrieval_stats=retrieval_stats,
        )

        elapsed = time.time() - start_time
        response = decorate_answer(payload["answer"], elapsed, retrieval_stats, payload["stats"])
        sidebar_history = update_sidebar(sidebar_history, message, payload, elapsed)
        citations_md = format_citations(payload["citations"])
        assistant_response = response

    except Exception as exc:  # pragma: no cover - defensive path
        fallback = build_local_answer([])
        response = f"⚠️ {exc}\n\n{fallback}"
        citations_md = "_Citations unavailable due to error._"
        elapsed = time.time() - start_time
        assistant_response = response

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": assistant_response})
    sidebar_md = render_sidebar(sidebar_history)
    return chat_history, "", sidebar_md, sidebar_history, citations_md


CUSTOM_CSS = """
#chat-app {
    background: linear-gradient(135deg, #0d1b2a, #1b263b);
    min-height: 100vh;
    color: #e0e6ed;
}
#chat-app .gr-block.gr-box {
    border: none;
}
#history-panel {
    background: rgba(15, 23, 42, 0.8);
    border-radius: 16px;
    padding: 16px;
    color: #cbd5f5;
}
#history-panel h1, #history-panel h2, #history-panel strong {
    color: #f8fafc;
}
#chatbot-panel .gr-chatbot {
    background: #0b1624;
    border-radius: 18px;
}
.gr-chatbot-message.user {
    background: #1e2a3a;
}
.gr-chatbot-message.bot {
    background: #122033;
}
#input-panel textarea {
    border-radius: 12px;
}
"""


with gr.Blocks(title="SharePoint RAG Assistant", elem_id="chat-app") as demo:
    gr.Markdown("# SharePoint RAG Assistant")

    with gr.Row():
        with gr.Column(scale=1, elem_id="history-panel"):
            sidebar_state = gr.State([])
            sidebar_markdown = gr.Markdown(render_sidebar([]), elem_id="sidebar-md")
            citations_box = gr.Markdown("**Citations**\n_Start asking to see sources._")

        with gr.Column(scale=3, elem_id="chatbot-panel"):
            chatbot = gr.Chatbot(
                height=520,
                show_label=False,
                avatar_images=(None, None),
            )
            msg = gr.Textbox(placeholder="Ask anything about the indexed governance docs...", show_label=False)
            helper = gr.Markdown("↵ Enter to send · Shift+Enter for newline", elem_id="input-panel")

    msg.submit(
        fn=answer_question,
        inputs=[msg, chatbot, sidebar_state],
        outputs=[chatbot, msg, sidebar_markdown, sidebar_state, citations_box],
    )


def launch_demo():
    launch_kwargs = {
        "server_name": "127.0.0.1",
        "server_port": 8080,
        "share": False,
    }

    try:
        demo.launch(css=CUSTOM_CSS, theme="soft", **launch_kwargs)
    except OSError as exc:
        if "Cannot find empty port" not in str(exc):
            raise

        print("Port 8080 is busy. Retrying on the next available port...")
        launch_kwargs["server_port"] = None
        demo.launch(css=CUSTOM_CSS, theme="soft", **launch_kwargs)


launch_demo()
