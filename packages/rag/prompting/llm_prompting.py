from typing import List
from openai import AzureOpenAI
from configs.settings import settings

class LLMPrompting:
    def __init__(self, model: str | None = None):
        self.model = model or settings.AZURE_CHAT_DEPLOYMENT
        self.client = AzureOpenAI(
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
        )

    def generate_prompt(self, query: str, chunks: List[str]) -> list[dict]:
        if chunks:
            sources = "\n\n".join([f"### S{i+1}\n{c.strip()}" for i, c in enumerate(chunks)])
        else:
            sources = "### S1\n[NO_SOURCES]"

        system_msg = (
            "You are a retrieval-augmented assistant. "
            "Use ONLY the provided sources. "
            "If the answer is not fully supported, say you don’t know. "
            "Cite sources inline as [S1], [S2], etc."
        )
        user_msg = f"# Question\n{query.strip()}\n\n# Sources\n{sources}\n\nAnswer using only these sources."
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    def get_response(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=700,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    query = "What is the process for document ingestion?"
    chunks = [
        "Document ingestion involves parsing PDFs.",
        "It also includes extracting text from Word documents.",
    ]
    llm = LLMPrompting()
    messages = llm.generate_prompt(query, chunks)
    response = llm.get_response(messages)
    print(response)
