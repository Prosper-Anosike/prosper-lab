from typing import List
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

class LLMPrompting:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint="https://byupwai5996918872.openai.azure.com/",
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def generate_prompt(self, query: str, chunks: List[str]) -> str:
        """Generate a prompt for the LLM with citations."""
        citations = "\n".join([f"Source {i+1}: {chunk}" for i, chunk in enumerate(chunks)])
        prompt = f"Answer the following query with references:\n\nQuery: {query}\n\n{citations}\n\nAnswer:"
        return prompt

    def get_response(self, prompt: str) -> str:
        """Get response from the LLM."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

if __name__ == "__main__":
    query = "What is the process for document ingestion?"
    chunks = ["Document ingestion involves parsing PDFs.", "It also includes extracting text from Word documents."]
    llm = LLMPrompting()
    prompt = llm.generate_prompt(query, chunks)
    response = llm.get_response(prompt)
    print(response)
