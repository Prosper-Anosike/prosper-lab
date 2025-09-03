from typing import List
import openai

class LLMPrompting:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def generate_prompt(self, query: str, chunks: List[str]) -> str:
        """Generate a prompt for the LLM with citations."""
        citations = "\n".join([f"Source {i+1}: {chunk}" for i, chunk in enumerate(chunks)])
        prompt = f"Answer the following query with references:\n\nQuery: {query}\n\n{citations}\n\nAnswer:"
        return prompt

    def get_response(self, prompt: str) -> str:
        """Get response from the LLM."""
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            max_tokens=512,
            temperature=0.7
        )
        return response.choices[0].text.strip()

if __name__ == "__main__":
    query = "What is the process for document ingestion?"
    chunks = ["Document ingestion involves parsing PDFs.", "It also includes extracting text from Word documents."]
    llm = LLMPrompting()
    prompt = llm.generate_prompt(query, chunks)
    response = llm.get_response(prompt)
    print(response)
