from typing import List
from openai import AzureOpenAI
from configs.settings import settings
from utils.RAGLogger import RAGLogger
import time

class LLMPrompting:
    def __init__(self, model: str | None = None):
        self.logger = RAGLogger('LLMPrompting')
        self.model = model or settings.AZURE_CHAT_DEPLOYMENT
        
        try:
            self.client = AzureOpenAI(
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
            )
            
            self.logger.info(
                "LLMPrompting initialized successfully",
                model=self.model,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT[:50] + "..." if len(settings.AZURE_OPENAI_ENDPOINT) > 50 else settings.AZURE_OPENAI_ENDPOINT,
                api_key_configured=bool(settings.AZURE_OPENAI_API_KEY)
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to initialize LLMPrompting",
                model=self.model,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise

    def generate_prompt(self, query: str, chunks: List[str]) -> list[dict]:
        start_time = time.time()
        
        self.logger.debug(
            "Starting prompt generation",
            query_preview=query[:50] + "..." if len(query) > 50 else query,
            query_length=len(query),
            query_words=len(query.split()),
            chunks_provided=len(chunks),
            has_sources=bool(chunks)
        )
        
        # Process chunks into sources
        if chunks:
            sources = "\n\n".join([f"### S{i+1}\n{c.strip()}" for i, c in enumerate(chunks)])
            
            # Calculate source statistics
            total_source_chars = sum(len(c.strip()) for c in chunks)
            avg_chunk_length = round(total_source_chars / len(chunks), 1) if chunks else 0
            
            self.logger.debug(
                "Sources processed for prompt",
                sources_count=len(chunks),
                total_source_characters=total_source_chars,
                average_chunk_length=avg_chunk_length,
                longest_chunk=max(len(c.strip()) for c in chunks) if chunks else 0,
                shortest_chunk=min(len(c.strip()) for c in chunks) if chunks else 0
            )
        else:
            sources = "### S1\n[NO_SOURCES]"
            self.logger.warning(
                "No sources provided for prompt generation",
                query_preview=query[:50] + "..." if len(query) > 50 else query
            )

        # Generate system and user messages
        system_msg = (
            "You are a retrieval-augmented assistant. "
            "Use ONLY the provided sources. "
            "If the answer is not fully supported, say you don't know. "
            "Cite sources inline as [S1], [S2], etc."
        )
        user_msg = f"# Question\n{query.strip()}\n\n# Sources\n{sources}\n\nAnswer using only these sources."
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        
        # Calculate prompt statistics
        total_prompt_chars = len(system_msg) + len(user_msg)
        estimated_tokens = round(total_prompt_chars / 4)  # Rough estimate: 4 chars per token
        generation_time = time.time() - start_time
        
        self.logger.info(
            "Prompt generation completed",
            query_preview=query[:50] + "..." if len(query) > 50 else query,
            sources_included=len(chunks),
            system_message_length=len(system_msg),
            user_message_length=len(user_msg),
            total_prompt_characters=total_prompt_chars,
            estimated_prompt_tokens=estimated_tokens,
            generation_time_seconds=round(generation_time, 3)
        )
        
        return messages

    def get_response(self, messages: list[dict]) -> str:
        start_time = time.time()
        
        # Calculate input statistics
        total_input_chars = sum(len(msg["content"]) for msg in messages)
        estimated_input_tokens = round(total_input_chars / 4)  # Rough estimate
        
        self.logger.debug(
            "Starting LLM response generation",
            model=self.model,
            messages_count=len(messages),
            total_input_characters=total_input_chars,
            estimated_input_tokens=estimated_input_tokens,
            max_tokens=700,
            temperature=0.0
        )
        
        try:
            api_start_time = time.time()
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=700,
                temperature=0.0,
            )
            api_time = time.time() - api_start_time
            
            response_content = resp.choices[0].message.content.strip()
            total_time = time.time() - start_time
            
            # Extract token usage if available
            usage = resp.usage if hasattr(resp, 'usage') else None
            prompt_tokens = usage.prompt_tokens if usage else estimated_input_tokens
            completion_tokens = usage.completion_tokens if usage else round(len(response_content) / 4)
            total_tokens = usage.total_tokens if usage else prompt_tokens + completion_tokens
            
            # Estimate cost (rough calculation for GPT-4)
            estimated_cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000
            
            # Analyze response quality
            response_words = len(response_content.split())
            has_citations = any(f"[S{i}]" in response_content for i in range(1, 10))
            
            self.logger.info(
                "LLM response generated successfully",
                model=self.model,
                response_length=len(response_content),
                response_words=response_words,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                api_call_time_seconds=round(api_time, 3),
                total_time_seconds=round(total_time, 3),
                estimated_cost_usd=round(estimated_cost, 4),
                has_source_citations=has_citations,
                finish_reason=resp.choices[0].finish_reason if hasattr(resp.choices[0], 'finish_reason') else "unknown"
            )
            
            return response_content
            
        except Exception as e:
            total_time = time.time() - start_time
            
            self.logger.error(
                "Failed to generate LLM response",
                model=self.model,
                messages_count=len(messages),
                estimated_input_tokens=estimated_input_tokens,
                error_type=type(e).__name__,
                error_message=str(e),
                total_time_seconds=round(total_time, 3)
            )
            raise

    def generate_rag_response(self, query: str, chunks: List[str]) -> str:
        """Complete RAG response generation with comprehensive logging."""
        start_time = time.time()
        
        self.logger.info(
            "Starting complete RAG response generation",
            query_preview=query[:50] + "..." if len(query) > 50 else query,
            query_length=len(query),
            chunks_provided=len(chunks)
        )
        
        try:
            # Step 1: Generate prompt
            prompt_start_time = time.time()
            messages = self.generate_prompt(query, chunks)
            prompt_time = time.time() - prompt_start_time
            
            # Step 2: Get LLM response
            response_start_time = time.time()
            response = self.get_response(messages)
            response_time = time.time() - response_start_time
            
            total_time = time.time() - start_time
            
            self.logger.info(
                "RAG response generation completed successfully",
                query_preview=query[:50] + "..." if len(query) > 50 else query,
                response_preview=response[:100] + "..." if len(response) > 100 else response,
                response_length=len(response),
                prompt_generation_time_seconds=round(prompt_time, 3),
                llm_response_time_seconds=round(response_time, 3),
                total_time_seconds=round(total_time, 3)
            )
            
            return response
            
        except Exception as e:
            total_time = time.time() - start_time
            
            self.logger.error(
                "RAG response generation failed",
                query_preview=query[:50] + "..." if len(query) > 50 else query,
                chunks_provided=len(chunks),
                error_type=type(e).__name__,
                error_message=str(e),
                total_time_seconds=round(total_time, 3)
            )
            raise


if __name__ == "__main__":
    logger = RAGLogger('LLMPrompting-Main')
    logger.info("Starting LLM prompting script execution")
    
    query = "What is the process for document ingestion?"
    chunks = [
        "Document ingestion involves parsing PDFs.",
        "It also includes extracting text from Word documents.",
    ]
    
    logger.info(
        "Script configuration",
        query=query,
        chunks_count=len(chunks),
        total_chunk_characters=sum(len(chunk) for chunk in chunks)
    )
    
    try:
        llm = LLMPrompting()
        
        # Method 1: Step by step
        logger.info("Testing step-by-step approach")
        messages = llm.generate_prompt(query, chunks)
        response = llm.get_response(messages)
        
        # Method 2: Complete RAG response
        logger.info("Testing complete RAG response approach")
        rag_response = llm.generate_rag_response(query, chunks)
        
        logger.info(
            "LLM prompting script execution completed successfully",
            step_by_step_response_length=len(response),
            rag_response_length=len(rag_response),
            responses_match=response == rag_response
        )
        
        print("\n=== Generated Response ===")
        print(response)
        
    except Exception as e:
        logger.error(
            "LLM prompting script execution failed",
            error_type=type(e).__name__,
            error_message=str(e)
        )
        raise
