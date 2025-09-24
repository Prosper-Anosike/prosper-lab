import os
from pathlib import Path
from typing import List
import tiktoken
from utils.RAGLogger import RAGLogger
import time

class TextChunker:
    def __init__(self, input_dir: str, output_dir: str, chunk_size: int = 512, overlap: int = 256):
        self.logger = RAGLogger('TextChunker')
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_text_files(self) -> List[Path]:
        """Load text files from the input directory."""
        self.logger.debug("Starting text file discovery")
        return [file for file in self.input_dir.iterdir() if file.suffix == '.txt']

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        start_time = time.time()
        
        self.logger.debug(
            f"Starting text chunking",
            text_length = len(text),
            chunk_size = self.chunk_size,
            overlap=self.overlap
        )
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text)
        chunks = []

        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk = tokens[i:i + self.chunk_size]
            chunks.append(tokenizer.decode(chunk))

        processing_time = time.time() - start_time

        self.logger.debug(
            "Text chunking completed",
            original_tokens=len(tokens),
            chunks_created = len(chunks),
            average_chunk_tokens=len(tokens) // len(chunks) if chunks else 0,
            processing_time_seconds=round(processing_time,3),
            token_efficiency=round(len(tokens) / len(chunks) * self.chunk_size, 3) if chunks else 0

        )

        return chunks

    def save_chunks(self, file_name: str, chunks: List[str]):
        """Save text chunks to the output directory."""
        try:
            output_path = self.output_dir / f"{file_name}_chunks.txt"

            self.logger.debug(
                "Saving chunks to file",
                original_filename=file_name,
                output_path=str(output_path),
                chunks_count=len(chunks),
                total_characters=sum(len(chunk) for chunk in chunks)
            )

            with open(output_path, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(chunk + "\n\n")
            
            # Verify file creation

            if output_path.exists():
                file_size = output_path.stat().st_size
                self.logger.debug(
                    "Chunks saved successfully",
                    output_file=str(output_path),
                    file_size_bytes=file_size,
                    chunks_written=len(chunks)
                )
            else:
                self.logger.error(f"Failed to create chunks files: {output_path}")
        except Exception as e:
            self.logger.error(
                "Failed to save chunks",
                filename=file_name,
                error_message=str(e),
                chunks_count=len(chunks)
            )
            raise

    def process_text_files(self):
        """Process all text files in the input directory."""
        start_time = time.time()
        files = self.load_text_files()

        if not files:
            self.logger.warning(
                "No text files found for processing",
                input_directory = str(self.input_dir)
            )
            return
        
        self.logger.info(
            f"Starting text chunking batch",
            total_files=len(files),
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            batch_id=f"chunk_batch_{int(start_time)}"
        )
        
        successful_files = 0
        failed_files = 0
        total_chunks = 0
        total_characters = 0


        for file in files:
            file_start_time = time.time()
            try:
                self.logger.info(
                    "Processing text file",
                    filename=file.name,
                    file_path=str(file),
                    file_size_bytes=file.stat().st_size
                )

                with open(file, 'r', encoding='utf-8') as f:
                    text = f.read()

                chunks = self.chunk_text(text)
                self.save_chunks(file.stem, chunks)

                file_processing_time = time.time() - file_start_time
                total_chunks += len(chunks)
                total_characters += len(text)

                self.logger.info(
                    "Text file processed successfully",
                    filename=file.name,
                    chunks_created=len(chunks),
                    original_characters=len(text),
                    processing_time_seconds=round(file_processing_time,3),
                    success=True
                )

                successful_files += 1

            except Exception as e:
                file_processing_time = time.time() - file_start_time

                self.logger.error(
                    "Failed to process text file",
                    filename=file.name,
                    file_path=str(file),
                    error_message=str(e),
                    processing_time_seconds=round(file_processing_time, 3),
                    success=False
                )

                failed_files += 1

        total_time = time.time() - start_time

        self.logger.info(
            "Text Chunking batch completed",
            total_files_processed=len(files),
            successful_files=successful_files,
            failed_files=failed_files,
            total_chunks_created = total_chunks,
            total_characters_processed  = total_characters,
            total_processing_time_seconds = round(total_time, 3),
            average_time_per_file=round(total_time / len(files), 3) if files else 0,
            average_chunks_per_file = round(total_chunks / successful_files, 1) if successful_files else 0
        )


if __name__ == "__main__":
    input_dir = "data/raw"
    output_dir = "data/index"
    chunker = TextChunker(input_dir, output_dir)
    chunker.process_text_files()
