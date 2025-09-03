import os
from pathlib import Path
from typing import List
import tiktoken

class TextChunker:
    def __init__(self, input_dir: str, output_dir: str, chunk_size: int = 512, overlap: int = 256):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_text_files(self) -> List[Path]:
        """Load text files from the input directory."""
        return [file for file in self.input_dir.iterdir() if file.suffix == '.txt']

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text)
        chunks = []

        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk = tokens[i:i + self.chunk_size]
            chunks.append(tokenizer.decode(chunk))

        return chunks

    def save_chunks(self, file_name: str, chunks: List[str]):
        """Save text chunks to the output directory."""
        output_path = self.output_dir / f"{file_name}_chunks.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(chunk + "\n\n")

    def process_text_files(self):
        """Process all text files in the input directory."""
        for file in self.load_text_files():
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()

            chunks = self.chunk_text(text)
            self.save_chunks(file.stem, chunks)

if __name__ == "__main__":
    input_dir = "data/raw"
    output_dir = "data/index"
    chunker = TextChunker(input_dir, output_dir)
    chunker.process_text_files()
