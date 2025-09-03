import os
from pathlib import Path
from typing import List
from pypdf import PdfReader
from docx import Document
from pptx import Presentation
from openpyxl import load_workbook

class DocumentLoader:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_documents(self) -> List[Path]:
        """Recursively load documents from the input directory and subfolders."""
        supported_exts = {'.pdf', '.docx', '.pptx', '.xlsx'}
        documents = []
        all_files = list(self.input_dir.rglob('*'))
        print(f"DEBUG: Found {len(all_files)} files in raw folder and subfolders.")
        for file in all_files:
            print(f"DEBUG: {file} (suffix: '{file.suffix}')")
            if file.is_file() and file.suffix.lower() in supported_exts:
                documents.append(file)
        return documents

    def parse_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF file."""
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() for page in reader.pages)
        return text

    def parse_docx(self, file_path: Path) -> str:
        """Extract text from a Word document."""
        doc = Document(file_path)
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        return text

    def parse_pptx(self, file_path: Path) -> str:
        """Extract text from a PowerPoint file."""
        presentation = Presentation(file_path)
        text = "\n".join(shape.text for slide in presentation.slides for shape in slide.shapes if hasattr(shape, 'text'))
        return text

    def parse_xlsx(self, file_path: Path) -> str:
        """Extract text from an Excel file."""
        workbook = load_workbook(file_path, data_only=True)
        text = "\n".join(
            "\n".join(str(cell.value) for cell in row if cell.value is not None)
            for sheet in workbook.worksheets
            for row in sheet.iter_rows()
        )
        return text

    def save_text(self, file_name: str, text: str):
        """Save extracted text to the output directory."""
        # Replace path separators with underscores for unique file names
        safe_name = file_name.replace(os.sep, '_')
        output_path = self.output_dir / f"{safe_name}.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

    def process_documents(self):
        """Process all documents in the input directory and subfolders."""
        files = self.load_documents()
        print(f"Found {len(files)} supported files to process.")
        for file in files:
            print(f"Processing: {file}")
            try:
                if file.suffix == '.pdf':
                    text = self.parse_pdf(file)
                elif file.suffix == '.docx':
                    text = self.parse_docx(file)
                elif file.suffix == '.pptx':
                    text = self.parse_pptx(file)
                elif file.suffix == '.xlsx':
                    text = self.parse_xlsx(file)
                else:
                    print(f"Skipping unsupported file: {file}")
                    continue
                # Use relative path from input_dir for unique file name
                rel_path = str(file.relative_to(self.input_dir)).replace(os.sep, '_')
                self.save_text(rel_path, text)
                print(f"Saved extracted text: {self.output_dir / (rel_path + '.txt')}")
            except Exception as e:
                print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    import os
    # Set project root to three levels up from this script (sharepoint-rag-bot)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    input_dir = os.path.join(root_dir, "data", "raw")
    output_dir = input_dir
    print(f"Using input_dir: {input_dir}")
    loader = DocumentLoader(input_dir, output_dir)
    loader.process_documents()
