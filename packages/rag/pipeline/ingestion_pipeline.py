import os
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from configs.settings import settings
from packages.rag.loaders.document_loader import DocumentLoader
from packages.rag.chunking.text_chunker import TextChunker
from packages.rag.tracking.file_tracker import FileTracker
from utils.RAGLogger import RAGLogger


class IngestionPipeline:
    def __init__(self, input_dir: str = None, output_dir: str = None, manifest_path: str = None):
        """Initialize the ingestion pipeline with configurable paths"""
        self.logger = RAGLogger('IngestionPipeline')
        
        # Use settings from .env if not provided
        self.input_dir = input_dir or settings.INPUT_DIR
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.manifest_path = manifest_path or settings.MANIFEST_FILE
        
        # Initialize components
        self.document_loader = DocumentLoader(self.input_dir, self.output_dir)
        self.text_chunker = TextChunker(self.output_dir, self.output_dir)
        self.file_tracker = FileTracker(self.manifest_path)
        
        self.logger.info(
            "IngestionPipeline initialized",
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            manifest_path=self.manifest_path,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def run_ingestion(self) -> Dict[str, Any]:
        """
        Run the complete ingestion pipeline.
        Returns summary of processing results.
        """
        start_time = time.time()
        pipeline_id = f"pipeline_{int(start_time)}"
        
        self.logger.info(
            "Starting ingestion pipeline",
            pipeline_id=pipeline_id,
            input_directory=self.input_dir
        )
        
        try:
            # Step 1: Load manifest
            manifest = self.file_tracker.load_manifest()
            
            # Step 2: Discover files
            all_files = self.document_loader.load_documents()
            
            # Step 3: Filter files that need processing
            files_to_process = []
            skipped_files = []
            
            for file_path in all_files:
                if self.file_tracker.should_process_file(file_path, manifest):
                    files_to_process.append(file_path)
                else:
                    skipped_files.append(file_path)
            
            self.logger.info(
                "File filtering completed",
                total_files=len(all_files),
                files_to_process=len(files_to_process),
                skipped_files=len(skipped_files)
            )
            
            # Step 4: Process files that need processing
            processing_results = {
                "successful_files": [],
                "failed_files": [],
                "skipped_files": skipped_files,
                "total_files": len(all_files),
                "files_processed": len(files_to_process)
            }
            
            if files_to_process:
                # Process each file individually for better error handling
                for file_path in files_to_process:
                    file_result = self.process_single_file(file_path, manifest)
                    
                    if file_result["status"] == "success":
                        processing_results["successful_files"].append(file_result)
                    else:
                        processing_results["failed_files"].append(file_result)
            
            # Step 5: Save updated manifest
            self.file_tracker.save_manifest(manifest)
            
            # Step 6: Generate summary
            total_time = time.time() - start_time
            summary = self.generate_summary_report(processing_results, total_time)
            
            self.logger.info(
                "Ingestion pipeline completed",
                pipeline_id=pipeline_id,
                total_time_seconds=round(total_time, 3),
                files_processed=len(files_to_process),
                successful_files=len(processing_results["successful_files"]),
                failed_files=len(processing_results["failed_files"]),
                skipped_files=len(skipped_files)
            )
            
            return summary
            
        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(
                "Ingestion pipeline failed",
                pipeline_id=pipeline_id,
                error_message=str(e),
                total_time_seconds=round(total_time, 3)
            )
            raise

    def process_single_file(self, file_path: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single file through the complete pipeline.
        Returns processing result with metadata.
        """
        file_start_time = time.time()
        
        try:
            self.logger.info(f"Processing file: {file_path.name}")
            
            # Step 1: Parse document
            text, metadata = self.document_loader.parse_single_file(file_path)
            
            if metadata["status"] != "success":
                # Document parsing failed
                processing_time = time.time() - file_start_time
                self.file_tracker.update_file_record(
                    manifest, file_path, 0, "failed", metadata["error_message"]
                )
                
                return {
                    "file_path": file_path,
                    "status": "failed",
                    "error_message": metadata["error_message"],
                    "processing_time": processing_time,
                    "chunks_created": 0
                }
            
            # Step 2: Save extracted text
            rel_path = str(file_path.relative_to(Path(self.input_dir))).replace(os.sep, '_')
            self.document_loader.save_text(rel_path, text)
            
            # Step 3: Chunk the text
            chunks, chunk_count = self.text_chunker.chunk_text(text)
            
            # Step 4: Save chunks
            self.text_chunker.save_chunks(rel_path, chunks)
            
            # Step 5: Update manifest
            processing_time = time.time() - file_start_time
            self.file_tracker.update_file_record(
                manifest, file_path, chunk_count, "success"
            )
            
            self.logger.info(
                f"Successfully processed: {file_path.name}",
                chunks_created=chunk_count,
                processing_time_seconds=round(processing_time, 3),
                text_length=len(text)
            )
            
            return {
                "file_path": file_path,
                "status": "success",
                "error_message": None,
                "processing_time": processing_time,
                "chunks_created": chunk_count,
                "text_length": len(text)
            }
            
        except Exception as e:
            # Handle unexpected errors
            processing_time = time.time() - file_start_time
            error_msg = f"Unexpected error: {str(e)}"
            
            self.file_tracker.update_file_record(
                manifest, file_path, 0, "failed", error_msg
            )
            
            self.logger.error(
                f"Failed to process file: {file_path.name}",
                error_message=error_msg,
                processing_time_seconds=round(processing_time, 3)
            )
            
            return {
                "file_path": file_path,
                "status": "failed",
                "error_message": error_msg,
                "processing_time": processing_time,
                "chunks_created": 0
            }

    def generate_summary_report(self, results: Dict[str, Any], total_time: float) -> Dict[str, Any]:
        """Generate a comprehensive summary report of the ingestion process"""
        
        successful_files = results["successful_files"]
        failed_files = results["failed_files"]
        skipped_files = results["skipped_files"]
        
        total_chunks = sum(file_result["chunks_created"] for file_result in successful_files)
        total_text_length = sum(file_result.get("text_length", 0) for file_result in successful_files)
        
        summary = {
            "pipeline_info": {
                "completed_at": datetime.now().isoformat(),
                "total_processing_time": round(total_time, 3),
                "configuration": {
                    "chunk_size": settings.CHUNK_SIZE,
                    "chunk_overlap": settings.CHUNK_OVERLAP,
                    "input_dir": self.input_dir,
                    "output_dir": self.output_dir
                }
            },
            "file_statistics": {
                "total_files_discovered": results["total_files"],
                "files_processed": results["files_processed"],
                "successful_files": len(successful_files),
                "failed_files": len(failed_files),
                "skipped_files": len(skipped_files),
                "success_rate": round((len(successful_files) / results["files_processed"]) * 100, 2) if results["files_processed"] > 0 else 100
            },
            "content_statistics": {
                "total_chunks_created": total_chunks,
                "total_text_length": total_text_length,
                "average_chunks_per_file": round(total_chunks / len(successful_files), 1) if successful_files else 0,
                "average_processing_time": round(total_time / results["files_processed"], 3) if results["files_processed"] > 0 else 0
            },
            "detailed_results": {
                "successful_files": [
                    {
                        "file_name": result["file_path"].name,
                        "chunks_created": result["chunks_created"],
                        "processing_time": round(result["processing_time"], 3)
                    }
                    for result in successful_files
                ],
                "failed_files": [
                    {
                        "file_name": result["file_path"].name,
                        "error_message": result["error_message"],
                        "processing_time": round(result["processing_time"], 3)
                    }
                    for result in failed_files
                ],
                "skipped_files": [file_path.name for file_path in skipped_files]
            }
        }
        
        return summary

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current status of the pipeline and manifest"""
        manifest = self.file_tracker.load_manifest()
        summary = self.file_tracker.get_processing_summary(manifest)
        
        return {
            "manifest_path": self.manifest_path,
            "manifest_exists": Path(self.manifest_path).exists(),
            "last_run": summary.get("last_run"),
            "total_files_tracked": summary.get("total_files", 0),
            "successful_files": summary.get("successful_files", 0),
            "failed_files": summary.get("failed_files", 0),
            "total_chunks": summary.get("total_chunks", 0),
            "success_rate": summary.get("success_rate", 0)
        }


if __name__ == "__main__":
    # Simple test run
    pipeline = IngestionPipeline()
    
    print("Pipeline Status Before:")
    status = pipeline.get_pipeline_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\nRunning ingestion pipeline...")
    results = pipeline.run_ingestion()
    
    print("\nPipeline Results:")
    print(f"  Files processed: {results['file_statistics']['files_processed']}")
    print(f"  Successful: {results['file_statistics']['successful_files']}")
    print(f"  Failed: {results['file_statistics']['failed_files']}")
    print(f"  Skipped: {results['file_statistics']['skipped_files']}")
    print(f"  Total chunks: {results['content_statistics']['total_chunks_created']}")
    print(f"  Processing time: {results['pipeline_info']['total_processing_time']}s")
