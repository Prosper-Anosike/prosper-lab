#!/usr/bin/env python3
"""
Ingestion Pipeline Runner

This script runs the complete document ingestion pipeline for the SharePoint RAG bot.
It processes documents from the configured input directory, chunks them, and tracks
everything in a manifest file for efficient re-processing.

Usage:
    python scripts/run_ingestion.py [options]

Options:
    --input-dir PATH     Override input directory (default: from .env)
    --output-dir PATH    Override output directory (default: from .env)
    --manifest PATH      Override manifest file path (default: from .env)
    --status            Show current pipeline status without running
    --verbose           Enable verbose logging
    --help              Show this help message

Examples:
    # Run with default settings from .env
    python scripts/run_ingestion.py
    
    # Check status without running
    python scripts/run_ingestion.py --status
    
    # Run with custom input directory
    python scripts/run_ingestion.py --input-dir /path/to/documents
    
    # Run with verbose logging
    python scripts/run_ingestion.py --verbose
"""

import sys
import argparse
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from packages.rag.pipeline.ingestion_pipeline import IngestionPipeline
from utils.RAGLogger import RAGLogger


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run the document ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory containing documents to process"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        help="Output directory for processed files"
    )
    
    parser.add_argument(
        "--manifest",
        type=str,
        help="Path to manifest file for tracking processed files"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current pipeline status without running ingestion"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output"
    )
    
    return parser.parse_args()


def print_status(pipeline: IngestionPipeline):
    """Print current pipeline status"""
    print("=" * 60)
    print("INGESTION PIPELINE STATUS")
    print("=" * 60)
    
    status = pipeline.get_pipeline_status()
    
    print(f"Manifest File: {status['manifest_path']}")
    print(f"Manifest Exists: {'✓' if status['manifest_exists'] else '✗'}")
    print(f"Last Run: {status['last_run'] or 'Never'}")
    print()
    
    print("File Statistics:")
    print(f"  Total Files Tracked: {status['total_files_tracked']}")
    print(f"  Successful Files: {status['successful_files']}")
    print(f"  Failed Files: {status['failed_files']}")
    print(f"  Success Rate: {status['success_rate']}%")
    print()
    
    print(f"Total Chunks Created: {status['total_chunks']}")
    print()


def print_results(results: dict):
    """Print pipeline execution results"""
    print("=" * 60)
    print("INGESTION PIPELINE RESULTS")
    print("=" * 60)
    
    # Pipeline info
    pipeline_info = results["pipeline_info"]
    print(f"Completed At: {pipeline_info['completed_at']}")
    print(f"Total Processing Time: {pipeline_info['total_processing_time']}s")
    print()
    
    # Configuration
    config = pipeline_info["configuration"]
    print("Configuration:")
    print(f"  Chunk Size: {config['chunk_size']} tokens")
    print(f"  Chunk Overlap: {config['chunk_overlap']} tokens")
    print(f"  Input Directory: {config['input_dir']}")
    print(f"  Output Directory: {config['output_dir']}")
    print()
    
    # File statistics
    file_stats = results["file_statistics"]
    print("File Processing:")
    print(f"  Total Files Discovered: {file_stats['total_files_discovered']}")
    print(f"  Files Processed: {file_stats['files_processed']}")
    print(f"  Successful: {file_stats['successful_files']} ✓")
    print(f"  Failed: {file_stats['failed_files']} ✗")
    print(f"  Skipped (unchanged): {file_stats['skipped_files']} ⏭")
    print(f"  Success Rate: {file_stats['success_rate']}%")
    print()
    
    # Content statistics
    content_stats = results["content_statistics"]
    print("Content Processing:")
    print(f"  Total Chunks Created: {content_stats['total_chunks_created']}")
    print(f"  Total Text Length: {content_stats['total_text_length']:,} characters")
    print(f"  Average Chunks per File: {content_stats['average_chunks_per_file']}")
    print(f"  Average Processing Time: {content_stats['average_processing_time']}s per file")
    print()
    
    # Detailed results
    detailed = results["detailed_results"]
    
    if detailed["successful_files"]:
        print("Successfully Processed Files:")
        for file_info in detailed["successful_files"]:
            print(f"  ✓ {file_info['file_name']} ({file_info['chunks_created']} chunks, {file_info['processing_time']}s)")
        print()
    
    if detailed["failed_files"]:
        print("Failed Files:")
        for file_info in detailed["failed_files"]:
            print(f"  ✗ {file_info['file_name']}: {file_info['error_message']}")
        print()
    
    if detailed["skipped_files"]:
        print("Skipped Files (unchanged):")
        for file_name in detailed["skipped_files"]:
            print(f"  ⏭ {file_name}")
        print()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup logging
    logger = RAGLogger('IngestionRunner')
    
    if args.verbose:
        print("Verbose logging enabled")
        print()
    
    try:
        # Initialize pipeline with optional overrides
        pipeline = IngestionPipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            manifest_path=args.manifest
        )
        
        if args.status:
            # Just show status
            print_status(pipeline)
            return
        
        # Show status before running
        print_status(pipeline)
        
        # Run the pipeline
        print("Starting ingestion pipeline...")
        print()
        
        results = pipeline.run_ingestion()
        
        # Show results
        print_results(results)
        
        # Summary message
        file_stats = results["file_statistics"]
        if file_stats["files_processed"] == 0:
            print("🎉 No new files to process - all files are up to date!")
        elif file_stats["failed_files"] == 0:
            print("🎉 All files processed successfully!")
        else:
            print(f"⚠️  Processing completed with {file_stats['failed_files']} failed files.")
            print("   Check the logs above for error details.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        print("Check the logs for more details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
