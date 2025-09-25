import json
import hashlib
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from utils.RAGLogger import RAGLogger

class FileTracker:
    def __init__(self, manifest_path: str):
        """Initiaize the file tracker with manifest path"""
        self.logger = RAGLogger('FileTracker')
        self.manifest_path=Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True,exist_ok=True)
        self.logger.info(f"FileTracker initialized with manifest: {self.manifest_path}")

    def load_manifest(self) -> Dict[str, Any]:
        """Load existing manifest file or create a new empty one"""
        try:
            if self.manifest_path.exists():
                with open(self.manifest_path,'r',encoding='utf-8') as f:
                    manifest= json.load(f)
                self.logger.info(f"Loaded existing manifest with {len(manifest.get('files',{}))} files")
                return manifest
            else:
                # Create a new manifest structure
                manifest = {
                    "pipeline_info": {
                        "version": "1.0",
                        "created_at": datetime.now().isoformat(),
                        "last_run": None,
                        "total_files": 0,
                        "successful_files": 0,
                        "failed_files": 0
                    },
                    "files": {}
                }
                self.logger.info("Created new manifest structure")
                return manifest
        except Exception as e:
            self.logger.error(f"Error loading manifest: {e}")
            # Return empty manifest if loading fails
            return {
                "pipeline_info": {
                    "version": "1.0",
                    "created_at": datetime.now().isoformat(),
                    "last_run": None,
                    "total_files": 0,
                    "successful_files": 0,
                    "failed_files": 0
                },
                "files": {}
            }
    
    def save_manifest(self, manifest: Dict[str, Any]) -> None:
        """Save manifest to disk"""
        try:
            # updating pipeline info before saving
            manifest["pipeline_info"]["last_run"] = datetime.now().isoformat()

            with open(self.manifest_path, 'w', encoding="utf-8") as f:
                json.dump(manifest,f,indent=2,ensure_ascii=False)

            self.logger.info(f"Manifest saved successfully to {self.manifest_path}")
        except Exception as e:
            self.logger.error(f"Error saving manifest: {e}")
            raise

    def get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                # Read file in chunks to handle Large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            
            file_hash = hash_sha256.hexdigest()
            self.logger.debug(f"Calculated hash for {file_path.name} : {file_hash[:16]}...")
            return file_hash
        except Exception as e:
            self.logger.error(f"Error calculating hash for {file_path} : {e}")
            raise

    def get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get fie metadata like size and modification time"""

        try:
            stat = file_path.stat()
            metadata = {
                "file_size": stat.st_size,
                "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
            return metadata
        except Exception as e:
            self.logger.error(f"Error getting metadata for {file_path}: {e}")
            return {"file_size": 0, "last_modified": None}
    
     
    def should_process_file(self, file_path: Path, manifest: Dict[str, Any]) -> bool:
        """ Check if file needs processing by comparing hashes"""
        try:
            
            file_key = str(file_path.name)

            # if file not in manifest, it's new
            if file_key not in manifest["files"]:
                self.logger.info(f"New file detected: {file_path.name}")
                return True
            
            # Calculate current hash
            current_hash = self.get_file_hash(file_path)
            stored_hash = manifest["files"][file_key].get("sha256")

            # if has is different, file was modified
            if current_hash != stored_hash:
                self.logger.info(f"File modified: {file_path.name}")
                return True
            
            self.logger.info(f"File unchanged, skipping: {file_path.name}")
            return False
        
        except Exception as e:
            self.logger.error(f"Error checking if the file should be proccessed: {e}")
            # If can't determine, process it just to be safe
            return True
        
    def update_file_record(self, manifest: Dict[str, Any], file_path: Path, chunk_count: int, status: str, error_msg: str = None) -> None:
        """Update manifest with file processing results"""
        try:
            file_key = str(file_path.name)

            # Get file metada
            metadata = self.get_file_metadata(file_path)
            file_hash = self.get_file_hash(file_path) if status == "success" else None

            # Update file Record
            manifest["files"][file_key] = {
                "sha256": file_hash,
                "chunk_count": chunk_count,
                "file_size": metadata["file_size"],
                "last_modified": metadata["last_modified"],
                "processed_at": datetime.now().isoformat(),
                "status": status,
                "error_message": error_msg
            }

            if status == "success":
                manifest["pipeline_info"]["successful_files"] = manifest["pipeline_info"].get("successful_files", 0 ) + 1
            else:
                 manifest["pipeline_info"]["failed_files"] = manifest["pipeline_info"].get("failed_files", 0) + 1
            
            manifest["pipeline_info"]["total_files"] = len(manifest["files"])
            
            self.logger.info(f"Updated record for {file_path.name}: {status}")
            
        except Exception as e:
            self.logger.error(f"Error updating file record: {e}")
            raise

    def get_processing_summary(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from manifest"""
        pipeline_info = manifest.get("pipeline_info", {})
        files = manifest.get("files", {})
        
        summary = {
            "total_files": len(files),
            "successful_files": pipeline_info.get("successful_files", 0),
            "failed_files": pipeline_info.get("failed_files", 0),
            "total_chunks": sum(file_info.get("chunk_count", 0) for file_info in files.values()),
            "last_run": pipeline_info.get("last_run"),
            "success_rate": 0
        }
        
        if summary["total_files"] > 0:
            summary["success_rate"] = round((summary["successful_files"] / summary["total_files"]) * 100, 2)
        
        return summary
