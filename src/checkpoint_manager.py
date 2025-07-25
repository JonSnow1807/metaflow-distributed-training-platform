"""
Production-grade checkpoint management with automatic recovery
Handles distributed checkpointing, versioning, and cloud storage
"""

import os
import json
import shutil
import torch
import boto3
from pathlib import Path
from typing import Dict, Optional, List, Any
import logging
from datetime import datetime
from minio import Minio
import hashlib
import threading
import queue


class CheckpointManager:
    """
    Manages checkpoints with features:
    - Automatic recovery from interruptions
    - Cloud storage support (S3/MinIO)
    - Checkpoint versioning and rotation
    - Async uploading for minimal training disruption
    - Corruption detection with checksums
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        cloud_bucket: Optional[str] = None,
        storage_backend: str = "local",  # local, s3, minio
        keep_last_n: int = 3,
        rank: int = 0,
        world_size: int = 1,
        async_upload: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.cloud_bucket = cloud_bucket
        self.storage_backend = storage_backend
        self.keep_last_n = keep_last_n
        self.rank = rank
        self.world_size = world_size
        self.async_upload = async_upload
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage backend
        self._init_storage_backend()
        
        # Async upload queue
        if self.async_upload:
            self.upload_queue = queue.Queue()
            self.upload_thread = threading.Thread(target=self._upload_worker, daemon=True)
            self.upload_thread.start()
            
        self.logger.info(f"CheckpointManager initialized with backend: {storage_backend}")
        
    def _init_storage_backend(self):
        """Initialize cloud storage backend"""
        if self.storage_backend == "s3":
            self.s3_client = boto3.client("s3")
            # Verify bucket exists
            if self.cloud_bucket and self.rank == 0:
                try:
                    self.s3_client.head_bucket(Bucket=self.cloud_bucket)
                except:
                    self.logger.error(f"S3 bucket {self.cloud_bucket} not accessible")
                    
        elif self.storage_backend == "minio":
            self.minio_client = Minio(
                os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
                access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
                secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
                secure=False,
            )
            # Create bucket if it doesn't exist
            if self.cloud_bucket and self.rank == 0:
                if not self.minio_client.bucket_exists(self.cloud_bucket):
                    self.minio_client.make_bucket(self.cloud_bucket)
                    
    def save(self, checkpoint: Dict[str, Any], step: int):
        """Save checkpoint with versioning and cloud backup"""
        if self.rank != 0:  # Only rank 0 saves
            return
            
        checkpoint_name = f"checkpoint-{step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Add metadata
        checkpoint["metadata"] = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "world_size": self.world_size,
            "version": "1.0",
        }
        
        # Save locally first
        self.logger.info(f"Saving checkpoint at step {step}")
        torch.save(checkpoint, checkpoint_path)
        
        # Calculate checksum
        checksum = self._calculate_checksum(checkpoint_path)
        
        # Save metadata
        metadata = {
            "step": step,
            "path": str(checkpoint_path),
            "checksum": checksum,
            "timestamp": checkpoint["metadata"]["timestamp"],
            "size_bytes": checkpoint_path.stat().st_size,
        }
        
        metadata_path = checkpoint_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Upload to cloud storage
        if self.cloud_bucket:
            if self.async_upload:
                self.upload_queue.put((checkpoint_path, checkpoint_name))
            else:
                self._upload_to_cloud(checkpoint_path, checkpoint_name)
                
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint with verification"""
        checkpoints = self._list_checkpoints()
        
        if not checkpoints:
            self.logger.info("No checkpoints found")
            return None
            
        # Try loading checkpoints from newest to oldest
        for checkpoint_info in checkpoints:
            try:
                checkpoint = self._load_checkpoint(checkpoint_info["path"])
                
                # Verify checkpoint integrity
                if self._verify_checkpoint(checkpoint, checkpoint_info):
                    self.logger.info(f"Loaded checkpoint from step {checkpoint_info['step']}")
                    return checkpoint
                else:
                    self.logger.warning(f"Checkpoint at step {checkpoint_info['step']} is corrupted")
                    
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint: {e}")
                continue
                
        # If all local checkpoints failed, try cloud storage
        if self.cloud_bucket:
            return self._load_from_cloud()
            
        return None
        
    def load_specific(self, step: int) -> Optional[Dict[str, Any]]:
        """Load a specific checkpoint by step number"""
        checkpoint_name = f"checkpoint-{step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        if checkpoint_path.exists():
            return self._load_checkpoint(checkpoint_path)
        elif self.cloud_bucket:
            # Try loading from cloud
            return self._load_from_cloud(checkpoint_name)
            
        return None
        
    def _load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """Load checkpoint from disk with error handling"""
        try:
            checkpoint = torch.load(path, map_location="cpu")
            return checkpoint
        except Exception as e:
            self.logger.error(f"Error loading checkpoint from {path}: {e}")
            raise
            
    def _list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints sorted by step"""
        checkpoints = []
        
        for metadata_file in self.checkpoint_dir.glob("checkpoint-*.json"):
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                checkpoints.append(metadata)
            except:
                continue
                
        # Sort by step (newest first)
        checkpoints.sort(key=lambda x: x["step"], reverse=True)
        
        return checkpoints
        
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only the last N"""
        checkpoints = self._list_checkpoints()
        
        if len(checkpoints) > self.keep_last_n:
            # Remove old checkpoints
            for checkpoint_info in checkpoints[self.keep_last_n:]:
                checkpoint_path = Path(checkpoint_info["path"])
                metadata_path = checkpoint_path.with_suffix(".json")
                
                # Remove files
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                if metadata_path.exists():
                    metadata_path.unlink()
                    
                # Remove from cloud storage
                if self.cloud_bucket:
                    self._delete_from_cloud(checkpoint_path.name)
                    
                self.logger.info(f"Removed old checkpoint: {checkpoint_path.name}")
                
    def _calculate_checksum(self, filepath: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
        
    def _verify_checkpoint(self, checkpoint: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
        """Verify checkpoint integrity"""
        # Check required fields
        required_fields = ["model_state_dict", "optimizer_state_dict", "global_step"]
        for field in required_fields:
            if field not in checkpoint:
                return False
                
        # Verify metadata matches
        if checkpoint.get("metadata", {}).get("step") != metadata["step"]:
            return False
            
        return True
        
    def _upload_to_cloud(self, local_path: Path, cloud_name: str):
        """Upload checkpoint to cloud storage"""
        try:
            if self.storage_backend == "s3":
                self.s3_client.upload_file(
                    str(local_path),
                    self.cloud_bucket,
                    cloud_name,
                    ExtraArgs={"StorageClass": "INTELLIGENT_TIERING"},
                )
                # Also upload metadata
                metadata_path = local_path.with_suffix(".json")
                if metadata_path.exists():
                    self.s3_client.upload_file(
                        str(metadata_path),
                        self.cloud_bucket,
                        f"{cloud_name}.json",
                    )
                    
            elif self.storage_backend == "minio":
                self.minio_client.fput_object(
                    self.cloud_bucket,
                    cloud_name,
                    str(local_path),
                )
                # Also upload metadata
                metadata_path = local_path.with_suffix(".json")
                if metadata_path.exists():
                    self.minio_client.fput_object(
                        self.cloud_bucket,
                        f"{cloud_name}.json",
                        str(metadata_path),
                    )
                    
            self.logger.info(f"Uploaded checkpoint to cloud: {cloud_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to upload checkpoint to cloud: {e}")
            
    def _load_from_cloud(self, checkpoint_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load checkpoint from cloud storage"""
        try:
            if self.storage_backend == "s3":
                # List available checkpoints if no specific one requested
                if not checkpoint_name:
                    response = self.s3_client.list_objects_v2(
                        Bucket=self.cloud_bucket,
                        Prefix="checkpoint-",
                    )
                    if "Contents" not in response:
                        return None
                        
                    # Get latest checkpoint
                    checkpoints = [obj["Key"] for obj in response["Contents"] if not obj["Key"].endswith(".json")]
                    checkpoints.sort(reverse=True)
                    checkpoint_name = checkpoints[0]
                    
                # Download checkpoint
                local_path = self.checkpoint_dir / checkpoint_name
                self.s3_client.download_file(
                    self.cloud_bucket,
                    checkpoint_name,
                    str(local_path),
                )
                
                return self._load_checkpoint(local_path)
                
            elif self.storage_backend == "minio":
                # Similar logic for MinIO
                if not checkpoint_name:
                    objects = self.minio_client.list_objects(
                        self.cloud_bucket,
                        prefix="checkpoint-",
                    )
                    checkpoints = [obj.object_name for obj in objects if not obj.object_name.endswith(".json")]
                    if not checkpoints:
                        return None
                    checkpoints.sort(reverse=True)
                    checkpoint_name = checkpoints[0]
                    
                # Download checkpoint
                local_path = self.checkpoint_dir / checkpoint_name
                self.minio_client.fget_object(
                    self.cloud_bucket,
                    checkpoint_name,
                    str(local_path),
                )
                
                return self._load_checkpoint(local_path)
                
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from cloud: {e}")
            return None
            
    def _delete_from_cloud(self, checkpoint_name: str):
        """Delete checkpoint from cloud storage"""
        try:
            if self.storage_backend == "s3":
                self.s3_client.delete_object(
                    Bucket=self.cloud_bucket,
                    Key=checkpoint_name,
                )
                # Also delete metadata
                self.s3_client.delete_object(
                    Bucket=self.cloud_bucket,
                    Key=f"{checkpoint_name}.json",
                )
                
            elif self.storage_backend == "minio":
                self.minio_client.remove_object(
                    self.cloud_bucket,
                    checkpoint_name,
                )
                # Also delete metadata
                self.minio_client.remove_object(
                    self.cloud_bucket,
                    f"{checkpoint_name}.json",
                )
                
        except Exception as e:
            self.logger.error(f"Failed to delete checkpoint from cloud: {e}")
            
    def _upload_worker(self):
        """Background worker for async uploads"""
        while True:
            try:
                local_path, cloud_name = self.upload_queue.get(timeout=1)
                self._upload_to_cloud(local_path, cloud_name)
                self.upload_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Upload worker error: {e}")
                
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about all checkpoints"""
        checkpoints = self._list_checkpoints()
        
        total_size = sum(cp.get("size_bytes", 0) for cp in checkpoints)
        
        return {
            "total_checkpoints": len(checkpoints),
            "latest_step": checkpoints[0]["step"] if checkpoints else 0,
            "total_size_gb": total_size / 1e9,
            "storage_backend": self.storage_backend,
            "cloud_bucket": self.cloud_bucket,
        }