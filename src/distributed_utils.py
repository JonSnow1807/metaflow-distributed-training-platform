"""
Distributed training utilities for multi-node coordination
Handles fault tolerance, communication optimization, and multi-cloud setups
"""

import os
import socket
import time
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any, List, Tuple, Union
import logging
import subprocess
import json
import threading
from datetime import datetime, timedelta
from contextlib import contextmanager
import signal
import psutil


class DistributedCoordinator:
    """
    Manages distributed training coordination across nodes
    
    Features:
    - Automatic node discovery and registration
    - Health monitoring and failure detection
    - Dynamic node addition/removal
    - Cross-cloud communication optimization
    """
    
    def __init__(
        self,
        backend: str = "nccl",
        init_method: Optional[str] = None,
        timeout: timedelta = timedelta(minutes=30),
        health_check_interval: int = 30,
        elastic: bool = True,
    ):
        self.backend = backend
        self.init_method = init_method or "env://"
        self.timeout = timeout
        self.health_check_interval = health_check_interval
        self.elastic = elastic
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Node information
        self.rank = None
        self.world_size = None
        self.local_rank = None
        self.node_name = socket.gethostname()
        
        # Health monitoring
        self.healthy_nodes = set()
        self.failed_nodes = set()
        self.last_health_check = {}
        
        # Initialize distributed environment
        self._init_distributed()
        
        # Start health monitoring if elastic
        if self.elastic and self.rank == 0:
            self.health_thread = threading.Thread(target=self._monitor_health, daemon=True)
            self.health_thread.start()
            
    def _init_distributed(self):
        """Initialize distributed training environment"""
        if not dist.is_initialized():
            # Get environment variables
            self.rank = int(os.environ.get("RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            
            # Set CUDA device
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                
            # Initialize process group
            dist.init_process_group(
                backend=self.backend,
                init_method=self.init_method,
                world_size=self.world_size,
                rank=self.rank,
                timeout=self.timeout,
            )
            
            self.logger.info(
                f"Initialized distributed training: "
                f"rank={self.rank}/{self.world_size}, "
                f"local_rank={self.local_rank}, "
                f"node={self.node_name}"
            )
            
        else:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            
    def barrier(self, tag: Optional[str] = None):
        """Synchronization barrier with optional tagging"""
        if tag:
            self.logger.debug(f"Barrier: {tag}")
            
        start_time = time.time()
        dist.barrier()
        duration = time.time() - start_time
        
        if duration > 5:  # Log slow barriers
            self.logger.warning(f"Slow barrier ({tag}): {duration:.2f}s")
            
    def all_gather_object(self, obj: Any) -> List[Any]:
        """Gather objects from all ranks"""
        output = [None] * self.world_size
        dist.all_gather_object(output, obj)
        return output
        
    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """Broadcast object from source rank to all ranks"""
        output = [obj] if self.rank == src else [None]
        dist.broadcast_object_list(output, src=src)
        return output[0]
        
    def get_node_info(self) -> Dict[str, Any]:
        """Get information about current node"""
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "node_name": self.node_name,
            "hostname": socket.gethostname(),
            "ip_address": socket.gethostbyname(socket.gethostname()),
            "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / 1e9,
        }
        
    def gather_node_info(self) -> List[Dict[str, Any]]:
        """Gather node information from all ranks"""
        local_info = self.get_node_info()
        all_info = self.all_gather_object(local_info)
        return all_info
        
    def _monitor_health(self):
        """Monitor health of all nodes (rank 0 only)"""
        while True:
            try:
                # Perform health check
                health_status = self._check_all_nodes_health()
                
                # Update healthy/failed nodes
                for rank, status in enumerate(health_status):
                    if status["healthy"]:
                        self.healthy_nodes.add(rank)
                        self.failed_nodes.discard(rank)
                    else:
                        self.failed_nodes.add(rank)
                        self.healthy_nodes.discard(rank)
                        self.logger.error(f"Node {rank} is unhealthy: {status['reason']}")
                        
                # Sleep before next check
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(self.health_check_interval)
                
    def _check_all_nodes_health(self) -> List[Dict[str, Any]]:
        """Check health of all nodes"""
        # Each node reports its health
        local_health = {
            "rank": self.rank,
            "healthy": True,
            "timestamp": datetime.now().isoformat(),
            "gpu_ok": self._check_gpu_health(),
            "memory_ok": self._check_memory_health(),
            "reason": "OK",
        }
        
        # Gather health from all nodes
        try:
            all_health = self.all_gather_object(local_health)
            return all_health
        except Exception as e:
            # If gather fails, assume communication issue
            return [{"rank": i, "healthy": False, "reason": str(e)} 
                   for i in range(self.world_size)]
                   
    def _check_gpu_health(self) -> bool:
        """Check if GPUs are healthy"""
        if not torch.cuda.is_available():
            return True  # No GPUs to check
            
        try:
            # Try to allocate small tensor on each GPU
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    test_tensor = torch.zeros(1, device=f"cuda:{i}")
                    del test_tensor
            return True
        except Exception:
            return False
            
    def _check_memory_health(self) -> bool:
        """Check if memory usage is healthy"""
        memory = psutil.virtual_memory()
        return memory.percent < 95  # Less than 95% used
        
    @contextmanager
    def fault_tolerant_section(self, name: str, max_retries: int = 3):
        """
        Context manager for fault-tolerant execution sections
        Automatically handles retries and node failures
        """
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                yield
                break  # Success, exit loop
            except Exception as e:
                retry_count += 1
                last_error = e
                self.logger.warning(
                    f"Error in {name} (attempt {retry_count}/{max_retries}): {e}"
                )
                
                if retry_count < max_retries:
                    # Wait before retry with exponential backoff
                    wait_time = 2 ** retry_count
                    time.sleep(wait_time)
                    
                    # Re-sync all nodes before retry
                    try:
                        self.barrier(f"fault_recovery_{name}")
                    except:
                        pass  # Barrier might fail if nodes are down
                        
        else:
            # All retries failed
            self.logger.error(f"Failed {name} after {max_retries} attempts")
            if last_error:
                raise last_error
                

class CrossCloudCommunicator:
    """
    Optimizes communication across different cloud providers
    Handles network topology awareness and bandwidth optimization
    """
    
    def __init__(
        self,
        node_info: List[Dict[str, Any]],
        enable_compression: bool = True,
        optimize_topology: bool = True,
    ):
        self.node_info = node_info
        self.enable_compression = enable_compression
        self.optimize_topology = optimize_topology
        
        # Build network topology
        self.topology = self._build_network_topology()
        
        # Setup compression if enabled
        if self.enable_compression:
            self._setup_compression()
            
    def _build_network_topology(self) -> Dict[str, Any]:
        """Build network topology map"""
        topology = {
            "nodes": {},
            "regions": {},
            "providers": {},
        }
        
        for info in self.node_info:
            rank = info["rank"]
            
            # Detect cloud provider and region
            provider, region = self._detect_cloud_provider(info)
            
            topology["nodes"][rank] = {
                "provider": provider,
                "region": region,
                "ip": info["ip_address"],
            }
            
            # Group by provider and region
            if provider not in topology["providers"]:
                topology["providers"][provider] = []
            topology["providers"][provider].append(rank)
            
            region_key = f"{provider}_{region}"
            if region_key not in topology["regions"]:
                topology["regions"][region_key] = []
            topology["regions"][region_key].append(rank)
            
        return topology
        
    def _detect_cloud_provider(self, node_info: Dict[str, Any]) -> Tuple[str, str]:
        """Detect cloud provider and region from node information"""
        hostname = node_info["hostname"]
        
        # AWS detection
        if "ec2" in hostname or "aws" in hostname:
            # Try to get region from metadata
            try:
                region = subprocess.check_output(
                    ["curl", "-s", "http://169.254.169.254/latest/meta-data/placement/region"],
                    timeout=2
                ).decode().strip()
                return "aws", region
            except:
                return "aws", "unknown"
                
        # GCP detection
        elif "gcp" in hostname or "google" in hostname:
            try:
                zone = subprocess.check_output(
                    ["curl", "-s", "http://metadata.google.internal/computeMetadata/v1/instance/zone",
                     "-H", "Metadata-Flavor: Google"],
                    timeout=2
                ).decode().strip().split("/")[-1]
                region = "-".join(zone.split("-")[:-1])
                return "gcp", region
            except:
                return "gcp", "unknown"
                
        # Azure detection
        elif "azure" in hostname:
            return "azure", "unknown"
            
        # On-premise or unknown
        else:
            return "on-premise", "local"
            
    def _setup_compression(self):
        """Setup gradient compression for cross-region communication"""
        # This would implement gradient compression
        # For now, we'll use PyTorch's built-in compression
        pass
        
    def get_optimal_communication_groups(self) -> List[List[int]]:
        """
        Get optimal communication groups based on network topology
        Groups nodes by proximity for hierarchical allreduce
        """
        groups = []
        
        # First level: nodes in same region
        for region_nodes in self.topology["regions"].values():
            if len(region_nodes) > 1:
                groups.append(region_nodes)
                
        # Second level: nodes in same provider
        for provider_nodes in self.topology["providers"].values():
            if len(provider_nodes) > 1:
                groups.append(provider_nodes)
                
        return groups
        
    def estimate_communication_cost(self, data_size_gb: float) -> Dict[str, float]:
        """Estimate communication cost for data transfer"""
        costs = {
            "intra_region": 0.0,  # Usually free
            "inter_region": 0.02 * data_size_gb,  # ~$0.02/GB
            "inter_provider": 0.12 * data_size_gb,  # ~$0.12/GB
            "total": 0.0,
        }
        
        # Calculate based on topology
        # This is simplified - real calculation would consider actual communication patterns
        num_inter_region = len(self.topology["regions"]) - 1
        num_inter_provider = len(self.topology["providers"]) - 1
        
        if num_inter_region > 0:
            costs["total"] += costs["inter_region"] * num_inter_region
            
        if num_inter_provider > 0:
            costs["total"] += costs["inter_provider"] * num_inter_provider
            
        return costs


class ElasticTrainingManager:
    """
    Manages elastic training with dynamic node scaling
    Supports spot instance preemption and automatic recovery
    """
    
    def __init__(
        self,
        min_nodes: int = 1,
        max_nodes: int = 16,
        checkpoint_interval: int = 1000,
        preemption_warning_time: int = 120,  # seconds
    ):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.checkpoint_interval = checkpoint_interval
        self.preemption_warning_time = preemption_warning_time
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Current state
        self.active_nodes = set()
        self.pending_preemption = set()
        
        # Setup signal handlers for spot instance termination
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Setup handlers for spot instance termination signals"""
        def handle_termination(signum, frame):
            self.logger.warning("Received termination signal!")
            self._handle_preemption()
            
        # AWS sends SIGTERM before termination
        signal.signal(signal.SIGTERM, handle_termination)
        
        # Also monitor metadata endpoint
        self.termination_thread = threading.Thread(
            target=self._monitor_termination,
            daemon=True
        )
        self.termination_thread.start()
        
    def _monitor_termination(self):
        """Monitor for spot instance termination notices"""
        while True:
            try:
                # Check AWS metadata for termination notice
                response = subprocess.run(
                    ["curl", "-s", "http://169.254.169.254/latest/meta-data/spot/termination-time"],
                    capture_output=True,
                    timeout=2
                )
                
                if response.returncode == 0 and response.stdout:
                    termination_time = response.stdout.decode().strip()
                    self.logger.warning(f"Spot instance termination scheduled: {termination_time}")
                    self._handle_preemption()
                    
            except Exception:
                pass  # Not on AWS or error checking
                
            time.sleep(5)  # Check every 5 seconds
            
    def _handle_preemption(self):
        """Handle node preemption gracefully"""
        self.logger.info("Handling node preemption...")
        
        # Trigger immediate checkpoint
        self._trigger_emergency_checkpoint()
        
        # Notify other nodes
        if dist.is_initialized():
            rank = dist.get_rank()
            self.pending_preemption.add(rank)
            
            # Broadcast preemption notice
            preemption_info = {
                "rank": rank,
                "timestamp": datetime.now().isoformat(),
                "estimated_termination": (
                    datetime.now() + timedelta(seconds=self.preemption_warning_time)
                ).isoformat(),
            }
            
            # Use non-blocking communication
            for i in range(dist.get_world_size()):
                if i != rank:
                    try:
                        dist.send(
                            torch.tensor([rank], dtype=torch.int32),
                            dst=i,
                            tag=9999,  # Special tag for preemption
                        )
                    except:
                        pass
                        
    def _trigger_emergency_checkpoint(self):
        """Trigger emergency checkpoint before termination"""
        checkpoint_flag = torch.tensor([1], dtype=torch.int32)
        
        if dist.is_initialized():
            # Notify all nodes to checkpoint
            dist.broadcast(checkpoint_flag, src=dist.get_rank())
            
        self.logger.info("Emergency checkpoint triggered")
        
    def can_continue_training(self) -> bool:
        """Check if training can continue with current nodes"""
        if dist.is_initialized():
            world_size = dist.get_world_size()
            failed_count = len(self.pending_preemption)
            active_count = world_size - failed_count
            
            return active_count >= self.min_nodes
        else:
            return True
            
    def rebalance_workload(self, total_samples: int) -> Dict[int, int]:
        """
        Rebalance workload among remaining nodes
        Returns mapping of rank to number of samples
        """
        if not dist.is_initialized():
            return {0: total_samples}
            
        world_size = dist.get_world_size()
        active_ranks = [
            i for i in range(world_size) 
            if i not in self.pending_preemption
        ]
        
        if not active_ranks:
            return {}
            
        # Distribute samples evenly among active nodes
        samples_per_node = total_samples // len(active_ranks)
        remainder = total_samples % len(active_ranks)
        
        distribution = {}
        for i, rank in enumerate(active_ranks):
            distribution[rank] = samples_per_node
            if i < remainder:
                distribution[rank] += 1
                
        return distribution