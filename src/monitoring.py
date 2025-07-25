"""
Production monitoring with Prometheus metrics and Grafana integration
Tracks training performance, system resources, and cost metrics
"""

import time
import psutil
import pynvml
import torch
import torch.distributed as dist
from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, push_to_gateway
from typing import Dict, Optional, List, Any
import logging
import threading
import os
from datetime import datetime
import json


class MetricsCollector:
    """
    Comprehensive metrics collection for distributed training:
    - Training metrics (loss, throughput, gradients)
    - System metrics (GPU, CPU, memory, network)
    - Cost metrics (spot pricing, accumulated cost)
    - Model metrics (parameters, memory usage)
    """
    
    def __init__(
        self,
        rank: int = 0,
        prometheus_gateway: Optional[str] = None,
        push_interval: int = 30,
        job_name: str = "fsdp_training",
    ):
        self.rank = rank
        self.prometheus_gateway = prometheus_gateway or os.environ.get(
            "PROMETHEUS_GATEWAY", "localhost:9091"
        )
        self.push_interval = push_interval
        self.job_name = job_name
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize NVIDIA ML
        pynvml.nvmlInit()
        self.gpu_count = pynvml.nvmlDeviceGetCount()
        self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.gpu_count)]
        
        # Create Prometheus registry
        self.registry = CollectorRegistry()
        
        # Initialize metrics
        self._init_metrics()
        
        # Start background metric collection
        self.stop_event = threading.Event()
        self.collection_thread = threading.Thread(target=self._collect_system_metrics, daemon=True)
        self.collection_thread.start()
        
        self.logger.info(f"MetricsCollector initialized on rank {rank}")
        
    def _init_metrics(self):
        """Initialize Prometheus metrics"""
        # Training metrics
        self.training_loss = Gauge(
            "training_loss",
            "Current training loss",
            ["rank", "node"],
            registry=self.registry,
        )
        
        self.training_throughput = Gauge(
            "training_throughput_tokens_per_sec",
            "Training throughput in tokens per second",
            ["rank", "node"],
            registry=self.registry,
        )
        
        self.gradient_norm = Gauge(
            "gradient_norm",
            "L2 norm of gradients",
            ["rank", "node"],
            registry=self.registry,
        )
        
        self.learning_rate = Gauge(
            "learning_rate",
            "Current learning rate",
            ["rank", "node"],
            registry=self.registry,
        )
        
        self.batch_processing_time = Histogram(
            "batch_processing_time_seconds",
            "Time to process one batch",
            ["rank", "node"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
            registry=self.registry,
        )
        
        # GPU metrics
        self.gpu_utilization = Gauge(
            "gpu_utilization_percent",
            "GPU utilization percentage",
            ["rank", "gpu_id", "node"],
            registry=self.registry,
        )
        
        self.gpu_memory_used = Gauge(
            "gpu_memory_used_bytes",
            "GPU memory used in bytes",
            ["rank", "gpu_id", "node"],
            registry=self.registry,
        )
        
        self.gpu_memory_total = Gauge(
            "gpu_memory_total_bytes",
            "Total GPU memory in bytes",
            ["rank", "gpu_id", "node"],
            registry=self.registry,
        )
        
        self.gpu_temperature = Gauge(
            "gpu_temperature_celsius",
            "GPU temperature in Celsius",
            ["rank", "gpu_id", "node"],
            registry=self.registry,
        )
        
        self.gpu_power_usage = Gauge(
            "gpu_power_usage_watts",
            "GPU power usage in watts",
            ["rank", "gpu_id", "node"],
            registry=self.registry,
        )
        
        # System metrics
        self.cpu_utilization = Gauge(
            "cpu_utilization_percent",
            "CPU utilization percentage",
            ["rank", "node"],
            registry=self.registry,
        )
        
        self.memory_used = Gauge(
            "memory_used_bytes",
            "System memory used in bytes",
            ["rank", "node"],
            registry=self.registry,
        )
        
        self.disk_io_read = Counter(
            "disk_io_read_bytes_total",
            "Total disk read bytes",
            ["rank", "node"],
            registry=self.registry,
        )
        
        self.disk_io_write = Counter(
            "disk_io_write_bytes_total",
            "Total disk write bytes",
            ["rank", "node"],
            registry=self.registry,
        )
        
        self.network_sent = Counter(
            "network_sent_bytes_total",
            "Total network bytes sent",
            ["rank", "node"],
            registry=self.registry,
        )
        
        self.network_recv = Counter(
            "network_recv_bytes_total",
            "Total network bytes received",
            ["rank", "node"],
            registry=self.registry,
        )
        
        # Model metrics
        self.model_parameters = Gauge(
            "model_parameters_total",
            "Total number of model parameters",
            ["rank", "node"],
            registry=self.registry,
        )
        
        self.model_memory_usage = Gauge(
            "model_memory_usage_bytes",
            "Model memory usage in bytes",
            ["rank", "node"],
            registry=self.registry,
        )
        
        # Distributed training metrics
        self.allreduce_time = Histogram(
            "allreduce_time_seconds",
            "Time spent in allreduce operations",
            ["rank", "node"],
            buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 5.0),
            registry=self.registry,
        )
        
        self.checkpoint_save_time = Histogram(
            "checkpoint_save_time_seconds",
            "Time to save checkpoint",
            ["rank", "node"],
            buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 300.0),
            registry=self.registry,
        )
        
        # Cost metrics
        self.spot_price_per_hour = Gauge(
            "spot_price_per_hour_usd",
            "Current spot instance price per hour",
            ["instance_type", "availability_zone"],
            registry=self.registry,
        )
        
        self.accumulated_cost = Gauge(
            "accumulated_cost_usd",
            "Total accumulated training cost",
            ["rank", "node"],
            registry=self.registry,
        )
        
    def update_training_metrics(
        self,
        loss: float,
        throughput: float,
        learning_rate: float,
        gradient_norm: Optional[float] = None,
        batch_time: Optional[float] = None,
    ):
        """Update training-related metrics"""
        node_name = os.environ.get("NODE_NAME", f"node-{self.rank}")
        
        self.training_loss.labels(rank=self.rank, node=node_name).set(loss)
        self.training_throughput.labels(rank=self.rank, node=node_name).set(throughput)
        self.learning_rate.labels(rank=self.rank, node=node_name).set(learning_rate)
        
        if gradient_norm is not None:
            self.gradient_norm.labels(rank=self.rank, node=node_name).set(gradient_norm)
            
        if batch_time is not None:
            self.batch_processing_time.labels(rank=self.rank, node=node_name).observe(batch_time)
            
    def update_model_metrics(self, model: torch.nn.Module):
        """Update model-related metrics"""
        node_name = os.environ.get("NODE_NAME", f"node-{self.rank}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        self.model_parameters.labels(rank=self.rank, node=node_name).set(total_params)
        
        # Calculate model memory usage
        model_memory = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) + sum(
            b.numel() * b.element_size() for b in model.buffers()
        )
        self.model_memory_usage.labels(rank=self.rank, node=node_name).set(model_memory)
        
    def record_allreduce_time(self, duration: float):
        """Record time spent in allreduce operation"""
        node_name = os.environ.get("NODE_NAME", f"node-{self.rank}")
        self.allreduce_time.labels(rank=self.rank, node=node_name).observe(duration)
        
    def record_checkpoint_time(self, duration: float):
        """Record checkpoint save time"""
        node_name = os.environ.get("NODE_NAME", f"node-{self.rank}")
        self.checkpoint_save_time.labels(rank=self.rank, node=node_name).observe(duration)
        
    def update_cost_metrics(self, spot_price: float, accumulated_cost: float):
        """Update cost-related metrics"""
        instance_type = os.environ.get("INSTANCE_TYPE", "p4d.24xlarge")
        availability_zone = os.environ.get("AWS_AVAILABILITY_ZONE", "us-east-1a")
        node_name = os.environ.get("NODE_NAME", f"node-{self.rank}")
        
        self.spot_price_per_hour.labels(
            instance_type=instance_type,
            availability_zone=availability_zone,
        ).set(spot_price)
        
        self.accumulated_cost.labels(rank=self.rank, node=node_name).set(accumulated_cost)
        
    def _collect_system_metrics(self):
        """Background thread to collect system metrics"""
        node_name = os.environ.get("NODE_NAME", f"node-{self.rank}")
        
        # Get initial network and disk stats
        net_io_start = psutil.net_io_counters()
        disk_io_start = psutil.disk_io_counters()
        
        while not self.stop_event.is_set():
            try:
                # CPU and memory
                self.cpu_utilization.labels(rank=self.rank, node=node_name).set(
                    psutil.cpu_percent(interval=1)
                )
                
                memory = psutil.virtual_memory()
                self.memory_used.labels(rank=self.rank, node=node_name).set(memory.used)
                
                # Network I/O
                net_io = psutil.net_io_counters()
                self.network_sent.labels(rank=self.rank, node=node_name).inc(
                    net_io.bytes_sent - net_io_start.bytes_sent
                )
                self.network_recv.labels(rank=self.rank, node=node_name).inc(
                    net_io.bytes_recv - net_io_start.bytes_recv
                )
                net_io_start = net_io
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                self.disk_io_read.labels(rank=self.rank, node=node_name).inc(
                    disk_io.read_bytes - disk_io_start.read_bytes
                )
                self.disk_io_write.labels(rank=self.rank, node=node_name).inc(
                    disk_io.write_bytes - disk_io_start.write_bytes
                )
                disk_io_start = disk_io
                
                # GPU metrics
                for gpu_id, handle in enumerate(self.gpu_handles):
                    # Utilization
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.gpu_utilization.labels(
                        rank=self.rank,
                        gpu_id=gpu_id,
                        node=node_name,
                    ).set(utilization.gpu)
                    
                    # Memory
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.gpu_memory_used.labels(
                        rank=self.rank,
                        gpu_id=gpu_id,
                        node=node_name,
                    ).set(mem_info.used)
                    self.gpu_memory_total.labels(
                        rank=self.rank,
                        gpu_id=gpu_id,
                        node=node_name,
                    ).set(mem_info.total)
                    
                    # Temperature
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    self.gpu_temperature.labels(
                        rank=self.rank,
                        gpu_id=gpu_id,
                        node=node_name,
                    ).set(temp)
                    
                    # Power usage
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                        self.gpu_power_usage.labels(
                            rank=self.rank,
                            gpu_id=gpu_id,
                            node=node_name,
                        ).set(power)
                    except pynvml.NVMLError:
                        pass  # Not all GPUs support power monitoring
                        
                # Push metrics to Prometheus gateway
                if self.rank == 0:  # Only rank 0 pushes to avoid duplicates
                    self._push_metrics()
                    
                # Sleep until next collection
                self.stop_event.wait(self.push_interval)
                
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")
                self.stop_event.wait(self.push_interval)
                
    def _push_metrics(self):
        """Push metrics to Prometheus gateway"""
        try:
            push_to_gateway(
                self.prometheus_gateway,
                job=self.job_name,
                registry=self.registry,
            )
        except Exception as e:
            self.logger.error(f"Failed to push metrics to Prometheus: {e}")
            
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current snapshot of all metrics"""
        node_name = os.environ.get("NODE_NAME", f"node-{self.rank}")
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "rank": self.rank,
            "node": node_name,
            "training": {
                "loss": self._get_metric_value(self.training_loss, rank=self.rank, node=node_name),
                "throughput": self._get_metric_value(self.training_throughput, rank=self.rank, node=node_name),
                "learning_rate": self._get_metric_value(self.learning_rate, rank=self.rank, node=node_name),
                "gradient_norm": self._get_metric_value(self.gradient_norm, rank=self.rank, node=node_name),
            },
            "system": {
                "cpu_percent": self._get_metric_value(self.cpu_utilization, rank=self.rank, node=node_name),
                "memory_used_gb": self._get_metric_value(self.memory_used, rank=self.rank, node=node_name) / 1e9,
            },
            "gpu": [],
            "cost": {
                "accumulated_usd": self._get_metric_value(self.accumulated_cost, rank=self.rank, node=node_name),
            },
        }
        
        # Add GPU metrics
        for gpu_id in range(self.gpu_count):
            gpu_metrics = {
                "gpu_id": gpu_id,
                "utilization_percent": self._get_metric_value(
                    self.gpu_utilization,
                    rank=self.rank,
                    gpu_id=gpu_id,
                    node=node_name,
                ),
                "memory_used_gb": self._get_metric_value(
                    self.gpu_memory_used,
                    rank=self.rank,
                    gpu_id=gpu_id,
                    node=node_name,
                ) / 1e9,
                "temperature_c": self._get_metric_value(
                    self.gpu_temperature,
                    rank=self.rank,
                    gpu_id=gpu_id,
                    node=node_name,
                ),
                "power_watts": self._get_metric_value(
                    self.gpu_power_usage,
                    rank=self.rank,
                    gpu_id=gpu_id,
                    node=node_name,
                ),
            }
            metrics["gpu"].append(gpu_metrics)
            
        return metrics
        
    def _get_metric_value(self, metric: Gauge, **labels) -> float:
        """Get current value of a metric"""
        try:
            return metric.labels(**labels)._value.get()
        except:
            return 0.0
            
    def shutdown(self):
        """Cleanup resources"""
        self.stop_event.set()
        self.collection_thread.join(timeout=5)
        pynvml.nvmlShutdown()
        
    def export_metrics_summary(self, output_path: str):
        """Export metrics summary to file"""
        summary = {
            "job_name": self.job_name,
            "rank": self.rank,
            "final_metrics": self.get_current_metrics(),
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"Exported metrics summary to {output_path}")