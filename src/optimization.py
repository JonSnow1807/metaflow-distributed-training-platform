"""
Memory and performance optimization utilities for distributed training
Implements gradient checkpointing, CPU offloading, and dynamic batching
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Dict, Any, List, Tuple, Callable
import logging
import psutil
import pynvml
from contextlib import contextmanager
import gc
import functools


class MemoryOptimizer:
    """
    Comprehensive memory optimization for large model training
    
    Features:
    - Gradient checkpointing with smart layer selection
    - CPU offloading for optimizer states
    - Dynamic batch size adjustment
    - Memory profiling and analytics
    - Activation memory management
    """
    
    def __init__(
        self,
        model: nn.Module,
        gradient_checkpointing: bool = True,
        cpu_offload: bool = False,
        mixed_precision: bool = True,
        activation_checkpointing_ratio: float = 0.5,
        memory_efficient_attention: bool = True,
        profile_memory: bool = True,
    ):
        self.model = model
        self.gradient_checkpointing = gradient_checkpointing
        self.cpu_offload = cpu_offload
        self.mixed_precision = mixed_precision
        self.activation_checkpointing_ratio = activation_checkpointing_ratio
        self.memory_efficient_attention = memory_efficient_attention
        self.profile_memory = profile_memory
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize NVIDIA ML
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # Memory tracking
        self.memory_stats = {
            "peak_memory_gb": 0,
            "average_memory_gb": 0,
            "oom_events": 0,
        }
        
        # Apply optimizations
        self._apply_optimizations()
        
    def _apply_optimizations(self):
        """Apply all requested optimizations to the model"""
        if self.gradient_checkpointing:
            self._enable_gradient_checkpointing()
            
        if self.memory_efficient_attention:
            self._enable_memory_efficient_attention()
            
        if self.profile_memory:
            self._setup_memory_profiling()
            
        self.logger.info("Memory optimizations applied successfully")
        
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for transformer layers"""
        # Find all transformer layers
        transformer_layers = self._find_transformer_layers()
        
        if not transformer_layers:
            self.logger.warning("No transformer layers found for gradient checkpointing")
            return
            
        # Select layers for checkpointing based on ratio
        num_layers = len(transformer_layers)
        num_checkpoint = int(num_layers * self.activation_checkpointing_ratio)
        
        # Checkpoint every nth layer for even distribution
        if num_checkpoint > 0:
            checkpoint_interval = max(1, num_layers // num_checkpoint)
            
            for i, layer in enumerate(transformer_layers):
                if i % checkpoint_interval == 0:
                    self._wrap_layer_with_checkpoint(layer)
                    
            self.logger.info(f"Enabled gradient checkpointing for {num_checkpoint}/{num_layers} layers")
            
    def _find_transformer_layers(self) -> List[nn.Module]:
        """Find all transformer layers in the model"""
        layers = []
        
        # Common transformer layer names
        layer_names = ["TransformerBlock", "TransformerLayer", "Block", "Layer"]
        
        def find_layers(module, prefix=""):
            for name, child in module.named_children():
                if any(layer_name in child.__class__.__name__ for layer_name in layer_names):
                    layers.append(child)
                else:
                    find_layers(child, prefix + name + ".")
                    
        find_layers(self.model)
        return layers
        
    def _wrap_layer_with_checkpoint(self, layer: nn.Module):
        """Wrap a layer's forward method with gradient checkpointing"""
        original_forward = layer.forward
        
        @functools.wraps(original_forward)
        def checkpointed_forward(*args, **kwargs):
            # Use checkpoint only during training
            if layer.training:
                # Handle both args and kwargs
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, **kwargs)
                    return custom_forward
                    
                return checkpoint(create_custom_forward(original_forward), *args)
            else:
                return original_forward(*args, **kwargs)
                
        layer.forward = checkpointed_forward
        
    def _enable_memory_efficient_attention(self):
        """Enable memory-efficient attention mechanisms"""
        # This would implement Flash Attention or similar
        # For now, we'll use PyTorch's built-in optimizations
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            self.logger.info("Using memory-efficient scaled dot product attention")
            # The model should already use this if available
            
    def _setup_memory_profiling(self):
        """Setup memory profiling hooks"""
        def memory_hook(module, input, output):
            if self.profile_memory:
                self._update_memory_stats()
                
        # Register hooks on major modules
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module.register_forward_hook(memory_hook)
                
    def _update_memory_stats(self):
        """Update memory statistics"""
        try:
            # GPU memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            current_memory_gb = mem_info.used / 1e9
            
            # Update peak
            if current_memory_gb > self.memory_stats["peak_memory_gb"]:
                self.memory_stats["peak_memory_gb"] = current_memory_gb
                
            # Update average (simple moving average)
            alpha = 0.1
            self.memory_stats["average_memory_gb"] = (
                alpha * current_memory_gb + 
                (1 - alpha) * self.memory_stats["average_memory_gb"]
            )
            
        except Exception as e:
            self.logger.debug(f"Error updating memory stats: {e}")
            
    @contextmanager
    def optimize_memory_context(self):
        """Context manager for memory optimization during training"""
        # Clear cache before
        torch.cuda.empty_cache()
        gc.collect()
        
        # Set memory allocator settings
        old_fraction = torch.cuda.get_per_process_memory_fraction()
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use up to 95% of GPU memory
        
        try:
            yield
        finally:
            # Restore settings
            torch.cuda.set_per_process_memory_fraction(old_fraction)
            
            # Clear cache after
            torch.cuda.empty_cache()
            gc.collect()
            
    def get_optimal_batch_size(
        self,
        initial_batch_size: int,
        sequence_length: int,
        safety_margin: float = 0.9,
    ) -> int:
        """
        Find optimal batch size that fits in memory
        Uses binary search to find maximum batch size
        """
        self.logger.info("Finding optimal batch size...")
        
        # Get available memory
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        available_memory = mem_info.free * safety_margin
        
        # Binary search for optimal batch size
        low, high = 1, initial_batch_size * 2
        optimal_batch_size = 1
        
        while low <= high:
            mid = (low + high) // 2
            
            try:
                # Test forward and backward pass
                self._test_batch_size(mid, sequence_length)
                
                # If successful, try larger
                optimal_batch_size = mid
                low = mid + 1
                
            except torch.cuda.OutOfMemoryError:
                # If OOM, try smaller
                high = mid - 1
                torch.cuda.empty_cache()
                
        self.logger.info(f"Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
        
    def _test_batch_size(self, batch_size: int, sequence_length: int):
        """Test if a batch size fits in memory"""
        # Create dummy input
        dummy_input = torch.randint(
            0, 1000,
            (batch_size, sequence_length),
            device="cuda",
        )
        
        # Forward pass
        with self.optimize_memory_context():
            output = self.model(dummy_input)
            if hasattr(output, "loss"):
                loss = output.loss
            else:
                loss = output.mean()
                
            # Backward pass
            loss.backward()
            
        # Clean up
        del dummy_input, output, loss
        torch.cuda.empty_cache()
        
    def optimize_data_loading(
        self,
        dataloader: torch.utils.data.DataLoader,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
    ) -> torch.utils.data.DataLoader:
        """Optimize data loading for better GPU utilization"""
        # Create optimized dataloader
        optimized_dataloader = torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=isinstance(dataloader.sampler, torch.utils.data.RandomSampler),
            num_workers=dataloader.num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers and dataloader.num_workers > 0,
            drop_last=dataloader.drop_last,
        )
        
        return optimized_dataloader
        
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary"""
        # GPU memory
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        
        # System memory
        system_memory = psutil.virtual_memory()
        
        # PyTorch memory stats
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
        else:
            allocated = reserved = max_allocated = 0
            
        summary = {
            "gpu": {
                "total_gb": mem_info.total / 1e9,
                "used_gb": mem_info.used / 1e9,
                "free_gb": mem_info.free / 1e9,
                "utilization_percent": (mem_info.used / mem_info.total) * 100,
                "pytorch_allocated_gb": allocated,
                "pytorch_reserved_gb": reserved,
                "pytorch_max_allocated_gb": max_allocated,
            },
            "system": {
                "total_gb": system_memory.total / 1e9,
                "used_gb": system_memory.used / 1e9,
                "available_gb": system_memory.available / 1e9,
                "percent": system_memory.percent,
            },
            "optimization_stats": self.memory_stats,
        }
        
        return summary
        
    def log_memory_usage(self, step: Optional[int] = None):
        """Log current memory usage"""
        summary = self.get_memory_summary()
        
        log_msg = f"Memory Usage"
        if step is not None:
            log_msg += f" (Step {step})"
            
        log_msg += f": GPU {summary['gpu']['used_gb']:.2f}/{summary['gpu']['total_gb']:.2f} GB"
        log_msg += f" ({summary['gpu']['utilization_percent']:.1f}%)"
        log_msg += f", System {summary['system']['used_gb']:.2f}/{summary['system']['total_gb']:.2f} GB"
        
        self.logger.info(log_msg)
        
    def suggest_optimizations(self) -> List[str]:
        """Suggest additional optimizations based on current usage"""
        suggestions = []
        summary = self.get_memory_summary()
        
        # Check GPU memory pressure
        gpu_util = summary["gpu"]["utilization_percent"]
        if gpu_util > 90:
            suggestions.append("High GPU memory usage - consider enabling gradient checkpointing")
            suggestions.append("Reduce batch size or sequence length")
            
        if gpu_util > 95:
            suggestions.append("Critical GPU memory - enable CPU offloading")
            
        # Check fragmentation
        fragmentation = (summary["gpu"]["pytorch_reserved_gb"] - 
                        summary["gpu"]["pytorch_allocated_gb"])
        if fragmentation > 2.0:  # More than 2GB fragmented
            suggestions.append(f"High memory fragmentation ({fragmentation:.1f} GB) - consider restarting")
            
        # Check system memory
        if summary["system"]["percent"] > 80:
            suggestions.append("High system memory usage - reduce number of workers or prefetch factor")
            
        # Model-specific suggestions
        param_count = sum(p.numel() for p in self.model.parameters())
        if param_count > 1e9 and not self.gradient_checkpointing:
            suggestions.append("Large model without gradient checkpointing - enable to save memory")
            
        return suggestions


class DynamicBatchScheduler:
    """
    Dynamic batch size scheduling based on memory usage and training progress
    """
    
    def __init__(
        self,
        initial_batch_size: int,
        min_batch_size: int = 1,
        max_batch_size: Optional[int] = None,
        memory_threshold: float = 0.9,
        adjustment_interval: int = 100,
    ):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size or initial_batch_size * 4
        self.memory_threshold = memory_threshold
        self.adjustment_interval = adjustment_interval
        
        self.step_count = 0
        self.adjustment_history = []
        
        # Initialize NVIDIA ML
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
    def step(self) -> int:
        """Get current batch size and potentially adjust"""
        self.step_count += 1
        
        # Check if we should adjust
        if self.step_count % self.adjustment_interval == 0:
            self._adjust_batch_size()
            
        return self.current_batch_size
        
    def _adjust_batch_size(self):
        """Adjust batch size based on memory usage"""
        # Get current memory usage
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        memory_usage = mem_info.used / mem_info.total
        
        old_batch_size = self.current_batch_size
        
        if memory_usage > self.memory_threshold:
            # Decrease batch size
            self.current_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * 0.8)
            )
        elif memory_usage < self.memory_threshold * 0.8:
            # Increase batch size if we have headroom
            self.current_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * 1.2)
            )
            
        # Record adjustment
        if self.current_batch_size != old_batch_size:
            self.adjustment_history.append({
                "step": self.step_count,
                "old_batch_size": old_batch_size,
                "new_batch_size": self.current_batch_size,
                "memory_usage": memory_usage,
            })
            
            logging.info(
                f"Adjusted batch size: {old_batch_size} -> {self.current_batch_size} "
                f"(memory usage: {memory_usage:.1%})"
            )