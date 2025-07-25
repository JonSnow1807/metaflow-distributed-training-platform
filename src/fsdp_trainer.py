"""
Production-ready FSDP Trainer with automatic recovery and optimization
Designed for Netflix-scale distributed training workloads
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import wandb
from typing import Dict, Optional, Any, List
import logging
from datetime import datetime
import psutil
import pynvml

from .checkpoint_manager import CheckpointManager
from .monitoring import MetricsCollector
from .cost_tracker import CostTracker
from .optimization import MemoryOptimizer


class FSDPTrainer:
    """
    Production-grade FSDP trainer with comprehensive features:
    - Automatic checkpoint recovery
    - Memory optimization
    - Cost tracking
    - Real-time monitoring
    - Multi-node coordination
    """
    
    def __init__(
        self,
        model_name: str,
        dataset: str,
        num_nodes: int = 1,
        checkpoint_interval: int = 1000,
        use_mixed_precision: bool = True,
        gradient_checkpointing: bool = True,
        cpu_offload: bool = False,
        checkpoint_dir: str = "checkpoints",
        wandb_project: str = "fsdp-training",
        cost_aware: bool = True,
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.dataset = dataset
        self.num_nodes = num_nodes
        self.checkpoint_interval = checkpoint_interval
        self.use_mixed_precision = use_mixed_precision
        self.gradient_checkpointing = gradient_checkpointing
        self.cpu_offload = cpu_offload
        self.checkpoint_dir = checkpoint_dir
        self.wandb_project = wandb_project
        self.cost_aware = cost_aware
        self.max_retries = max_retries
        
        # Initialize components
        self._setup_logging()
        self._init_distributed()
        self._init_monitoring()
        
        # Load model and tokenizer
        self.device = torch.device(f"cuda:{self.local_rank}")
        self._load_model_and_tokenizer()
        
        # Setup FSDP
        self._setup_fsdp()
        
        # Initialize managers
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            rank=self.rank,
            world_size=self.world_size,
        )
        self.metrics_collector = MetricsCollector(rank=self.rank)
        self.cost_tracker = CostTracker(
            instance_type=os.environ.get("INSTANCE_TYPE", "p4d.24xlarge"),
            use_spot=os.environ.get("USE_SPOT", "true").lower() == "true",
        )
        
        # Memory optimizer
        self.memory_optimizer = MemoryOptimizer(
            model=self.model,
            gradient_checkpointing=gradient_checkpointing,
        )
        
        self.logger.info(f"FSDP Trainer initialized on rank {self.rank}/{self.world_size}")
        
    def _setup_logging(self):
        """Configure production logging"""
        logging.basicConfig(
            level=logging.INFO,
            format=f'%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'training_rank_{os.environ.get("RANK", 0)}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _init_distributed(self):
        """Initialize distributed training environment"""
        if not dist.is_initialized():
            # Get distributed training parameters
            self.rank = int(os.environ.get("RANK", 0))
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            
            # Initialize process group
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=self.world_size,
                rank=self.rank,
            )
            
            # Set CUDA device
            torch.cuda.set_device(self.local_rank)
            
        else:
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = dist.get_world_size()
            
    def _init_monitoring(self):
        """Initialize monitoring and tracking"""
        if self.rank == 0:
            # Initialize wandb
            wandb.init(
                project=self.wandb_project,
                name=f"{self.model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={
                    "model": self.model_name,
                    "dataset": self.dataset,
                    "num_nodes": self.num_nodes,
                    "world_size": self.world_size,
                    "mixed_precision": self.use_mixed_precision,
                    "gradient_checkpointing": self.gradient_checkpointing,
                    "cpu_offload": self.cpu_offload,
                }
            )
            
        # Initialize NVIDIA ML
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.local_rank)
        
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer with error handling"""
        try:
            self.logger.info(f"Loading model {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load model with memory efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if self.use_mixed_precision else torch.float32,
                use_cache=False,  # Disable KV cache for training
            )
            
            # Enable gradient checkpointing if requested
            if self.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
            
    def _setup_fsdp(self):
        """Configure and wrap model with FSDP"""
        # Mixed precision configuration
        if self.use_mixed_precision:
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        else:
            mp_policy = None
            
        # CPU offload configuration
        if self.cpu_offload:
            cpu_offload_policy = CPUOffload(offload_params=True)
        else:
            cpu_offload_policy = None
            
        # Auto wrap policy for transformer models
        auto_wrap_policy = transformer_auto_wrap_policy(
            self.model,
            transformer_layer_cls={
                type(self.model.model.layers[0])  # Get transformer layer class
            },
        )
        
        # Wrap model with FSDP
        self.model = FSDP(
            self.model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            cpu_offload=cpu_offload_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=self.device,
            sync_module_states=True,
            use_orig_params=True,
        )
        
        self.logger.info("Model wrapped with FSDP successfully")
        
    def _prepare_dataset(self, batch_size: int) -> DataLoader:
        """Prepare distributed dataset loader"""
        # This is a placeholder - in production, load actual dataset
        from datasets import load_dataset
        
        # Load dataset
        dataset = load_dataset(self.dataset, split="train")
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
            )
            
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        # Create distributed sampler
        sampler = DistributedSampler(
            tokenized_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        
        # Create dataloader
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        
        return dataloader
        
    def train(
        self,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        warmup_steps: int = 1000,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        save_total_limit: int = 3,
    ) -> Dict[str, Any]:
        """
        Main training loop with production features
        """
        self.logger.info(f"Starting training for {epochs} epochs")
        
        # Prepare data
        train_dataloader = self._prepare_dataset(batch_size)
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        
        num_training_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        # Setup gradient scaler for mixed precision
        if self.use_mixed_precision:
            scaler = GradScaler()
        else:
            scaler = None
            
        # Training metrics
        global_step = 0
        total_loss = 0
        start_time = time.time()
        
        # Check for checkpoint to resume
        checkpoint = self.checkpoint_manager.load_latest()
        if checkpoint:
            self.logger.info(f"Resuming from checkpoint at step {checkpoint['global_step']}")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            global_step = checkpoint['global_step']
            
        # Main training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            for step, batch in enumerate(train_dataloader):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass with mixed precision
                    if self.use_mixed_precision:
                        with autocast():
                            outputs = self.model(**batch)
                            loss = outputs.loss / gradient_accumulation_steps
                    else:
                        outputs = self.model(**batch)
                        loss = outputs.loss / gradient_accumulation_steps
                        
                    # Backward pass
                    if self.use_mixed_precision:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                        
                    # Gradient accumulation
                    if (step + 1) % gradient_accumulation_steps == 0:
                        # Gradient clipping
                        if self.use_mixed_precision:
                            scaler.unscale_(optimizer)
                            
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_grad_norm,
                        )
                        
                        # Optimizer step
                        if self.use_mixed_precision:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                            
                        scheduler.step()
                        optimizer.zero_grad()
                        
                    # Update metrics
                    total_loss += loss.item() * gradient_accumulation_steps
                    epoch_loss += loss.item() * gradient_accumulation_steps
                    global_step += 1
                    
                    # Collect metrics
                    if global_step % 10 == 0:
                        metrics = self._collect_metrics(
                            loss=total_loss / global_step,
                            learning_rate=scheduler.get_last_lr()[0],
                            throughput=self._calculate_throughput(start_time, global_step, batch_size),
                        )
                        
                        if self.rank == 0:
                            wandb.log(metrics, step=global_step)
                            
                    # Checkpoint
                    if global_step % self.checkpoint_interval == 0:
                        self._save_checkpoint(
                            epoch=epoch,
                            global_step=global_step,
                            model=self.model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            loss=total_loss / global_step,
                        )
                        
                    # Log progress
                    if global_step % 100 == 0 and self.rank == 0:
                        self.logger.info(
                            f"Epoch: {epoch}, Step: {global_step}, "
                            f"Loss: {total_loss / global_step:.4f}, "
                            f"LR: {scheduler.get_last_lr()[0]:.2e}"
                        )
                        
                except Exception as e:
                    self.logger.error(f"Training error at step {global_step}: {e}")
                    if self.max_retries > 0:
                        self.max_retries -= 1
                        self.logger.info(f"Retrying... ({self.max_retries} retries left)")
                        continue
                    else:
                        raise
                        
            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            self.logger.info(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")
            
        # Training completed
        total_time = time.time() - start_time
        final_metrics = {
            "loss": total_loss / global_step,
            "total_time": total_time,
            "throughput": self._calculate_throughput(start_time, global_step, batch_size),
            "total_cost": self.cost_tracker.get_total_cost(total_time),
        }
        
        self.logger.info(f"Training completed. Final metrics: {final_metrics}")
        
        return final_metrics
        
    def _collect_metrics(self, loss: float, learning_rate: float, throughput: float) -> Dict[str, float]:
        """Collect comprehensive training metrics"""
        # GPU metrics
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        metrics = {
            "train/loss": loss,
            "train/learning_rate": learning_rate,
            "train/throughput_tokens_per_sec": throughput,
            "system/gpu_memory_used_gb": gpu_info.used / 1e9,
            "system/gpu_memory_percent": (gpu_info.used / gpu_info.total) * 100,
            "system/gpu_utilization": gpu_utilization.gpu,
            "system/cpu_percent": cpu_percent,
            "system/memory_percent": memory_percent,
            "cost/accumulated_usd": self.cost_tracker.get_accumulated_cost(),
        }
        
        return metrics
        
    def _calculate_throughput(self, start_time: float, steps: int, batch_size: int) -> float:
        """Calculate training throughput in tokens/sec"""
        elapsed_time = time.time() - start_time
        total_tokens = steps * batch_size * 512 * self.world_size  # seq_len = 512
        return total_tokens / elapsed_time
        
    def _save_checkpoint(
        self,
        epoch: int,
        global_step: int,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        loss: float,
    ):
        """Save training checkpoint with FSDP state"""
        if self.rank == 0:
            self.logger.info(f"Saving checkpoint at step {global_step}")
            
        # Configure FSDP state dict
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            save_policy,
        ):
            state_dict = self.model.state_dict()
            
        # Only save on rank 0
        if self.rank == 0:
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss,
                "config": {
                    "model_name": self.model_name,
                    "world_size": self.world_size,
                },
            }
            
            self.checkpoint_manager.save(checkpoint, global_step)
            
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.eval()
        total_loss = 0
        total_steps = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                total_loss += loss.item()
                total_steps += 1
                
        avg_loss = total_loss / total_steps
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity.item(),
        }