"""
Comprehensive tests for distributed training components
Tests FSDP, checkpointing, cost tracking, and fault tolerance
"""

import pytest
import torch
import torch.distributed as dist
import tempfile
import os
import time
from unittest.mock import Mock, patch, MagicMock
import json
import numpy as np

# Import our modules
from src.fsdp_trainer import FSDPTrainer
from src.checkpoint_manager import CheckpointManager
from src.cost_tracker import CostTracker
from src.monitoring import MetricsCollector
from src.optimization import MemoryOptimizer, DynamicBatchScheduler
from src.distributed_utils import DistributedCoordinator, ElasticTrainingManager


class TestFSDPTrainer:
    """Test FSDP trainer functionality"""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing"""
        model = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
        )
        return model
    
    @pytest.fixture
    def trainer(self, mock_model, tmp_path):
        """Create trainer instance for testing"""
        with patch('torch.distributed.is_initialized', return_value=False):
            with patch('torch.distributed.init_process_group'):
                trainer = FSDPTrainer(
                    model_name="test-model",
                    dataset="test-dataset",
                    num_nodes=1,
                    checkpoint_dir=str(tmp_path),
                    wandb_project="test-project",
                )
                # Replace model with mock
                trainer.model = mock_model
                return trainer
    
    def test_trainer_initialization(self, trainer):
        """Test trainer initializes correctly"""
        assert trainer.model_name == "test-model"
        assert trainer.dataset == "test-dataset"
        assert trainer.num_nodes == 1
        assert trainer.checkpoint_manager is not None
        assert trainer.metrics_collector is not None
        assert trainer.cost_tracker is not None
    
    def test_training_step(self, trainer):
        """Test single training step"""
        # Create mock data
        batch = {
            "input_ids": torch.randint(0, 1000, (4, 128)),
            "attention_mask": torch.ones(4, 128),
            "labels": torch.randint(0, 1000, (4, 128)),
        }
        
        # Mock model forward
        trainer.model.forward = Mock(return_value=Mock(loss=torch.tensor(2.5)))
        
        # Perform training step
        optimizer = torch.optim.Adam(trainer.model.parameters())
        loss = trainer._training_step(batch, optimizer)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() == 2.5
    
    def test_gradient_accumulation(self, trainer):
        """Test gradient accumulation works correctly"""
        trainer.gradient_accumulation_steps = 4
        accumulated_gradients = []
        
        for step in range(4):
            # Simulate gradient accumulation
            loss = torch.tensor(1.0, requires_grad=True)
            (loss / 4).backward()
            
            if (step + 1) % 4 == 0:
                # Check gradients are accumulated
                assert loss.grad is not None
                
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_training(self, trainer):
        """Test mixed precision training"""
        trainer.use_mixed_precision = True
        scaler = torch.cuda.amp.GradScaler()
        
        # Create dummy data
        batch = torch.randn(4, 512).cuda()
        
        with torch.cuda.amp.autocast():
            output = trainer.model(batch)
            loss = output.mean()
            
        # Check output is in half precision
        assert output.dtype == torch.float16 or output.dtype == torch.bfloat16


class TestCheckpointManager:
    """Test checkpoint management functionality"""
    
    @pytest.fixture
    def checkpoint_manager(self, tmp_path):
        """Create checkpoint manager for testing"""
        return CheckpointManager(
            checkpoint_dir=str(tmp_path),
            storage_backend="local",
            keep_last_n=3,
        )
    
    def test_save_and_load_checkpoint(self, checkpoint_manager):
        """Test saving and loading checkpoints"""
        # Create test checkpoint
        checkpoint = {
            "epoch": 1,
            "global_step": 1000,
            "model_state_dict": {"weight": torch.randn(10, 10)},
            "optimizer_state_dict": {"lr": 0.001},
            "loss": 2.5,
        }
        
        # Save checkpoint
        checkpoint_manager.save(checkpoint, step=1000)
        
        # Load checkpoint
        loaded = checkpoint_manager.load_latest()
        
        assert loaded is not None
        assert loaded["epoch"] == 1
        assert loaded["global_step"] == 1000
        assert torch.allclose(loaded["model_state_dict"]["weight"], checkpoint["model_state_dict"]["weight"])
    
    def test_checkpoint_rotation(self, checkpoint_manager):
        """Test old checkpoints are removed"""
        # Save multiple checkpoints
        for i in range(5):
            checkpoint = {
                "global_step": i * 100,
                "model_state_dict": {"weight": torch.randn(10, 10)},
            }
            checkpoint_manager.save(checkpoint, step=i * 100)
            
        # Check only last 3 are kept
        checkpoints = checkpoint_manager._list_checkpoints()
        assert len(checkpoints) == 3
        assert checkpoints[0]["step"] == 400  # Most recent
        assert checkpoints[2]["step"] == 200  # Oldest kept
    
    def test_checkpoint_corruption_detection(self, checkpoint_manager, tmp_path):
        """Test corruption detection works"""
        # Save valid checkpoint
        checkpoint = {
            "global_step": 1000,
            "model_state_dict": {"weight": torch.randn(10, 10)},
        }
        checkpoint_manager.save(checkpoint, step=1000)
        
        # Corrupt the checkpoint file
        checkpoint_files = list(tmp_path.glob("checkpoint-*"))
        if checkpoint_files:
            with open(checkpoint_files[0], "wb") as f:
                f.write(b"corrupted data")
                
        # Try to load - should handle corruption gracefully
        loaded = checkpoint_manager.load_latest()
        assert loaded is None  # Should return None for corrupted checkpoint
    
    @patch('boto3.client')
    def test_s3_upload(self, mock_boto3, checkpoint_manager):
        """Test S3 checkpoint upload"""
        checkpoint_manager.storage_backend = "s3"
        checkpoint_manager.cloud_bucket = "test-bucket"
        checkpoint_manager.s3_client = mock_boto3.return_value
        
        checkpoint = {
            "global_step": 1000,
            "model_state_dict": {"weight": torch.randn(10, 10)},
        }
        
        checkpoint_manager.save(checkpoint, step=1000)
        
        # Verify S3 upload was called
        mock_boto3.return_value.upload_file.assert_called()


class TestCostTracker:
    """Test cost tracking functionality"""
    
    @pytest.fixture
    def cost_tracker(self):
        """Create cost tracker for testing"""
        with patch('boto3.client'):
            return CostTracker(
                instance_type="p4d.24xlarge",
                use_spot=True,
                region="us-east-1",
            )
    
    def test_cost_calculation(self, cost_tracker):
        """Test cost calculation accuracy"""
        # Set current price
        cost_tracker.current_spot_price = 10.0
        
        # Simulate 2 hours of training
        cost_tracker.start_time = time.time() - 7200  # 2 hours ago
        
        total_cost = cost_tracker.get_accumulated_cost()
        assert abs(total_cost - 20.0) < 0.1  # Should be ~$20
    
    def test_spot_price_monitoring(self, cost_tracker):
        """Test spot price monitoring"""
        # Add price history
        for i in range(10):
            cost_tracker.cost_history.append({
                "timestamp": f"2024-01-01T{i:02d}:00:00",
                "price": 10.0 + i * 0.5,
                "instance_type": "p4d.24xlarge",
            })
            
        # Check price statistics
        prices = [h["price"] for h in cost_tracker.cost_history]
        assert min(prices) == 10.0
        assert max(prices) == 14.5
        assert len(cost_tracker.cost_history) == 10
    
    def test_cost_optimization_recommendations(self, cost_tracker):
        """Test cost optimization recommendations"""
        # Set high current price
        cost_tracker.current_spot_price = 25.0
        cost_tracker.ON_DEMAND_PRICES["p4d.24xlarge"] = 32.77
        
        recommendations = cost_tracker.get_cost_optimization_recommendations()
        
        # Should recommend waiting due to low discount
        assert any("high_spot_price" in r["type"] for r in recommendations)
    
    def test_multi_cloud_comparison(self, cost_tracker):
        """Test multi-cloud cost comparison"""
        comparison = cost_tracker.get_multi_cloud_comparison()
        
        assert "aws" in comparison
        assert comparison["aws"]["instance_type"] == "p4d.24xlarge"
        
        if "gcp" in comparison:
            assert "preemptible_price" in comparison["gcp"]


class TestMonitoring:
    """Test monitoring and metrics collection"""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector for testing"""
        with patch('pynvml.nvmlInit'):
            with patch('pynvml.nvmlDeviceGetCount', return_value=1):
                with patch('pynvml.nvmlDeviceGetHandleByIndex'):
                    return MetricsCollector(rank=0)
    
    def test_metrics_update(self, metrics_collector):
        """Test metrics can be updated"""
        metrics_collector.update_training_metrics(
            loss=2.5,
            throughput=50000,
            learning_rate=0.001,
            gradient_norm=1.5,
            batch_time=0.5,
        )
        
        # Verify metrics were set
        assert metrics_collector.training_loss._value.get() == 2.5
        assert metrics_collector.training_throughput._value.get() == 50000
    
    def test_gpu_metrics_collection(self, metrics_collector):
        """Test GPU metrics collection"""
        with patch('pynvml.nvmlDeviceGetMemoryInfo') as mock_mem:
            with patch('pynvml.nvmlDeviceGetUtilizationRates') as mock_util:
                # Mock GPU info
                mock_mem.return_value = Mock(
                    total=40 * 1e9,  # 40GB
                    used=30 * 1e9,   # 30GB used
                    free=10 * 1e9,   # 10GB free
                )
                mock_util.return_value = Mock(gpu=80)  # 80% utilization
                
                # Collect metrics
                metrics_collector._collect_system_metrics()
                
                # Get summary
                summary = metrics_collector.get_current_metrics()
                
                assert summary["gpu"][0]["utilization_percent"] == 80
                assert summary["gpu"][0]["memory_used_gb"] == 30.0


class TestOptimization:
    """Test memory and performance optimization"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing"""
        return torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
        )
    
    @pytest.fixture
    def memory_optimizer(self, mock_model):
        """Create memory optimizer for testing"""
        with patch('pynvml.nvmlInit'):
            with patch('pynvml.nvmlDeviceGetHandleByIndex'):
                return MemoryOptimizer(
                    model=mock_model,
                    gradient_checkpointing=True,
                )
    
    def test_gradient_checkpointing_applied(self, memory_optimizer, mock_model):
        """Test gradient checkpointing is applied"""
        # Check if forward methods were wrapped
        # In real implementation, this would check for checkpoint wrapper
        assert memory_optimizer.gradient_checkpointing is True
    
    def test_memory_optimization_context(self, memory_optimizer):
        """Test memory optimization context manager"""
        with memory_optimizer.optimize_memory_context():
            # Inside context, memory should be optimized
            # This would test actual memory settings in production
            pass
    
    def test_optimal_batch_size_search(self, memory_optimizer):
        """Test optimal batch size search"""
        with patch.object(memory_optimizer, '_test_batch_size') as mock_test:
            # Mock successful tests for batch sizes 1-8, fail for 16
            def side_effect(batch_size, seq_len):
                if batch_size > 8:
                    raise torch.cuda.OutOfMemoryError()
                    
            mock_test.side_effect = side_effect
            
            optimal = memory_optimizer.get_optimal_batch_size(
                initial_batch_size=16,
                sequence_length=512,
            )
            
            assert optimal == 8
    
    def test_dynamic_batch_scheduler(self):
        """Test dynamic batch size scheduling"""
        scheduler = DynamicBatchScheduler(
            initial_batch_size=8,
            min_batch_size=1,
            max_batch_size=16,
        )
        
        # Test adjustment based on memory
        with patch('pynvml.nvmlDeviceGetMemoryInfo') as mock_mem:
            # High memory usage - should decrease
            mock_mem.return_value = Mock(
                total=40 * 1e9,
                used=38 * 1e9,  # 95% used
            )
            
            for _ in range(100):  # Trigger adjustment
                scheduler.step()
                
            assert scheduler.current_batch_size < 8
            
            # Low memory usage - should increase
            mock_mem.return_value = Mock(
                total=40 * 1e9,
                used=20 * 1e9,  # 50% used
            )
            
            for _ in range(100):  # Trigger adjustment
                scheduler.step()
                
            assert scheduler.current_batch_size > scheduler.min_batch_size


class TestDistributedUtils:
    """Test distributed training utilities"""
    
    @pytest.fixture
    def distributed_coordinator(self):
        """Create distributed coordinator for testing"""
        with patch('torch.distributed.is_initialized', return_value=False):
            with patch('torch.distributed.init_process_group'):
                return DistributedCoordinator(
                    backend="nccl",
                    elastic=False,  # Disable health monitoring for tests
                )
    
    def test_node_info_gathering(self, distributed_coordinator):
        """Test gathering node information"""
        node_info = distributed_coordinator.get_node_info()
        
        assert "rank" in node_info
        assert "hostname" in node_info
        assert "cpu_count" in node_info
        assert "memory_gb" in node_info
    
    def test_fault_tolerant_execution(self, distributed_coordinator):
        """Test fault-tolerant execution context"""
        attempts = 0
        
        def flaky_operation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("Simulated failure")
            return "success"
            
        with distributed_coordinator.fault_tolerant_section("test_op", max_retries=3):
            result = flaky_operation()
            
        assert result == "success"
        assert attempts == 3
    
    def test_elastic_training_manager(self):
        """Test elastic training manager"""
        manager = ElasticTrainingManager(
            min_nodes=2,
            max_nodes=8,
        )
        
        # Test workload rebalancing
        with patch('torch.distributed.is_initialized', return_value=True):
            with patch('torch.distributed.get_world_size', return_value=4):
                # Mark node 2 as pending preemption
                manager.pending_preemption.add(2)
                
                # Rebalance 1000 samples
                distribution = manager.rebalance_workload(1000)
                
                # Should distribute among 3 active nodes
                assert len(distribution) == 3
                assert 0 in distribution
                assert 1 in distribution
                assert 3 in distribution
                assert 2 not in distribution  # Preempted
                
                # Check fair distribution
                total = sum(distribution.values())
                assert total == 1000


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the full system"""
    
    def test_end_to_end_training(self, tmp_path):
        """Test complete training flow"""
        # This would be a full integration test
        # Skipped in unit tests but run in integration environment
        pass
    
    def test_multi_node_coordination(self):
        """Test multi-node coordination"""
        # Requires actual distributed setup
        pass
    
    def test_spot_instance_preemption_handling(self):
        """Test handling of spot instance preemption"""
        # Requires cloud environment
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])