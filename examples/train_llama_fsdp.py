"""
Example: Train LLaMA model with FSDP using Metaflow
Demonstrates production features including automatic recovery and cost optimization
"""

from metaflow import FlowSpec, step, Parameter, kubernetes, resources, retry, catch
from metaflow.cards import Card, Markdown, Artifact, Table, Image
import os
import json
import torch
import time
from datetime import datetime


class LLaMATrainingFlow(FlowSpec):
    """
    Production-ready distributed training flow for LLaMA models
    
    Features:
    - Multi-node FSDP training
    - Automatic checkpoint recovery
    - Spot instance orchestration
    - Real-time monitoring
    - Cost optimization
    """
    
    # Flow parameters
    model_name = Parameter(
        "model",
        default="meta-llama/Llama-2-7b-hf",
        help="Model name from HuggingFace Hub",
    )
    
    dataset = Parameter(
        "dataset",
        default="c4",
        help="Dataset name (c4, openwebtext, pile)",
    )
    
    num_nodes = Parameter(
        "num_nodes",
        default=4,
        type=int,
        help="Number of training nodes",
    )
    
    batch_size = Parameter(
        "batch_size",
        default=32,
        type=int,
        help="Batch size per GPU",
    )
    
    learning_rate = Parameter(
        "lr",
        default=3e-4,
        type=float,
        help="Learning rate",
    )
    
    epochs = Parameter(
        "epochs",
        default=3,
        type=int,
        help="Number of training epochs",
    )
    
    use_spot = Parameter(
        "use_spot",
        default=True,
        type=bool,
        help="Use spot instances for cost savings",
    )
    
    checkpoint_interval = Parameter(
        "checkpoint_interval",
        default=1000,
        type=int,
        help="Checkpoint interval in steps",
    )
    
    mixed_precision = Parameter(
        "mixed_precision",
        default=True,
        type=bool,
        help="Use mixed precision training",
    )
    
    gradient_checkpointing = Parameter(
        "gradient_checkpointing",
        default=True,
        type=bool,
        help="Enable gradient checkpointing",
    )
    
    wandb_project = Parameter(
        "wandb_project",
        default="llama-fsdp-training",
        help="Weights & Biases project name",
    )
    
    @step
    def start(self):
        """
        Initialize training job and validate configuration
        """
        print(f"🚀 Starting LLaMA FSDP training")
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset}")
        print(f"Nodes: {self.num_nodes}")
        print(f"Spot instances: {self.use_spot}")
        
        # Set up training configuration
        self.training_config = {
            "model_name": self.model_name,
            "dataset": self.dataset,
            "num_nodes": self.num_nodes,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "use_spot": self.use_spot,
            "checkpoint_interval": self.checkpoint_interval,
            "mixed_precision": self.mixed_precision,
            "gradient_checkpointing": self.gradient_checkpointing,
            "wandb_project": self.wandb_project,
            "job_id": f"llama-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        }
        
        # Estimate training time and cost
        self._estimate_training_cost()
        
        self.next(self.setup_infrastructure)
        
    @step
    @catch(var="infrastructure_error")
    def setup_infrastructure(self):
        """
        Set up distributed training infrastructure
        """
        print("🔧 Setting up infrastructure")
        
        # Configure Kubernetes resources
        self.k8s_config = {
            "namespace": "ml-training",
            "service_account": "training-sa",
            "node_selector": {
                "node.kubernetes.io/instance-type": "p4d.24xlarge" if not self.use_spot else "spot-p4d.24xlarge",
            },
            "tolerations": [
                {
                    "key": "nvidia.com/gpu",
                    "operator": "Exists",
                    "effect": "NoSchedule",
                },
                {
                    "key": "spot",
                    "operator": "Equal",
                    "value": "true",
                    "effect": "NoSchedule",
                } if self.use_spot else None,
            ],
        }
        
        # Set up monitoring
        self.monitoring_config = {
            "prometheus_gateway": os.environ.get("PROMETHEUS_GATEWAY", "prometheus-pushgateway:9091"),
            "grafana_url": os.environ.get("GRAFANA_URL", "http://grafana:3000"),
            "dashboard_uid": "fsdp-training",
        }
        
        # Check for existing checkpoints
        self._check_existing_checkpoints()
        
        self.next(self.distributed_training, num_parallel=self.num_nodes)
        
    @step
    @kubernetes(
        gpu=8,
        cpu=96,
        memory=768,
        image="nvcr.io/nvidia/pytorch:23.10-py3",
    )
    @resources(
        use_spot=True,
        max_retries=3,
        retry_delay=60,
    )
    @retry(times=3, minutes_between_retries=5)
    def distributed_training(self):
        """
        Main distributed training step
        """
        import sys
        sys.path.append("/app/src")
        
        from fsdp_trainer import FSDPTrainer
        from monitoring import MetricsCollector
        from cost_tracker import CostTracker
        
        # Get distributed training rank
        node_index = self.parallel_step_index
        os.environ["RANK"] = str(node_index)
        os.environ["LOCAL_RANK"] = str(0)  # Single GPU per task for this example
        os.environ["WORLD_SIZE"] = str(self.num_nodes)
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "distributed-training-master")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        
        print(f"🎯 Starting training on node {node_index}/{self.num_nodes}")
        
        try:
            # Initialize trainer
            trainer = FSDPTrainer(
                model_name=self.model_name,
                dataset=self.dataset,
                num_nodes=self.num_nodes,
                checkpoint_interval=self.checkpoint_interval,
                use_mixed_precision=self.mixed_precision,
                gradient_checkpointing=self.gradient_checkpointing,
                checkpoint_dir=f"s3://your-bucket/checkpoints/{self.training_config['job_id']}",
                wandb_project=self.wandb_project,
                cost_aware=True,
                max_retries=3,
            )
            
            # Start training
            start_time = time.time()
            
            metrics = trainer.train(
                epochs=self.epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                warmup_steps=1000,
                gradient_accumulation_steps=4,
                max_grad_norm=1.0,
            )
            
            training_time = time.time() - start_time
            
            # Save final metrics
            self.training_metrics = {
                "node_index": node_index,
                "final_loss": metrics["loss"],
                "throughput": metrics["throughput"],
                "training_time": training_time,
                "total_cost": metrics["total_cost"],
            }
            
            # Export detailed metrics
            if node_index == 0:
                trainer.metrics_collector.export_metrics_summary(
                    f"metrics_summary_{self.training_config['job_id']}.json"
                )
                trainer.cost_tracker.export_cost_data(
                    f"cost_report_{self.training_config['job_id']}.json"
                )
                
            print(f"✅ Training completed on node {node_index}")
            print(f"   Final loss: {metrics['loss']:.4f}")
            print(f"   Throughput: {metrics['throughput']:.0f} tokens/sec")
            print(f"   Cost: ${metrics['total_cost']:.2f}")
            
        except Exception as e:
            print(f"❌ Training failed on node {node_index}: {e}")
            self.training_error = str(e)
            raise
            
        finally:
            # Cleanup
            if 'trainer' in locals():
                trainer.metrics_collector.shutdown()
                trainer.cost_tracker.shutdown()
                
        self.next(self.aggregate_results)
        
    @step
    def aggregate_results(self, inputs):
        """
        Aggregate results from all training nodes
        """
        print("📊 Aggregating results from all nodes")
        
        # Merge results from all nodes
        all_metrics = [inp.training_metrics for inp in inputs if hasattr(inp, 'training_metrics')]
        
        if not all_metrics:
            print("❌ No successful training runs")
            self.final_metrics = None
            self.next(self.generate_report)
            return
            
        # Calculate aggregate metrics
        total_cost = sum(m["total_cost"] for m in all_metrics)
        avg_throughput = sum(m["throughput"] for m in all_metrics) / len(all_metrics)
        final_loss = min(m["final_loss"] for m in all_metrics)  # Best loss
        total_time = max(m["training_time"] for m in all_metrics)  # Longest node
        
        self.final_metrics = {
            "num_nodes": len(all_metrics),
            "final_loss": final_loss,
            "average_throughput_per_node": avg_throughput,
            "total_throughput": avg_throughput * len(all_metrics),
            "total_training_time_hours": total_time / 3600,
            "total_cost": total_cost,
            "cost_per_epoch": total_cost / self.epochs,
        }
        
        print("📈 Training Summary:")
        print(f"   Nodes completed: {len(all_metrics)}/{self.num_nodes}")
        print(f"   Final loss: {final_loss:.4f}")
        print(f"   Total throughput: {self.final_metrics['total_throughput']:.0f} tokens/sec")
        print(f"   Training time: {self.final_metrics['total_training_time_hours']:.2f} hours")
        print(f"   Total cost: ${total_cost:.2f}")
        print(f"   Cost per epoch: ${self.final_metrics['cost_per_epoch']:.2f}")
        
        self.next(self.generate_report)
        
    @step
    @card
    def generate_report(self):
        """
        Generate training report and visualizations
        """
        print("📝 Generating training report")
        
        # Create Metaflow card with results
        if self.final_metrics:
            # Calculate cost savings
            on_demand_cost = self._calculate_on_demand_cost()
            savings = on_demand_cost - self.final_metrics["total_cost"]
            savings_percent = (savings / on_demand_cost * 100) if on_demand_cost > 0 else 0
            
            report_md = f"""
# 🚀 LLaMA FSDP Training Report

## Model Configuration
- **Model**: {self.model_name}
- **Dataset**: {self.dataset}
- **Nodes**: {self.num_nodes}
- **Batch Size**: {self.batch_size} per GPU
- **Learning Rate**: {self.learning_rate}
- **Epochs**: {self.epochs}

## Training Results
- **Final Loss**: {self.final_metrics['final_loss']:.4f}
- **Total Throughput**: {self.final_metrics['total_throughput']:.0f} tokens/sec
- **Training Time**: {self.final_metrics['total_training_time_hours']:.2f} hours
- **Tokens Processed**: ~{self.final_metrics['total_throughput'] * self.final_metrics['total_training_time_hours'] * 3600 / 1e9:.2f}B

## Cost Analysis
- **Total Cost**: ${self.final_metrics['total_cost']:.2f}
- **On-Demand Cost**: ${on_demand_cost:.2f}
- **Savings**: ${savings:.2f} ({savings_percent:.1f}%)
- **Cost per Epoch**: ${self.final_metrics['cost_per_epoch']:.2f}
- **Cost per 1B Tokens**: ${self.final_metrics['total_cost'] / (self.final_metrics['total_throughput'] * self.final_metrics['total_training_time_hours'] * 3600 / 1e9):.2f}

## Infrastructure
- **Instance Type**: p4d.24xlarge (8x A100 40GB)
- **Spot Instances**: {'✅ Enabled' if self.use_spot else '❌ Disabled'}
- **Mixed Precision**: {'✅ Enabled' if self.mixed_precision else '❌ Disabled'}
- **Gradient Checkpointing**: {'✅ Enabled' if self.gradient_checkpointing else '❌ Disabled'}

## Monitoring
- **Grafana Dashboard**: [View Dashboard]({self.monitoring_config['grafana_url']}/d/{self.monitoring_config['dashboard_uid']})
- **W&B Project**: [View Project](https://wandb.ai/{self.wandb_project})

## Recommendations
1. {'✅ Spot instances provided significant cost savings' if savings_percent > 50 else '⚠️ Consider adjusting spot instance strategy'}
2. {'✅ Training completed successfully on all nodes' if self.final_metrics['num_nodes'] == self.num_nodes else f'⚠️ Only {self.final_metrics["num_nodes"]}/{self.num_nodes} nodes completed successfully'}
3. {'✅ Excellent throughput achieved' if self.final_metrics['total_throughput'] > 100000 else '⚠️ Consider optimizing batch size or gradient accumulation'}

---
*Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}*
            """
            
            current_card = self.card
            current_card.append(Markdown(report_md))
            
            # Add metrics table
            metrics_table = Table([
                ["Metric", "Value"],
                ["Model Parameters", "7B"],
                ["Training Nodes", str(self.num_nodes)],
                ["GPUs Total", str(self.num_nodes * 8)],
                ["Batch Size (Global)", str(self.batch_size * self.num_nodes * 8)],
                ["Training Duration", f"{self.final_metrics['total_training_time_hours']:.2f} hours"],
                ["Cost Efficiency", f"${self.final_metrics['total_cost'] / self.final_metrics['total_training_time_hours']:.2f}/hour"],
            ])
            current_card.append(metrics_table)
            
        else:
            current_card = self.card
            current_card.append(Markdown("# ❌ Training Failed\n\nNo successful training runs completed."))
            
        print("✅ Report generation complete")
        self.next(self.end)
        
    @step
    def end(self):
        """
        Finalize training job
        """
        print("🎉 Training job completed!")
        
        if self.final_metrics:
            print("\n📊 Final Summary:")
            print(f"   Model: {self.model_name}")
            print(f"   Final Loss: {self.final_metrics['final_loss']:.4f}")
            print(f"   Total Cost: ${self.final_metrics['total_cost']:.2f}")
            print(f"   Training Time: {self.final_metrics['total_training_time_hours']:.2f} hours")
            
            # Save summary to file
            summary = {
                "job_id": self.training_config["job_id"],
                "config": self.training_config,
                "results": self.final_metrics,
                "timestamp": datetime.now().isoformat(),
            }
            
            with open(f"training_summary_{self.training_config['job_id']}.json", "w") as f:
                json.dump(summary, f, indent=2)
                
        print("\n🔗 Resources:")
        print(f"   Grafana: {self.monitoring_config['grafana_url']}")
        print(f"   W&B: https://wandb.ai/{self.wandb_project}")
        print(f"   Checkpoints: s3://your-bucket/checkpoints/{self.training_config['job_id']}")
        
    def _estimate_training_cost(self):
        """Estimate training cost before starting"""
        # Rough estimates
        tokens_per_epoch = 1e9  # 1B tokens
        throughput_per_node = 20000  # tokens/sec
        total_throughput = throughput_per_node * self.num_nodes
        
        training_time_hours = (tokens_per_epoch * self.epochs) / (total_throughput * 3600)
        
        # Cost estimates
        if self.use_spot:
            cost_per_hour = 12.0 * self.num_nodes  # ~$12/hour for p4d.24xlarge spot
        else:
            cost_per_hour = 32.77 * self.num_nodes  # On-demand price
            
        estimated_cost = training_time_hours * cost_per_hour
        
        print(f"\n💰 Cost Estimate:")
        print(f"   Estimated training time: {training_time_hours:.1f} hours")
        print(f"   Estimated cost: ${estimated_cost:.2f}")
        print(f"   Cost per epoch: ${estimated_cost / self.epochs:.2f}")
        
    def _check_existing_checkpoints(self):
        """Check for existing checkpoints from previous runs"""
        # In production, check S3/storage for existing checkpoints
        print("🔍 Checking for existing checkpoints...")
        # Implementation would go here
        
    def _calculate_on_demand_cost(self):
        """Calculate equivalent on-demand cost"""
        # p4d.24xlarge on-demand price
        on_demand_price_per_hour = 32.77
        total_hours = self.final_metrics["total_training_time_hours"]
        return on_demand_price_per_hour * self.num_nodes * total_hours


if __name__ == "__main__":
    LLaMATrainingFlow()