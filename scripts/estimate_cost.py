#!/usr/bin/env python3
"""
Estimate training costs for different model sizes and configurations
Helps with budget planning and resource allocation
"""

import click
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import matplotlib.pyplot as plt
import numpy as np

console = Console()

# Model configurations
MODEL_CONFIGS = {
    "1.3B": {"params": 1.3e9, "tokens_per_param": 20, "memory_gb": 8},
    "7B": {"params": 7e9, "tokens_per_param": 20, "memory_gb": 28},
    "13B": {"params": 13e9, "tokens_per_param": 20, "memory_gb": 52},
    "30B": {"params": 30e9, "tokens_per_param": 20, "memory_gb": 120},
    "70B": {"params": 70e9, "tokens_per_param": 20, "memory_gb": 280},
    "175B": {"params": 175e9, "tokens_per_param": 20, "memory_gb": 700},
}

# Instance configurations
INSTANCE_CONFIGS = {
    "p4d.24xlarge": {
        "gpus": 8,
        "gpu_type": "A100-40GB",
        "memory_per_gpu": 40,
        "on_demand_price": 32.77,
        "typical_spot_price": 11.80,
        "network_bandwidth": 400,  # Gbps
        "optimal_batch_size": 8,
    },
    "p4de.24xlarge": {
        "gpus": 8,
        "gpu_type": "A100-80GB", 
        "memory_per_gpu": 80,
        "on_demand_price": 40.96,
        "typical_spot_price": 15.20,
        "network_bandwidth": 400,
        "optimal_batch_size": 16,
    },
    "p3dn.24xlarge": {
        "gpus": 8,
        "gpu_type": "V100-32GB",
        "memory_per_gpu": 32,
        "on_demand_price": 31.22,
        "typical_spot_price": 9.68,
        "network_bandwidth": 100,
        "optimal_batch_size": 4,
    },
    "g5.48xlarge": {
        "gpus": 8,
        "gpu_type": "A10G",
        "memory_per_gpu": 24,
        "on_demand_price": 16.29,
        "typical_spot_price": 5.87,
        "network_bandwidth": 100,
        "optimal_batch_size": 2,
    },
}

# Training throughput estimates (tokens/sec per GPU)
THROUGHPUT_ESTIMATES = {
    ("A100-40GB", "7B"): 2500,
    ("A100-40GB", "13B"): 1400,
    ("A100-40GB", "30B"): 600,
    ("A100-40GB", "70B"): 250,
    ("A100-80GB", "7B"): 2800,
    ("A100-80GB", "13B"): 1600,
    ("A100-80GB", "30B"): 700,
    ("A100-80GB", "70B"): 300,
    ("A100-80GB", "175B"): 120,
    ("V100-32GB", "7B"): 1200,
    ("V100-32GB", "13B"): 700,
    ("V100-32GB", "30B"): 300,
    ("A10G", "7B"): 800,
    ("A10G", "13B"): 450,
}

@click.command()
@click.option('--model-size', '-m', 
              type=click.Choice(['1.3B', '7B', '13B', '30B', '70B', '175B']),
              default='7B',
              help='Model size to train')
@click.option('--dataset-size', '-d',
              type=float,
              default=1000,
              help='Dataset size in billions of tokens')
@click.option('--num-nodes', '-n',
              type=int,
              default=4,
              help='Number of nodes to use')
@click.option('--instance-type', '-i',
              type=click.Choice(list(INSTANCE_CONFIGS.keys())),
              default='p4d.24xlarge',
              help='Instance type to use')
@click.option('--use-spot/--on-demand',
              default=True,
              help='Use spot instances vs on-demand')
@click.option('--regions', '-r',
              default='us-east-1,us-west-2,eu-west-1',
              help='Regions to consider (comma-separated)')
@click.option('--include-storage/--no-storage',
              default=True,
              help='Include storage costs in estimate')
@click.option('--include-network/--no-network', 
              default=True,
              help='Include network transfer costs')
@click.option('--output', '-o',
              type=click.Choice(['summary', 'detailed', 'json']),
              default='summary',
              help='Output format')
@click.option('--save-plot', '-p',
              help='Save cost breakdown plot to file')
def main(model_size, dataset_size, num_nodes, instance_type, use_spot, regions,
         include_storage, include_network, output, save_plot):
    """
    Estimate training costs for large language models.
    
    Examples:
        # Estimate cost for 7B model on 1T tokens
        python estimate_cost.py -m 7B -d 1000
        
        # Compare spot vs on-demand for 13B model
        python estimate_cost.py -m 13B --on-demand
        
        # Detailed breakdown with plot
        python estimate_cost.py -m 30B -o detailed -p cost_breakdown.png
    """
    
    console.print(Panel.fit(
        "[bold cyan]ML Training Cost Estimator[/bold cyan]",
        box=box.DOUBLE
    ))
    
    # Validate configuration
    model_config = MODEL_CONFIGS[model_size]
    instance_config = INSTANCE_CONFIGS[instance_type]
    
    # Check if model fits
    total_gpus = num_nodes * instance_config['gpus']
    memory_per_gpu_needed = model_config['memory_gb'] / total_gpus
    
    if memory_per_gpu_needed > instance_config['memory_per_gpu'] * 0.9:  # 90% threshold
        console.print(f"[red]Warning: Model may not fit in memory![/red]")
        console.print(f"Need {memory_per_gpu_needed:.1f}GB per GPU, "
                     f"have {instance_config['memory_per_gpu']}GB")
        
        # Suggest alternatives
        min_gpus_needed = int(np.ceil(model_config['memory_gb'] / 
                                     (instance_config['memory_per_gpu'] * 0.9)))
        min_nodes_needed = int(np.ceil(min_gpus_needed / instance_config['gpus']))
        console.print(f"[yellow]Suggestion: Use at least {min_nodes_needed} nodes[/yellow]")
    
    # Calculate training time
    gpu_type = instance_config['gpu_type']
    throughput_per_gpu = THROUGHPUT_ESTIMATES.get((gpu_type, model_size), 1000)
    total_throughput = throughput_per_gpu * total_gpus
    
    total_tokens = dataset_size * 1e9
    training_time_seconds = total_tokens / total_throughput
    training_time_hours = training_time_seconds / 3600
    
    # Calculate costs
    regions_list = [r.strip() for r in regions.split(',')]
    
    # Compute costs
    compute_cost = calculate_compute_cost(
        training_time_hours, 
        num_nodes, 
        instance_type, 
        use_spot
    )
    
    # Storage costs
    storage_cost = 0
    if include_storage:
        storage_cost = calculate_storage_cost(
            model_size,
            training_time_hours,
            num_checkpoints=int(training_time_hours / 2)  # Checkpoint every 2 hours
        )
    
    # Network costs
    network_cost = 0
    if include_network:
        network_cost = calculate_network_cost(
            model_size,
            num_nodes,
            training_time_hours
        )
    
    # Total cost
    total_cost = compute_cost + storage_cost + network_cost
    
    # Cost savings vs on-demand
    on_demand_cost = training_time_hours * num_nodes * instance_config['on_demand_price']
    savings = on_demand_cost - total_cost
    savings_percent = (savings / on_demand_cost * 100) if on_demand_cost > 0 else 0
    
    # Display results
    if output == 'summary':
        display_summary(
            model_size, dataset_size, num_nodes, instance_type,
            training_time_hours, total_cost, on_demand_cost,
            compute_cost, storage_cost, network_cost,
            use_spot, total_throughput
        )
    elif output == 'detailed':
        display_detailed(
            model_size, dataset_size, num_nodes, instance_type,
            training_time_hours, total_cost, on_demand_cost,
            compute_cost, storage_cost, network_cost,
            use_spot, total_throughput, model_config, instance_config
        )
    else:  # json
        result = {
            "configuration": {
                "model_size": model_size,
                "dataset_size_billions": dataset_size,
                "num_nodes": num_nodes,
                "instance_type": instance_type,
                "use_spot": use_spot,
                "total_gpus": total_gpus,
            },
            "performance": {
                "throughput_tokens_per_sec": total_throughput,
                "training_time_hours": training_time_hours,
                "training_time_days": training_time_hours / 24,
            },
            "costs": {
                "compute": compute_cost,
                "storage": storage_cost,
                "network": network_cost,
                "total": total_cost,
                "on_demand_equivalent": on_demand_cost,
                "savings": savings,
                "savings_percent": savings_percent,
            },
            "metrics": {
                "cost_per_billion_tokens": total_cost / dataset_size,
                "cost_per_gpu_hour": total_cost / (total_gpus * training_time_hours),
                "tokens_per_dollar": dataset_size * 1e9 / total_cost,
            }
        }
        print(json.dumps(result, indent=2))
    
    # Generate plot if requested
    if save_plot:
        generate_cost_plot(
            compute_cost, storage_cost, network_cost,
            on_demand_cost, save_plot
        )
        console.print(f"[green]Cost breakdown plot saved to {save_plot}[/green]")

def calculate_compute_cost(hours: float, nodes: int, instance_type: str, 
                          use_spot: bool) -> float:
    """Calculate compute costs"""
    instance_config = INSTANCE_CONFIGS[instance_type]
    
    if use_spot:
        hourly_rate = instance_config['typical_spot_price']
    else:
        hourly_rate = instance_config['on_demand_price']
    
    return hours * nodes * hourly_rate

def calculate_storage_cost(model_size: str, training_hours: float, 
                          num_checkpoints: int) -> float:
    """Calculate storage costs for checkpoints and data"""
    # S3 storage costs
    s3_storage_per_gb_month = 0.023  # Standard storage
    s3_put_per_1000 = 0.005
    s3_get_per_1000 = 0.0004
    
    # Checkpoint sizes
    checkpoint_size_gb = MODEL_CONFIGS[model_size]['memory_gb'] * 4  # Model + optimizer states
    
    # Storage usage
    total_storage_gb = checkpoint_size_gb * min(num_checkpoints, 10)  # Keep last 10
    storage_months = training_hours / (24 * 30)
    
    # Calculate costs
    storage_cost = total_storage_gb * s3_storage_per_gb_month * storage_months
    
    # API costs (uploads/downloads)
    put_requests = num_checkpoints * 100  # ~100 parts per checkpoint
    get_requests = num_checkpoints * 20   # Some reads for validation
    
    api_cost = (put_requests / 1000 * s3_put_per_1000 + 
                get_requests / 1000 * s3_get_per_1000)
    
    return storage_cost + api_cost

def calculate_network_cost(model_size: str, num_nodes: int, 
                          training_hours: float) -> float:
    """Calculate network transfer costs"""
    # AWS data transfer costs
    transfer_per_gb = 0.02  # Inter-AZ transfer
    
    # Estimate data transfer (gradient synchronization)
    model_gb = MODEL_CONFIGS[model_size]['memory_gb']
    
    # Gradient syncs per hour (assuming 1 sync per 10 seconds)
    syncs_per_hour = 360
    
    # Data per sync (gradients are same size as model)
    data_per_sync_gb = model_gb * (num_nodes - 1) / num_nodes  # All-reduce pattern
    
    total_transfer_gb = data_per_sync_gb * syncs_per_hour * training_hours
    
    return total_transfer_gb * transfer_per_gb

def display_summary(model_size, dataset_size, num_nodes, instance_type,
                   training_hours, total_cost, on_demand_cost,
                   compute_cost, storage_cost, network_cost,
                   use_spot, throughput):
    """Display summary of cost estimates"""
    
    # Configuration table
    config_table = Table(title="Training Configuration", box=box.ROUNDED)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="white")
    
    config_table.add_row("Model Size", model_size)
    config_table.add_row("Dataset Size", f"{dataset_size}B tokens")
    config_table.add_row("Instance Type", instance_type)
    config_table.add_row("Number of Nodes", str(num_nodes))
    config_table.add_row("Total GPUs", str(num_nodes * INSTANCE_CONFIGS[instance_type]['gpus']))
    config_table.add_row("Pricing", "Spot" if use_spot else "On-Demand")
    
    console.print(config_table)
    
    # Performance table
    perf_table = Table(title="Performance Estimates", box=box.ROUNDED)
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="white")
    
    perf_table.add_row("Throughput", f"{throughput:,.0f} tokens/sec")
    perf_table.add_row("Training Time", f"{training_hours:.1f} hours ({training_hours/24:.1f} days)")
    perf_table.add_row("Tokens per GPU-hour", f"{dataset_size * 1e9 / (num_nodes * 8 * training_hours):,.0f}")
    
    console.print(perf_table)
    
    # Cost table
    cost_table = Table(title="Cost Breakdown", box=box.ROUNDED)
    cost_table.add_column("Component", style="cyan")
    cost_table.add_column("Cost", style="green", justify="right")
    cost_table.add_column("Percentage", style="yellow", justify="right")
    
    cost_table.add_row("Compute", f"${compute_cost:,.2f}", f"{compute_cost/total_cost*100:.1f}%")
    cost_table.add_row("Storage", f"${storage_cost:,.2f}", f"{storage_cost/total_cost*100:.1f}%")
    cost_table.add_row("Network", f"${network_cost:,.2f}", f"{network_cost/total_cost*100:.1f}%")
    cost_table.add_row("", "", "")
    cost_table.add_row("[bold]Total", f"[bold]${total_cost:,.2f}", "[bold]100.0%")
    
    console.print(cost_table)
    
    # Savings summary
    if use_spot:
        savings_panel = Panel(
            f"[bold green]💰 Estimated Savings[/bold green]\n\n"
            f"On-Demand Cost: ${on_demand_cost:,.2f}\n"
            f"Your Cost: ${total_cost:,.2f}\n"
            f"Savings: ${on_demand_cost - total_cost:,.2f} ({(on_demand_cost - total_cost)/on_demand_cost*100:.1f}%)",
            box=box.DOUBLE,
            border_style="green"
        )
        console.print(savings_panel)
    
    # Key metrics
    console.print("\n[bold]Key Metrics:[/bold]")
    console.print(f"• Cost per billion tokens: ${total_cost / dataset_size:.2f}")
    console.print(f"• Cost per training day: ${total_cost / (training_hours / 24):.2f}")
    console.print(f"• Tokens per dollar: {dataset_size * 1e9 / total_cost:,.0f}")

def display_detailed(model_size, dataset_size, num_nodes, instance_type,
                    training_hours, total_cost, on_demand_cost,
                    compute_cost, storage_cost, network_cost,
                    use_spot, throughput, model_config, instance_config):
    """Display detailed cost analysis"""
    
    # First show summary
    display_summary(model_size, dataset_size, num_nodes, instance_type,
                   training_hours, total_cost, on_demand_cost,
                   compute_cost, storage_cost, network_cost,
                   use_spot, throughput)
    
    # Additional details
    console.print("\n[bold]Detailed Analysis:[/bold]\n")
    
    # Resource utilization
    util_table = Table(title="Resource Utilization", box=box.ROUNDED)
    util_table.add_column("Resource", style="cyan")
    util_table.add_column("Usage", style="white")
    util_table.add_column("Efficiency", style="green")
    
    total_gpus = num_nodes * instance_config['gpus']
    memory_efficiency = (model_config['memory_gb'] / 
                        (total_gpus * instance_config['memory_per_gpu']) * 100)
    
    util_table.add_row("GPU Memory", 
                      f"{model_config['memory_gb']}GB / {total_gpus * instance_config['memory_per_gpu']}GB",
                      f"{memory_efficiency:.1f}%")
    
    # Estimate compute efficiency
    compute_efficiency = min(95, 70 + num_nodes * 2)  # Simplified estimate
    util_table.add_row("Compute", 
                      f"{compute_efficiency:.0f}% utilized",
                      "Good" if compute_efficiency > 80 else "Fair")
    
    # Network utilization
    network_util = 30 if num_nodes > 1 else 0
    util_table.add_row("Network", 
                      f"{network_util}% of {instance_config['network_bandwidth']}Gbps",
                      "Good" if network_util < 50 else "High")
    
    console.print(util_table)
    
    # Optimization suggestions
    console.print("\n[bold]Optimization Suggestions:[/bold]")
    
    suggestions = []
    
    # Memory optimization
    if memory_efficiency < 70:
        suggestions.append("• Consider using smaller instances or more nodes for better memory efficiency")
    elif memory_efficiency > 90:
        suggestions.append("• ⚠️  High memory usage - enable gradient checkpointing for safety")
    
    # Compute optimization  
    if throughput < 50000 and model_size in ['7B', '13B']:
        suggestions.append("• Throughput seems low - check batch size and mixed precision settings")
    
    # Cost optimization
    if not use_spot:
        spot_savings = (on_demand_cost - compute_cost * 0.36) / on_demand_cost * 100
        suggestions.append(f"• Switch to spot instances for ~{spot_savings:.0f}% savings")
    
    if training_hours > 168:  # 1 week
        suggestions.append("• Consider checkpointing to S3 Glacier for long-term storage savings")
    
    if num_nodes == 1 and model_size in ['7B', '13B']:
        suggestions.append("• Single node training is less efficient - consider 2-4 nodes")
    
    for suggestion in suggestions:
        console.print(suggestion)
    
    # Time breakdown
    console.print("\n[bold]Timeline:[/bold]")
    console.print(f"• Start: Now")
    console.print(f"• First checkpoint: ~{training_hours * 0.1:.1f} hours")
    console.print(f"• 50% complete: ~{training_hours * 0.5:.1f} hours ({training_hours * 0.5 / 24:.1f} days)")
    console.print(f"• Completion: ~{training_hours:.1f} hours ({training_hours / 24:.1f} days)")
    
    # Budget alerts
    console.print("\n[bold]Budget Alerts:[/bold]")
    thresholds = [100, 500, 1000, 5000, 10000]
    for threshold in thresholds:
        if total_cost > threshold * 0.8 and total_cost < threshold * 1.2:
            console.print(f"⚠️  Cost is close to ${threshold} threshold")
            break

def generate_cost_plot(compute_cost: float, storage_cost: float, 
                      network_cost: float, on_demand_cost: float,
                      output_file: str):
    """Generate cost breakdown visualization"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Pie chart of cost breakdown
    costs = [compute_cost, storage_cost, network_cost]
    labels = ['Compute', 'Storage', 'Network']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Remove zero costs
    non_zero = [(c, l, col) for c, l, col in zip(costs, labels, colors) if c > 0]
    if non_zero:
        costs, labels, colors = zip(*non_zero)
    
    ax1.pie(costs, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Cost Breakdown')
    
    # Bar chart comparing spot vs on-demand
    total_cost = sum(costs)
    x = ['Spot/Preemptible', 'On-Demand']
    y = [total_cost, on_demand_cost]
    colors = ['#2ECC71', '#E74C3C']
    
    bars = ax2.bar(x, y, color=colors)
    
    # Add value labels on bars
    for bar, value in zip(bars, y):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${value:,.0f}',
                ha='center', va='bottom')
    
    ax2.set_ylabel('Cost (USD)')
    ax2.set_title('Spot vs On-Demand Pricing')
    
    # Add savings annotation
    savings = on_demand_cost - total_cost
    savings_pct = savings / on_demand_cost * 100
    ax2.text(0.5, max(y) * 0.9, f'Savings: ${savings:,.0f} ({savings_pct:.1f}%)',
            transform=ax2.transAxes, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('ML Training Cost Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()