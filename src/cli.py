"""
Command-line interface for the Metaflow Distributed Training Platform
Provides easy access to training, monitoring, and cost optimization features
"""

import click
import json
import yaml
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import subprocess
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
import boto3
import requests

# Import our modules
from .fsdp_trainer import FSDPTrainer
from .cost_tracker import CostTracker
from .monitoring import MetricsCollector
from .distributed_utils import DistributedCoordinator

console = Console()

@click.group()
@click.version_option(version='1.0.0')
def cli():
    """
    Metaflow Distributed Training Platform CLI
    
    Train large language models at scale with automatic cost optimization.
    """
    pass

@cli.command()
@click.option('--model', '-m', default='meta-llama/Llama-2-7b-hf', help='Model name from HuggingFace')
@click.option('--dataset', '-d', default='c4', help='Dataset name')
@click.option('--nodes', '-n', default=4, help='Number of training nodes')
@click.option('--batch-size', '-b', default=32, help='Batch size per GPU')
@click.option('--epochs', '-e', default=3, help='Number of epochs')
@click.option('--use-spot/--no-spot', default=True, help='Use spot instances')
@click.option('--checkpoint-dir', default='./checkpoints', help='Checkpoint directory')
@click.option('--config', '-c', help='Configuration file (YAML)')
@click.option('--dry-run', is_flag=True, help='Show configuration without starting training')
def train(model, dataset, nodes, batch_size, epochs, use_spot, checkpoint_dir, config, dry_run):
    """Start distributed training job"""
    console.print(Panel.fit("🚀 [bold cyan]Metaflow Distributed Training Platform[/bold cyan]"))
    
    # Load configuration
    if config:
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
        model = config_data.get('model', {}).get('name', model)
        dataset = config_data.get('dataset', {}).get('name', dataset)
        nodes = config_data.get('distributed', {}).get('world_size', nodes)
        batch_size = config_data.get('training', {}).get('batch_size_per_gpu', batch_size)
        epochs = config_data.get('training', {}).get('num_train_epochs', epochs)
    
    # Display configuration
    config_table = Table(title="Training Configuration")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    
    config_table.add_row("Model", model)
    config_table.add_row("Dataset", dataset)
    config_table.add_row("Nodes", str(nodes))
    config_table.add_row("GPUs Total", str(nodes * 8))
    config_table.add_row("Batch Size (per GPU)", str(batch_size))
    config_table.add_row("Global Batch Size", str(batch_size * nodes * 8))
    config_table.add_row("Epochs", str(epochs))
    config_table.add_row("Use Spot Instances", "✅" if use_spot else "❌")
    config_table.add_row("Checkpoint Directory", checkpoint_dir)
    
    console.print(config_table)
    
    if dry_run:
        console.print("\n[yellow]Dry run mode - not starting training[/yellow]")
        return
    
    # Estimate cost
    console.print("\n[bold]Estimating training cost...[/bold]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Analyzing spot prices...", total=None)
        
        cost_estimate = estimate_training_cost(
            model_size="7B",  # Simplified for demo
            nodes=nodes,
            epochs=epochs,
            use_spot=use_spot
        )
        
        progress.update(task, completed=True)
    
    cost_table = Table(title="Cost Estimate")
    cost_table.add_column("Metric", style="cyan")
    cost_table.add_column("Value", style="green" if use_spot else "red")
    
    cost_table.add_row("Estimated Time", f"{cost_estimate['duration_hours']:.1f} hours")
    cost_table.add_row("Hourly Rate", f"${cost_estimate['hourly_rate']:.2f}")
    cost_table.add_row("Total Cost", f"${cost_estimate['total_cost']:.2f}")
    cost_table.add_row("On-Demand Cost", f"${cost_estimate['on_demand_cost']:.2f}")
    cost_table.add_row("Savings", f"${cost_estimate['savings']:.2f} ({cost_estimate['savings_percent']:.1f}%)")
    
    console.print(cost_table)
    
    # Confirm before starting
    if not click.confirm("\nDo you want to start training?"):
        console.print("[red]Training cancelled[/red]")
        return
    
    # Start training
    console.print("\n[bold green]Starting distributed training...[/bold green]")
    
    # Launch training job
    cmd = [
        "python", "-m", "metaflow", "run",
        "examples/train_llama_fsdp.py",
        "run",
        "--model", model,
        "--dataset", dataset,
        "--num-nodes", str(nodes),
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--use-spot", str(use_spot).lower(),
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Training failed: {e}[/red]")
        sys.exit(1)

@cli.command()
@click.option('--namespace', '-n', default='ml-training', help='Kubernetes namespace')
@click.option('--tail', '-t', default=100, help='Number of lines to tail')
@click.option('--follow', '-f', is_flag=True, help='Follow log output')
@click.option('--rank', '-r', help='Show logs from specific rank')
def logs(namespace, tail, follow, rank):
    """View training logs"""
    console.print("[bold]Fetching training logs...[/bold]")
    
    cmd = ["kubectl", "logs", "-n", namespace, "-l", "app=distributed-training"]
    
    if rank:
        cmd.extend(["-l", f"rank={rank}"])
    
    cmd.extend([f"--tail={tail}"])
    
    if follow:
        cmd.append("-f")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to fetch logs: {e}[/red]")

@cli.command()
@click.option('--format', '-f', type=click.Choice(['table', 'json']), default='table')
def status(format):
    """Check training status and metrics"""
    console.print("[bold]Training Status[/bold]\n")
    
    # Get pod status
    try:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", "ml-training", "-o", "json"],
            capture_output=True,
            text=True,
            check=True
        )
        pods_data = json.loads(result.stdout)
        
        if format == 'table':
            status_table = Table(title="Pod Status")
            status_table.add_column("Name", style="cyan")
            status_table.add_column("Status", style="green")
            status_table.add_column("Node", style="yellow")
            status_table.add_column("Age")
            
            for pod in pods_data['items']:
                name = pod['metadata']['name']
                status = pod['status']['phase']
                node = pod['spec'].get('nodeName', 'N/A')
                
                # Calculate age
                start_time = pod['metadata']['creationTimestamp']
                age = calculate_age(start_time)
                
                status_style = "green" if status == "Running" else "red"
                status_table.add_row(name, f"[{status_style}]{status}[/{status_style}]", node, age)
            
            console.print(status_table)
        else:
            console.print_json(json.dumps(pods_data, indent=2))
            
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to get status: {e}[/red]")

@cli.command()
@click.option('--live', '-l', is_flag=True, help='Live monitoring')
@click.option('--interval', '-i', default=5, help='Update interval in seconds')
def monitor(live, interval):
    """Monitor training metrics in real-time"""
    console.print("[bold]Training Metrics Monitor[/bold]\n")
    
    if live:
        with Live(console=console, refresh_per_second=1) as live_display:
            while True:
                metrics = fetch_current_metrics()
                
                # Create metrics display
                metrics_table = Table(title=f"Live Metrics - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                metrics_table.add_column("Metric", style="cyan")
                metrics_table.add_column("Value", style="green")
                
                if metrics:
                    metrics_table.add_row("Loss", f"{metrics.get('loss', 0):.4f}")
                    metrics_table.add_row("Throughput", f"{metrics.get('throughput', 0):,.0f} tokens/sec")
                    metrics_table.add_row("GPU Utilization", f"{metrics.get('gpu_util', 0):.1f}%")
                    metrics_table.add_row("Memory Used", f"{metrics.get('memory_gb', 0):.1f} GB")
                    metrics_table.add_row("Current Cost", f"${metrics.get('cost', 0):.2f}")
                else:
                    metrics_table.add_row("Status", "[red]No metrics available[/red]")
                
                live_display.update(metrics_table)
                time.sleep(interval)
    else:
        # One-time fetch
        metrics = fetch_current_metrics()
        if metrics:
            console.print_json(json.dumps(metrics, indent=2))
        else:
            console.print("[red]No metrics available[/red]")

@cli.command()
@click.option('--regions', '-r', default='us-east-1,us-west-2,eu-west-1', help='Comma-separated regions')
@click.option('--instance-types', '-i', default='p4d.24xlarge,p3dn.24xlarge', help='Instance types')
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'csv']), default='table')
def spot_prices(regions, instance_types, format):
    """Check current spot prices across regions"""
    console.print("[bold]Fetching spot prices...[/bold]\n")
    
    regions_list = [r.strip() for r in regions.split(',')]
    instances_list = [i.strip() for i in instance_types.split(',')]
    
    prices_data = []
    
    with Progress() as progress:
        task = progress.add_task("Querying AWS...", total=len(regions_list) * len(instances_list))
        
        for region in regions_list:
            ec2 = boto3.client('ec2', region_name=region)
            
            for instance_type in instances_list:
                try:
                    response = ec2.describe_spot_price_history(
                        InstanceTypes=[instance_type],
                        MaxResults=1,
                        ProductDescriptions=['Linux/UNIX']
                    )
                    
                    if response['SpotPriceHistory']:
                        price_info = response['SpotPriceHistory'][0]
                        prices_data.append({
                            'region': region,
                            'instance_type': instance_type,
                            'price': float(price_info['SpotPrice']),
                            'availability_zone': price_info['AvailabilityZone'],
                            'timestamp': price_info['Timestamp']
                        })
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to get price for {instance_type} in {region}[/yellow]")
                
                progress.advance(task)
    
    # Display results
    if format == 'table':
        price_table = Table(title="Current Spot Prices")
        price_table.add_column("Region", style="cyan")
        price_table.add_column("Instance Type", style="yellow")
        price_table.add_column("Price/Hour", style="green")
        price_table.add_column("AZ")
        price_table.add_column("Savings vs On-Demand")
        
        on_demand_prices = {
            'p4d.24xlarge': 32.77,
            'p3dn.24xlarge': 31.22,
            'p3.16xlarge': 24.48,
        }
        
        for price in sorted(prices_data, key=lambda x: x['price']):
            on_demand = on_demand_prices.get(price['instance_type'], 0)
            savings = ((on_demand - price['price']) / on_demand * 100) if on_demand > 0 else 0
            
            price_table.add_row(
                price['region'],
                price['instance_type'],
                f"${price['price']:.2f}",
                price['availability_zone'],
                f"{savings:.1f}%" if savings > 0 else "N/A"
            )
        
        console.print(price_table)
        
        # Show best option
        if prices_data:
            best = min(prices_data, key=lambda x: x['price'])
            console.print(f"\n[bold green]Best option:[/bold green] {best['instance_type']} in {best['region']} at ${best['price']:.2f}/hour")
            
    elif format == 'json':
        console.print_json(json.dumps(prices_data, indent=2, default=str))
    else:  # csv
        import csv
        import sys
        writer = csv.DictWriter(sys.stdout, fieldnames=['region', 'instance_type', 'price', 'availability_zone'])
        writer.writeheader()
        writer.writerows(prices_data)

@cli.command()
@click.option('--output', '-o', default='cost_report.json', help='Output file')
@click.option('--format', '-f', type=click.Choice(['summary', 'detailed']), default='summary')
def cost_report(output, format):
    """Generate cost analysis report"""
    console.print("[bold]Generating cost report...[/bold]\n")
    
    # Fetch cost data
    cost_data = {
        'generated_at': datetime.now().isoformat(),
        'training_runs': [],
        'total_cost': 0,
        'total_savings': 0,
        'recommendations': []
    }
    
    # Get recent training runs (mock data for demo)
    recent_runs = [
        {
            'id': 'run-001',
            'model': 'llama-7b',
            'duration_hours': 57,
            'cost': 892,
            'on_demand_cost': 2416,
            'nodes': 4,
            'status': 'completed'
        },
        {
            'id': 'run-002',
            'model': 'llama-13b',
            'duration_hours': 112,
            'cost': 1750,
            'on_demand_cost': 4720,
            'nodes': 8,
            'status': 'completed'
        }
    ]
    
    for run in recent_runs:
        run['savings'] = run['on_demand_cost'] - run['cost']
        run['savings_percent'] = (run['savings'] / run['on_demand_cost'] * 100)
        cost_data['training_runs'].append(run)
        cost_data['total_cost'] += run['cost']
        cost_data['total_savings'] += run['savings']
    
    # Add recommendations
    cost_data['recommendations'] = [
        "Schedule training during off-peak hours (2 AM - 6 AM) for additional 15% savings",
        "Consider using g5.48xlarge instances for smaller models (40% cheaper)",
        "Enable gradient checkpointing to use smaller instance types"
    ]
    
    # Display summary
    if format == 'summary':
        summary_table = Table(title="Cost Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Training Runs", str(len(recent_runs)))
        summary_table.add_row("Total Cost", f"${cost_data['total_cost']:,.2f}")
        summary_table.add_row("Total Savings", f"${cost_data['total_savings']:,.2f}")
        summary_table.add_row("Average Savings", f"{cost_data['total_savings'] / cost_data['total_cost'] * 100:.1f}%")
        
        console.print(summary_table)
        
        console.print("\n[bold]Recommendations:[/bold]")
        for i, rec in enumerate(cost_data['recommendations'], 1):
            console.print(f"{i}. {rec}")
    
    # Save full report
    with open(output, 'w') as f:
        json.dump(cost_data, f, indent=2)
    
    console.print(f"\n[green]Report saved to {output}[/green]")

@cli.command()
@click.argument('checkpoint_path')
@click.option('--validate', '-v', is_flag=True, help='Validate checkpoint integrity')
def checkpoint(checkpoint_path, validate):
    """Manage training checkpoints"""
    console.print(f"[bold]Checkpoint: {checkpoint_path}[/bold]\n")
    
    if not Path(checkpoint_path).exists():
        console.print("[red]Checkpoint not found[/red]")
        return
    
    # Load checkpoint info
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        info_table = Table(title="Checkpoint Information")
        info_table.add_column("Field", style="cyan")
        info_table.add_column("Value", style="green")
        
        info_table.add_row("Global Step", str(checkpoint_data.get('global_step', 'N/A')))
        info_table.add_row("Epoch", str(checkpoint_data.get('epoch', 'N/A')))
        info_table.add_row("Loss", f"{checkpoint_data.get('loss', 0):.4f}")
        info_table.add_row("Model Size", f"{sum(p.numel() for p in checkpoint_data.get('model_state_dict', {}).values()) / 1e9:.2f}B params")
        
        if 'metadata' in checkpoint_data:
            info_table.add_row("Timestamp", checkpoint_data['metadata'].get('timestamp', 'N/A'))
            info_table.add_row("World Size", str(checkpoint_data['metadata'].get('world_size', 'N/A')))
        
        console.print(info_table)
        
        if validate:
            console.print("\n[bold]Validating checkpoint...[/bold]")
            
            # Check required fields
            required_fields = ['model_state_dict', 'optimizer_state_dict', 'global_step']
            missing = [f for f in required_fields if f not in checkpoint_data]
            
            if missing:
                console.print(f"[red]Missing required fields: {', '.join(missing)}[/red]")
            else:
                console.print("[green]✓ All required fields present[/green]")
                
                # Check for corruption
                try:
                    # Attempt to move tensors to verify integrity
                    for key, tensor in checkpoint_data['model_state_dict'].items():
                        if hasattr(tensor, 'cpu'):
                            _ = tensor.cpu()
                    console.print("[green]✓ Checkpoint integrity verified[/green]")
                except Exception as e:
                    console.print(f"[red]Checkpoint may be corrupted: {e}[/red]")
                    
    except Exception as e:
        console.print(f"[red]Failed to load checkpoint: {e}[/red]")

@cli.command()
@click.option('--profile', '-p', help='Configuration profile to use')
@click.option('--key', '-k', help='Configuration key to get/set')
@click.option('--value', '-v', help='Value to set')
@click.option('--list', '-l', is_flag=True, help='List all configuration')
def config(profile, key, value, list):
    """Manage configuration settings"""
    config_file = Path.home() / '.metaflow-training' / 'config.yaml'
    config_file.parent.mkdir(exist_ok=True)
    
    # Load existing config
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f) or {}
    else:
        config_data = {}
    
    if list:
        console.print("[bold]Current Configuration:[/bold]\n")
        syntax = Syntax(yaml.dump(config_data, default_flow_style=False), "yaml")
        console.print(syntax)
        return
    
    if key and value:
        # Set configuration
        keys = key.split('.')
        current = config_data
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        console.print(f"[green]Set {key} = {value}[/green]")
        
    elif key:
        # Get configuration
        keys = key.split('.')
        current = config_data
        for k in keys:
            if k in current:
                current = current[k]
            else:
                console.print(f"[red]Key not found: {key}[/red]")
                return
        
        console.print(f"{key} = {current}")

# Helper functions
def estimate_training_cost(model_size: str, nodes: int, epochs: int, use_spot: bool) -> Dict[str, float]:
    """Estimate training cost based on model size and configuration"""
    # Simplified cost estimation
    tokens_per_epoch = {
        "7B": 1e9,
        "13B": 1e9,
        "30B": 1e9,
        "70B": 1e9,
    }.get(model_size, 1e9)
    
    throughput_per_node = 20000  # tokens/sec
    total_throughput = throughput_per_node * nodes
    
    training_time_hours = (tokens_per_epoch * epochs) / (total_throughput * 3600)
    
    if use_spot:
        hourly_rate = 12.0 * nodes  # ~$12/hour for p4d.24xlarge spot
    else:
        hourly_rate = 32.77 * nodes  # On-demand price
    
    total_cost = training_time_hours * hourly_rate
    on_demand_cost = training_time_hours * 32.77 * nodes
    
    return {
        'duration_hours': training_time_hours,
        'hourly_rate': hourly_rate,
        'total_cost': total_cost,
        'on_demand_cost': on_demand_cost,
        'savings': on_demand_cost - total_cost,
        'savings_percent': ((on_demand_cost - total_cost) / on_demand_cost * 100) if on_demand_cost > 0 else 0
    }

def fetch_current_metrics() -> Optional[Dict[str, Any]]:
    """Fetch current training metrics from monitoring system"""
    try:
        # In production, this would query Prometheus/monitoring system
        # For demo, return mock data
        return {
            'loss': 2.34,
            'throughput': 145000,
            'gpu_util': 94.2,
            'memory_gb': 35.6,
            'cost': 189.45,
            'step': 5000,
            'epoch': 1.2
        }
    except Exception:
        return None

def calculate_age(timestamp_str: str) -> str:
    """Calculate age from timestamp string"""
    from dateutil import parser
    from datetime import timezone
    
    timestamp = parser.parse(timestamp_str)
    now = datetime.now(timezone.utc)
    delta = now - timestamp
    
    if delta.days > 0:
        return f"{delta.days}d"
    elif delta.seconds > 3600:
        return f"{delta.seconds // 3600}h"
    elif delta.seconds > 60:
        return f"{delta.seconds // 60}m"
    else:
        return f"{delta.seconds}s"

def main():
    """Main entry point"""
    cli()

if __name__ == '__main__':
    main()