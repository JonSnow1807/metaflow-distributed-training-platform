#!/usr/bin/env python3
"""
Check current spot prices across regions and instance types
Helps find the best deals for distributed training
"""

import boto3
import click
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import concurrent.futures

console = Console()

# Default instance types for ML training
DEFAULT_INSTANCES = [
    "p4d.24xlarge",   # 8x A100 40GB
    "p4de.24xlarge",  # 8x A100 80GB
    "p3dn.24xlarge",  # 8x V100 32GB
    "p3.16xlarge",    # 8x V100 16GB
    "g5.48xlarge",    # 8x A10G
    "g4dn.12xlarge",  # 4x T4
]

# On-demand prices for comparison
ON_DEMAND_PRICES = {
    "p4d.24xlarge": 32.77,
    "p4de.24xlarge": 40.96,
    "p3dn.24xlarge": 31.22,
    "p3.16xlarge": 24.48,
    "g5.48xlarge": 16.29,
    "g4dn.12xlarge": 3.91,
}

@click.command()
@click.option('--regions', '-r', 
              default='us-east-1,us-west-2,eu-west-1,ap-southeast-1',
              help='Comma-separated list of AWS regions')
@click.option('--instance-types', '-i',
              default=','.join(DEFAULT_INSTANCES),
              help='Comma-separated list of instance types')
@click.option('--output', '-o', 
              type=click.Choice(['table', 'json', 'csv']),
              default='table',
              help='Output format')
@click.option('--max-price', '-m',
              type=float,
              help='Maximum acceptable price per hour')
@click.option('--min-savings', '-s',
              type=float,
              default=50.0,
              help='Minimum savings percentage vs on-demand')
@click.option('--availability-zones', '-z',
              help='Check specific availability zones')
@click.option('--sort-by', 
              type=click.Choice(['price', 'savings', 'region']),
              default='price',
              help='Sort results by')
def main(regions, instance_types, output, max_price, min_savings, availability_zones, sort_by):
    """
    Check spot prices across AWS regions and find the best deals.
    
    Examples:
        # Check default instances in major regions
        python check_spot_prices.py
        
        # Check specific instance in all US regions
        python check_spot_prices.py -r us-east-1,us-west-1,us-west-2 -i p4d.24xlarge
        
        # Find instances with >70% savings
        python check_spot_prices.py -s 70
        
        # Export results as JSON
        python check_spot_prices.py -o json > prices.json
    """
    
    regions_list = [r.strip() for r in regions.split(',')]
    instances_list = [i.strip() for i in instance_types.split(',')]
    az_list = [z.strip() for z in availability_zones.split(',')] if availability_zones else None
    
    console.print(f"[bold cyan]Checking spot prices across {len(regions_list)} regions...[/bold cyan]")
    
    # Collect price data
    all_prices = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Fetching prices...", total=len(regions_list))
        
        # Use ThreadPoolExecutor for parallel requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_region = {
                executor.submit(
                    fetch_region_prices, 
                    region, 
                    instances_list, 
                    az_list
                ): region 
                for region in regions_list
            }
            
            for future in concurrent.futures.as_completed(future_to_region):
                region = future_to_region[future]
                try:
                    prices = future.result()
                    all_prices.extend(prices)
                except Exception as e:
                    console.print(f"[red]Error fetching prices for {region}: {e}[/red]")
                finally:
                    progress.advance(task)
    
    # Filter results
    filtered_prices = []
    for price in all_prices:
        # Calculate savings
        on_demand = ON_DEMAND_PRICES.get(price['instance_type'], 0)
        if on_demand > 0:
            savings_amount = on_demand - price['spot_price']
            savings_percent = (savings_amount / on_demand) * 100
            price['on_demand_price'] = on_demand
            price['savings_amount'] = savings_amount
            price['savings_percent'] = savings_percent
        else:
            price['on_demand_price'] = 0
            price['savings_amount'] = 0
            price['savings_percent'] = 0
        
        # Apply filters
        if max_price and price['spot_price'] > max_price:
            continue
        if min_savings and price['savings_percent'] < min_savings:
            continue
            
        filtered_prices.append(price)
    
    # Sort results
    if sort_by == 'price':
        filtered_prices.sort(key=lambda x: x['spot_price'])
    elif sort_by == 'savings':
        filtered_prices.sort(key=lambda x: x['savings_percent'], reverse=True)
    else:  # region
        filtered_prices.sort(key=lambda x: (x['region'], x['spot_price']))
    
    # Display results
    if output == 'table':
        display_table(filtered_prices)
    elif output == 'json':
        print(json.dumps(filtered_prices, indent=2, default=str))
    else:  # csv
        display_csv(filtered_prices)
    
    # Show summary
    if output == 'table' and filtered_prices:
        console.print("\n[bold green]Summary:[/bold green]")
        best_price = min(filtered_prices, key=lambda x: x['spot_price'])
        best_savings = max(filtered_prices, key=lambda x: x['savings_percent'])
        
        console.print(f"🏆 Lowest price: [cyan]{best_price['instance_type']}[/cyan] in "
                     f"[yellow]{best_price['region']}[/yellow] at "
                     f"[green]${best_price['spot_price']:.2f}/hr[/green]")
        
        console.print(f"💰 Best savings: [cyan]{best_savings['instance_type']}[/cyan] in "
                     f"[yellow]{best_savings['region']}[/yellow] with "
                     f"[green]{best_savings['savings_percent']:.1f}% off[/green]")
        
        # Recommendations
        console.print("\n[bold]Recommendations:[/bold]")
        if any(p['instance_type'] == 'p4d.24xlarge' and p['savings_percent'] > 60 for p in filtered_prices):
            console.print("✅ Excellent p4d.24xlarge prices available - ideal for large model training")
        
        if any(p['savings_percent'] > 70 for p in filtered_prices):
            console.print("✅ Found instances with >70% savings - great time to train!")
        
        volatile_regions = check_price_volatility(filtered_prices)
        if volatile_regions:
            console.print(f"⚠️  High price volatility in: {', '.join(volatile_regions)}")

def fetch_region_prices(region: str, instance_types: List[str], 
                       availability_zones: List[str] = None) -> List[Dict[str, Any]]:
    """Fetch spot prices for a specific region"""
    ec2 = boto3.client('ec2', region_name=region)
    prices = []
    
    try:
        # Get spot price history
        params = {
            'InstanceTypes': instance_types,
            'ProductDescriptions': ['Linux/UNIX', 'Linux/UNIX (Amazon VPC)'],
            'StartTime': datetime.utcnow() - timedelta(hours=1),
            'EndTime': datetime.utcnow(),
            'MaxResults': 100,
        }
        
        if availability_zones:
            params['AvailabilityZones'] = availability_zones
        
        response = ec2.describe_spot_price_history(**params)
        
        # Process results
        seen = set()  # To avoid duplicates
        for item in response['SpotPriceHistory']:
            key = (item['InstanceType'], item['AvailabilityZone'])
            if key not in seen:
                seen.add(key)
                prices.append({
                    'region': region,
                    'availability_zone': item['AvailabilityZone'],
                    'instance_type': item['InstanceType'],
                    'spot_price': float(item['SpotPrice']),
                    'timestamp': item['Timestamp'],
                    'product_description': item['ProductDescription'],
                })
        
    except Exception as e:
        console.print(f"[yellow]Warning: Error fetching prices for {region}: {e}[/yellow]")
    
    return prices

def display_table(prices: List[Dict[str, Any]]):
    """Display prices in a formatted table"""
    if not prices:
        console.print("[red]No prices found matching criteria[/red]")
        return
    
    table = Table(title="Spot Instance Prices", show_lines=True)
    table.add_column("Region", style="cyan", no_wrap=True)
    table.add_column("AZ", style="dim")
    table.add_column("Instance Type", style="yellow")
    table.add_column("Spot Price", style="green", justify="right")
    table.add_column("On-Demand", style="dim", justify="right")
    table.add_column("Savings", style="bold green", justify="right")
    table.add_column("Updated", style="dim")
    
    for price in prices[:50]:  # Limit to 50 rows for readability
        # Format timestamp
        timestamp = price['timestamp']
        if isinstance(timestamp, str):
            time_str = timestamp
        else:
            time_str = timestamp.strftime("%H:%M")
        
        # Color code savings
        savings_pct = price['savings_percent']
        if savings_pct >= 70:
            savings_style = "bold green"
        elif savings_pct >= 50:
            savings_style = "green"
        else:
            savings_style = "yellow"
        
        table.add_row(
            price['region'],
            price['availability_zone'].split('-')[-1],  # Just show the AZ letter
            price['instance_type'],
            f"${price['spot_price']:.3f}",
            f"${price['on_demand_price']:.2f}" if price['on_demand_price'] > 0 else "N/A",
            f"[{savings_style}]{savings_pct:.1f}%[/{savings_style}]" if savings_pct > 0 else "N/A",
            time_str,
        )
    
    console.print(table)
    
    if len(prices) > 50:
        console.print(f"\n[dim]Showing 50 of {len(prices)} results[/dim]")

def display_csv(prices: List[Dict[str, Any]]):
    """Display prices in CSV format"""
    if not prices:
        return
    
    # Header
    print("region,availability_zone,instance_type,spot_price,on_demand_price,savings_percent,timestamp")
    
    # Data rows
    for price in prices:
        print(f"{price['region']},"
              f"{price['availability_zone']},"
              f"{price['instance_type']},"
              f"{price['spot_price']:.4f},"
              f"{price['on_demand_price']:.2f},"
              f"{price['savings_percent']:.2f},"
              f"{price['timestamp']}")

def check_price_volatility(prices: List[Dict[str, Any]]) -> List[str]:
    """Check for regions with high price volatility"""
    volatile_regions = []
    
    # Group by region and instance type
    from collections import defaultdict
    region_prices = defaultdict(list)
    
    for price in prices:
        key = f"{price['region']}-{price['instance_type']}"
        region_prices[key].append(price['spot_price'])
    
    # Check volatility
    for key, price_list in region_prices.items():
        if len(price_list) > 1:
            min_price = min(price_list)
            max_price = max(price_list)
            volatility = (max_price - min_price) / min_price * 100
            
            if volatility > 20:  # 20% variation
                region = key.split('-')[0]
                if region not in volatile_regions:
                    volatile_regions.append(region)
    
    return volatile_regions

if __name__ == "__main__":
    main()