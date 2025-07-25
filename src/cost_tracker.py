"""
Cost tracking and optimization for distributed training
Monitors spot instance pricing and provides cost-aware scheduling
"""

import boto3
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import requests
from collections import deque
import os


class CostTracker:
    """
    Tracks training costs with features:
    - Real-time spot instance pricing
    - Cost prediction and budgeting
    - Multi-cloud cost comparison
    - Automatic cost optimization recommendations
    """
    
    # Instance pricing (on-demand prices for comparison)
    ON_DEMAND_PRICES = {
        # AWS instances
        "p4d.24xlarge": 32.77,  # 8x A100 40GB
        "p4de.24xlarge": 40.96,  # 8x A100 80GB
        "p3.16xlarge": 24.48,   # 8x V100
        "p3dn.24xlarge": 31.22,  # 8x V100 32GB
        "g5.48xlarge": 16.29,    # 8x A10G
        "g4dn.12xlarge": 3.91,   # 4x T4
        
        # GCP instances (approximate USD)
        "a2-highgpu-8g": 29.32,  # 8x A100 40GB
        "a2-ultragpu-8g": 43.98, # 8x A100 80GB
        "n1-highmem-96-v100x8": 22.14,  # 8x V100
        "a2-megagpu-16g": 58.64,  # 16x A100 40GB
    }
    
    def __init__(
        self,
        instance_type: str = "p4d.24xlarge",
        use_spot: bool = True,
        region: str = "us-east-1",
        availability_zones: Optional[List[str]] = None,
        cost_threshold: Optional[float] = None,
        update_interval: int = 300,  # 5 minutes
    ):
        self.instance_type = instance_type
        self.use_spot = use_spot
        self.region = region
        self.availability_zones = availability_zones or ["us-east-1a", "us-east-1b", "us-east-1c"]
        self.cost_threshold = cost_threshold
        self.update_interval = update_interval
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Cost tracking
        self.start_time = time.time()
        self.accumulated_cost = 0.0
        self.cost_history = deque(maxlen=1000)  # Keep last 1000 price points
        self.current_spot_price = 0.0
        
        # Initialize AWS client
        self.ec2_client = boto3.client("ec2", region_name=self.region)
        
        # Start price monitoring thread
        self.stop_event = threading.Event()
        self.price_thread = threading.Thread(target=self._monitor_prices, daemon=True)
        self.price_thread.start()
        
        # Get initial price
        self._update_spot_price()
        
        self.logger.info(f"CostTracker initialized for {instance_type} in {region}")
        
    def _monitor_prices(self):
        """Background thread to monitor spot prices"""
        while not self.stop_event.is_set():
            try:
                self._update_spot_price()
                self.stop_event.wait(self.update_interval)
            except Exception as e:
                self.logger.error(f"Error monitoring prices: {e}")
                self.stop_event.wait(self.update_interval)
                
    def _update_spot_price(self):
        """Fetch current spot price from AWS"""
        if not self.use_spot:
            self.current_spot_price = self.ON_DEMAND_PRICES.get(self.instance_type, 0.0)
            return
            
        try:
            # Get spot price history
            response = self.ec2_client.describe_spot_price_history(
                InstanceTypes=[self._get_ec2_instance_type()],
                ProductDescriptions=["Linux/UNIX"],
                AvailabilityZones=self.availability_zones,
                StartTime=datetime.utcnow() - timedelta(minutes=5),
                MaxResults=20,
            )
            
            if response["SpotPriceHistory"]:
                # Get the most recent price
                prices = [float(p["SpotPrice"]) for p in response["SpotPriceHistory"]]
                self.current_spot_price = min(prices)  # Use lowest available price
                
                # Record price point
                self.cost_history.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "price": self.current_spot_price,
                    "instance_type": self.instance_type,
                })
                
                # Check if price exceeds threshold
                if self.cost_threshold and self.current_spot_price > self.cost_threshold:
                    self.logger.warning(
                        f"Spot price ${self.current_spot_price:.2f} exceeds threshold "
                        f"${self.cost_threshold:.2f}"
                    )
                    
        except Exception as e:
            self.logger.error(f"Failed to get spot price: {e}")
            # Fall back to on-demand price
            self.current_spot_price = self.ON_DEMAND_PRICES.get(self.instance_type, 0.0)
            
    def _get_ec2_instance_type(self) -> str:
        """Convert instance type to EC2 format"""
        # Handle special cases where names differ
        instance_map = {
            "p4de.24xlarge": "p4de.24xlarge",
            # Add more mappings as needed
        }
        return instance_map.get(self.instance_type, self.instance_type)
        
    def get_current_price(self) -> float:
        """Get current price per hour"""
        return self.current_spot_price
        
    def get_accumulated_cost(self) -> float:
        """Calculate total accumulated cost"""
        elapsed_hours = (time.time() - self.start_time) / 3600
        # Simple calculation - in production, integrate over price history
        self.accumulated_cost = elapsed_hours * self.current_spot_price
        return self.accumulated_cost
        
    def get_total_cost(self, duration_seconds: float) -> float:
        """Calculate total cost for given duration"""
        duration_hours = duration_seconds / 3600
        return duration_hours * self.current_spot_price
        
    def get_cost_projection(self, remaining_steps: int, seconds_per_step: float) -> Dict[str, float]:
        """Project remaining training cost"""
        remaining_seconds = remaining_steps * seconds_per_step
        remaining_hours = remaining_seconds / 3600
        
        # Calculate projections
        current_rate_cost = remaining_hours * self.current_spot_price
        
        # Calculate average price from history
        if self.cost_history:
            avg_price = sum(h["price"] for h in self.cost_history) / len(self.cost_history)
            avg_rate_cost = remaining_hours * avg_price
        else:
            avg_rate_cost = current_rate_cost
            
        # Calculate on-demand cost for comparison
        on_demand_price = self.ON_DEMAND_PRICES.get(self.instance_type, self.current_spot_price)
        on_demand_cost = remaining_hours * on_demand_price
        
        return {
            "current_rate_cost": current_rate_cost,
            "average_rate_cost": avg_rate_cost,
            "on_demand_cost": on_demand_cost,
            "potential_savings": on_demand_cost - avg_rate_cost,
            "savings_percentage": ((on_demand_cost - avg_rate_cost) / on_demand_cost * 100) if on_demand_cost > 0 else 0,
            "estimated_completion_time": datetime.utcnow() + timedelta(seconds=remaining_seconds),
        }
        
    def get_cost_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        # Check if spot prices are high
        on_demand_price = self.ON_DEMAND_PRICES.get(self.instance_type, float('inf'))
        spot_discount = (on_demand_price - self.current_spot_price) / on_demand_price * 100
        
        if spot_discount < 30:  # Less than 30% discount
            recommendations.append({
                "type": "high_spot_price",
                "severity": "warning",
                "message": f"Spot discount is only {spot_discount:.1f}%. Consider waiting or switching regions.",
                "action": "wait_or_relocate",
            })
            
        # Check price volatility
        if len(self.cost_history) > 10:
            recent_prices = [h["price"] for h in list(self.cost_history)[-10:]]
            price_std = self._calculate_std(recent_prices)
            avg_price = sum(recent_prices) / len(recent_prices)
            volatility = (price_std / avg_price * 100) if avg_price > 0 else 0
            
            if volatility > 20:  # High volatility
                recommendations.append({
                    "type": "high_volatility",
                    "severity": "info",
                    "message": f"Price volatility is {volatility:.1f}%. Consider using multiple AZs.",
                    "action": "diversify_availability_zones",
                })
                
        # Suggest alternative instance types
        alternatives = self._get_alternative_instances()
        if alternatives:
            best_alternative = alternatives[0]
            if best_alternative["savings_percent"] > 20:
                recommendations.append({
                    "type": "better_instance_available",
                    "severity": "info",
                    "message": f"Instance {best_alternative['instance_type']} could save {best_alternative['savings_percent']:.1f}%",
                    "action": "consider_instance_change",
                    "details": best_alternative,
                })
                
        # Time-based recommendations
        current_hour = datetime.utcnow().hour
        if 9 <= current_hour <= 17:  # Business hours UTC
            recommendations.append({
                "type": "peak_hours",
                "severity": "info",
                "message": "Training during business hours. Consider scheduling for off-peak times.",
                "action": "schedule_off_peak",
            })
            
        return recommendations
        
    def _get_alternative_instances(self) -> List[Dict[str, Any]]:
        """Find alternative instance types with better pricing"""
        alternatives = []
        
        # Define instance families with similar capabilities
        instance_families = {
            "p4d.24xlarge": ["p4de.24xlarge", "p3dn.24xlarge", "g5.48xlarge"],
            "p3.16xlarge": ["p3dn.24xlarge", "g5.48xlarge"],
            # Add more mappings
        }
        
        current_family = instance_families.get(self.instance_type, [])
        
        for alt_instance in current_family:
            try:
                # Get spot price for alternative
                response = self.ec2_client.describe_spot_price_history(
                    InstanceTypes=[alt_instance],
                    ProductDescriptions=["Linux/UNIX"],
                    AvailabilityZones=self.availability_zones,
                    StartTime=datetime.utcnow() - timedelta(minutes=5),
                    MaxResults=1,
                )
                
                if response["SpotPriceHistory"]:
                    alt_price = float(response["SpotPriceHistory"][0]["SpotPrice"])
                    savings = self.current_spot_price - alt_price
                    savings_percent = (savings / self.current_spot_price * 100) if self.current_spot_price > 0 else 0
                    
                    if savings > 0:
                        alternatives.append({
                            "instance_type": alt_instance,
                            "spot_price": alt_price,
                            "savings": savings,
                            "savings_percent": savings_percent,
                        })
                        
            except Exception as e:
                self.logger.debug(f"Could not get price for {alt_instance}: {e}")
                
        # Sort by savings
        alternatives.sort(key=lambda x: x["savings_percent"], reverse=True)
        
        return alternatives
        
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
        
    def get_cost_report(self) -> Dict[str, Any]:
        """Generate comprehensive cost report"""
        current_cost = self.get_accumulated_cost()
        on_demand_price = self.ON_DEMAND_PRICES.get(self.instance_type, self.current_spot_price)
        
        # Calculate what on-demand would have cost
        elapsed_hours = (time.time() - self.start_time) / 3600
        on_demand_cost = elapsed_hours * on_demand_price
        
        report = {
            "instance_type": self.instance_type,
            "region": self.region,
            "use_spot": self.use_spot,
            "current_spot_price": self.current_spot_price,
            "on_demand_price": on_demand_price,
            "spot_discount_percent": ((on_demand_price - self.current_spot_price) / on_demand_price * 100) if on_demand_price > 0 else 0,
            "accumulated_cost": current_cost,
            "on_demand_equivalent_cost": on_demand_cost,
            "total_savings": on_demand_cost - current_cost,
            "savings_percent": ((on_demand_cost - current_cost) / on_demand_cost * 100) if on_demand_cost > 0 else 0,
            "runtime_hours": elapsed_hours,
            "recommendations": self.get_cost_optimization_recommendations(),
        }
        
        # Add price history stats
        if self.cost_history:
            prices = [h["price"] for h in self.cost_history]
            report["price_stats"] = {
                "min": min(prices),
                "max": max(prices),
                "avg": sum(prices) / len(prices),
                "current": self.current_spot_price,
                "volatility": self._calculate_std(prices) / (sum(prices) / len(prices)) * 100 if prices else 0,
            }
            
        return report
        
    def export_cost_data(self, filepath: str):
        """Export cost data for analysis"""
        data = {
            "summary": self.get_cost_report(),
            "price_history": list(self.cost_history),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
            
        self.logger.info(f"Exported cost data to {filepath}")
        
    def should_pause_training(self) -> bool:
        """Determine if training should pause due to high costs"""
        if not self.cost_threshold:
            return False
            
        # Check if current price exceeds threshold
        if self.current_spot_price > self.cost_threshold:
            return True
            
        # Check if price is trending up rapidly
        if len(self.cost_history) >= 5:
            recent_prices = [h["price"] for h in list(self.cost_history)[-5:]]
            price_increase = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100
            
            if price_increase > 50:  # 50% increase in last 5 checks
                self.logger.warning(f"Rapid price increase detected: {price_increase:.1f}%")
                return True
                
        return False
        
    def get_multi_cloud_comparison(self) -> Dict[str, Dict[str, float]]:
        """Compare costs across multiple cloud providers"""
        comparison = {}
        
        # AWS pricing (already have this)
        comparison["aws"] = {
            "instance_type": self.instance_type,
            "spot_price": self.current_spot_price,
            "on_demand_price": self.ON_DEMAND_PRICES.get(self.instance_type, 0.0),
            "region": self.region,
        }
        
        # GCP equivalent (simplified - in production, use GCP API)
        gcp_equivalents = {
            "p4d.24xlarge": "a2-highgpu-8g",
            "p3.16xlarge": "n1-highmem-96-v100x8",
        }
        
        if self.instance_type in gcp_equivalents:
            gcp_instance = gcp_equivalents[self.instance_type]
            comparison["gcp"] = {
                "instance_type": gcp_instance,
                "preemptible_price": self.ON_DEMAND_PRICES.get(gcp_instance, 0.0) * 0.3,  # ~70% discount
                "on_demand_price": self.ON_DEMAND_PRICES.get(gcp_instance, 0.0),
                "region": "us-central1",
            }
            
        return comparison
        
    def shutdown(self):
        """Cleanup resources"""
        self.stop_event.set()
        self.price_thread.join(timeout=5)
        
        # Save final report
        final_report_path = f"cost_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        self.export_cost_data(final_report_path)