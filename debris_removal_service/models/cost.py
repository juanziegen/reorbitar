"""
Cost calculation data models for mission planning.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PropellantMass:
    """Propellant mass requirements for a mission or maneuver."""
    fuel_kg: float
    oxidizer_kg: float
    total_kg: float
    
    def __post_init__(self):
        """Validate propellant mass data."""
        if any(mass < 0 for mass in [self.fuel_kg, self.oxidizer_kg, self.total_kg]):
            raise ValueError("Propellant masses cannot be negative")
        
        # Verify total is sum of components (with small tolerance for floating point)
        calculated_total = self.fuel_kg + self.oxidizer_kg
        if abs(calculated_total - self.total_kg) > 0.001:
            raise ValueError("Total mass must equal sum of fuel and oxidizer masses")
    
    def get_mass_ratio(self) -> float:
        """Get oxidizer to fuel mass ratio."""
        if self.fuel_kg == 0:
            return float('inf')
        return self.oxidizer_kg / self.fuel_kg
    
    def get_volume_estimate(self, fuel_density: float = 0.789, oxidizer_density: float = 1.45) -> tuple[float, float]:
        """Estimate fuel and oxidizer volumes in liters."""
        fuel_volume_l = self.fuel_kg / fuel_density
        oxidizer_volume_l = self.oxidizer_kg / oxidizer_density
        return fuel_volume_l, oxidizer_volume_l


@dataclass
class DetailedCost:
    """Detailed cost breakdown for mission components."""
    propellant_cost: float  # USD
    operational_cost: float  # USD
    processing_cost: float  # USD
    storage_cost: float  # USD
    total_cost: float  # USD
    cost_breakdown: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Validate cost data and create breakdown if not provided."""
        if any(cost < 0 for cost in [self.propellant_cost, self.operational_cost, 
                                   self.processing_cost, self.storage_cost, self.total_cost]):
            raise ValueError("Costs cannot be negative")
        
        # Verify total matches sum of components
        calculated_total = (self.propellant_cost + self.operational_cost + 
                          self.processing_cost + self.storage_cost)
        if abs(calculated_total - self.total_cost) > 0.01:
            raise ValueError("Total cost must equal sum of component costs")
        
        # Create breakdown if not provided
        if self.cost_breakdown is None:
            self.cost_breakdown = {
                'propellant': self.propellant_cost,
                'operational': self.operational_cost,
                'processing': self.processing_cost,
                'storage': self.storage_cost
            }
    
    def get_cost_percentages(self) -> Dict[str, float]:
        """Get cost breakdown as percentages of total."""
        if self.total_cost == 0:
            return {key: 0.0 for key in self.cost_breakdown.keys()}
        
        return {
            key: (value / self.total_cost) * 100 
            for key, value in self.cost_breakdown.items()
        }
    
    def get_largest_cost_component(self) -> tuple[str, float]:
        """Get the largest cost component and its value."""
        if not self.cost_breakdown:
            return "unknown", 0.0
        
        max_component = max(self.cost_breakdown.items(), key=lambda x: x[1])
        return max_component


@dataclass
class MissionCost:
    """High-level mission cost summary."""
    collection_cost: float  # USD - cost to collect satellites
    processing_cost: float  # USD - cost to process materials
    storage_cost: float  # USD - cost to store materials
    operational_overhead: float  # USD - operational overhead
    total_cost: float  # USD
    cost_per_satellite: float  # USD per satellite collected
    propellant_mass: Optional[PropellantMass] = None
    detailed_breakdown: Optional[DetailedCost] = None
    
    def __post_init__(self):
        """Validate mission cost data."""
        if any(cost < 0 for cost in [self.collection_cost, self.processing_cost, 
                                   self.storage_cost, self.operational_overhead, 
                                   self.total_cost, self.cost_per_satellite]):
            raise ValueError("Costs cannot be negative")
        
        # Verify total matches sum of components
        calculated_total = (self.collection_cost + self.processing_cost + 
                          self.storage_cost + self.operational_overhead)
        if abs(calculated_total - self.total_cost) > 0.01:
            raise ValueError("Total cost must equal sum of component costs")
    
    def get_cost_summary(self) -> Dict[str, float]:
        """Get a summary of mission costs."""
        return {
            'collection_cost_usd': self.collection_cost,
            'processing_cost_usd': self.processing_cost,
            'storage_cost_usd': self.storage_cost,
            'operational_overhead_usd': self.operational_overhead,
            'total_cost_usd': self.total_cost,
            'cost_per_satellite_usd': self.cost_per_satellite
        }
    
    def get_cost_efficiency_metrics(self, satellites_collected: int, total_delta_v: float) -> Dict[str, float]:
        """Calculate various cost efficiency metrics."""
        metrics = {}
        
        if satellites_collected > 0:
            metrics['cost_per_satellite'] = self.total_cost / satellites_collected
        
        if total_delta_v > 0:
            metrics['cost_per_delta_v_ms'] = self.total_cost / total_delta_v
        
        if self.propellant_mass and self.propellant_mass.total_kg > 0:
            metrics['cost_per_kg_propellant'] = self.total_cost / self.propellant_mass.total_kg
        
        return metrics