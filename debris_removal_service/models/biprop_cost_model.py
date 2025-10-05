"""
Biprop cost calculation model for satellite debris removal missions.

This module implements accurate cost calculations based on the rocket equation
and biprop propulsion system parameters, using the established $1.27 per m/s
delta-v cost model.
"""

import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .cost import PropellantMass, DetailedCost, MissionCost


@dataclass
class BipropParameters:
    """Biprop propulsion system parameters."""
    isp: float = 320.0  # Specific impulse in seconds
    dry_mass: float = 14.35  # Dry mass in kg
    fuel_cost_per_kg: float = 200.0  # USD per kg of fuel
    oxidizer_cost_per_kg: float = 320.0  # USD per kg of oxidizer
    o_f_ratio: float = 1.9  # Oxidizer to fuel mass ratio
    fuel_density: float = 0.789  # kg/L (RP-1 kerosene)
    oxidizer_density: float = 1.45  # kg/L (LOX)
    operational_overhead_factor: float = 0.15  # 15% operational overhead
    mission_complexity_factor: float = 1.0  # Mission complexity multiplier


class BipropCostModel:
    """
    Accurate biprop cost calculation model using rocket equation.
    
    Implements the $1.27 per m/s delta-v cost formula with detailed
    propellant mass calculations and operational overhead factors.
    """
    
    # Standard gravitational acceleration (m/s²)
    G0 = 9.80665
    
    def __init__(self, parameters: Optional[BipropParameters] = None):
        """Initialize the cost model with biprop parameters."""
        self.params = parameters or BipropParameters()
        
        # Validate parameters
        if self.params.isp <= 0:
            raise ValueError("Specific impulse must be positive")
        if self.params.dry_mass <= 0:
            raise ValueError("Dry mass must be positive")
        if self.params.o_f_ratio <= 0:
            raise ValueError("O/F ratio must be positive")
    
    def calculate_delta_v_cost(self, delta_v_ms: float) -> float:
        """
        Calculate total cost for a given delta-v using the $1.27/m/s formula.
        
        Args:
            delta_v_ms: Delta-v requirement in m/s
            
        Returns:
            Total cost in USD
        """
        if delta_v_ms < 0:
            raise ValueError("Delta-v cannot be negative")
        
        # Base cost using the established formula
        base_cost = delta_v_ms * 1.27
        
        # Apply mission complexity factor
        total_cost = base_cost * self.params.mission_complexity_factor
        
        return total_cost
    
    def get_propellant_requirements(self, delta_v_ms: float) -> PropellantMass:
        """
        Calculate propellant mass requirements using the rocket equation.
        
        Uses the Tsiolkovsky rocket equation:
        Δv = Isp * g0 * ln(m_initial / m_final)
        
        Args:
            delta_v_ms: Delta-v requirement in m/s
            
        Returns:
            PropellantMass object with fuel, oxidizer, and total masses
        """
        if delta_v_ms < 0:
            raise ValueError("Delta-v cannot be negative")
        
        if delta_v_ms == 0:
            return PropellantMass(fuel_kg=0.0, oxidizer_kg=0.0, total_kg=0.0)
        
        # Calculate exhaust velocity
        ve = self.params.isp * self.G0
        
        # Calculate mass ratio using rocket equation
        mass_ratio = math.exp(delta_v_ms / ve)
        
        # Calculate initial mass (wet mass)
        initial_mass = self.params.dry_mass * mass_ratio
        
        # Calculate propellant mass
        propellant_mass = initial_mass - self.params.dry_mass
        
        # Split propellant into fuel and oxidizer based on O/F ratio
        # Total propellant = fuel + oxidizer
        # oxidizer = fuel * o_f_ratio
        # Therefore: propellant_mass = fuel + fuel * o_f_ratio = fuel * (1 + o_f_ratio)
        fuel_mass = propellant_mass / (1 + self.params.o_f_ratio)
        oxidizer_mass = fuel_mass * self.params.o_f_ratio
        
        return PropellantMass(
            fuel_kg=fuel_mass,
            oxidizer_kg=oxidizer_mass,
            total_kg=propellant_mass
        )
    
    def calculate_propellant_cost(self, propellant_mass: PropellantMass) -> float:
        """
        Calculate the cost of propellant based on mass requirements.
        
        Args:
            propellant_mass: PropellantMass object
            
        Returns:
            Propellant cost in USD
        """
        fuel_cost = propellant_mass.fuel_kg * self.params.fuel_cost_per_kg
        oxidizer_cost = propellant_mass.oxidizer_kg * self.params.oxidizer_cost_per_kg
        
        return fuel_cost + oxidizer_cost
    
    def calculate_operational_overhead(self, base_cost: float) -> float:
        """
        Calculate operational overhead based on base mission cost.
        
        Args:
            base_cost: Base mission cost in USD
            
        Returns:
            Operational overhead cost in USD
        """
        return base_cost * self.params.operational_overhead_factor
    
    def calculate_detailed_cost(self, delta_v_ms: float, 
                              processing_cost: float = 0.0,
                              storage_cost: float = 0.0) -> DetailedCost:
        """
        Calculate detailed cost breakdown for a mission.
        
        Args:
            delta_v_ms: Delta-v requirement in m/s
            processing_cost: Additional processing cost in USD
            storage_cost: Additional storage cost in USD
            
        Returns:
            DetailedCost object with complete breakdown
        """
        # Get propellant requirements and cost
        propellant_mass = self.get_propellant_requirements(delta_v_ms)
        propellant_cost = self.calculate_propellant_cost(propellant_mass)
        
        # Calculate operational overhead
        base_cost = propellant_cost + processing_cost + storage_cost
        operational_cost = self.calculate_operational_overhead(base_cost)
        
        # Calculate total cost
        total_cost = propellant_cost + operational_cost + processing_cost + storage_cost
        
        return DetailedCost(
            propellant_cost=propellant_cost,
            operational_cost=operational_cost,
            processing_cost=processing_cost,
            storage_cost=storage_cost,
            total_cost=total_cost
        )
    
    def calculate_mission_cost(self, delta_v_ms: float, 
                             satellites_collected: int,
                             processing_cost: float = 0.0,
                             storage_cost: float = 0.0) -> MissionCost:
        """
        Calculate complete mission cost summary.
        
        Args:
            delta_v_ms: Total delta-v requirement in m/s
            satellites_collected: Number of satellites collected
            processing_cost: Additional processing cost in USD
            storage_cost: Additional storage cost in USD
            
        Returns:
            MissionCost object with complete mission summary
        """
        if satellites_collected <= 0:
            raise ValueError("Number of satellites collected must be positive")
        
        # Get detailed cost breakdown
        detailed_cost = self.calculate_detailed_cost(delta_v_ms, processing_cost, storage_cost)
        
        # Get propellant mass
        propellant_mass = self.get_propellant_requirements(delta_v_ms)
        
        # Collection cost is propellant cost only (operational overhead is separate)
        collection_cost = detailed_cost.propellant_cost
        
        # Cost per satellite
        cost_per_satellite = detailed_cost.total_cost / satellites_collected
        
        return MissionCost(
            collection_cost=collection_cost,
            processing_cost=processing_cost,
            storage_cost=storage_cost,
            operational_overhead=detailed_cost.operational_cost,
            total_cost=detailed_cost.total_cost,
            cost_per_satellite=cost_per_satellite,
            propellant_mass=propellant_mass,
            detailed_breakdown=detailed_cost
        )
    
    def validate_against_csv_data(self, delta_v_ms: float, expected_cost: float, 
                                tolerance: float = 0.01) -> bool:
        """
        Validate cost calculation against CSV reference data.
        
        Args:
            delta_v_ms: Delta-v in m/s
            expected_cost: Expected cost from CSV data
            tolerance: Tolerance for cost comparison (default 1%)
            
        Returns:
            True if calculation matches within tolerance
        """
        calculated_cost = self.calculate_delta_v_cost(delta_v_ms)
        relative_error = abs(calculated_cost - expected_cost) / expected_cost
        
        return relative_error <= tolerance
    
    def get_cost_optimization_suggestions(self, delta_v_ms: float) -> Dict[str, str]:
        """
        Provide cost optimization suggestions for different mission profiles.
        
        Args:
            delta_v_ms: Delta-v requirement in m/s
            
        Returns:
            Dictionary of optimization suggestions
        """
        suggestions = {}
        
        propellant_mass = self.get_propellant_requirements(delta_v_ms)
        cost = self.calculate_delta_v_cost(delta_v_ms)
        
        # High delta-v missions
        if delta_v_ms > 1000:
            suggestions['high_delta_v'] = (
                "Consider multi-stage approach or refueling to reduce propellant mass ratio"
            )
        
        # High propellant mass
        if propellant_mass.total_kg > 100:
            suggestions['high_propellant'] = (
                "Large propellant requirement - consider orbital refueling or staging"
            )
        
        # Cost efficiency
        cost_per_ms = cost / delta_v_ms if delta_v_ms > 0 else 0
        if cost_per_ms > 1.5:
            suggestions['cost_efficiency'] = (
                "Cost per m/s is high - review mission profile for optimization opportunities"
            )
        
        # O/F ratio optimization
        if self.params.o_f_ratio != 1.9:
            suggestions['of_ratio'] = (
                f"Current O/F ratio is {self.params.o_f_ratio:.2f} - "
                "consider optimizing for propellant cost balance"
            )
        
        return suggestions
    
    def get_performance_metrics(self, delta_v_ms: float) -> Dict[str, float]:
        """
        Calculate performance metrics for the mission.
        
        Args:
            delta_v_ms: Delta-v requirement in m/s
            
        Returns:
            Dictionary of performance metrics
        """
        propellant_mass = self.get_propellant_requirements(delta_v_ms)
        cost = self.calculate_delta_v_cost(delta_v_ms)
        
        # Mass ratio
        mass_ratio = (self.params.dry_mass + propellant_mass.total_kg) / self.params.dry_mass
        
        # Propellant fraction
        propellant_fraction = propellant_mass.total_kg / (self.params.dry_mass + propellant_mass.total_kg)
        
        # Cost efficiency
        cost_per_kg_propellant = cost / propellant_mass.total_kg if propellant_mass.total_kg > 0 else 0
        
        return {
            'mass_ratio': mass_ratio,
            'propellant_fraction': propellant_fraction,
            'cost_per_ms_usd': cost / delta_v_ms if delta_v_ms > 0 else 0,
            'cost_per_kg_propellant_usd': cost_per_kg_propellant,
            'exhaust_velocity_ms': self.params.isp * self.G0,
            'total_propellant_kg': propellant_mass.total_kg
        }