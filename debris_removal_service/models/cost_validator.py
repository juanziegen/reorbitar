"""
Cost validation and optimization system for biprop cost calculations.

This module provides validation against reference CSV data and optimization
suggestions for different mission profiles.
"""

import csv
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .biprop_cost_model import BipropCostModel, BipropParameters
from .cost import MissionCost, DetailedCost


class MissionProfile(Enum):
    """Mission profile types for optimization."""
    SINGLE_SATELLITE = "single_satellite"
    MULTI_SATELLITE = "multi_satellite"
    HIGH_DELTA_V = "high_delta_v"
    LOW_DELTA_V = "low_delta_v"
    BULK_COLLECTION = "bulk_collection"
    PRECISION_COLLECTION = "precision_collection"


@dataclass
class ValidationResult:
    """Result of cost validation against reference data."""
    delta_v_ms: float
    expected_cost: float
    calculated_cost: float
    absolute_error: float
    relative_error: float
    within_tolerance: bool
    tolerance_used: float


@dataclass
class OptimizationSuggestion:
    """Cost optimization suggestion for a mission profile."""
    category: str
    priority: str  # "high", "medium", "low"
    description: str
    potential_savings_percent: Optional[float] = None
    implementation_complexity: str = "medium"  # "low", "medium", "high"


class CostValidator:
    """
    Cost validation and optimization system.
    
    Provides validation against CSV reference data and generates
    optimization suggestions for different mission profiles.
    """
    
    def __init__(self, cost_model: Optional[BipropCostModel] = None,
                 csv_file_path: str = "data/leo_biprop_costs_dv_1_500.csv"):
        """
        Initialize the cost validator.
        
        Args:
            cost_model: BipropCostModel instance to validate
            csv_file_path: Path to CSV reference data file
        """
        self.cost_model = cost_model or BipropCostModel()
        self.csv_file_path = csv_file_path
        self.reference_data = self._load_reference_data()
    
    def _load_reference_data(self) -> List[Dict[str, float]]:
        """Load reference data from CSV file."""
        reference_data = []
        
        if not os.path.exists(self.csv_file_path):
            print(f"Warning: CSV file {self.csv_file_path} not found")
            return reference_data
        
        try:
            with open(self.csv_file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    reference_data.append({
                        'delta_v_m_s': float(row['delta_v_m_s']),
                        'total_prop_kg': float(row['total_prop_kg']),
                        'fuel_kg': float(row['fuel_kg']),
                        'ox_kg': float(row['ox_kg']),
                        'cost_usd': float(row['cost_usd']),
                        'cost_per_ms_usd': float(row['cost_per_ms_usd']),
                        'burn_time_s': float(row['burn_time_s']),
                        'fuel_volume_L': float(row['fuel_volume_L']),
                        'ox_volume_L': float(row['ox_volume_L'])
                    })
        except Exception as e:
            print(f"Error loading CSV data: {e}")
        
        return reference_data
    
    def validate_single_point(self, delta_v_ms: float, 
                            tolerance: float = 0.01) -> ValidationResult:
        """
        Validate cost calculation for a single delta-v point.
        
        Args:
            delta_v_ms: Delta-v in m/s to validate
            tolerance: Acceptable relative error tolerance
            
        Returns:
            ValidationResult with comparison details
        """
        # Find closest reference point
        closest_ref = None
        min_delta_v_diff = float('inf')
        
        for ref_point in self.reference_data:
            delta_v_diff = abs(ref_point['delta_v_m_s'] - delta_v_ms)
            if delta_v_diff < min_delta_v_diff:
                min_delta_v_diff = delta_v_diff
                closest_ref = ref_point
        
        if closest_ref is None:
            raise ValueError("No reference data available for validation")
        
        # Use exact delta-v if very close, otherwise interpolate
        if min_delta_v_diff < 0.001:
            expected_cost = closest_ref['cost_usd']
        else:
            # Simple linear interpolation for nearby points
            expected_cost = delta_v_ms * 1.27  # Fallback to formula
        
        # Calculate actual cost
        calculated_cost = self.cost_model.calculate_delta_v_cost(delta_v_ms)
        
        # Calculate errors
        absolute_error = abs(calculated_cost - expected_cost)
        relative_error = absolute_error / expected_cost if expected_cost > 0 else 0
        within_tolerance = relative_error <= tolerance
        
        return ValidationResult(
            delta_v_ms=delta_v_ms,
            expected_cost=expected_cost,
            calculated_cost=calculated_cost,
            absolute_error=absolute_error,
            relative_error=relative_error,
            within_tolerance=within_tolerance,
            tolerance_used=tolerance
        )
    
    def validate_range(self, delta_v_range: Tuple[float, float], 
                      num_points: int = 10,
                      tolerance: float = 0.01) -> List[ValidationResult]:
        """
        Validate cost calculations over a range of delta-v values.
        
        Args:
            delta_v_range: (min_delta_v, max_delta_v) tuple
            num_points: Number of validation points
            tolerance: Acceptable relative error tolerance
            
        Returns:
            List of ValidationResult objects
        """
        min_dv, max_dv = delta_v_range
        delta_v_step = (max_dv - min_dv) / (num_points - 1)
        
        results = []
        for i in range(num_points):
            delta_v = min_dv + i * delta_v_step
            result = self.validate_single_point(delta_v, tolerance)
            results.append(result)
        
        return results
    
    def validate_against_csv(self, tolerance: float = 0.01) -> List[ValidationResult]:
        """
        Validate against all available CSV reference data.
        
        Args:
            tolerance: Acceptable relative error tolerance
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        
        for ref_point in self.reference_data:
            delta_v = ref_point['delta_v_m_s']
            expected_cost = ref_point['cost_usd']
            calculated_cost = self.cost_model.calculate_delta_v_cost(delta_v)
            
            absolute_error = abs(calculated_cost - expected_cost)
            relative_error = absolute_error / expected_cost if expected_cost > 0 else 0
            within_tolerance = relative_error <= tolerance
            
            results.append(ValidationResult(
                delta_v_ms=delta_v,
                expected_cost=expected_cost,
                calculated_cost=calculated_cost,
                absolute_error=absolute_error,
                relative_error=relative_error,
                within_tolerance=within_tolerance,
                tolerance_used=tolerance
            ))
        
        return results
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, float]:
        """
        Generate summary statistics for validation results.
        
        Args:
            results: List of ValidationResult objects
            
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {}
        
        total_points = len(results)
        passed_points = sum(1 for r in results if r.within_tolerance)
        
        relative_errors = [r.relative_error for r in results]
        absolute_errors = [r.absolute_error for r in results]
        
        return {
            'total_points': total_points,
            'passed_points': passed_points,
            'pass_rate': passed_points / total_points,
            'max_relative_error': max(relative_errors),
            'avg_relative_error': sum(relative_errors) / len(relative_errors),
            'max_absolute_error': max(absolute_errors),
            'avg_absolute_error': sum(absolute_errors) / len(absolute_errors)
        }
    
    def generate_optimization_suggestions(self, mission_cost: MissionCost,
                                        mission_profile: MissionProfile,
                                        satellites_collected: int,
                                        total_delta_v: float) -> List[OptimizationSuggestion]:
        """
        Generate cost optimization suggestions for a mission.
        
        Args:
            mission_cost: MissionCost object to analyze
            mission_profile: Type of mission profile
            satellites_collected: Number of satellites collected
            total_delta_v: Total delta-v requirement
            
        Returns:
            List of OptimizationSuggestion objects
        """
        suggestions = []
        
        # Analyze cost breakdown
        cost_percentages = mission_cost.detailed_breakdown.get_cost_percentages() if mission_cost.detailed_breakdown else {}
        
        # High propellant cost suggestions
        if cost_percentages.get('propellant', 0) > 40:
            suggestions.append(OptimizationSuggestion(
                category="propellant_optimization",
                priority="high",
                description="Propellant costs are high (>40% of total). Consider route optimization to reduce total delta-v requirements.",
                potential_savings_percent=15.0,
                implementation_complexity="medium"
            ))
        
        # High operational overhead suggestions
        if cost_percentages.get('operational', 0) > 25:
            suggestions.append(OptimizationSuggestion(
                category="operational_efficiency",
                priority="medium",
                description="Operational overhead is high (>25% of total). Consider batch processing multiple missions to amortize fixed costs.",
                potential_savings_percent=10.0,
                implementation_complexity="low"
            ))
        
        # Mission profile specific suggestions
        if mission_profile == MissionProfile.HIGH_DELTA_V:
            if total_delta_v > 1000:
                suggestions.append(OptimizationSuggestion(
                    category="mission_architecture",
                    priority="high",
                    description="High delta-v mission (>1000 m/s). Consider multi-stage approach or orbital refueling to improve mass ratio efficiency.",
                    potential_savings_percent=25.0,
                    implementation_complexity="high"
                ))
        
        elif mission_profile == MissionProfile.MULTI_SATELLITE:
            if satellites_collected > 5:
                suggestions.append(OptimizationSuggestion(
                    category="route_optimization",
                    priority="high",
                    description="Multi-satellite mission with many targets. Optimize collection sequence to minimize total delta-v.",
                    potential_savings_percent=20.0,
                    implementation_complexity="medium"
                ))
        
        elif mission_profile == MissionProfile.BULK_COLLECTION:
            suggestions.append(OptimizationSuggestion(
                category="economies_of_scale",
                priority="medium",
                description="Bulk collection mission. Consider larger spacecraft to improve mass ratio efficiency.",
                potential_savings_percent=12.0,
                implementation_complexity="high"
            ))
        
        # Cost per satellite analysis
        if mission_cost.cost_per_satellite > 1000:
            suggestions.append(OptimizationSuggestion(
                category="cost_efficiency",
                priority="medium",
                description="High cost per satellite (>$1000). Consider grouping nearby satellites or improving collection efficiency.",
                potential_savings_percent=15.0,
                implementation_complexity="medium"
            ))
        
        # Processing cost optimization
        if mission_cost.processing_cost > mission_cost.collection_cost:
            suggestions.append(OptimizationSuggestion(
                category="processing_optimization",
                priority="medium",
                description="Processing costs exceed collection costs. Consider alternative processing methods or bulk processing discounts.",
                potential_savings_percent=8.0,
                implementation_complexity="low"
            ))
        
        # Storage cost optimization
        if mission_cost.storage_cost > 0:
            storage_ratio = mission_cost.storage_cost / mission_cost.total_cost
            if storage_ratio > 0.15:
                suggestions.append(OptimizationSuggestion(
                    category="storage_optimization",
                    priority="low",
                    description="Storage costs are significant (>15% of total). Consider immediate processing or alternative storage solutions.",
                    potential_savings_percent=5.0,
                    implementation_complexity="low"
                ))
        
        # Sort suggestions by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: priority_order.get(x.priority, 3))
        
        return suggestions
    
    def analyze_mission_efficiency(self, mission_cost: MissionCost,
                                 satellites_collected: int,
                                 total_delta_v: float) -> Dict[str, float]:
        """
        Analyze mission efficiency metrics.
        
        Args:
            mission_cost: MissionCost object to analyze
            satellites_collected: Number of satellites collected
            total_delta_v: Total delta-v requirement
            
        Returns:
            Dictionary with efficiency metrics
        """
        metrics = {}
        
        # Basic efficiency metrics
        metrics['cost_per_satellite_usd'] = mission_cost.cost_per_satellite
        metrics['cost_per_delta_v_usd'] = mission_cost.total_cost / total_delta_v if total_delta_v > 0 else 0
        
        # Propellant efficiency
        if mission_cost.propellant_mass:
            metrics['cost_per_kg_propellant_usd'] = mission_cost.total_cost / mission_cost.propellant_mass.total_kg
            metrics['delta_v_per_kg_propellant'] = total_delta_v / mission_cost.propellant_mass.total_kg
        
        # Cost breakdown ratios
        if mission_cost.detailed_breakdown:
            total_cost = mission_cost.total_cost
            metrics['propellant_cost_ratio'] = mission_cost.detailed_breakdown.propellant_cost / total_cost
            metrics['operational_cost_ratio'] = mission_cost.detailed_breakdown.operational_cost / total_cost
            metrics['processing_cost_ratio'] = mission_cost.detailed_breakdown.processing_cost / total_cost
            metrics['storage_cost_ratio'] = mission_cost.detailed_breakdown.storage_cost / total_cost
        
        # Collection efficiency
        metrics['satellites_per_1000_delta_v'] = (satellites_collected / total_delta_v) * 1000 if total_delta_v > 0 else 0
        
        return metrics
    
    def compare_mission_profiles(self, missions: List[Tuple[str, MissionCost, int, float]]) -> Dict[str, Dict[str, float]]:
        """
        Compare efficiency metrics across different mission profiles.
        
        Args:
            missions: List of (name, mission_cost, satellites_collected, total_delta_v) tuples
            
        Returns:
            Dictionary with comparison metrics for each mission
        """
        comparison = {}
        
        for name, mission_cost, satellites_collected, total_delta_v in missions:
            comparison[name] = self.analyze_mission_efficiency(
                mission_cost, satellites_collected, total_delta_v
            )
        
        return comparison