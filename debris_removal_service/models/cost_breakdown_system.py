"""
Comprehensive cost breakdown system for satellite debris removal missions.

This module provides a complete cost analysis system that integrates
biprop cost calculations, validation, and optimization suggestions.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .biprop_cost_model import BipropCostModel, BipropParameters
from .cost_validator import CostValidator, MissionProfile, OptimizationSuggestion
from .cost import MissionCost, DetailedCost, PropellantMass


class CostCategory(Enum):
    """Cost categories for detailed breakdown."""
    PROPELLANT = "propellant"
    OPERATIONAL = "operational"
    PROCESSING = "processing"
    STORAGE = "storage"
    OVERHEAD = "overhead"


@dataclass
class CostBreakdownItem:
    """Individual cost breakdown item."""
    category: CostCategory
    description: str
    cost_usd: float
    percentage_of_total: float
    unit_cost: Optional[float] = None
    units: Optional[str] = None


@dataclass
class ComprehensiveCostAnalysis:
    """Comprehensive cost analysis result."""
    mission_cost: MissionCost
    cost_breakdown: List[CostBreakdownItem]
    efficiency_metrics: Dict[str, float]
    optimization_suggestions: List[OptimizationSuggestion]
    validation_results: Optional[Dict[str, float]] = None
    comparison_benchmarks: Optional[Dict[str, float]] = None


class CostBreakdownSystem:
    """
    Comprehensive cost breakdown and analysis system.
    
    Integrates biprop cost calculations, validation against reference data,
    and optimization suggestions for different mission profiles.
    """
    
    def __init__(self, cost_model: Optional[BipropCostModel] = None,
                 validator: Optional[CostValidator] = None):
        """
        Initialize the cost breakdown system.
        
        Args:
            cost_model: BipropCostModel instance
            validator: CostValidator instance
        """
        self.cost_model = cost_model or BipropCostModel()
        self.validator = validator or CostValidator(self.cost_model)
    
    def analyze_mission_cost(self, delta_v_ms: float,
                           satellites_collected: int,
                           processing_cost: float = 0.0,
                           storage_cost: float = 0.0,
                           mission_profile: MissionProfile = MissionProfile.MULTI_SATELLITE,
                           include_validation: bool = True,
                           include_benchmarks: bool = True) -> ComprehensiveCostAnalysis:
        """
        Perform comprehensive cost analysis for a mission.
        
        Args:
            delta_v_ms: Total delta-v requirement in m/s
            satellites_collected: Number of satellites collected
            processing_cost: Additional processing cost in USD
            storage_cost: Additional storage cost in USD
            mission_profile: Type of mission profile
            include_validation: Whether to include validation results
            include_benchmarks: Whether to include benchmark comparisons
            
        Returns:
            ComprehensiveCostAnalysis with complete analysis
        """
        # Calculate mission cost
        mission_cost = self.cost_model.calculate_mission_cost(
            delta_v_ms, satellites_collected, processing_cost, storage_cost
        )
        
        # Generate detailed cost breakdown
        cost_breakdown = self._generate_cost_breakdown(mission_cost)
        
        # Calculate efficiency metrics
        efficiency_metrics = self.validator.analyze_mission_efficiency(
            mission_cost, satellites_collected, delta_v_ms
        )
        
        # Generate optimization suggestions
        optimization_suggestions = self.validator.generate_optimization_suggestions(
            mission_cost, mission_profile, satellites_collected, delta_v_ms
        )
        
        # Validation results (optional)
        validation_results = None
        if include_validation:
            validation_results = self._get_validation_summary(delta_v_ms)
        
        # Benchmark comparisons (optional)
        comparison_benchmarks = None
        if include_benchmarks:
            comparison_benchmarks = self._generate_benchmarks(
                mission_cost, satellites_collected, delta_v_ms
            )
        
        return ComprehensiveCostAnalysis(
            mission_cost=mission_cost,
            cost_breakdown=cost_breakdown,
            efficiency_metrics=efficiency_metrics,
            optimization_suggestions=optimization_suggestions,
            validation_results=validation_results,
            comparison_benchmarks=comparison_benchmarks
        )
    
    def _generate_cost_breakdown(self, mission_cost: MissionCost) -> List[CostBreakdownItem]:
        """Generate detailed cost breakdown items."""
        breakdown_items = []
        total_cost = mission_cost.total_cost
        
        if total_cost == 0:
            return breakdown_items
        
        # Collection cost breakdown
        breakdown_items.append(CostBreakdownItem(
            category=CostCategory.PROPELLANT,
            description="Propellant costs (fuel and oxidizer)",
            cost_usd=mission_cost.collection_cost,
            percentage_of_total=(mission_cost.collection_cost / total_cost) * 100,
            unit_cost=mission_cost.collection_cost / mission_cost.propellant_mass.total_kg if mission_cost.propellant_mass and mission_cost.propellant_mass.total_kg > 0 else None,
            units="USD/kg propellant"
        ))
        
        # Operational overhead
        breakdown_items.append(CostBreakdownItem(
            category=CostCategory.OPERATIONAL,
            description="Operational overhead and mission management",
            cost_usd=mission_cost.operational_overhead,
            percentage_of_total=(mission_cost.operational_overhead / total_cost) * 100
        ))
        
        # Processing costs
        if mission_cost.processing_cost > 0:
            breakdown_items.append(CostBreakdownItem(
                category=CostCategory.PROCESSING,
                description="Material processing and recycling",
                cost_usd=mission_cost.processing_cost,
                percentage_of_total=(mission_cost.processing_cost / total_cost) * 100
            ))
        
        # Storage costs
        if mission_cost.storage_cost > 0:
            breakdown_items.append(CostBreakdownItem(
                category=CostCategory.STORAGE,
                description="HEO storage and inventory management",
                cost_usd=mission_cost.storage_cost,
                percentage_of_total=(mission_cost.storage_cost / total_cost) * 100
            ))
        
        # Sort by cost (highest first)
        breakdown_items.sort(key=lambda x: x.cost_usd, reverse=True)
        
        return breakdown_items
    
    def _get_validation_summary(self, delta_v_ms: float) -> Dict[str, float]:
        """Get validation summary for the delta-v value."""
        try:
            validation_result = self.validator.validate_single_point(delta_v_ms)
            return {
                'expected_cost_usd': validation_result.expected_cost,
                'calculated_cost_usd': validation_result.calculated_cost,
                'absolute_error_usd': validation_result.absolute_error,
                'relative_error_percent': validation_result.relative_error * 100,
                'within_tolerance': float(validation_result.within_tolerance)
            }
        except Exception:
            return {}
    
    def _generate_benchmarks(self, mission_cost: MissionCost,
                           satellites_collected: int,
                           delta_v_ms: float) -> Dict[str, float]:
        """Generate benchmark comparisons."""
        benchmarks = {}
        
        # Industry benchmarks (hypothetical values for comparison)
        industry_cost_per_satellite = 800.0  # USD
        industry_cost_per_delta_v = 2.0  # USD per m/s
        
        # Calculate performance vs benchmarks
        benchmarks['cost_per_satellite_vs_industry'] = (
            mission_cost.cost_per_satellite / industry_cost_per_satellite
        )
        
        cost_per_delta_v = mission_cost.total_cost / delta_v_ms if delta_v_ms > 0 else 0
        benchmarks['cost_per_delta_v_vs_industry'] = (
            cost_per_delta_v / industry_cost_per_delta_v if industry_cost_per_delta_v > 0 else 0
        )
        
        # Efficiency benchmarks
        if mission_cost.propellant_mass:
            # Theoretical minimum propellant cost (no overhead)
            theoretical_min_cost = self.cost_model.calculate_propellant_cost(mission_cost.propellant_mass)
            benchmarks['efficiency_vs_theoretical'] = theoretical_min_cost / mission_cost.total_cost
        
        # Mission complexity factor
        base_cost = delta_v_ms * 1.27  # Base formula cost
        benchmarks['complexity_factor'] = mission_cost.total_cost / base_cost if base_cost > 0 else 0
        
        return benchmarks
    
    def compare_mission_scenarios(self, scenarios: List[Tuple[str, float, int, float, float]]) -> Dict[str, ComprehensiveCostAnalysis]:
        """
        Compare multiple mission scenarios.
        
        Args:
            scenarios: List of (name, delta_v, satellites, processing_cost, storage_cost) tuples
            
        Returns:
            Dictionary mapping scenario names to their analyses
        """
        analyses = {}
        
        for name, delta_v, satellites, processing_cost, storage_cost in scenarios:
            analysis = self.analyze_mission_cost(
                delta_v_ms=delta_v,
                satellites_collected=satellites,
                processing_cost=processing_cost,
                storage_cost=storage_cost,
                include_validation=False,  # Skip validation for comparisons
                include_benchmarks=False   # Skip benchmarks for comparisons
            )
            analyses[name] = analysis
        
        return analyses
    
    def generate_cost_report(self, analysis: ComprehensiveCostAnalysis,
                           mission_name: str = "Mission") -> str:
        """
        Generate a formatted cost report.
        
        Args:
            analysis: ComprehensiveCostAnalysis to report on
            mission_name: Name of the mission
            
        Returns:
            Formatted cost report string
        """
        report = []
        report.append(f"=== {mission_name.upper()} COST ANALYSIS REPORT ===\n")
        
        # Mission summary
        mc = analysis.mission_cost
        report.append("MISSION SUMMARY:")
        report.append(f"  Total Cost: ${mc.total_cost:,.2f}")
        report.append(f"  Cost per Satellite: ${mc.cost_per_satellite:,.2f}")
        if mc.propellant_mass:
            report.append(f"  Total Propellant: {mc.propellant_mass.total_kg:.2f} kg")
        report.append("")
        
        # Cost breakdown
        report.append("DETAILED COST BREAKDOWN:")
        for item in analysis.cost_breakdown:
            report.append(f"  {item.description}:")
            report.append(f"    Cost: ${item.cost_usd:,.2f} ({item.percentage_of_total:.1f}%)")
            if item.unit_cost and item.units:
                report.append(f"    Unit Cost: ${item.unit_cost:.2f} {item.units}")
        report.append("")
        
        # Efficiency metrics
        report.append("EFFICIENCY METRICS:")
        for metric, value in analysis.efficiency_metrics.items():
            if 'ratio' in metric:
                report.append(f"  {metric.replace('_', ' ').title()}: {value:.1%}")
            else:
                report.append(f"  {metric.replace('_', ' ').title()}: {value:.2f}")
        report.append("")
        
        # Optimization suggestions
        if analysis.optimization_suggestions:
            report.append("OPTIMIZATION SUGGESTIONS:")
            for i, suggestion in enumerate(analysis.optimization_suggestions, 1):
                report.append(f"  {i}. [{suggestion.priority.upper()}] {suggestion.category.replace('_', ' ').title()}")
                report.append(f"     {suggestion.description}")
                if suggestion.potential_savings_percent:
                    report.append(f"     Potential Savings: {suggestion.potential_savings_percent}%")
                report.append(f"     Implementation: {suggestion.implementation_complexity.title()} complexity")
                report.append("")
        
        # Validation results
        if analysis.validation_results:
            report.append("VALIDATION RESULTS:")
            vr = analysis.validation_results
            report.append(f"  Model Accuracy: {100 - vr.get('relative_error_percent', 0):.1f}%")
            report.append(f"  Expected Cost: ${vr.get('expected_cost_usd', 0):.2f}")
            report.append(f"  Calculated Cost: ${vr.get('calculated_cost_usd', 0):.2f}")
            report.append("")
        
        # Benchmarks
        if analysis.comparison_benchmarks:
            report.append("BENCHMARK COMPARISONS:")
            cb = analysis.comparison_benchmarks
            for benchmark, value in cb.items():
                if value < 1.0:
                    status = "BETTER than benchmark"
                elif value > 1.2:
                    status = "WORSE than benchmark"
                else:
                    status = "Similar to benchmark"
                report.append(f"  {benchmark.replace('_', ' ').title()}: {value:.2f}x ({status})")
            report.append("")
        
        return "\n".join(report)
    
    def export_cost_data(self, analysis: ComprehensiveCostAnalysis) -> Dict:
        """
        Export cost analysis data in a structured format.
        
        Args:
            analysis: ComprehensiveCostAnalysis to export
            
        Returns:
            Dictionary with structured cost data
        """
        return {
            'mission_cost': {
                'total_cost_usd': analysis.mission_cost.total_cost,
                'collection_cost_usd': analysis.mission_cost.collection_cost,
                'processing_cost_usd': analysis.mission_cost.processing_cost,
                'storage_cost_usd': analysis.mission_cost.storage_cost,
                'operational_overhead_usd': analysis.mission_cost.operational_overhead,
                'cost_per_satellite_usd': analysis.mission_cost.cost_per_satellite
            },
            'propellant_mass': {
                'total_kg': analysis.mission_cost.propellant_mass.total_kg if analysis.mission_cost.propellant_mass else 0,
                'fuel_kg': analysis.mission_cost.propellant_mass.fuel_kg if analysis.mission_cost.propellant_mass else 0,
                'oxidizer_kg': analysis.mission_cost.propellant_mass.oxidizer_kg if analysis.mission_cost.propellant_mass else 0
            },
            'cost_breakdown': [
                {
                    'category': item.category.value,
                    'description': item.description,
                    'cost_usd': item.cost_usd,
                    'percentage': item.percentage_of_total,
                    'unit_cost': item.unit_cost,
                    'units': item.units
                }
                for item in analysis.cost_breakdown
            ],
            'efficiency_metrics': analysis.efficiency_metrics,
            'optimization_suggestions': [
                {
                    'category': suggestion.category,
                    'priority': suggestion.priority,
                    'description': suggestion.description,
                    'potential_savings_percent': suggestion.potential_savings_percent,
                    'implementation_complexity': suggestion.implementation_complexity
                }
                for suggestion in analysis.optimization_suggestions
            ],
            'validation_results': analysis.validation_results,
            'comparison_benchmarks': analysis.comparison_benchmarks
        }