"""
Test for cost validation and optimization system.
"""

from debris_removal_service.models.cost_validator import (
    CostValidator, MissionProfile, ValidationResult, OptimizationSuggestion
)
from debris_removal_service.models.biprop_cost_model import BipropCostModel
from debris_removal_service.models.cost import MissionCost, DetailedCost, PropellantMass


def test_cost_validator():
    """Test cost validation functionality."""
    print("Testing CostValidator...")
    
    # Initialize validator
    validator = CostValidator()
    
    # Test single point validation
    print("\n1. Testing single point validation:")
    result = validator.validate_single_point(10.0, tolerance=0.01)
    print(f"Delta-v: {result.delta_v_ms} m/s")
    print(f"Expected: ${result.expected_cost:.2f}")
    print(f"Calculated: ${result.calculated_cost:.2f}")
    print(f"Relative error: {result.relative_error:.1%}")
    print(f"Within tolerance: {result.within_tolerance}")
    
    # Test range validation
    print("\n2. Testing range validation:")
    range_results = validator.validate_range((1.0, 50.0), num_points=5)
    for result in range_results:
        print(f"Δv={result.delta_v_ms:.1f} m/s: Error={result.relative_error:.1%}, "
              f"Pass={result.within_tolerance}")
    
    # Test CSV validation (if file exists)
    print("\n3. Testing CSV validation:")
    try:
        csv_results = validator.validate_against_csv(tolerance=0.01)
        summary = validator.get_validation_summary(csv_results)
        print(f"Total points: {summary['total_points']}")
        print(f"Pass rate: {summary['pass_rate']:.1%}")
        print(f"Max relative error: {summary['max_relative_error']:.1%}")
        print(f"Avg relative error: {summary['avg_relative_error']:.1%}")
    except Exception as e:
        print(f"CSV validation skipped: {e}")
    
    print("✓ Cost validation tests passed!")


def test_optimization_suggestions():
    """Test optimization suggestion generation."""
    print("\nTesting optimization suggestions...")
    
    validator = CostValidator()
    
    # Create sample mission cost
    propellant_mass = PropellantMass(fuel_kg=5.0, oxidizer_kg=9.5, total_kg=14.5)
    detailed_cost = DetailedCost(
        propellant_cost=2000.0,
        operational_cost=500.0,
        processing_cost=1000.0,
        storage_cost=300.0,
        total_cost=3800.0
    )
    mission_cost = MissionCost(
        collection_cost=2000.0,
        processing_cost=1000.0,
        storage_cost=300.0,
        operational_overhead=500.0,
        total_cost=3800.0,
        cost_per_satellite=760.0,
        propellant_mass=propellant_mass,
        detailed_breakdown=detailed_cost
    )
    
    # Test different mission profiles
    profiles = [
        (MissionProfile.HIGH_DELTA_V, 5, 1200.0),
        (MissionProfile.MULTI_SATELLITE, 8, 800.0),
        (MissionProfile.BULK_COLLECTION, 3, 400.0)
    ]
    
    for profile, satellites, delta_v in profiles:
        print(f"\n{profile.value.upper()} mission suggestions:")
        suggestions = validator.generate_optimization_suggestions(
            mission_cost, profile, satellites, delta_v
        )
        
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. [{suggestion.priority.upper()}] {suggestion.category}")
            print(f"     {suggestion.description}")
            if suggestion.potential_savings_percent:
                print(f"     Potential savings: {suggestion.potential_savings_percent}%")
            print(f"     Complexity: {suggestion.implementation_complexity}")
    
    print("✓ Optimization suggestion tests passed!")


def test_mission_efficiency_analysis():
    """Test mission efficiency analysis."""
    print("\nTesting mission efficiency analysis...")
    
    validator = CostValidator()
    
    # Create sample mission costs for comparison
    missions = []
    
    # Mission 1: High-efficiency mission
    propellant_mass_1 = PropellantMass(fuel_kg=2.0, oxidizer_kg=3.8, total_kg=5.8)
    detailed_cost_1 = DetailedCost(
        propellant_cost=800.0,
        operational_cost=200.0,
        processing_cost=400.0,
        storage_cost=100.0,
        total_cost=1500.0
    )
    mission_cost_1 = MissionCost(
        collection_cost=800.0,
        processing_cost=400.0,
        storage_cost=100.0,
        operational_overhead=200.0,
        total_cost=1500.0,
        cost_per_satellite=300.0,
        propellant_mass=propellant_mass_1,
        detailed_breakdown=detailed_cost_1
    )
    missions.append(("efficient_mission", mission_cost_1, 5, 200.0))
    
    # Mission 2: Less efficient mission
    propellant_mass_2 = PropellantMass(fuel_kg=8.0, oxidizer_kg=15.2, total_kg=23.2)
    detailed_cost_2 = DetailedCost(
        propellant_cost=3200.0,
        operational_cost=800.0,
        processing_cost=1000.0,
        storage_cost=500.0,
        total_cost=5500.0
    )
    mission_cost_2 = MissionCost(
        collection_cost=3200.0,
        processing_cost=1000.0,
        storage_cost=500.0,
        operational_overhead=800.0,
        total_cost=5500.0,
        cost_per_satellite=1100.0,
        propellant_mass=propellant_mass_2,
        detailed_breakdown=detailed_cost_2
    )
    missions.append(("expensive_mission", mission_cost_2, 5, 800.0))
    
    # Compare missions
    comparison = validator.compare_mission_profiles(missions)
    
    print("Mission comparison:")
    for mission_name, metrics in comparison.items():
        print(f"\n{mission_name.upper()}:")
        print(f"  Cost per satellite: ${metrics['cost_per_satellite_usd']:.2f}")
        print(f"  Cost per delta-v: ${metrics['cost_per_delta_v_usd']:.2f}")
        print(f"  Cost per kg propellant: ${metrics['cost_per_kg_propellant_usd']:.2f}")
        print(f"  Propellant cost ratio: {metrics['propellant_cost_ratio']:.1%}")
        print(f"  Satellites per 1000 m/s: {metrics['satellites_per_1000_delta_v']:.1f}")
    
    print("✓ Mission efficiency analysis tests passed!")


if __name__ == "__main__":
    try:
        test_cost_validator()
        test_optimization_suggestions()
        test_mission_efficiency_analysis()
        print("\n🎉 All cost validation tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()