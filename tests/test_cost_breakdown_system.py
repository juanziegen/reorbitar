"""
Test for comprehensive cost breakdown system.
"""

from debris_removal_service.models.cost_breakdown_system import (
    CostBreakdownSystem, CostCategory, ComprehensiveCostAnalysis
)
from debris_removal_service.models.cost_validator import MissionProfile


def test_comprehensive_cost_analysis():
    """Test comprehensive cost analysis functionality."""
    print("Testing comprehensive cost analysis...")
    
    # Initialize system
    cost_system = CostBreakdownSystem()
    
    # Test mission analysis
    analysis = cost_system.analyze_mission_cost(
        delta_v_ms=300.0,
        satellites_collected=4,
        processing_cost=800.0,
        storage_cost=200.0,
        mission_profile=MissionProfile.MULTI_SATELLITE
    )
    
    print(f"Total mission cost: ${analysis.mission_cost.total_cost:,.2f}")
    print(f"Cost per satellite: ${analysis.mission_cost.cost_per_satellite:,.2f}")
    
    # Test cost breakdown
    print("\nCost breakdown:")
    for item in analysis.cost_breakdown:
        print(f"  {item.category.value}: ${item.cost_usd:,.2f} ({item.percentage_of_total:.1f}%)")
        print(f"    {item.description}")
    
    # Test efficiency metrics
    print("\nEfficiency metrics:")
    for metric, value in analysis.efficiency_metrics.items():
        if 'ratio' in metric:
            print(f"  {metric}: {value:.1%}")
        else:
            print(f"  {metric}: {value:.2f}")
    
    # Test optimization suggestions
    print(f"\nOptimization suggestions ({len(analysis.optimization_suggestions)} found):")
    for i, suggestion in enumerate(analysis.optimization_suggestions, 1):
        print(f"  {i}. [{suggestion.priority}] {suggestion.category}")
        print(f"     {suggestion.description}")
        if suggestion.potential_savings_percent:
            print(f"     Savings: {suggestion.potential_savings_percent}%")
    
    print("✓ Comprehensive cost analysis test passed!")


def test_mission_comparison():
    """Test mission scenario comparison."""
    print("\nTesting mission scenario comparison...")
    
    cost_system = CostBreakdownSystem()
    
    # Define scenarios to compare
    scenarios = [
        ("low_cost_mission", 150.0, 3, 300.0, 100.0),
        ("high_efficiency_mission", 200.0, 5, 400.0, 0.0),
        ("complex_mission", 500.0, 8, 1200.0, 600.0)
    ]
    
    # Compare scenarios
    comparisons = cost_system.compare_mission_scenarios(scenarios)
    
    print("Mission comparison results:")
    for name, analysis in comparisons.items():
        mc = analysis.mission_cost
        print(f"\n{name.upper()}:")
        print(f"  Total Cost: ${mc.total_cost:,.2f}")
        print(f"  Cost/Satellite: ${mc.cost_per_satellite:,.2f}")
        print(f"  Efficiency Score: {analysis.efficiency_metrics.get('satellites_per_1000_delta_v', 0):.1f}")
        print(f"  Suggestions: {len(analysis.optimization_suggestions)}")
    
    print("✓ Mission comparison test passed!")


def test_cost_report_generation():
    """Test cost report generation."""
    print("\nTesting cost report generation...")
    
    cost_system = CostBreakdownSystem()
    
    # Generate analysis
    analysis = cost_system.analyze_mission_cost(
        delta_v_ms=250.0,
        satellites_collected=3,
        processing_cost=600.0,
        storage_cost=150.0,
        mission_profile=MissionProfile.HIGH_DELTA_V
    )
    
    # Generate report
    report = cost_system.generate_cost_report(analysis, "Test Mission Alpha")
    
    print("Generated cost report:")
    print("-" * 60)
    print(report)
    print("-" * 60)
    
    # Verify report contains key sections
    assert "MISSION SUMMARY" in report
    assert "COST BREAKDOWN" in report
    assert "EFFICIENCY METRICS" in report
    assert "OPTIMIZATION SUGGESTIONS" in report
    
    print("✓ Cost report generation test passed!")


def test_data_export():
    """Test structured data export."""
    print("\nTesting data export...")
    
    cost_system = CostBreakdownSystem()
    
    # Generate analysis
    analysis = cost_system.analyze_mission_cost(
        delta_v_ms=180.0,
        satellites_collected=2,
        processing_cost=400.0,
        storage_cost=100.0
    )
    
    # Export data
    exported_data = cost_system.export_cost_data(analysis)
    
    # Verify export structure
    required_keys = [
        'mission_cost', 'propellant_mass', 'cost_breakdown',
        'efficiency_metrics', 'optimization_suggestions'
    ]
    
    for key in required_keys:
        assert key in exported_data, f"Missing key: {key}"
    
    print("Exported data structure:")
    for key, value in exported_data.items():
        if isinstance(value, dict):
            print(f"  {key}: {len(value)} items")
        elif isinstance(value, list):
            print(f"  {key}: {len(value)} items")
        else:
            print(f"  {key}: {type(value).__name__}")
    
    # Verify mission cost data
    mission_cost_data = exported_data['mission_cost']
    assert mission_cost_data['total_cost_usd'] > 0
    assert mission_cost_data['cost_per_satellite_usd'] > 0
    
    print("✓ Data export test passed!")


def test_different_mission_profiles():
    """Test analysis with different mission profiles."""
    print("\nTesting different mission profiles...")
    
    cost_system = CostBreakdownSystem()
    
    profiles = [
        MissionProfile.SINGLE_SATELLITE,
        MissionProfile.MULTI_SATELLITE,
        MissionProfile.HIGH_DELTA_V,
        MissionProfile.BULK_COLLECTION
    ]
    
    for profile in profiles:
        print(f"\nTesting {profile.value} profile:")
        
        # Adjust parameters based on profile
        if profile == MissionProfile.SINGLE_SATELLITE:
            delta_v, satellites = 100.0, 1
        elif profile == MissionProfile.HIGH_DELTA_V:
            delta_v, satellites = 1200.0, 3
        elif profile == MissionProfile.BULK_COLLECTION:
            delta_v, satellites = 800.0, 10
        else:  # MULTI_SATELLITE
            delta_v, satellites = 400.0, 5
        
        analysis = cost_system.analyze_mission_cost(
            delta_v_ms=delta_v,
            satellites_collected=satellites,
            processing_cost=300.0,
            storage_cost=100.0,
            mission_profile=profile
        )
        
        print(f"  Total cost: ${analysis.mission_cost.total_cost:,.2f}")
        print(f"  Suggestions: {len(analysis.optimization_suggestions)}")
        
        # Verify profile-specific suggestions
        suggestion_categories = [s.category for s in analysis.optimization_suggestions]
        
        if profile == MissionProfile.HIGH_DELTA_V and delta_v > 1000:
            assert any('mission_architecture' in cat for cat in suggestion_categories), \
                "High delta-v missions should have architecture suggestions"
        
        if profile == MissionProfile.MULTI_SATELLITE and satellites > 5:
            assert any('route_optimization' in cat for cat in suggestion_categories), \
                "Multi-satellite missions should have route optimization suggestions"
    
    print("✓ Different mission profiles test passed!")


if __name__ == "__main__":
    try:
        test_comprehensive_cost_analysis()
        test_mission_comparison()
        test_cost_report_generation()
        test_data_export()
        test_different_mission_profiles()
        print("\n🎉 All cost breakdown system tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()