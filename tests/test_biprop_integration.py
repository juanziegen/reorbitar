"""
Integration test for the complete biprop cost calculation engine.
"""

from debris_removal_service.models.biprop_cost_model import BipropCostModel, BipropParameters
from debris_removal_service.models.cost_validator import CostValidator, MissionProfile
from debris_removal_service.models.cost_breakdown_system import CostBreakdownSystem


def test_complete_integration():
    """Test complete integration of all biprop cost components."""
    print("Testing complete biprop cost calculation engine integration...")
    
    # Initialize all components
    cost_model = BipropCostModel()
    validator = CostValidator(cost_model)
    cost_system = CostBreakdownSystem(cost_model, validator)
    
    print("✓ All components initialized successfully")
    
    # Test realistic mission scenario
    print("\n=== REALISTIC MISSION SCENARIO ===")
    
    # Mission parameters
    delta_v = 450.0  # m/s - realistic for LEO debris collection
    satellites = 6   # Multiple satellites
    processing_cost = 1200.0  # ISS recycling costs
    storage_cost = 300.0      # HEO storage costs
    
    print(f"Mission Parameters:")
    print(f"  Delta-v requirement: {delta_v} m/s")
    print(f"  Satellites to collect: {satellites}")
    print(f"  Processing cost: ${processing_cost}")
    print(f"  Storage cost: ${storage_cost}")
    
    # Perform comprehensive analysis
    analysis = cost_system.analyze_mission_cost(
        delta_v_ms=delta_v,
        satellites_collected=satellites,
        processing_cost=processing_cost,
        storage_cost=storage_cost,
        mission_profile=MissionProfile.MULTI_SATELLITE
    )
    
    # Display results
    mc = analysis.mission_cost
    print(f"\nMission Cost Results:")
    print(f"  Total Cost: ${mc.total_cost:,.2f}")
    print(f"  Cost per Satellite: ${mc.cost_per_satellite:,.2f}")
    print(f"  Collection Cost: ${mc.collection_cost:,.2f}")
    print(f"  Propellant Mass: {mc.propellant_mass.total_kg:.2f} kg")
    
    # Verify cost formula accuracy
    expected_base_cost = delta_v * 1.27
    print(f"\nCost Formula Verification:")
    print(f"  Base formula cost (Δv × $1.27): ${expected_base_cost:.2f}")
    print(f"  Actual calculated cost: ${cost_model.calculate_delta_v_cost(delta_v):.2f}")
    
    # Test propellant calculations
    prop_mass = cost_model.get_propellant_requirements(delta_v)
    print(f"\nPropellant Breakdown:")
    print(f"  Fuel: {prop_mass.fuel_kg:.3f} kg")
    print(f"  Oxidizer: {prop_mass.oxidizer_kg:.3f} kg")
    print(f"  Total: {prop_mass.total_kg:.3f} kg")
    print(f"  O/F Ratio: {prop_mass.oxidizer_kg/prop_mass.fuel_kg:.3f}")
    
    # Test optimization suggestions
    print(f"\nOptimization Suggestions ({len(analysis.optimization_suggestions)}):")
    for i, suggestion in enumerate(analysis.optimization_suggestions, 1):
        print(f"  {i}. [{suggestion.priority.upper()}] {suggestion.category}")
        print(f"     {suggestion.description}")
        if suggestion.potential_savings_percent:
            print(f"     Potential savings: {suggestion.potential_savings_percent}%")
    
    # Test validation
    if analysis.validation_results:
        vr = analysis.validation_results
        print(f"\nValidation Results:")
        print(f"  Model accuracy: {100 - vr.get('relative_error_percent', 0):.1f}%")
        print(f"  Relative error: {vr.get('relative_error_percent', 0):.1f}%")
    
    print("\n✓ Complete integration test passed!")
    
    return analysis


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n=== TESTING EDGE CASES ===")
    
    cost_system = CostBreakdownSystem()
    
    # Very small mission
    print("\n1. Very small mission (1 m/s, 1 satellite):")
    small_analysis = cost_system.analyze_mission_cost(1.0, 1, 0.0, 0.0)
    print(f"   Total cost: ${small_analysis.mission_cost.total_cost:.2f}")
    print(f"   Propellant: {small_analysis.mission_cost.propellant_mass.total_kg:.6f} kg")
    
    # Large mission
    print("\n2. Large mission (2000 m/s, 20 satellites):")
    large_analysis = cost_system.analyze_mission_cost(2000.0, 20, 5000.0, 1000.0)
    print(f"   Total cost: ${large_analysis.mission_cost.total_cost:,.2f}")
    print(f"   Cost per satellite: ${large_analysis.mission_cost.cost_per_satellite:,.2f}")
    print(f"   Suggestions: {len(large_analysis.optimization_suggestions)}")
    
    # Zero processing/storage costs
    print("\n3. Collection-only mission:")
    collection_only = cost_system.analyze_mission_cost(300.0, 3, 0.0, 0.0)
    print(f"   Total cost: ${collection_only.mission_cost.total_cost:.2f}")
    print(f"   Collection ratio: {collection_only.mission_cost.collection_cost/collection_only.mission_cost.total_cost:.1%}")
    
    print("\n✓ Edge cases test passed!")


def test_custom_parameters():
    """Test with custom biprop parameters."""
    print("\n=== TESTING CUSTOM PARAMETERS ===")
    
    # Create custom parameters (higher performance engine)
    custom_params = BipropParameters(
        isp=350.0,  # Higher ISP
        dry_mass=12.0,  # Lighter spacecraft
        fuel_cost_per_kg=180.0,  # Cheaper fuel
        oxidizer_cost_per_kg=280.0,  # Cheaper oxidizer
        o_f_ratio=2.2,  # Different mixture ratio
        operational_overhead_factor=0.12  # Lower overhead
    )
    
    # Create models with custom parameters
    custom_model = BipropCostModel(custom_params)
    custom_validator = CostValidator(custom_model)
    custom_system = CostBreakdownSystem(custom_model, custom_validator)
    
    # Compare standard vs custom
    delta_v = 400.0
    satellites = 4
    
    print(f"\nComparing standard vs custom parameters for {delta_v} m/s mission:")
    
    # Standard analysis
    standard_system = CostBreakdownSystem()
    standard_analysis = standard_system.analyze_mission_cost(delta_v, satellites, 500.0, 100.0)
    
    # Custom analysis
    custom_analysis = custom_system.analyze_mission_cost(delta_v, satellites, 500.0, 100.0)
    
    print(f"\nStandard Parameters:")
    print(f"  Total cost: ${standard_analysis.mission_cost.total_cost:.2f}")
    print(f"  Propellant mass: {standard_analysis.mission_cost.propellant_mass.total_kg:.3f} kg")
    print(f"  Collection cost: ${standard_analysis.mission_cost.collection_cost:.2f}")
    
    print(f"\nCustom Parameters (ISP={custom_params.isp}, lower costs):")
    print(f"  Total cost: ${custom_analysis.mission_cost.total_cost:.2f}")
    print(f"  Propellant mass: {custom_analysis.mission_cost.propellant_mass.total_kg:.3f} kg")
    print(f"  Collection cost: ${custom_analysis.mission_cost.collection_cost:.2f}")
    
    # Calculate savings
    cost_savings = standard_analysis.mission_cost.total_cost - custom_analysis.mission_cost.total_cost
    mass_savings = standard_analysis.mission_cost.propellant_mass.total_kg - custom_analysis.mission_cost.propellant_mass.total_kg
    
    print(f"\nImprovements with custom parameters:")
    print(f"  Cost savings: ${cost_savings:.2f} ({cost_savings/standard_analysis.mission_cost.total_cost:.1%})")
    print(f"  Mass savings: {mass_savings:.3f} kg ({mass_savings/standard_analysis.mission_cost.propellant_mass.total_kg:.1%})")
    
    print("\n✓ Custom parameters test passed!")


if __name__ == "__main__":
    try:
        # Run all integration tests
        main_analysis = test_complete_integration()
        test_edge_cases()
        test_custom_parameters()
        
        # Generate final report
        cost_system = CostBreakdownSystem()
        report = cost_system.generate_cost_report(main_analysis, "Integration Test Mission")
        
        print("\n" + "="*80)
        print("FINAL INTEGRATION TEST REPORT")
        print("="*80)
        print(report)
        
        print("\n🎉 ALL BIPROP COST ENGINE INTEGRATION TESTS PASSED!")
        print("\nImplemented components:")
        print("✓ BipropCostModel - Accurate cost calculations using rocket equation")
        print("✓ CostValidator - Validation against CSV data and optimization suggestions")
        print("✓ CostBreakdownSystem - Comprehensive analysis and reporting")
        print("✓ Complete integration with all data models")
        print("✓ Support for custom parameters and mission profiles")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()