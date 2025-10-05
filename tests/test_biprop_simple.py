"""
Simple test for BipropCostModel implementation without pytest.
"""

import math
import csv
from debris_removal_service.models.biprop_cost_model import BipropCostModel, BipropParameters
from debris_removal_service.models.cost import PropellantMass


def test_basic_functionality():
    """Test basic BipropCostModel functionality."""
    print("Testing BipropCostModel basic functionality...")
    
    # Initialize model
    model = BipropCostModel()
    
    # Test 1: Basic cost calculation
    cost_1ms = model.calculate_delta_v_cost(1.0)
    print(f"Cost for 1 m/s delta-v: ${cost_1ms:.2f}")
    assert abs(cost_1ms - 1.27) < 0.001, f"Expected 1.27, got {cost_1ms}"
    
    cost_10ms = model.calculate_delta_v_cost(10.0)
    print(f"Cost for 10 m/s delta-v: ${cost_10ms:.2f}")
    assert abs(cost_10ms - 12.7) < 0.001, f"Expected 12.7, got {cost_10ms}"
    
    # Test 2: Propellant mass calculation
    prop_mass = model.get_propellant_requirements(100.0)
    print(f"Propellant mass for 100 m/s: {prop_mass.total_kg:.3f} kg")
    print(f"  Fuel: {prop_mass.fuel_kg:.3f} kg")
    print(f"  Oxidizer: {prop_mass.oxidizer_kg:.3f} kg")
    
    # Verify O/F ratio
    actual_ratio = prop_mass.oxidizer_kg / prop_mass.fuel_kg
    expected_ratio = model.params.o_f_ratio
    print(f"O/F ratio: {actual_ratio:.3f} (expected: {expected_ratio})")
    assert abs(actual_ratio - expected_ratio) < 0.001, f"O/F ratio mismatch"
    
    # Test 3: Rocket equation verification
    delta_v = 100.0
    initial_mass = model.params.dry_mass + prop_mass.total_kg
    mass_ratio = initial_mass / model.params.dry_mass
    ve = model.params.isp * model.G0
    calculated_delta_v = ve * math.log(mass_ratio)
    print(f"Rocket equation verification: {calculated_delta_v:.3f} m/s (expected: {delta_v})")
    assert abs(calculated_delta_v - delta_v) < 0.001, f"Rocket equation error"
    
    # Test 4: Mission cost calculation
    mission_cost = model.calculate_mission_cost(
        delta_v_ms=200.0,
        satellites_collected=3,
        processing_cost=500.0,
        storage_cost=200.0
    )
    print(f"Mission cost breakdown:")
    print(f"  Collection: ${mission_cost.collection_cost:.2f}")
    print(f"  Processing: ${mission_cost.processing_cost:.2f}")
    print(f"  Storage: ${mission_cost.storage_cost:.2f}")
    print(f"  Overhead: ${mission_cost.operational_overhead:.2f}")
    print(f"  Total: ${mission_cost.total_cost:.2f}")
    print(f"  Cost per satellite: ${mission_cost.cost_per_satellite:.2f}")
    
    # Test 5: Performance metrics
    metrics = model.get_performance_metrics(100.0)
    print(f"Performance metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")
    
    print("✓ All basic functionality tests passed!")


def test_csv_validation():
    """Test validation against CSV data."""
    print("\nTesting CSV validation...")
    
    model = BipropCostModel()
    
    # Read first few lines of CSV data
    try:
        with open('data/leo_biprop_costs_dv_1_500.csv', 'r') as f:
            reader = csv.DictReader(f)
            test_count = 0
            
            for row in reader:
                if test_count >= 10:  # Test first 10 rows
                    break
                
                delta_v = float(row['delta_v_m_s'])
                expected_cost = float(row['cost_usd'])
                
                calculated_cost = model.calculate_delta_v_cost(delta_v)
                error = abs(calculated_cost - expected_cost)
                relative_error = error / expected_cost if expected_cost > 0 else 0
                
                print(f"Δv={delta_v} m/s: Expected=${expected_cost:.2f}, "
                      f"Calculated=${calculated_cost:.2f}, Error={relative_error:.1%}")
                
                # Allow 1% tolerance
                assert relative_error < 0.01, f"Cost calculation error too large: {relative_error:.1%}"
                
                test_count += 1
        
        print("✓ CSV validation tests passed!")
        
    except FileNotFoundError:
        print("⚠ CSV file not found, skipping validation test")


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\nTesting edge cases...")
    
    model = BipropCostModel()
    
    # Test zero delta-v
    cost_zero = model.calculate_delta_v_cost(0.0)
    assert cost_zero == 0.0, f"Zero delta-v should cost $0, got ${cost_zero}"
    
    prop_zero = model.get_propellant_requirements(0.0)
    assert prop_zero.total_kg == 0.0, f"Zero delta-v should need 0 kg propellant"
    
    # Test very small delta-v
    small_dv = 0.001
    prop_small = model.get_propellant_requirements(small_dv)
    assert prop_small.total_kg > 0, f"Small delta-v should need some propellant"
    
    # Test negative delta-v (should raise error)
    try:
        model.calculate_delta_v_cost(-10.0)
        assert False, "Negative delta-v should raise ValueError"
    except ValueError:
        pass  # Expected
    
    # Test invalid parameters
    try:
        BipropCostModel(BipropParameters(isp=-100))
        assert False, "Negative ISP should raise ValueError"
    except ValueError:
        pass  # Expected
    
    print("✓ Edge case tests passed!")


def test_optimization_suggestions():
    """Test cost optimization suggestions."""
    print("\nTesting optimization suggestions...")
    
    model = BipropCostModel()
    
    # High delta-v mission
    suggestions_high = model.get_cost_optimization_suggestions(1500.0)
    print(f"High delta-v suggestions: {len(suggestions_high)} items")
    for key, value in suggestions_high.items():
        print(f"  {key}: {value}")
    
    # Low delta-v mission
    suggestions_low = model.get_cost_optimization_suggestions(10.0)
    print(f"Low delta-v suggestions: {len(suggestions_low)} items")
    
    print("✓ Optimization suggestion tests passed!")


if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_csv_validation()
        test_edge_cases()
        test_optimization_suggestions()
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()