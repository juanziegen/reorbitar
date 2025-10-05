"""
Simple test for RouteSimulator constraint handling implementation.
"""

import sys
import traceback
from datetime import datetime, timedelta

from debris_removal_service.models.satellite import Satellite, OrbitalElements
from debris_removal_service.models.service_request import (
    ServiceRequest, TimelineConstraints, BudgetConstraints, 
    ProcessingPreferences, ProcessingType
)
from debris_removal_service.services.route_simulator import RouteSimulator


def create_test_satellites():
    """Create test satellites with orbital elements."""
    satellites = []
    
    for i in range(3):
        orbital_elements = OrbitalElements(
            semi_major_axis=7000 + i * 100,  # km
            eccentricity=0.01 + i * 0.005,
            inclination=45 + i * 10,  # degrees
            raan=i * 30,  # degrees
            argument_of_perigee=i * 45,  # degrees
            mean_anomaly=i * 60,  # degrees
            mean_motion=15.0 - i * 0.5,  # revolutions per day
            epoch=datetime.now()
        )
        
        satellite = Satellite(
            id=str(i + 1),
            name=f"TestSat-{i + 1}",
            tle_line1=f"1 {i+1:05d}U 21001A   21001.00000000  .00000000  00000-0  00000-0 0  999{i}",
            tle_line2=f"2 {i+1:05d}  45.000{i:03d} 000.0000 0010000 000.0000 000.0000 15.0000000{i:05d}{i}",
            mass=500.0 + i * 100,
            material_composition={"aluminum": 0.6, "steel": 0.3, "other": 0.1},
            decommission_date=datetime.now() + timedelta(days=30),
            orbital_elements=orbital_elements
        )
        satellites.append(satellite)
    
    return satellites


def create_test_service_request():
    """Create test service request."""
    timeline_constraints = TimelineConstraints(
        earliest_start=datetime.now() + timedelta(days=1),
        latest_completion=datetime.now() + timedelta(days=30),
        preferred_duration=timedelta(days=10)
    )
    
    budget_constraints = BudgetConstraints(
        max_total_cost=50000.0,
        preferred_cost=40000.0
    )
    
    processing_preferences = ProcessingPreferences(
        preferred_processing_types=[ProcessingType.ISS_RECYCLING],
        processing_timeline=timedelta(days=5)
    )
    
    return ServiceRequest(
        client_id="test_client",
        satellites=["1", "2", "3"],
        timeline_requirements=timeline_constraints,
        budget_constraints=budget_constraints,
        processing_preferences=processing_preferences
    )


def test_constraint_analysis():
    """Test constraint analysis functionality."""
    print("Testing constraint analysis...")
    
    route_simulator = RouteSimulator()
    satellites = create_test_satellites()
    service_request = create_test_service_request()
    
    try:
        analysis = route_simulator._analyze_constraints(service_request, satellites)
        
        assert isinstance(analysis, dict), "Analysis should return a dictionary"
        assert 'feasible' in analysis, "Analysis should include feasible flag"
        assert 'constraint_violations' in analysis, "Analysis should include violations"
        assert 'warnings' in analysis, "Analysis should include warnings"
        assert 'recommendations' in analysis, "Analysis should include recommendations"
        
        print(f"✓ Constraint analysis completed: feasible={analysis['feasible']}")
        print(f"  Violations: {len(analysis['constraint_violations'])}")
        print(f"  Warnings: {len(analysis['warnings'])}")
        print(f"  Recommendations: {len(analysis['recommendations'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Constraint analysis failed: {str(e)}")
        traceback.print_exc()
        return False


def test_budget_analysis():
    """Test budget constraint analysis."""
    print("\nTesting budget constraint analysis...")
    
    route_simulator = RouteSimulator()
    satellites = create_test_satellites()
    service_request = create_test_service_request()
    
    try:
        analysis = route_simulator._analyze_budget_constraints(service_request, satellites)
        
        assert 'constraint_violations' in analysis
        assert 'warnings' in analysis
        assert 'recommendations' in analysis
        assert 'estimated_costs' in analysis
        
        estimated_costs = analysis['estimated_costs']
        assert 'collection_cost' in estimated_costs
        assert 'processing_cost' in estimated_costs
        assert 'total_cost' in estimated_costs
        assert 'budget_utilization' in estimated_costs
        
        print(f"✓ Budget analysis completed")
        print(f"  Estimated total cost: ${estimated_costs['total_cost']:.2f}")
        print(f"  Budget utilization: {estimated_costs['budget_utilization']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"✗ Budget analysis failed: {str(e)}")
        traceback.print_exc()
        return False


def test_timeline_analysis():
    """Test timeline constraint analysis."""
    print("\nTesting timeline constraint analysis...")
    
    route_simulator = RouteSimulator()
    satellites = create_test_satellites()
    service_request = create_test_service_request()
    
    try:
        analysis = route_simulator._analyze_timeline_constraints(service_request, satellites)
        
        assert 'constraint_violations' in analysis
        assert 'warnings' in analysis
        assert 'recommendations' in analysis
        assert 'estimated_timeline' in analysis
        
        estimated_timeline = analysis['estimated_timeline']
        assert 'estimated_duration_hours' in estimated_timeline
        assert 'available_duration_hours' in estimated_timeline
        assert 'timeline_utilization' in estimated_timeline
        assert 'is_urgent' in estimated_timeline
        
        print(f"✓ Timeline analysis completed")
        print(f"  Estimated duration: {estimated_timeline['estimated_duration_hours']:.1f} hours")
        print(f"  Available duration: {estimated_timeline['available_duration_hours']:.1f} hours")
        print(f"  Timeline utilization: {estimated_timeline['timeline_utilization']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"✗ Timeline analysis failed: {str(e)}")
        traceback.print_exc()
        return False


def test_orbital_analysis():
    """Test orbital feasibility analysis."""
    print("\nTesting orbital feasibility analysis...")
    
    route_simulator = RouteSimulator()
    satellites = create_test_satellites()
    
    try:
        analysis = route_simulator._analyze_orbital_feasibility(satellites)
        
        assert 'constraint_violations' in analysis
        assert 'warnings' in analysis
        assert 'recommendations' in analysis
        assert 'orbital_characteristics' in analysis
        
        orbital_chars = analysis['orbital_characteristics']
        assert 'satellites_with_elements' in orbital_chars
        assert 'inclination_range_deg' in orbital_chars
        assert 'altitude_range_km' in orbital_chars
        
        print(f"✓ Orbital analysis completed")
        print(f"  Satellites with elements: {orbital_chars['satellites_with_elements']}")
        print(f"  Inclination range: {orbital_chars['inclination_range_deg']:.1f}°")
        print(f"  Altitude range: {orbital_chars['altitude_range_km']:.1f} km")
        
        return True
        
    except Exception as e:
        print(f"✗ Orbital analysis failed: {str(e)}")
        traceback.print_exc()
        return False


def test_constraint_setup():
    """Test constraint setup for optimization."""
    print("\nTesting constraint setup...")
    
    route_simulator = RouteSimulator()
    satellites = create_test_satellites()
    service_request = create_test_service_request()
    
    try:
        ga_config, route_constraints = route_simulator._setup_constrained_optimization(
            service_request, satellites
        )
        
        assert ga_config is not None, "GA config should not be None"
        assert route_constraints is not None, "Route constraints should not be None"
        
        # Check route constraints
        assert 'max_deltav_budget_ms' in route_constraints
        assert 'max_mission_duration_seconds' in route_constraints
        assert 'max_budget_usd' in route_constraints
        assert 'budget_violation_penalty' in route_constraints
        assert 'timeline_violation_penalty' in route_constraints
        assert 'constraint_tolerance' in route_constraints
        
        print(f"✓ Constraint setup completed")
        print(f"  Max budget: ${route_constraints['max_budget_usd']:.2f}")
        print(f"  Max delta-v budget: {route_constraints['max_deltav_budget_ms']:.0f} m/s")
        print(f"  Max duration: {route_constraints['max_mission_duration_seconds']/3600:.1f} hours")
        
        return True
        
    except Exception as e:
        print(f"✗ Constraint setup failed: {str(e)}")
        traceback.print_exc()
        return False


def test_fitness_evaluation():
    """Test constrained fitness evaluation."""
    print("\nTesting constrained fitness evaluation...")
    
    route_simulator = RouteSimulator()
    satellites = create_test_satellites()
    service_request = create_test_service_request()
    
    try:
        # Create mock route and cost
        class MockRoute:
            def __init__(self):
                self.total_cost = 45000.0
                self.total_delta_v = 2000.0
                self.mission_duration = timedelta(days=15)
        
        class MockCost:
            def __init__(self):
                self.total_cost = 45000.0
        
        route = MockRoute()
        cost = MockCost()
        
        constraints = route_simulator._create_route_constraints_with_penalties(
            service_request, satellites
        )
        
        fitness = route_simulator._evaluate_constrained_fitness(
            route, cost, service_request, constraints
        )
        
        assert isinstance(fitness, float), "Fitness should be a float"
        
        print(f"✓ Fitness evaluation completed")
        print(f"  Fitness score: {fitness:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Fitness evaluation failed: {str(e)}")
        traceback.print_exc()
        return False


def test_constraint_satisfaction():
    """Test constraint satisfaction evaluation."""
    print("\nTesting constraint satisfaction evaluation...")
    
    route_simulator = RouteSimulator()
    satellites = create_test_satellites()
    service_request = create_test_service_request()
    
    try:
        # Create mock route within constraints
        class MockRoute:
            def __init__(self):
                self.total_cost = 40000.0  # Within budget
                self.total_delta_v = 1500.0  # Reasonable
                self.mission_duration = timedelta(days=20)  # Within timeline
        
        route = MockRoute()
        constraints = route_simulator._create_route_constraints_with_penalties(
            service_request, satellites
        )
        
        satisfaction = route_simulator._evaluate_constraint_satisfaction(
            route, service_request, constraints
        )
        
        assert isinstance(satisfaction, dict), "Satisfaction should be a dictionary"
        assert 'budget_satisfied' in satisfaction
        assert 'timeline_satisfied' in satisfaction
        assert 'deltav_satisfied' in satisfaction
        assert 'overall_satisfied' in satisfaction
        assert 'violations' in satisfaction
        
        print(f"✓ Constraint satisfaction evaluation completed")
        print(f"  Budget satisfied: {satisfaction['budget_satisfied']}")
        print(f"  Timeline satisfied: {satisfaction['timeline_satisfied']}")
        print(f"  Delta-v satisfied: {satisfaction['deltav_satisfied']}")
        print(f"  Overall satisfied: {satisfaction['overall_satisfied']}")
        print(f"  Violations: {len(satisfaction['violations'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Constraint satisfaction evaluation failed: {str(e)}")
        traceback.print_exc()
        return False


def test_full_optimization():
    """Test full constrained optimization workflow."""
    print("\nTesting full constrained optimization...")
    
    route_simulator = RouteSimulator()
    satellites = create_test_satellites()
    service_request = create_test_service_request()
    
    try:
        result = route_simulator.optimize_route_with_constraints(
            service_request, satellites
        )
        
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'optimization_id' in result
        assert 'success' in result
        
        print(f"✓ Full optimization completed")
        print(f"  Success: {result['success']}")
        print(f"  Optimization ID: {result['optimization_id']}")
        
        if result['success']:
            assert 'route' in result
            assert 'mission_cost' in result
            assert 'constraint_analysis' in result
            assert 'convergence_info' in result
            assert 'optimization_metadata' in result
            
            convergence_info = result['convergence_info']
            print(f"  Converged: {convergence_info['converged']}")
            print(f"  Generations run: {convergence_info['generations_run']}")
            print(f"  Convergence reason: {convergence_info['convergence_reason']}")
            
            metadata = result['optimization_metadata']
            print(f"  Execution time: {metadata['execution_time_seconds']:.2f} seconds")
        else:
            print(f"  Error/Message: {result.get('error', result.get('message', 'Unknown error'))}")
        
        return True
        
    except Exception as e:
        print(f"✗ Full optimization failed: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("RouteSimulator Constraint Handling Tests")
    print("=" * 60)
    
    tests = [
        test_constraint_analysis,
        test_budget_analysis,
        test_timeline_analysis,
        test_orbital_analysis,
        test_constraint_setup,
        test_fitness_evaluation,
        test_constraint_satisfaction,
        test_full_optimization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {str(e)}")
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())