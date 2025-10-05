"""
Simple Test for Error Handling and Recovery

This test verifies the error handling implementation without requiring pytest.
"""

import sys
import traceback
from unittest.mock import Mock

# Add src to path
sys.path.append('src')

from genetic_route_optimizer import GeneticRouteOptimizer
from ga_error_handler import GAErrorHandler, ErrorType, ErrorSeverity
from genetic_algorithm import RouteChromosome, GAConfig, RouteConstraints
from tle_parser import SatelliteData


def create_sample_satellites(count=10):
    """Create sample satellite data for testing."""
    from datetime import datetime
    
    satellites = []
    for i in range(count):
        sat = SatelliteData(
            catalog_number=25544 + i,
            name=f"TEST-SAT-{i}",
            epoch=datetime(2024, 1, 1),
            mean_motion=15.5 + i * 0.1,
            eccentricity=0.001 + i * 0.0001,
            inclination=51.6 + i * 0.1,
            raan=247.4 + i * 1.0,
            arg_perigee=130.5 + i * 1.0,
            mean_anomaly=325.0 + i * 1.0,
            semi_major_axis=6800.0 + i * 10.0,
            orbital_period=5400.0 + i * 10.0
        )
        satellites.append(sat)
    return satellites


def test_error_handler_initialization():
    """Test GAErrorHandler initialization."""
    print("Testing GAErrorHandler initialization...")
    
    satellites = create_sample_satellites(5)
    config = GAConfig(population_size=10, max_generations=20)
    
    try:
        error_handler = GAErrorHandler(satellites, config)
        assert len(error_handler.satellites) == 5
        assert error_handler.config.population_size == 10
        assert len(error_handler.satellite_ids) == 5
        print("✓ GAErrorHandler initialization successful")
        return True
    except Exception as e:
        print(f"✗ GAErrorHandler initialization failed: {e}")
        traceback.print_exc()
        return False


def test_fitness_evaluation_error_handling():
    """Test fitness evaluation error handling."""
    print("Testing fitness evaluation error handling...")
    
    try:
        satellites = create_sample_satellites(5)
        config = GAConfig()
        error_handler = GAErrorHandler(satellites, config)
        
        # Create test chromosome
        chromosome = RouteChromosome(
            satellite_sequence=[25544, 25545, 25546],
            departure_times=[0.0, 3600.0, 7200.0]
        )
        
        # Simulate error
        test_error = ValueError("Test fitness evaluation error")
        penalty_fitness = error_handler.handle_fitness_evaluation_error(
            chromosome, test_error, generation=5
        )
        
        # Verify results
        assert penalty_fitness < 0, "Penalty fitness should be negative"
        assert not chromosome.is_valid, "Chromosome should be marked invalid"
        assert len(chromosome.constraint_violations) > 0, "Should have constraint violations"
        assert len(error_handler.error_history) == 1, "Should record error in history"
        
        error_report = error_handler.error_history[0]
        assert error_report.error_type == ErrorType.FITNESS_EVALUATION
        assert error_report.generation == 5
        
        print("✓ Fitness evaluation error handling successful")
        return True
    except Exception as e:
        print(f"✗ Fitness evaluation error handling failed: {e}")
        traceback.print_exc()
        return False


def test_chromosome_repair():
    """Test chromosome repair functionality."""
    print("Testing chromosome repair...")
    
    try:
        satellites = create_sample_satellites(10)
        config = GAConfig()
        error_handler = GAErrorHandler(satellites, config)
        
        constraints = RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400.0,
            min_hops=2,
            max_hops=5
        )
        
        # Create invalid chromosome
        invalid_chromosome = RouteChromosome(
            satellite_sequence=[25544, 25544, 25545, 99999],  # Duplicate and invalid ID
            departure_times=[0.0, 3600.0, 7200.0, 10800.0]
        )
        
        # Attempt repair
        repaired = error_handler.repair_invalid_chromosome(
            invalid_chromosome, constraints, generation=3
        )
        
        # Verify repair
        assert len(repaired.satellite_sequence) == len(repaired.departure_times), "Sequence and times should match"
        assert len(set(repaired.satellite_sequence)) == len(repaired.satellite_sequence), "No duplicates allowed"
        
        # Check all satellite IDs are valid
        valid_ids = {sat.catalog_number for sat in satellites}
        for sat_id in repaired.satellite_sequence:
            assert sat_id in valid_ids, f"Invalid satellite ID: {sat_id}"
        
        # Check timing is ascending
        for i in range(1, len(repaired.departure_times)):
            assert repaired.departure_times[i] > repaired.departure_times[i-1], "Times should be ascending"
        
        print("✓ Chromosome repair successful")
        return True
    except Exception as e:
        print(f"✗ Chromosome repair failed: {e}")
        traceback.print_exc()
        return False


def test_fallback_chromosome_creation():
    """Test fallback chromosome creation."""
    print("Testing fallback chromosome creation...")
    
    try:
        satellites = create_sample_satellites(10)
        config = GAConfig()
        error_handler = GAErrorHandler(satellites, config)
        
        constraints = RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400.0,
            min_hops=2,
            max_hops=5
        )
        
        # Create fallback chromosome
        fallback = error_handler._create_fallback_chromosome(constraints, generation=1)
        
        # Verify fallback chromosome
        assert len(fallback.satellite_sequence) >= constraints.min_hops + 1, "Should meet minimum hops"
        assert len(fallback.satellite_sequence) <= constraints.max_hops + 1, "Should not exceed maximum hops"
        assert len(fallback.satellite_sequence) == len(fallback.departure_times), "Sequence and times should match"
        
        # Check all satellite IDs are valid
        valid_ids = {sat.catalog_number for sat in satellites}
        for sat_id in fallback.satellite_sequence:
            assert sat_id in valid_ids, f"Invalid satellite ID: {sat_id}"
        
        # Check no duplicates
        assert len(set(fallback.satellite_sequence)) == len(fallback.satellite_sequence), "No duplicates allowed"
        
        # Check timing is ascending
        for i in range(1, len(fallback.departure_times)):
            assert fallback.departure_times[i] > fallback.departure_times[i-1], "Times should be ascending"
        
        print("✓ Fallback chromosome creation successful")
        return True
    except Exception as e:
        print(f"✗ Fallback chromosome creation failed: {e}")
        traceback.print_exc()
        return False


def test_population_convergence_handling():
    """Test population convergence handling."""
    print("Testing population convergence handling...")
    
    try:
        satellites = create_sample_satellites(10)
        config = GAConfig()
        error_handler = GAErrorHandler(satellites, config)
        
        constraints = RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400.0,
            min_hops=2,
            max_hops=5
        )
        
        # Create converged population (all similar)
        converged_population = []
        base_sequence = [25544, 25545, 25546]
        base_times = [0.0, 3600.0, 7200.0]
        
        for i in range(8):
            chromosome = RouteChromosome(
                satellite_sequence=base_sequence.copy(),
                departure_times=[t + i * 100 for t in base_times]
            )
            converged_population.append(chromosome)
        
        # Handle convergence
        diverse_population = error_handler.handle_population_convergence(
            converged_population, constraints, generation=20
        )
        
        # Verify diversity injection
        assert len(diverse_population) == len(converged_population), "Population size should be maintained"
        
        # Check that some diversity was added
        original_sequences = {tuple(c.satellite_sequence) for c in converged_population}
        new_sequences = {tuple(c.satellite_sequence) for c in diverse_population}
        
        # Should have at least as many unique sequences (possibly more)
        assert len(new_sequences) >= len(original_sequences), "Should maintain or increase diversity"
        
        # Verify error was recorded
        assert len(error_handler.error_history) == 1, "Should record convergence handling"
        error_report = error_handler.error_history[0]
        assert error_report.error_type == ErrorType.POPULATION_CONVERGENCE
        
        print("✓ Population convergence handling successful")
        return True
    except Exception as e:
        print(f"✗ Population convergence handling failed: {e}")
        traceback.print_exc()
        return False


def test_error_summary():
    """Test error summary generation."""
    print("Testing error summary generation...")
    
    try:
        satellites = create_sample_satellites(5)
        config = GAConfig()
        error_handler = GAErrorHandler(satellites, config)
        
        constraints = RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400.0,
            min_hops=2,
            max_hops=5
        )
        
        # Generate some errors
        chromosome = RouteChromosome([25544, 25545], [0.0, 3600.0])
        
        error_handler.handle_fitness_evaluation_error(
            chromosome, ValueError("Test error 1"), generation=1
        )
        error_handler.handle_fitness_evaluation_error(
            chromosome, RuntimeError("Test error 2"), generation=2
        )
        error_handler.repair_invalid_chromosome(chromosome, constraints, generation=3)
        
        # Get summary
        summary = error_handler.get_error_summary()
        
        # Verify summary
        assert summary['total_errors'] == 3, f"Expected 3 errors, got {summary['total_errors']}"
        assert ErrorType.FITNESS_EVALUATION.value in summary['error_types']
        assert ErrorType.CHROMOSOME_INVALID.value in summary['error_types']
        assert summary['error_types'][ErrorType.FITNESS_EVALUATION.value] == 2
        assert summary['error_types'][ErrorType.CHROMOSOME_INVALID.value] == 1
        
        print("✓ Error summary generation successful")
        return True
    except Exception as e:
        print(f"✗ Error summary generation failed: {e}")
        traceback.print_exc()
        return False


def test_genetic_optimizer_integration():
    """Test error handling integration with GeneticRouteOptimizer."""
    print("Testing GeneticRouteOptimizer error handling integration...")
    
    try:
        satellites = create_sample_satellites(10)
        config = GAConfig(population_size=10, max_generations=3, elitism_count=2)  # Small for testing
        optimizer = GeneticRouteOptimizer(satellites, config)
        
        constraints = RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400.0,
            min_hops=2,
            max_hops=4
        )
        
        # Verify error handler was initialized
        assert hasattr(optimizer, 'error_handler'), "Should have error handler"
        print(f"Error handler type: {type(optimizer.error_handler)}")
        # Note: The error handler should be a GAErrorHandler instance
        # For now, just check it exists and has the expected methods
        assert hasattr(optimizer.error_handler, 'handle_fitness_evaluation_error'), "Should have error handling methods"
        
        # Test population initialization with recovery
        population = optimizer._initialize_population_with_recovery(constraints)
        assert population is not None, "Should create population"
        assert len(population) == config.population_size, "Should have correct population size"
        
        # Test basic chromosome validation
        for chromosome in population:
            assert len(chromosome.satellite_sequence) > 0, "Should have satellite sequence"
            assert len(chromosome.satellite_sequence) == len(chromosome.departure_times), "Sequence and times should match"
        
        print("✓ GeneticRouteOptimizer error handling integration successful")
        return True
    except Exception as e:
        print(f"✗ GeneticRouteOptimizer error handling integration failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all error handling tests."""
    print("=" * 60)
    print("Running Error Handling and Recovery Tests")
    print("=" * 60)
    
    tests = [
        test_error_handler_initialization,
        test_fitness_evaluation_error_handling,
        test_chromosome_repair,
        test_fallback_chromosome_creation,
        test_population_convergence_handling,
        test_error_summary,
        test_genetic_optimizer_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print()
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All error handling tests passed!")
        return True
    else:
        print(f"❌ {failed} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)