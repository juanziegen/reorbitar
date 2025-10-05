"""
Test Progress Tracking and Statistics for Genetic Route Optimizer

This test verifies the enhanced progress tracking, statistics calculation,
and convergence monitoring functionality.
"""

import sys
import os
sys.path.insert(0, 'src')

from genetic_route_optimizer import GeneticRouteOptimizer
from genetic_algorithm import GAConfig, RouteConstraints
from tle_parser import SatelliteData


def create_test_satellites():
    """Create sample satellite data for testing."""
    satellites = []
    for i in range(5):
        sat = SatelliteData(
            catalog_number=25544 + i,
            name=f"TEST-SAT-{i}",
            epoch="2024-01-01T00:00:00",
            mean_motion=15.5 + i * 0.1,  # Different orbital periods
            eccentricity=0.001 + i * 0.0001,
            inclination=51.6 + i * 2.0,  # Different inclinations
            raan=45.0 + i * 30.0,
            arg_perigee=90.0 + i * 10.0,
            mean_anomaly=0.0 + i * 36.0,
            semi_major_axis=6800.0 + i * 50.0,  # Different altitudes
            orbital_period=90.0 + i * 2.0
        )
        satellites.append(sat)
    return satellites


def test_progress_callback():
    """Test progress callback functionality."""
    print("Testing progress callback functionality...")
    
    satellites = create_test_satellites()
    config = GAConfig(
        population_size=10,
        max_generations=5,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    constraints = RouteConstraints(
        max_deltav_budget=5.0,
        max_mission_duration=86400.0,
        min_hops=2,
        max_hops=4
    )
    
    optimizer = GeneticRouteOptimizer(satellites, config)
    
    # Track progress updates
    progress_updates = []
    def callback(progress_info):
        progress_updates.append(progress_info.copy())
        print(f"  Generation {progress_info['generation']:2d}: "
              f"Best={progress_info['best_fitness']:6.3f}, "
              f"Avg={progress_info['average_fitness']:6.3f}, "
              f"Diversity={progress_info['population_diversity']:5.3f}, "
              f"Progress={progress_info['progress_percentage']:5.1f}%")
    
    optimizer.set_progress_callback(callback)
    
    try:
        result = optimizer.optimize_route(constraints)
        
        # Verify callback was called
        assert len(progress_updates) > 0, "Progress callback should be called"
        print(f"✓ Progress callback called {len(progress_updates)} times")
        
        # Verify progress information structure
        required_fields = [
            'generation', 'best_fitness', 'average_fitness', 
            'population_diversity', 'progress_percentage',
            'stagnant_generations', 'improvement_rate',
            'convergence_trend', 'adaptive_mutation_rate',
            'best_route_hops', 'best_route_deltav'
        ]
        
        for update in progress_updates:
            missing_fields = [field for field in required_fields if field not in update]
            if missing_fields:
                print(f"✗ Missing fields in progress update: {missing_fields}")
                return False
        
        print("✓ All required fields present in progress updates")
        return True
        
    except Exception as e:
        print(f"✗ Progress callback test failed: {e}")
        return False


def test_detailed_statistics():
    """Test detailed statistics compilation."""
    print("\nTesting detailed statistics compilation...")
    
    satellites = create_test_satellites()
    config = GAConfig(
        population_size=8,
        max_generations=6
    )
    constraints = RouteConstraints(
        max_deltav_budget=5.0,
        max_mission_duration=86400.0,
        min_hops=2,
        max_hops=4
    )
    
    optimizer = GeneticRouteOptimizer(satellites, config)
    
    try:
        result = optimizer.optimize_route(constraints)
        
        # Get detailed statistics
        detailed_stats = optimizer.get_detailed_statistics()
        
        # Verify main sections exist
        required_sections = [
            'optimization_summary', 'fitness_statistics', 
            'diversity_analysis', 'convergence_metrics',
            'performance_metrics', 'population_statistics'
        ]
        
        missing_sections = [section for section in required_sections if section not in detailed_stats]
        if missing_sections:
            print(f"✗ Missing statistics sections: {missing_sections}")
            return False
        
        print("✓ All required statistics sections present")
        
        # Verify key metrics
        summary = detailed_stats['optimization_summary']
        assert summary['total_generations'] > 0, "Should have completed some generations"
        
        fitness_stats = detailed_stats['fitness_statistics']
        assert len(fitness_stats['best_fitness_history']) > 0, "Should have fitness history"
        
        diversity_analysis = detailed_stats['diversity_analysis']
        assert len(diversity_analysis['diversity_history']) > 0, "Should have diversity history"
        
        print(f"✓ Statistics compilation successful")
        print(f"  - Total generations: {summary['total_generations']}")
        print(f"  - Final fitness: {summary['final_fitness']:.3f}")
        print(f"  - Fitness improvement: {summary['fitness_improvement']:.3f}")
        print(f"  - Average diversity: {diversity_analysis['average_diversity']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Statistics compilation test failed: {e}")
        return False


def test_convergence_metrics():
    """Test convergence metrics tracking."""
    print("\nTesting convergence metrics tracking...")
    
    satellites = create_test_satellites()
    config = GAConfig(
        population_size=8,
        max_generations=8
    )
    constraints = RouteConstraints(
        max_deltav_budget=5.0,
        max_mission_duration=86400.0,
        min_hops=2,
        max_hops=3
    )
    
    optimizer = GeneticRouteOptimizer(satellites, config)
    
    try:
        result = optimizer.optimize_route(constraints)
        
        # Verify convergence metrics are tracked
        assert len(optimizer.fitness_history) > 0, "Fitness history should be tracked"
        assert len(optimizer.diversity_history) > 0, "Diversity history should be tracked"
        
        print(f"✓ Convergence metrics tracked")
        print(f"  - Fitness history entries: {len(optimizer.fitness_history)}")
        print(f"  - Diversity history entries: {len(optimizer.diversity_history)}")
        
        # Verify convergence metrics structure
        metrics = optimizer.convergence_metrics
        required_metrics = ['improvement_rate', 'stagnation_periods', 'diversity_trend']
        missing_metrics = [metric for metric in required_metrics if metric not in metrics]
        if missing_metrics:
            print(f"✗ Missing convergence metrics: {missing_metrics}")
            return False
        
        print("✓ Convergence metrics structure correct")
        print(f"  - Improvement rate: {metrics['improvement_rate']:.3f}")
        print(f"  - Diversity trend: {metrics['diversity_trend']}")
        print(f"  - Stagnation periods: {len(metrics['stagnation_periods'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Convergence metrics test failed: {e}")
        return False


def test_detailed_logging():
    """Test detailed logging functionality."""
    print("\nTesting detailed logging functionality...")
    
    satellites = create_test_satellites()
    config = GAConfig(
        population_size=6,
        max_generations=4
    )
    constraints = RouteConstraints(
        max_deltav_budget=5.0,
        max_mission_duration=86400.0,
        min_hops=2,
        max_hops=3
    )
    
    optimizer = GeneticRouteOptimizer(satellites, config)
    optimizer.enable_detailed_logging(True)
    
    try:
        result = optimizer.optimize_route(constraints)
        print("✓ Detailed logging enabled and optimization completed")
        return True
        
    except Exception as e:
        print(f"✗ Detailed logging test failed: {e}")
        return False


def test_utility_methods():
    """Test utility methods for statistics calculation."""
    print("\nTesting utility methods...")
    
    satellites = create_test_satellites()
    config = GAConfig(population_size=10, max_generations=3, elitism_count=2)
    optimizer = GeneticRouteOptimizer(satellites, config)
    
    try:
        # Test standard deviation calculation
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        std_dev = optimizer._calculate_standard_deviation(values)
        assert std_dev > 0.0, "Standard deviation should be positive"
        print(f"✓ Standard deviation calculation: {std_dev:.3f}")
        
        # Test trend slope calculation
        increasing_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        slope = optimizer._calculate_trend_slope(increasing_values)
        assert slope > 0.0, "Slope should be positive for increasing values"
        print(f"✓ Trend slope calculation: {slope:.3f}")
        
        # Test with edge cases
        std_dev_single = optimizer._calculate_standard_deviation([1.0])
        assert std_dev_single == 0.0, "Single value should have zero std dev"
        
        slope_single = optimizer._calculate_trend_slope([1.0])
        assert slope_single == 0.0, "Single value should have zero slope"
        
        print("✓ Edge cases handled correctly")
        
        # Test convergence trend analysis
        trend = optimizer._analyze_convergence_trend()
        assert trend in ['insufficient_data', 'improving', 'stable', 'stagnating', 'degrading'], f"Invalid trend: {trend}"
        print(f"✓ Convergence trend analysis: {trend}")
        
        return True
        
    except Exception as e:
        print(f"✗ Utility methods test failed: {e}")
        return False


def main():
    """Run all progress tracking tests."""
    print("Running Progress Tracking and Statistics Tests")
    print("=" * 60)
    
    tests = [
        test_progress_callback,
        test_detailed_statistics,
        test_convergence_metrics,
        test_detailed_logging,
        test_utility_methods
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ ALL PROGRESS TRACKING TESTS PASSED!")
        return True
    else:
        print("✗ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)