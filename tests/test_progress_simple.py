"""
Simple test for progress tracking functionality without pytest dependency.
"""

import sys
import time
from unittest.mock import Mock

# Add src to path
sys.path.append('src')

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
            line1=f"1 {25544 + i:05d}U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
            line2=f"2 {25544 + i:05d}  51.6416 211.0220 0006703  69.9751 290.2127 15.72125391563537"
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
        print(f"Generation {progress_info['generation']}: "
              f"Best={progress_info['best_fitness']:.3f}, "
              f"Avg={progress_info['average_fitness']:.3f}, "
              f"Diversity={progress_info['population_diversity']:.3f}")
    
    optimizer.set_progress_callback(callback)
    
    try:
        result = optimizer.optimize_route(constraints)
        
        # Verify callback was called
        assert len(progress_updates) > 0, "Progress callback should be called"
        print(f"✓ Progress callback called {len(progress_updates)} times")
        
        # Verify progress information structure
        for update in progress_updates:
            required_fields = [
                'generation', 'best_fitness', 'average_fitness', 
                'population_diversity', 'progress_percentage'
            ]
            for field in required_fields:
                assert field in update, f"Missing field: {field}"
        
        print("✓ Progress callback functionality working correctly")
        
    except Exception as e:
        print(f"✗ Progress callback test failed: {e}")
        return False
    
    return True


def test_detailed_logging():
    """Test detailed logging functionality."""
    print("\nTesting detailed logging functionality...")
    
    satellites = create_test_satellites()
    config = GAConfig(
        population_size=8,
        max_generations=3
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
        
    except Exception as e:
        print(f"✗ Detailed logging test failed: {e}")
        return False
    
    return True


def test_statistics_compilation():
    """Test detailed statistics compilation."""
    print("\nTesting statistics compilation...")
    
    satellites = create_test_satellites()
    config = GAConfig(
        population_size=10,
        max_generations=8
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
        
        for section in required_sections:
            assert section in detailed_stats, f"Missing section: {section}"
        
        print("✓ All required statistics sections present")
        
        # Verify some key metrics
        summary = detailed_stats['optimization_summary']
        assert summary['total_generations'] > 0, "Should have completed some generations"
        
        fitness_stats = detailed_stats['fitness_statistics']
        assert len(fitness_stats['best_fitness_history']) > 0, "Should have fitness history"
        
        print("✓ Statistics compilation working correctly")
        print(f"  - Total generations: {summary['total_generations']}")
        print(f"  - Final fitness: {summary['final_fitness']:.3f}")
        print(f"  - Fitness improvement: {summary['fitness_improvement']:.3f}")
        
    except Exception as e:
        print(f"✗ Statistics compilation test failed: {e}")
        return False
    
    return True


def test_convergence_metrics():
    """Test convergence metrics tracking."""
    print("\nTesting convergence metrics tracking...")
    
    satellites = create_test_satellites()
    config = GAConfig(
        population_size=8,
        max_generations=6
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
        
        print("✓ Convergence metrics tracked")
        print(f"  - Fitness history entries: {len(optimizer.fitness_history)}")
        print(f"  - Diversity history entries: {len(optimizer.diversity_history)}")
        
        # Verify convergence metrics structure
        metrics = optimizer.convergence_metrics
        required_metrics = ['improvement_rate', 'stagnation_periods', 'diversity_trend']
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
        
        print("✓ Convergence metrics structure correct")
        print(f"  - Improvement rate: {metrics['improvement_rate']:.3f}")
        print(f"  - Diversity trend: {metrics['diversity_trend']}")
        
    except Exception as e:
        print(f"✗ Convergence metrics test failed: {e}")
        return False
    
    return True


def test_utility_methods():
    """Test utility methods for statistics calculation."""
    print("\nTesting utility methods...")
    
    satellites = create_test_satellites()
    config = GAConfig(population_size=5, max_generations=3)
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
        
        print("✓ Utility methods working correctly")
        
    except Exception as e:
        print(f"✗ Utility methods test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("Running progress tracking tests...\n")
    
    tests = [
        test_progress_callback,
        test_detailed_logging,
        test_statistics_compilation,
        test_convergence_metrics,
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
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All progress tracking tests passed!")
        return True
    else:
        print("✗ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)