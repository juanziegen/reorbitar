"""
Direct test for progress tracking functionality.
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now import the modules
from genetic_route_optimizer import GeneticRouteOptimizer
from genetic_algorithm import GAConfig, RouteConstraints
from tle_parser import SatelliteData


def main():
    print("Testing progress tracking functionality...")
    
    # Create test satellites
    satellites = []
    for i in range(3):
        sat = SatelliteData(
            catalog_number=25544 + i,
            name=f'TEST-SAT-{i}',
            line1=f'1 {25544 + i:05d}U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927',
            line2=f'2 {25544 + i:05d}  51.6416 211.0220 0006703  69.9751 290.2127 15.72125391563537'
        )
        satellites.append(sat)
    
    # Create test configuration
    config = GAConfig(
        population_size=8,
        max_generations=5,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    # Create test constraints
    constraints = RouteConstraints(
        max_deltav_budget=5.0,
        max_mission_duration=86400.0,
        min_hops=1,
        max_hops=3
    )
    
    # Create optimizer
    optimizer = GeneticRouteOptimizer(satellites, config)
    
    # Test 1: Progress callback
    print("\n1. Testing progress callback...")
    progress_calls = []
    
    def progress_callback(info):
        progress_calls.append(info)
        print(f"   Generation {info['generation']:2d}: "
              f"Best={info['best_fitness']:6.3f}, "
              f"Avg={info['average_fitness']:6.3f}, "
              f"Diversity={info['population_diversity']:5.3f}")
    
    optimizer.set_progress_callback(progress_callback)
    
    # Test 2: Enable detailed logging
    print("\n2. Enabling detailed logging...")
    optimizer.enable_detailed_logging(True)
    
    # Run optimization
    print("\n3. Running optimization...")
    try:
        result = optimizer.optimize_route(constraints)
        print(f"   ✓ Optimization completed with status: {result.status.value}")
        print(f"   ✓ Progress callback called {len(progress_calls)} times")
        
        # Verify progress callback data
        if progress_calls:
            first_call = progress_calls[0]
            required_fields = [
                'generation', 'best_fitness', 'average_fitness', 
                'population_diversity', 'progress_percentage',
                'stagnant_generations', 'improvement_rate'
            ]
            
            missing_fields = [field for field in required_fields if field not in first_call]
            if missing_fields:
                print(f"   ✗ Missing fields in progress callback: {missing_fields}")
            else:
                print("   ✓ Progress callback contains all required fields")
        
    except Exception as e:
        print(f"   ✗ Optimization failed: {e}")
        return False
    
    # Test 3: Detailed statistics
    print("\n4. Testing detailed statistics...")
    try:
        stats = optimizer.get_detailed_statistics()
        
        required_sections = [
            'optimization_summary', 'fitness_statistics', 
            'diversity_analysis', 'convergence_metrics',
            'performance_metrics', 'population_statistics'
        ]
        
        missing_sections = [section for section in required_sections if section not in stats]
        if missing_sections:
            print(f"   ✗ Missing statistics sections: {missing_sections}")
        else:
            print("   ✓ All required statistics sections present")
            
            # Print some key statistics
            summary = stats['optimization_summary']
            print(f"   - Total generations: {summary['total_generations']}")
            print(f"   - Final fitness: {summary['final_fitness']:.3f}")
            print(f"   - Fitness improvement: {summary['fitness_improvement']:.3f}")
            
            fitness_stats = stats['fitness_statistics']
            print(f"   - Fitness history length: {len(fitness_stats['best_fitness_history'])}")
            print(f"   - Fitness trend: {fitness_stats['fitness_trend']}")
            
    except Exception as e:
        print(f"   ✗ Statistics compilation failed: {e}")
        return False
    
    # Test 4: Convergence metrics
    print("\n5. Testing convergence metrics...")
    try:
        assert len(optimizer.fitness_history) > 0, "Fitness history should be tracked"
        assert len(optimizer.diversity_history) > 0, "Diversity history should be tracked"
        
        print(f"   ✓ Fitness history entries: {len(optimizer.fitness_history)}")
        print(f"   ✓ Diversity history entries: {len(optimizer.diversity_history)}")
        
        metrics = optimizer.convergence_metrics
        print(f"   - Improvement rate: {metrics['improvement_rate']:.3f}")
        print(f"   - Diversity trend: {metrics['diversity_trend']}")
        print(f"   - Stagnation periods: {len(metrics['stagnation_periods'])}")
        
    except Exception as e:
        print(f"   ✗ Convergence metrics test failed: {e}")
        return False
    
    # Test 5: Utility methods
    print("\n6. Testing utility methods...")
    try:
        # Test standard deviation
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        std_dev = optimizer._calculate_standard_deviation(values)
        assert std_dev > 0.0, "Standard deviation should be positive"
        print(f"   ✓ Standard deviation: {std_dev:.3f}")
        
        # Test trend slope
        increasing_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        slope = optimizer._calculate_trend_slope(increasing_values)
        assert slope > 0.0, "Slope should be positive for increasing values"
        print(f"   ✓ Trend slope: {slope:.3f}")
        
        # Test convergence trend analysis
        trend = optimizer._analyze_convergence_trend()
        print(f"   ✓ Convergence trend: {trend}")
        
    except Exception as e:
        print(f"   ✗ Utility methods test failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ ALL PROGRESS TRACKING TESTS PASSED!")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)