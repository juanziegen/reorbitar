#!/usr/bin/env python3
"""
Demo script to test genetic CLI and debug delta-v calculation issues.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.genetic_cli import GeneticCLI
from src.genetic_algorithm import GAConfig, RouteConstraints
from src.tle_parser import TLEParser


def test_simple_optimization():
    """Test a simple optimization to debug delta-v calculation."""
    print("Testing simple genetic algorithm optimization...")
    
    # Load satellites
    parser = TLEParser()
    satellites = parser.parse_tle_file('leo_satellites.txt')
    
    if not satellites:
        print("Error: No satellites loaded")
        return
    
    print(f"Loaded {len(satellites)} satellites")
    
    # Show first few satellites for reference
    print("\nFirst 5 satellites:")
    for i, sat in enumerate(satellites[:5]):
        altitude = sat.semi_major_axis - 6378.137
        print(f"  {sat.catalog_number}: {sat.name} at {altitude:.1f} km, inc={sat.inclination:.1f}°")
    
    # Create simple constraints
    constraints = RouteConstraints(
        max_deltav_budget=5.0,  # 5 km/s budget
        max_mission_duration=86400,  # 1 day
        min_hops=2,
        max_hops=5
    )
    
    print(f"\nConstraints:")
    print(f"  Delta-v budget: {constraints.max_deltav_budget} km/s")
    print(f"  Mission duration: {constraints.max_mission_duration/3600:.1f} hours")
    print(f"  Hop range: {constraints.min_hops}-{constraints.max_hops}")
    
    # Create GA config with small population for testing
    ga_config = GAConfig(
        population_size=20,
        max_generations=10,
        mutation_rate=0.2,
        crossover_rate=0.8
    )
    
    print(f"\nGA Configuration:")
    print(f"  Population: {ga_config.population_size}")
    print(f"  Generations: {ga_config.max_generations}")
    
    # Initialize and run optimization
    from src.genetic_route_optimizer import GeneticRouteOptimizer
    
    optimizer = GeneticRouteOptimizer(satellites, ga_config)
    
    # Add progress callback
    def progress_callback(data):
        gen = data.get('generation', 0)
        best_fitness = data.get('best_fitness', 0)
        avg_fitness = data.get('average_fitness', 0)
        valid_count = data.get('valid_solutions', 0)
        print(f"  Gen {gen:2d}: Best={best_fitness:.6f}, Avg={avg_fitness:.6f}, Valid={valid_count}")
    
    optimizer.progress_callback = progress_callback
    
    print("\nRunning optimization...")
    result = optimizer.optimize_route(constraints)
    
    print(f"\nOptimization completed!")
    print(f"Status: {result.status.value}")
    print(f"Execution time: {result.execution_time:.2f} seconds")
    
    if result.best_route:
        route = result.best_route
        print(f"\nBest route found:")
        print(f"  Satellites: {route.satellite_sequence}")
        print(f"  Departure times: {[t/3600 for t in route.departure_times]} hours")
        print(f"  Total delta-v: {route.total_deltav:.6f} km/s")
        print(f"  Hop count: {route.hop_count}")
        print(f"  Mission duration: {route.mission_duration/3600:.2f} hours")
        print(f"  Valid: {route.is_valid}")
        
        if route.constraint_violations:
            print(f"  Violations: {route.constraint_violations}")
        
        # Test individual fitness evaluation
        print(f"\nTesting individual fitness evaluation...")
        from src.route_fitness_evaluator import RouteFitnessEvaluator
        from src.orbital_propagator import OrbitalPropagator
        
        propagator = OrbitalPropagator(satellites)
        evaluator = RouteFitnessEvaluator(satellites, propagator)
        
        try:
            fitness_result = evaluator.evaluate_route(route, constraints)
            print(f"  Fitness score: {fitness_result.fitness_score:.6f}")
            print(f"  Total delta-v: {fitness_result.total_deltav:.6f} km/s")
            print(f"  Hop count: {fitness_result.hop_count}")
            print(f"  Mission duration: {fitness_result.mission_duration/3600:.2f} hours")
            print(f"  Valid: {fitness_result.is_valid}")
            print(f"  Violations: {fitness_result.constraint_violations}")
            
        except Exception as e:
            print(f"  Error in fitness evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print("No valid route found!")
    
    return result


def test_manual_route():
    """Test manual route creation and evaluation."""
    print("\n" + "="*60)
    print("Testing manual route creation and evaluation...")
    
    # Load satellites
    parser = TLEParser()
    satellites = parser.parse_tle_file('leo_satellites.txt')
    
    if len(satellites) < 3:
        print("Need at least 3 satellites for testing")
        return
    
    # Create a simple 3-satellite route
    from src.genetic_algorithm import RouteChromosome
    
    # Use first 3 satellites
    test_route = RouteChromosome(
        satellite_sequence=[satellites[0].catalog_number, satellites[1].catalog_number, satellites[2].catalog_number],
        departure_times=[0.0, 3600.0, 7200.0]  # 1 hour intervals
    )
    
    print(f"Test route:")
    print(f"  Satellites: {test_route.satellite_sequence}")
    print(f"  Times: {[t/3600 for t in test_route.departure_times]} hours")
    
    # Create constraints
    constraints = RouteConstraints(
        max_deltav_budget=10.0,  # Large budget
        max_mission_duration=86400,  # 1 day
        min_hops=1,
        max_hops=10
    )
    
    # Evaluate the route
    from src.route_fitness_evaluator import RouteFitnessEvaluator
    from src.orbital_propagator import OrbitalPropagator
    
    propagator = OrbitalPropagator(satellites)
    evaluator = RouteFitnessEvaluator(satellites, propagator)
    
    try:
        print(f"\nEvaluating route...")
        fitness_result = evaluator.evaluate_route(test_route, constraints)
        
        print(f"Results:")
        print(f"  Fitness score: {fitness_result.fitness_score:.6f}")
        print(f"  Total delta-v: {fitness_result.total_deltav:.6f} km/s")
        print(f"  Hop count: {fitness_result.hop_count}")
        print(f"  Mission duration: {fitness_result.mission_duration/3600:.2f} hours")
        print(f"  Valid: {fitness_result.is_valid}")
        
        if fitness_result.constraint_violations:
            print(f"  Violations:")
            for violation in fitness_result.constraint_violations:
                print(f"    - {violation}")
        
        # Test individual transfer calculations
        print(f"\nTesting individual transfers...")
        from src.transfer_calculator import calculate_transfer_deltav
        
        for i in range(len(test_route.satellite_sequence) - 1):
            source_id = test_route.satellite_sequence[i]
            target_id = test_route.satellite_sequence[i + 1]
            
            source_sat = next(s for s in satellites if s.catalog_number == source_id)
            target_sat = next(s for s in satellites if s.catalog_number == target_id)
            
            try:
                transfer_result = calculate_transfer_deltav(source_sat, target_sat)
                print(f"  Transfer {i+1}: {source_id} -> {target_id}")
                print(f"    Delta-v: {transfer_result.total_deltav:.3f} m/s ({transfer_result.total_deltav/1000:.6f} km/s)")
                print(f"    Transfer time: {transfer_result.transfer_time:.1f} minutes")
                
            except Exception as e:
                print(f"  Transfer {i+1}: {source_id} -> {target_id} - Error: {e}")
        
    except Exception as e:
        print(f"Error evaluating route: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Genetic Algorithm CLI Debug Demo")
    print("="*60)
    
    try:
        # Test simple optimization
        result = test_simple_optimization()
        
        # Test manual route
        test_manual_route()
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()