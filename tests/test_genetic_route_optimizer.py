"""
Test for GeneticRouteOptimizer class

This test verifies the basic functionality of the genetic algorithm engine
including initialization, optimization execution, and result generation.
"""

import math
from src.genetic_route_optimizer import GeneticRouteOptimizer
from src.genetic_algorithm import GAConfig, RouteConstraints, OptimizationStatus
from src.tle_parser import SatelliteData


def create_test_satellites():
    """Create a small set of test satellites for optimization."""
    satellites = []
    
    # Create satellites with different orbital characteristics
    for i in range(10):
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


def test_genetic_route_optimizer_initialization():
    """Test GeneticRouteOptimizer initialization."""
    satellites = create_test_satellites()
    
    # Test with default config
    optimizer = GeneticRouteOptimizer(satellites)
    assert optimizer.satellites == satellites
    assert optimizer.config.population_size == 100
    assert optimizer.orbital_propagator is not None
    assert optimizer.fitness_evaluator is not None
    
    # Test with custom config
    custom_config = GAConfig(
        population_size=50,
        max_generations=100,
        mutation_rate=0.15
    )
    optimizer = GeneticRouteOptimizer(satellites, custom_config)
    assert optimizer.config.population_size == 50
    assert optimizer.config.max_generations == 100
    assert optimizer.config.mutation_rate == 0.15


def test_genetic_route_optimizer_empty_satellites():
    """Test that empty satellites list raises error."""
    try:
        GeneticRouteOptimizer([])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Satellites list cannot be empty" in str(e)


def test_genetic_route_optimizer_basic_optimization():
    """Test basic optimization run with small problem."""
    satellites = create_test_satellites()
    
    # Use small config for fast test
    config = GAConfig(
        population_size=20,
        max_generations=10,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elitism_count=2
    )
    
    optimizer = GeneticRouteOptimizer(satellites, config)
    
    # Define simple constraints
    constraints = RouteConstraints(
        max_deltav_budget=5.0,  # 5 km/s budget
        max_mission_duration=86400.0,  # 24 hours
        min_hops=2,
        max_hops=5
    )
    
    # Run optimization
    result = optimizer.optimize_route(constraints)
    
    # Verify result structure
    assert result is not None
    assert result.status in [OptimizationStatus.SUCCESS, OptimizationStatus.CONVERGED, OptimizationStatus.MAX_GENERATIONS]
    assert result.execution_time > 0
    assert result.optimization_stats is not None
    assert result.convergence_history is not None
    assert len(result.convergence_history) > 0
    
    # If successful, verify best route
    if result.success and result.best_route:
        assert len(result.best_route.satellite_sequence) >= constraints.min_hops + 1
        assert len(result.best_route.satellite_sequence) <= constraints.max_hops + 1
        assert len(result.best_route.departure_times) == len(result.best_route.satellite_sequence)
        
        # Check departure times are ascending
        times = result.best_route.departure_times
        for i in range(1, len(times)):
            assert times[i] > times[i-1], "Departure times should be ascending"


def test_genetic_route_optimizer_with_constraints():
    """Test optimization with specific start/end constraints."""
    satellites = create_test_satellites()
    
    config = GAConfig(
        population_size=15,
        max_generations=5,
        mutation_rate=0.2
    )
    
    optimizer = GeneticRouteOptimizer(satellites, config)
    
    # Define constraints with fixed start and end
    constraints = RouteConstraints(
        max_deltav_budget=10.0,
        max_mission_duration=172800.0,  # 48 hours
        start_satellite_id=25544,  # First satellite
        end_satellite_id=25548,    # Fifth satellite
        min_hops=3,
        max_hops=6
    )
    
    result = optimizer.optimize_route(constraints)
    
    # Verify result
    assert result is not None
    assert result.execution_time > 0
    
    # If successful, verify constraints are satisfied
    if result.success and result.best_route:
        route = result.best_route
        assert route.satellite_sequence[0] == constraints.start_satellite_id
        assert route.satellite_sequence[-1] == constraints.end_satellite_id
        assert len(route.satellite_sequence) >= constraints.min_hops + 1


def test_genetic_route_optimizer_progress_tracking():
    """Test progress tracking functionality."""
    satellites = create_test_satellites()
    
    config = GAConfig(
        population_size=10,
        max_generations=3
    )
    
    optimizer = GeneticRouteOptimizer(satellites, config)
    
    # Check initial progress
    progress = optimizer.get_optimization_progress()
    assert progress['current_generation'] == 0
    assert progress['best_fitness'] == 0.0
    
    # Run short optimization
    constraints = RouteConstraints(
        max_deltav_budget=3.0,
        max_mission_duration=43200.0,  # 12 hours
        min_hops=1,
        max_hops=3
    )
    
    result = optimizer.optimize_route(constraints)
    
    # Check final progress
    final_progress = optimizer.get_optimization_progress()
    assert final_progress['current_generation'] > 0
    assert 'population_diversity' in final_progress
    assert 'constraint_satisfaction_rate' in final_progress


def test_genetic_route_optimizer_convergence_detection():
    """Test convergence detection with high convergence threshold."""
    satellites = create_test_satellites()
    
    # Config designed to converge quickly
    config = GAConfig(
        population_size=10,
        max_generations=50,
        convergence_threshold=1e-3,  # Loose convergence
        max_stagnant_generations=5   # Quick stagnation detection
    )
    
    optimizer = GeneticRouteOptimizer(satellites, config)
    
    constraints = RouteConstraints(
        max_deltav_budget=2.0,  # Tight budget to limit solutions
        max_mission_duration=21600.0,  # 6 hours
        min_hops=1,
        max_hops=2  # Very short routes
    )
    
    result = optimizer.optimize_route(constraints)
    
    # Should either converge or hit max generations
    assert result.status in [
        OptimizationStatus.CONVERGED, 
        OptimizationStatus.MAX_GENERATIONS,
        OptimizationStatus.SUCCESS
    ]
    
    # Should have some convergence history
    assert len(result.convergence_history) > 0
    
    # Check that statistics are reasonable
    stats = result.optimization_stats
    assert stats.generations_completed > 0
    assert stats.generations_completed <= config.max_generations


def test_genetic_route_optimizer_adaptive_parameters():
    """Test adaptive parameter adjustment during optimization."""
    satellites = create_test_satellites()
    
    config = GAConfig(
        population_size=15,
        max_generations=8,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    optimizer = GeneticRouteOptimizer(satellites, config)
    
    # Store initial parameters
    initial_mutation_rate = optimizer.adaptive_mutation_rate
    initial_crossover_rate = optimizer.adaptive_crossover_rate
    
    constraints = RouteConstraints(
        max_deltav_budget=4.0,
        max_mission_duration=64800.0,  # 18 hours
        min_hops=2,
        max_hops=4
    )
    
    result = optimizer.optimize_route(constraints)
    
    # Parameters may have been adapted during optimization
    final_progress = optimizer.get_optimization_progress()
    assert 'adaptive_mutation_rate' in final_progress
    assert 'adaptive_crossover_rate' in final_progress
    
    # Adaptive rates should be within reasonable bounds
    assert 0.01 <= final_progress['adaptive_mutation_rate'] <= 0.5
    assert 0.1 <= final_progress['adaptive_crossover_rate'] <= 1.0


if __name__ == "__main__":
    # Run basic tests
    test_genetic_route_optimizer_initialization()
    test_genetic_route_optimizer_basic_optimization()
    test_genetic_route_optimizer_with_constraints()
    test_genetic_route_optimizer_progress_tracking()
    
    print("All GeneticRouteOptimizer tests passed!")