"""
Test suite for genetic algorithm core data structures.

Tests the basic functionality and validation of the core data structures
used in the genetic algorithm route optimization system.
"""

from src.genetic_algorithm import (
    RouteChromosome, GAConfig, RouteConstraints, OptimizationResult,
    OptimizationStats, GenerationStats, OptimizationStatus, FitnessResult,
    ConstraintResult
)


def test_route_chromosome_creation():
    """Test RouteChromosome creation and basic properties."""
    chromosome = RouteChromosome(
        satellite_sequence=[1, 2, 3, 4],
        departure_times=[0.0, 100.0, 200.0, 300.0],
        total_deltav=5.2
    )
    
    assert chromosome.hop_count == 3
    assert chromosome.mission_duration == 300.0
    assert chromosome.is_valid == True
    assert len(chromosome.constraint_violations) == 0


def test_route_chromosome_validation():
    """Test RouteChromosome validation for mismatched lengths."""
    chromosome = RouteChromosome(
        satellite_sequence=[1, 2, 3],
        departure_times=[0.0, 100.0]  # Mismatched length
    )
    
    assert chromosome.is_valid == False
    assert len(chromosome.constraint_violations) == 1
    assert "same length" in chromosome.constraint_violations[0]


def test_ga_config_defaults():
    """Test GAConfig with default values."""
    config = GAConfig()
    
    assert config.population_size == 100
    assert config.max_generations == 500
    assert config.mutation_rate == 0.1
    assert config.crossover_rate == 0.8
    assert config.elitism_count == 5


def test_ga_config_validation():
    """Test GAConfig parameter validation."""
    # Test invalid population size
    try:
        GAConfig(population_size=1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Population size must be at least 2" in str(e)
    
    # Test invalid mutation rate
    try:
        GAConfig(mutation_rate=1.5)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Mutation rate must be between" in str(e)
    
    # Test invalid elitism count
    try:
        GAConfig(population_size=10, elitism_count=10)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Elitism count must be less than" in str(e)


def test_route_constraints_creation():
    """Test RouteConstraints creation and validation."""
    constraints = RouteConstraints(
        max_deltav_budget=10.0,
        max_mission_duration=86400.0,  # 1 day
        start_satellite_id=12345,
        min_hops=2,
        max_hops=10
    )
    
    assert constraints.max_deltav_budget == 10.0
    assert constraints.start_satellite_id == 12345
    assert constraints.end_satellite_id is None
    assert len(constraints.forbidden_satellites) == 0


def test_route_constraints_validation():
    """Test RouteConstraints parameter validation."""
    # Test invalid delta-v budget
    try:
        RouteConstraints(max_deltav_budget=-1.0, max_mission_duration=1000.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Delta-v budget must be positive" in str(e)
    
    # Test invalid hop constraints
    try:
        RouteConstraints(
            max_deltav_budget=10.0,
            max_mission_duration=1000.0,
            min_hops=5,
            max_hops=3
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Maximum hops must be >= minimum hops" in str(e)


def test_generation_stats():
    """Test GenerationStats properties."""
    stats = GenerationStats(
        generation=10,
        best_fitness=100.0,
        average_fitness=75.0,
        worst_fitness=50.0,
        diversity_metric=0.8,
        valid_solutions_count=85,
        constraint_satisfaction_rate=0.85
    )
    
    assert stats.fitness_range == 50.0  # 100.0 - 50.0


def test_optimization_stats():
    """Test OptimizationStats properties."""
    stats = OptimizationStats(
        generations_completed=100,
        best_fitness=150.0,
        average_fitness=100.0,
        population_diversity=0.6,
        constraint_satisfaction_rate=0.9
    )
    
    assert stats.convergence_efficiency == 1.5  # 150.0 / 100


def test_optimization_result():
    """Test OptimizationResult with successful result."""
    chromosome = RouteChromosome([1, 2, 3], [0.0, 100.0, 200.0], 5.0)
    stats = OptimizationStats(50, 100.0, 80.0, 0.7, 0.95)
    
    result = OptimizationResult(
        best_route=chromosome,
        optimization_stats=stats,
        convergence_history=[],
        execution_time=30.5,
        status=OptimizationStatus.SUCCESS
    )
    
    assert result.success == True
    assert result.total_hops == 2
    assert result.total_deltav == 5.0


def test_fitness_result():
    """Test FitnessResult properties."""
    result = FitnessResult(
        fitness_score=85.0,
        total_deltav=4.5,
        hop_count=3,
        mission_duration=300.0,
        constraint_violations=["Delta-v exceeded", "Time exceeded"],
        is_valid=False
    )
    
    assert result.penalty_score == 2000.0  # 2 violations * 1000


def test_constraint_result():
    """Test ConstraintResult utilization calculations."""
    result = ConstraintResult(
        is_valid=True,
        violations=[],
        deltav_usage=7.5,
        deltav_budget=10.0,
        duration_usage=3600.0,
        duration_budget=7200.0,
        hop_count=5,
        min_hops=2,
        max_hops=10
    )
    
    assert result.deltav_utilization == 75.0
    assert result.duration_utilization == 50.0


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_route_chromosome_creation,
        test_route_chromosome_validation,
        test_ga_config_defaults,
        test_ga_config_validation,
        test_route_constraints_creation,
        test_route_constraints_validation,
        test_generation_stats,
        test_optimization_stats,
        test_optimization_result,
        test_fitness_result,
        test_constraint_result
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    run_all_tests()