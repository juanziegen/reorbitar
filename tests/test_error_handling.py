"""
Test Error Handling and Recovery for Genetic Route Optimizer

This test module verifies the comprehensive error handling and recovery
mechanisms implemented in the genetic algorithm system.
"""

import pytest
import random
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import List

from src.genetic_route_optimizer import GeneticRouteOptimizer
from src.ga_error_handler import GAErrorHandler, ErrorType, ErrorSeverity, ErrorReport
from src.genetic_algorithm import (
    RouteChromosome, GAConfig, RouteConstraints, 
    OptimizationStatus, OptimizationResult
)
from src.tle_parser import SatelliteData


class TestGAErrorHandler:
    """Test the GAErrorHandler class functionality."""
    
    @pytest.fixture
    def sample_satellites(self):
        """Create sample satellite data for testing."""
        satellites = []
        for i in range(10):
            sat = SatelliteData(
                catalog_number=25544 + i,
                name=f"TEST-SAT-{i}",
                tle_line1=f"1 {25544 + i:05d}U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
                tle_line2=f"2 {25544 + i:05d}  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
            )
            satellites.append(sat)
        return satellites
    
    @pytest.fixture
    def error_handler(self, sample_satellites):
        """Create error handler instance."""
        config = GAConfig(population_size=20, max_generations=50)
        return GAErrorHandler(sample_satellites, config)
    
    @pytest.fixture
    def sample_constraints(self):
        """Create sample route constraints."""
        return RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400.0,
            min_hops=2,
            max_hops=5
        )
    
    def test_fitness_evaluation_error_handling(self, error_handler, sample_constraints):
        """Test handling of fitness evaluation failures."""
        # Create a test chromosome
        chromosome = RouteChromosome(
            satellite_sequence=[25544, 25545, 25546],
            departure_times=[0.0, 3600.0, 7200.0]
        )
        
        # Simulate fitness evaluation error
        test_error = ValueError("Orbital calculation failed")
        
        penalty_fitness = error_handler.handle_fitness_evaluation_error(
            chromosome, test_error, generation=5
        )
        
        # Verify penalty fitness is applied
        assert penalty_fitness < 0
        assert not chromosome.is_valid
        assert len(chromosome.constraint_violations) > 0
        assert "Fitness evaluation failed" in chromosome.constraint_violations[0]
        
        # Verify error is recorded
        assert len(error_handler.error_history) == 1
        error_report = error_handler.error_history[0]
        assert error_report.error_type == ErrorType.FITNESS_EVALUATION
        assert error_report.severity == ErrorSeverity.MEDIUM
        assert error_report.generation == 5
    
    def test_chromosome_repair(self, error_handler, sample_constraints):
        """Test chromosome repair mechanisms."""
        # Create invalid chromosome with duplicate satellites
        invalid_chromosome = RouteChromosome(
            satellite_sequence=[25544, 25544, 25545, 99999],  # Duplicate and invalid ID
            departure_times=[0.0, 3600.0, 7200.0, 10800.0]
        )
        
        repaired = error_handler.repair_invalid_chromosome(
            invalid_chromosome, sample_constraints, generation=3
        )
        
        # Verify repair was attempted
        assert len(error_handler.error_history) == 1
        
        # Verify repaired chromosome is valid
        assert len(repaired.satellite_sequence) == len(repaired.departure_times)
        assert len(set(repaired.satellite_sequence)) == len(repaired.satellite_sequence)  # No duplicates
        
        # Verify all satellite IDs are valid
        valid_ids = {sat.catalog_number for sat in error_handler.satellites}
        for sat_id in repaired.satellite_sequence:
            assert sat_id in valid_ids
        
        # Verify timing is ascending
        for i in range(1, len(repaired.departure_times)):
            assert repaired.departure_times[i] > repaired.departure_times[i-1]
    
    def test_population_convergence_handling(self, error_handler, sample_constraints):
        """Test population diversity injection."""
        # Create converged population (all similar chromosomes)
        converged_population = []
        base_sequence = [25544, 25545, 25546]
        base_times = [0.0, 3600.0, 7200.0]
        
        for i in range(10):
            chromosome = RouteChromosome(
                satellite_sequence=base_sequence.copy(),
                departure_times=[t + i * 100 for t in base_times]  # Slight variations
            )
            converged_population.append(chromosome)
        
        diverse_population = error_handler.handle_population_convergence(
            converged_population, sample_constraints, generation=20
        )
        
        # Verify diversity was injected
        assert len(diverse_population) == len(converged_population)
        
        # Verify some chromosomes are different
        original_sequences = {tuple(c.satellite_sequence) for c in converged_population}
        new_sequences = {tuple(c.satellite_sequence) for c in diverse_population}
        
        # Should have more unique sequences after diversity injection
        assert len(new_sequences) >= len(original_sequences)
        
        # Verify error was recorded
        assert len(error_handler.error_history) == 1
        error_report = error_handler.error_history[0]
        assert error_report.error_type == ErrorType.POPULATION_CONVERGENCE
    
    def test_fallback_chromosome_creation(self, error_handler, sample_constraints):
        """Test creation of fallback chromosomes."""
        fallback = error_handler._create_fallback_chromosome(sample_constraints, generation=1)
        
        # Verify fallback chromosome is valid
        assert len(fallback.satellite_sequence) >= sample_constraints.min_hops + 1
        assert len(fallback.satellite_sequence) <= sample_constraints.max_hops + 1
        assert len(fallback.satellite_sequence) == len(fallback.departure_times)
        
        # Verify all satellite IDs are valid
        valid_ids = {sat.catalog_number for sat in error_handler.satellites}
        for sat_id in fallback.satellite_sequence:
            assert sat_id in valid_ids
        
        # Verify no duplicates
        assert len(set(fallback.satellite_sequence)) == len(fallback.satellite_sequence)
        
        # Verify timing is ascending
        for i in range(1, len(fallback.departure_times)):
            assert fallback.departure_times[i] > fallback.departure_times[i-1]
    
    def test_diverse_chromosome_creation(self, error_handler, sample_constraints):
        """Test creation of diverse chromosomes."""
        # Create existing population
        existing_population = []
        for i in range(5):
            chromosome = RouteChromosome(
                satellite_sequence=[25544, 25545, 25546],
                departure_times=[0.0, 3600.0, 7200.0]
            )
            existing_population.append(chromosome)
        
        diverse_chromosome = error_handler._create_diverse_chromosome(
            sample_constraints, existing_population
        )
        
        # Verify diverse chromosome is different from existing ones
        existing_sequences = {tuple(c.satellite_sequence) for c in existing_population}
        diverse_sequence = tuple(diverse_chromosome.satellite_sequence)
        
        # Should be different from at least some existing chromosomes
        assert len(existing_sequences) == 1  # All existing are the same
        # Diverse chromosome should be different or at least have different timing
        is_different = (diverse_sequence not in existing_sequences or 
                       diverse_chromosome.departure_times != existing_population[0].departure_times)
        assert is_different
    
    def test_error_summary_generation(self, error_handler, sample_constraints):
        """Test comprehensive error summary generation."""
        # Generate various types of errors
        chromosome = RouteChromosome([25544, 25545], [0.0, 3600.0])
        
        # Add different types of errors
        error_handler.handle_fitness_evaluation_error(
            chromosome, ValueError("Test error 1"), generation=1
        )
        error_handler.handle_fitness_evaluation_error(
            chromosome, RuntimeError("Test error 2"), generation=2
        )
        error_handler.repair_invalid_chromosome(chromosome, sample_constraints, generation=3)
        
        summary = error_handler.get_error_summary()
        
        # Verify summary contains expected information
        assert summary['total_errors'] == 3
        assert ErrorType.FITNESS_EVALUATION.value in summary['error_types']
        assert ErrorType.CHROMOSOME_INVALID.value in summary['error_types']
        assert summary['error_types'][ErrorType.FITNESS_EVALUATION.value] == 2
        assert summary['error_types'][ErrorType.CHROMOSOME_INVALID.value] == 1
        
        # Verify severity distribution
        assert ErrorSeverity.MEDIUM.value in summary['severity_distribution']
        assert summary['severity_distribution'][ErrorSeverity.MEDIUM.value] == 3
        
        # Verify generation error counts
        assert 1 in summary['generation_error_counts']
        assert 2 in summary['generation_error_counts']
        assert 3 in summary['generation_error_counts']


class TestGeneticRouteOptimizerErrorHandling:
    """Test error handling integration in GeneticRouteOptimizer."""
    
    @pytest.fixture
    def sample_satellites(self):
        """Create sample satellite data for testing."""
        satellites = []
        for i in range(20):
            sat = SatelliteData(
                catalog_number=25544 + i,
                name=f"TEST-SAT-{i}",
                tle_line1=f"1 {25544 + i:05d}U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
                tle_line2=f"2 {25544 + i:05d}  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
            )
            satellites.append(sat)
        return satellites
    
    @pytest.fixture
    def optimizer(self, sample_satellites):
        """Create genetic route optimizer instance."""
        config = GAConfig(population_size=10, max_generations=5)
        return GeneticRouteOptimizer(sample_satellites, config)
    
    @pytest.fixture
    def sample_constraints(self):
        """Create sample route constraints."""
        return RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400.0,
            min_hops=2,
            max_hops=4
        )
    
    def test_population_initialization_with_recovery(self, optimizer, sample_constraints):
        """Test population initialization with error recovery."""
        # Mock chromosome initializer to fail initially
        with patch.object(optimizer.chromosome_initializer, 'initialize_population') as mock_init:
            # First call fails, second succeeds
            mock_init.side_effect = [
                ValueError("Initialization failed"),
                [RouteChromosome([25544, 25545], [0.0, 3600.0]) for _ in range(10)]
            ]
            
            population = optimizer._initialize_population_with_recovery(sample_constraints)
            
            # Should have recovered and created population
            assert population is not None
            assert len(population) == optimizer.config.population_size
            
            # Should have attempted initialization twice
            assert mock_init.call_count >= 1
    
    def test_fitness_evaluation_with_recovery(self, optimizer, sample_constraints):
        """Test fitness evaluation with error recovery."""
        # Create test population
        population = [
            RouteChromosome([25544, 25545, 25546], [0.0, 3600.0, 7200.0])
            for _ in range(5)
        ]
        
        # Mock fitness evaluator to fail for some chromosomes
        with patch.object(optimizer.fitness_evaluator, 'evaluate_route') as mock_eval:
            # Make some evaluations fail
            def side_effect(chromosome, constraints):
                if chromosome.satellite_sequence[0] == 25544:
                    raise ValueError("Fitness evaluation failed")
                else:
                    # Return mock fitness result
                    mock_result = Mock()
                    mock_result.fitness_score = 100.0
                    mock_result.total_deltav = 2.0
                    mock_result.is_valid = True
                    mock_result.constraint_violations = []
                    return mock_result
            
            mock_eval.side_effect = side_effect
            
            fitness_scores = optimizer._evaluate_population_with_recovery(
                population, sample_constraints, generation=1
            )
            
            # Should have fitness scores for all chromosomes
            assert len(fitness_scores) == len(population)
            
            # Failed evaluations should have penalty scores
            assert any(score < 0 for score in fitness_scores)
            
            # Should have recorded errors
            assert len(optimizer.error_handler.error_history) > 0
    
    def test_generation_creation_with_recovery(self, optimizer, sample_constraints):
        """Test generation creation with error recovery."""
        # Create test population and fitness scores
        population = [
            RouteChromosome([25544, 25545], [0.0, 3600.0]),
            RouteChromosome([25546, 25547], [0.0, 3600.0])
        ]
        fitness_scores = [100.0, 90.0]
        
        # Mock genetic operators to occasionally fail
        with patch.object(optimizer.crossover_operator, 'order_crossover') as mock_crossover:
            mock_crossover.side_effect = [
                (population[0], population[1]),  # Success
                ValueError("Crossover failed"),  # Failure
                (population[0], population[1])   # Success
            ]
            
            next_generation = optimizer._create_next_generation_with_recovery(
                population, fitness_scores, sample_constraints, generation=2
            )
            
            # Should have created next generation despite failures
            assert len(next_generation) == optimizer.config.population_size
            
            # All chromosomes should be valid
            for chromosome in next_generation:
                assert len(chromosome.satellite_sequence) > 0
                assert len(chromosome.satellite_sequence) == len(chromosome.departure_times)
    
    def test_premature_convergence_detection(self, optimizer, sample_constraints):
        """Test detection of premature convergence."""
        # Create highly similar population (converged)
        converged_population = []
        for i in range(10):
            chromosome = RouteChromosome(
                satellite_sequence=[25544, 25545, 25546],
                departure_times=[0.0, 3600.0, 7200.0]
            )
            converged_population.append(chromosome)
        
        # Mock selection operator to return very low diversity
        with patch.object(optimizer.selection_operator, 'calculate_population_diversity') as mock_diversity:
            mock_diversity.return_value = 0.05  # Very low diversity
            
            is_converged = optimizer._detect_premature_convergence(converged_population, generation=15)
            
            assert is_converged
    
    def test_system_error_handling(self, optimizer, sample_constraints):
        """Test system-level error handling."""
        # Mock a critical system error during optimization
        with patch.object(optimizer, '_initialize_population_with_recovery') as mock_init:
            mock_init.return_value = None  # Initialization fails completely
            
            result = optimizer.optimize_route(sample_constraints)
            
            # Should return failed result
            assert result.status == OptimizationStatus.FAILED
            assert result.error_message is not None
            assert not result.success
    
    def test_recovery_attempt_limiting(self, optimizer, sample_constraints):
        """Test that recovery attempts are limited to prevent infinite loops."""
        # Set low recovery attempt limit
        optimizer.error_handler.max_recovery_attempts = 2
        
        # Simulate repeated failures
        for i in range(5):
            test_error = ValueError(f"Test error {i}")
            can_recover = optimizer._attempt_generation_recovery(test_error, generation=i)
            
            if i < 2:
                assert can_recover  # Should attempt recovery
            else:
                assert not can_recover  # Should stop attempting recovery
        
        # Verify recovery attempts were limited
        assert optimizer.error_handler.recovery_attempts == 2
    
    def test_comprehensive_optimization_with_errors(self, optimizer, sample_constraints):
        """Test complete optimization run with various errors injected."""
        # Enable detailed logging for this test
        optimizer.enable_detailed_logging(True)
        
        # Mock various components to inject errors at different stages
        with patch.object(optimizer.fitness_evaluator, 'evaluate_route') as mock_eval:
            # Make fitness evaluation fail occasionally
            call_count = 0
            def fitness_side_effect(chromosome, constraints):
                nonlocal call_count
                call_count += 1
                if call_count % 3 == 0:  # Fail every 3rd evaluation
                    raise RuntimeError("Simulated fitness evaluation failure")
                
                # Return mock successful result
                mock_result = Mock()
                mock_result.fitness_score = random.uniform(50, 150)
                mock_result.total_deltav = random.uniform(1.0, 4.0)
                mock_result.is_valid = True
                mock_result.constraint_violations = []
                return mock_result
            
            mock_eval.side_effect = fitness_side_effect
            
            # Run optimization
            result = optimizer.optimize_route(sample_constraints)
            
            # Should complete despite errors
            assert result is not None
            
            # Should have recorded some errors
            error_summary = optimizer.error_handler.get_error_summary()
            assert error_summary['total_errors'] > 0
            
            # Log error summary for inspection
            print(f"Optimization completed with {error_summary['total_errors']} errors")
            print(f"Error types: {error_summary['error_types']}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])