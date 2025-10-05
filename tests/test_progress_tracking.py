"""
Test Progress Tracking and Statistics for Genetic Route Optimizer

This module tests the enhanced progress tracking, statistics calculation,
and convergence monitoring functionality.
"""

import pytest
import time
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from src.genetic_route_optimizer import GeneticRouteOptimizer
from src.genetic_algorithm import GAConfig, RouteConstraints, OptimizationStatus
from src.tle_parser import SatelliteData


class TestProgressTracking:
    """Test progress tracking and statistics functionality."""
    
    @pytest.fixture
    def sample_satellites(self):
        """Create sample satellite data for testing."""
        satellites = []
        for i in range(10):
            sat = SatelliteData(
                catalog_number=25544 + i,
                name=f"TEST-SAT-{i}",
                line1=f"1 {25544 + i:05d}U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
                line2=f"2 {25544 + i:05d}  51.6416 211.0220 0006703  69.9751 290.2127 15.72125391563537"
            )
            satellites.append(sat)
        return satellites
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration with small parameters for quick testing."""
        return GAConfig(
            population_size=20,
            max_generations=50,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elitism_count=2,
            tournament_size=3,
            convergence_threshold=1e-6,
            max_stagnant_generations=10
        )
    
    @pytest.fixture
    def test_constraints(self):
        """Create test constraints."""
        return RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400.0,  # 1 day
            min_hops=2,
            max_hops=5
        )
    
    def test_progress_callback_functionality(self, sample_satellites, test_config, test_constraints):
        """Test that progress callback receives correct information."""
        optimizer = GeneticRouteOptimizer(sample_satellites, test_config)
        
        # Mock callback to capture progress updates
        progress_updates = []
        def mock_callback(progress_info: Dict[str, Any]):
            progress_updates.append(progress_info.copy())
        
        optimizer.set_progress_callback(mock_callback)
        
        # Run optimization with limited generations
        test_config.max_generations = 5
        result = optimizer.optimize_route(test_constraints)
        
        # Verify callback was called
        assert len(progress_updates) > 0, "Progress callback should be called"
        
        # Verify progress information structure
        for update in progress_updates:
            assert 'generation' in update
            assert 'best_fitness' in update
            assert 'average_fitness' in update
            assert 'population_diversity' in update
            assert 'progress_percentage' in update
            assert 'stagnant_generations' in update
            assert 'improvement_rate' in update
            assert 'convergence_trend' in update
            
            # Verify data types and ranges
            assert isinstance(update['generation'], int)
            assert isinstance(update['best_fitness'], (int, float))
            assert isinstance(update['average_fitness'], (int, float))
            assert 0.0 <= update['population_diversity'] <= 1.0
            assert 0.0 <= update['progress_percentage'] <= 100.0
            assert update['stagnant_generations'] >= 0
    
    def test_detailed_logging_functionality(self, sample_satellites, test_config, test_constraints):
        """Test detailed logging functionality."""
        optimizer = GeneticRouteOptimizer(sample_satellites, test_config)
        
        # Enable detailed logging
        optimizer.enable_detailed_logging(True)
        assert optimizer.detailed_logging == True
        
        # Mock logger to capture log messages
        with patch.object(optimizer.logger, 'info') as mock_logger:
            test_config.max_generations = 3
            result = optimizer.optimize_route(test_constraints)
            
            # Verify logging was called
            assert mock_logger.call_count > 0, "Logger should be called for progress updates"
            
            # Check that generation progress was logged
            log_calls = [call.args[0] for call in mock_logger.call_args_list]
            generation_logs = [log for log in log_calls if 'Generation' in log]
            assert len(generation_logs) > 0, "Generation progress should be logged"
    
    def test_convergence_metrics_tracking(self, sample_satellites, test_config, test_constraints):
        """Test convergence metrics tracking."""
        optimizer = GeneticRouteOptimizer(sample_satellites, test_config)
        
        # Run optimization
        test_config.max_generations = 10
        result = optimizer.optimize_route(test_constraints)
        
        # Verify convergence metrics are tracked
        assert len(optimizer.fitness_history) > 0, "Fitness history should be tracked"
        assert len(optimizer.diversity_history) > 0, "Diversity history should be tracked"
        
        # Verify fitness history structure
        for entry in optimizer.fitness_history:
            assert 'generation' in entry
            assert 'best' in entry
            assert 'average' in entry
            assert 'improvement' in entry
            assert isinstance(entry['improvement'], bool)
        
        # Verify diversity history structure
        for entry in optimizer.diversity_history:
            assert 'generation' in entry
            assert 'diversity' in entry
            assert 0.0 <= entry['diversity'] <= 1.0
        
        # Verify convergence metrics
        metrics = optimizer.convergence_metrics
        assert 'improvement_rate' in metrics
        assert 'stagnation_periods' in metrics
        assert 'diversity_trend' in metrics
        assert metrics['diversity_trend'] in ['increasing', 'decreasing', 'stable']
    
    def test_detailed_statistics_compilation(self, sample_satellites, test_config, test_constraints):
        """Test detailed statistics compilation."""
        optimizer = GeneticRouteOptimizer(sample_satellites, test_config)
        
        # Run optimization
        test_config.max_generations = 15
        result = optimizer.optimize_route(test_constraints)
        
        # Get detailed statistics
        detailed_stats = optimizer.get_detailed_statistics()
        
        # Verify main sections exist
        assert 'optimization_summary' in detailed_stats
        assert 'fitness_statistics' in detailed_stats
        assert 'diversity_analysis' in detailed_stats
        assert 'convergence_metrics' in detailed_stats
        assert 'performance_metrics' in detailed_stats
        assert 'population_statistics' in detailed_stats
        
        # Verify optimization summary
        summary = detailed_stats['optimization_summary']
        assert 'total_generations' in summary
        assert 'peak_fitness' in summary
        assert 'final_fitness' in summary
        assert 'fitness_improvement' in summary
        assert summary['total_generations'] > 0
        
        # Verify fitness statistics
        fitness_stats = detailed_stats['fitness_statistics']
        assert 'best_fitness_history' in fitness_stats
        assert 'average_fitness_history' in fitness_stats
        assert 'fitness_variance' in fitness_stats
        assert 'fitness_trend' in fitness_stats
        assert len(fitness_stats['best_fitness_history']) > 0
        
        # Verify diversity analysis
        diversity_analysis = detailed_stats['diversity_analysis']
        assert 'diversity_history' in diversity_analysis
        assert 'average_diversity' in diversity_analysis
        assert 'min_diversity' in diversity_analysis
        assert 'max_diversity' in diversity_analysis
        assert len(diversity_analysis['diversity_history']) > 0
        
        # Verify performance metrics
        performance = detailed_stats['performance_metrics']
        assert 'stagnation_periods' in performance
        assert 'longest_stagnation' in performance
        assert 'improvement_rate' in performance
        assert 'convergence_efficiency' in performance
    
    def test_progress_info_compilation(self, sample_satellites, test_config, test_constraints):
        """Test progress information compilation."""
        optimizer = GeneticRouteOptimizer(sample_satellites, test_config)
        
        # Initialize some state
        optimizer._reset_optimization_state()
        optimizer._start_time = time.time()
        
        # Create mock population and fitness scores
        mock_population = [Mock() for _ in range(test_config.population_size)]
        for i, chrom in enumerate(mock_population):
            chrom.is_valid = i % 2 == 0  # Half valid, half invalid
            chrom.hop_count = i + 1
            chrom.total_deltav = (i + 1) * 0.5
        
        mock_fitness_scores = [float(i) for i in range(test_config.population_size)]
        
        # Mock selection operator
        optimizer.selection_operator.calculate_population_diversity = Mock(return_value=0.75)
        
        # Set best chromosome
        optimizer.best_chromosome = Mock()
        optimizer.best_chromosome.hop_count = 5
        optimizer.best_chromosome.total_deltav = 2.5
        optimizer.best_chromosome.is_valid = True
        
        # Compile progress info
        progress_info = optimizer._compile_progress_info(10, mock_fitness_scores, mock_population)
        
        # Verify all expected fields are present
        expected_fields = [
            'generation', 'max_generations', 'progress_percentage',
            'best_fitness', 'average_fitness', 'worst_fitness', 'fitness_std',
            'population_size', 'population_diversity', 'valid_solutions_count',
            'constraint_satisfaction_rate', 'stagnant_generations', 'improvement_rate',
            'convergence_trend', 'adaptive_mutation_rate', 'adaptive_crossover_rate',
            'best_route_hops', 'best_route_deltav', 'best_route_valid',
            'elapsed_time', 'estimated_remaining_time'
        ]
        
        for field in expected_fields:
            assert field in progress_info, f"Field '{field}' should be in progress info"
        
        # Verify data types and ranges
        assert progress_info['generation'] == 10
        assert progress_info['progress_percentage'] == (10 / test_config.max_generations) * 100
        assert progress_info['population_size'] == test_config.population_size
        assert progress_info['valid_solutions_count'] == test_config.population_size // 2
        assert progress_info['best_route_hops'] == 5
        assert progress_info['best_route_deltav'] == 2.5
        assert progress_info['best_route_valid'] == True
    
    def test_convergence_trend_analysis(self, sample_satellites, test_config, test_constraints):
        """Test convergence trend analysis."""
        optimizer = GeneticRouteOptimizer(sample_satellites, test_config)
        
        # Test with insufficient data
        assert optimizer._analyze_convergence_trend() == 'insufficient_data'
        
        # Add some fitness history
        optimizer.fitness_history = [
            {'generation': i, 'best': 10.0 + i * 2.0, 'average': 5.0 + i, 'improvement': True}
            for i in range(10)
        ]
        
        # Test improving trend
        trend = optimizer._analyze_convergence_trend()
        assert trend in ['improving', 'stable', 'stagnating']
        
        # Test stagnating trend
        optimizer.stagnant_generations = test_config.max_stagnant_generations
        trend = optimizer._analyze_convergence_trend()
        assert trend == 'stagnating'
    
    def test_time_estimation(self, sample_satellites, test_config, test_constraints):
        """Test remaining time estimation."""
        optimizer = GeneticRouteOptimizer(sample_satellites, test_config)
        
        # Test with no start time
        remaining_time = optimizer._estimate_remaining_time(5)
        assert remaining_time == 0.0
        
        # Test with start time
        optimizer._start_time = time.time() - 10.0  # 10 seconds ago
        remaining_time = optimizer._estimate_remaining_time(5)
        assert remaining_time > 0.0
        
        # Test at generation 0
        remaining_time = optimizer._estimate_remaining_time(0)
        assert remaining_time == 0.0
    
    def test_statistics_calculation_methods(self, sample_satellites, test_config, test_constraints):
        """Test various statistics calculation methods."""
        optimizer = GeneticRouteOptimizer(sample_satellites, test_config)
        
        # Test standard deviation calculation
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        std_dev = optimizer._calculate_standard_deviation(values)
        assert std_dev > 0.0
        
        # Test with single value
        std_dev = optimizer._calculate_standard_deviation([1.0])
        assert std_dev == 0.0
        
        # Test trend slope calculation
        increasing_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        slope = optimizer._calculate_trend_slope(increasing_values)
        assert slope > 0.0
        
        decreasing_values = [5.0, 4.0, 3.0, 2.0, 1.0]
        slope = optimizer._calculate_trend_slope(decreasing_values)
        assert slope < 0.0
        
        # Test with insufficient data
        slope = optimizer._calculate_trend_slope([1.0])
        assert slope == 0.0
    
    def test_stagnation_period_tracking(self, sample_satellites, test_config, test_constraints):
        """Test stagnation period tracking."""
        optimizer = GeneticRouteOptimizer(sample_satellites, test_config)
        
        # Initialize state
        optimizer._reset_optimization_state()
        
        # Simulate stagnation periods
        mock_population = [Mock() for _ in range(5)]
        mock_fitness_scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Start with no stagnation
        optimizer.stagnant_generations = 0
        optimizer._update_convergence_metrics(1, mock_fitness_scores, mock_population)
        
        # Start stagnation
        optimizer.stagnant_generations = 1
        optimizer._update_convergence_metrics(2, mock_fitness_scores, mock_population)
        
        # Continue stagnation
        optimizer.stagnant_generations = 2
        optimizer._update_convergence_metrics(3, mock_fitness_scores, mock_population)
        
        # End stagnation
        optimizer.stagnant_generations = 0
        optimizer._update_convergence_metrics(4, mock_fitness_scores, mock_population)
        
        # Verify stagnation period was recorded
        stagnation_periods = optimizer.convergence_metrics['stagnation_periods']
        assert len(stagnation_periods) > 0
        
        # Get longest stagnation period
        longest = optimizer._get_longest_stagnation_period()
        assert longest >= 0


if __name__ == "__main__":
    pytest.main([__file__])