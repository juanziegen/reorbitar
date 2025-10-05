"""
Unit tests for Route Fitness Evaluator

Tests fitness calculation accuracy, constraint violation detection,
and edge cases for the genetic algorithm route optimization system.
"""

import math
from unittest.mock import Mock, MagicMock
from src.route_fitness_evaluator import RouteFitnessEvaluator
from src.genetic_algorithm import (
    RouteChromosome, RouteConstraints, FitnessResult, ConstraintResult
)
from src.orbital_propagator import OrbitalPropagator, TransferWindow
from src.tle_parser import SatelliteData


def test_evaluator_initialization_success():
    """Test successful evaluator initialization."""
    # Create mock satellite data
    satellites = [
        SatelliteData(
            catalog_number=1,
            name="SAT-1",
            epoch=0.0,
            mean_motion=15.0,
            eccentricity=0.001,
            inclination=45.0,
            raan=0.0,
            arg_perigee=0.0,
            mean_anomaly=0.0,
            semi_major_axis=7000.0,
            orbital_period=6000.0
        ),
        SatelliteData(
            catalog_number=2,
            name="SAT-2",
            epoch=0.0,
            mean_motion=15.5,
            eccentricity=0.002,
            inclination=50.0,
            raan=10.0,
            arg_perigee=20.0,
            mean_anomaly=30.0,
            semi_major_axis=7100.0,
            orbital_period=6100.0
        )
    ]
    
    # Create mock orbital propagator
    mock_propagator = Mock(spec=OrbitalPropagator)
    mock_propagator.get_satellite_ids.return_value = [1, 2]
    
    # Create evaluator instance
    evaluator = RouteFitnessEvaluator(satellites, mock_propagator)
    
    assert len(evaluator.satellites) == 2
    assert evaluator.orbital_propagator == mock_propagator
    assert len(evaluator._deltav_cache) == 0


def test_evaluator_initialization_empty_satellites():
    """Test evaluator initialization with empty satellites list."""
    mock_propagator = Mock(spec=OrbitalPropagator)
    mock_propagator.get_satellite_ids.return_value = []
    
    try:
        RouteFitnessEvaluator([], mock_propagator)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Satellites list cannot be empty" in str(e)


def test_evaluator_initialization_none_propagator():
    """Test evaluator initialization with None propagator."""
    satellites = [
        SatelliteData(
            catalog_number=1, name="SAT-1", epoch=0.0, mean_motion=15.0,
            eccentricity=0.001, inclination=45.0, raan=0.0, arg_perigee=0.0,
            mean_anomaly=0.0, semi_major_axis=7000.0, orbital_period=6000.0
        )
    ]
    
    try:
        RouteFitnessEvaluator(satellites, None)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Orbital propagator cannot be None" in str(e)


def test_evaluator_initialization_propagator_satellite_mismatch():
    """Test evaluator initialization when propagator has satellites not in list."""
    satellites = [
        SatelliteData(
            catalog_number=1, name="SAT-1", epoch=0.0, mean_motion=15.0,
            eccentricity=0.001, inclination=45.0, raan=0.0, arg_perigee=0.0,
            mean_anomaly=0.0, semi_major_axis=7000.0, orbital_period=6000.0
        )
    ]
    
    mock_propagator = Mock(spec=OrbitalPropagator)
    mock_propagator.get_satellite_ids.return_value = [1, 2, 3, 4]  # Extra satellites
    
    try:
        RouteFitnessEvaluator(satellites, mock_propagator)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Propagator contains satellites not in satellite list" in str(e)


def setup_basic_evaluator():
    """Helper function to set up a basic evaluator for testing."""
    satellites = [
        SatelliteData(catalog_number=i, name=f"SAT-{i}", epoch=0.0, mean_motion=15.0,
                     eccentricity=0.001, inclination=45.0, raan=0.0, arg_perigee=0.0,
                     mean_anomaly=0.0, semi_major_axis=7000.0, orbital_period=6000.0)
        for i in range(1, 6)
    ]
    
    mock_propagator = Mock(spec=OrbitalPropagator)
    mock_propagator.get_satellite_ids.return_value = [1, 2, 3, 4, 5]
    
    return RouteFitnessEvaluator(satellites, mock_propagator), mock_propagator


def test_calculate_route_deltav_simple_route():
    """Test delta-v calculation for a simple two-satellite route."""
    evaluator, mock_propagator = setup_basic_evaluator()
    
    # Set up mock transfer window
    mock_propagator.calculate_transfer_window.return_value = TransferWindow(
        departure_deltav=1.5,
        arrival_deltav=1.2,
        transfer_time=3600.0,
        optimal_departure_time=0.0,
        transfer_efficiency=0.9
    )
    
    satellite_sequence = [1, 2]
    departure_times = [0.0, 3600.0]
    
    total_deltav = evaluator.calculate_route_deltav(satellite_sequence, departure_times)
    
    assert total_deltav == 2.7  # 1.5 + 1.2
    mock_propagator.calculate_transfer_window.assert_called_once_with(1, 2, 0.0)


def test_calculate_route_deltav_multi_hop_route():
    """Test delta-v calculation for a multi-hop route."""
    evaluator, mock_propagator = setup_basic_evaluator()
    
    # Set up different transfer windows for each hop
    transfer_windows = [
        TransferWindow(1.0, 1.0, 3600.0, 0.0, 0.8),      # 1->2
        TransferWindow(1.5, 1.2, 3600.0, 3600.0, 0.7),   # 2->3
        TransferWindow(0.8, 0.9, 3600.0, 7200.0, 0.9)    # 3->4
    ]
    
    mock_propagator.calculate_transfer_window.side_effect = transfer_windows
    
    satellite_sequence = [1, 2, 3, 4]
    departure_times = [0.0, 3600.0, 7200.0, 10800.0]
    
    total_deltav = evaluator.calculate_route_deltav(satellite_sequence, departure_times)
    
    expected_deltav = 2.0 + 2.7 + 1.7  # Sum of all hops
    assert total_deltav == expected_deltav
    assert mock_propagator.calculate_transfer_window.call_count == 3


def test_calculate_route_deltav_caching():
    """Test that delta-v calculations are cached properly."""
    evaluator, mock_propagator = setup_basic_evaluator()
    
    mock_propagator.calculate_transfer_window.return_value = TransferWindow(
        departure_deltav=1.0,
        arrival_deltav=1.0,
        transfer_time=3600.0,
        optimal_departure_time=0.0,
        transfer_efficiency=0.8
    )
    
    satellite_sequence = [1, 2]
    departure_times = [0.0, 3600.0]
    
    # First call should use propagator
    deltav1 = evaluator.calculate_route_deltav(satellite_sequence, departure_times)
    assert mock_propagator.calculate_transfer_window.call_count == 1
    
    # Second call should use cache
    deltav2 = evaluator.calculate_route_deltav(satellite_sequence, departure_times)
    assert mock_propagator.calculate_transfer_window.call_count == 1  # No additional calls
    assert deltav1 == deltav2


def test_calculate_route_deltav_single_satellite():
    """Test delta-v calculation for single satellite (no transfers)."""
    evaluator, mock_propagator = setup_basic_evaluator()
    
    satellite_sequence = [1]
    departure_times = [0.0]
    
    total_deltav = evaluator.calculate_route_deltav(satellite_sequence, departure_times)
    
    assert total_deltav == 0.0
    assert mock_propagator.calculate_transfer_window.call_count == 0


def test_calculate_route_deltav_empty_route():
    """Test delta-v calculation for empty route."""
    evaluator, mock_propagator = setup_basic_evaluator()
    
    satellite_sequence = []
    departure_times = []
    
    total_deltav = evaluator.calculate_route_deltav(satellite_sequence, departure_times)
    
    assert total_deltav == 0.0
    assert mock_propagator.calculate_transfer_window.call_count == 0


def test_evaluate_route_valid_route():
    """Test comprehensive route evaluation for a valid route."""
    evaluator, mock_propagator = setup_basic_evaluator()
    
    # Set up mock transfer window
    mock_propagator.calculate_transfer_window.return_value = TransferWindow(
        departure_deltav=1.0,
        arrival_deltav=1.0,
        transfer_time=3600.0,
        optimal_departure_time=0.0,
        transfer_efficiency=0.8
    )
    
    chromosome = RouteChromosome(
        satellite_sequence=[1, 2, 3],
        departure_times=[0.0, 3600.0, 7200.0],
        total_deltav=0.0,
        is_valid=True,
        constraint_violations=[]
    )
    
    constraints = RouteConstraints(
        max_deltav_budget=10.0,
        max_mission_duration=10800.0,
        min_hops=1,
        max_hops=10
    )
    
    result = evaluator.evaluate_route(chromosome, constraints)
    
    assert isinstance(result, FitnessResult)
    assert result.is_valid == True
    assert result.total_deltav == 4.0  # 2 hops * 2.0 deltav each
    assert result.hop_count == 2
    assert result.mission_duration == 7200.0
    assert len(result.constraint_violations) == 0
    assert result.fitness_score > 0


def test_check_constraints_hop_count_violations():
    """Test hop count constraint violations."""
    evaluator, mock_propagator = setup_basic_evaluator()
    mock_propagator.calculate_transfer_window.return_value = TransferWindow(
        departure_deltav=2.0, arrival_deltav=2.0, transfer_time=3600.0,
        optimal_departure_time=0.0, transfer_efficiency=0.8
    )
    
    chromosome = RouteChromosome(
        satellite_sequence=[1, 2],  # 1 hop
        departure_times=[0.0, 3600.0],
        total_deltav=0.0,
        is_valid=True
    )
    
    # Test minimum hops violation
    constraints = RouteConstraints(
        max_deltav_budget=10.0,
        max_mission_duration=10800.0,
        min_hops=3,  # Requires at least 3 hops
        max_hops=10
    )
    
    result = evaluator.check_constraints(chromosome, constraints)
    
    assert result.is_valid == False
    assert any("minimum required: 3" in violation for violation in result.violations)


def test_check_constraints_deltav_budget_violation():
    """Test delta-v budget constraint violation."""
    evaluator, mock_propagator = setup_basic_evaluator()
    mock_propagator.calculate_transfer_window.return_value = TransferWindow(
        departure_deltav=2.0, arrival_deltav=2.0, transfer_time=3600.0,
        optimal_departure_time=0.0, transfer_efficiency=0.8
    )
    
    chromosome = RouteChromosome(
        satellite_sequence=[1, 2, 3],
        departure_times=[0.0, 3600.0, 7200.0],
        total_deltav=0.0,
        is_valid=True
    )
    
    constraints = RouteConstraints(
        max_deltav_budget=5.0,  # Low budget
        max_mission_duration=10800.0,
        min_hops=1,
        max_hops=10
    )
    
    result = evaluator.check_constraints(chromosome, constraints)
    
    assert result.is_valid == False
    assert any("exceeds budget" in violation for violation in result.violations)
    assert result.deltav_usage == 8.0  # 2 hops * 4.0 deltav each
    assert result.deltav_budget == 5.0


def test_check_constraints_forbidden_satellites():
    """Test forbidden satellites constraint."""
    evaluator, mock_propagator = setup_basic_evaluator()
    mock_propagator.calculate_transfer_window.return_value = TransferWindow(
        departure_deltav=2.0, arrival_deltav=2.0, transfer_time=3600.0,
        optimal_departure_time=0.0, transfer_efficiency=0.8
    )
    
    chromosome = RouteChromosome(
        satellite_sequence=[1, 2, 3],  # Contains satellite 2
        departure_times=[0.0, 3600.0, 7200.0],
        total_deltav=0.0,
        is_valid=True
    )
    
    constraints = RouteConstraints(
        max_deltav_budget=20.0,
        max_mission_duration=10800.0,
        forbidden_satellites=[2, 5],  # Satellite 2 is forbidden
        min_hops=1,
        max_hops=10
    )
    
    result = evaluator.check_constraints(chromosome, constraints)
    
    assert result.is_valid == False
    assert any("forbidden satellites" in violation for violation in result.violations)
    assert any("2" in violation for violation in result.violations)


def test_duplicate_satellites_in_route():
    """Test detection of duplicate satellites in route."""
    evaluator, mock_propagator = setup_basic_evaluator()
    
    chromosome = RouteChromosome(
        satellite_sequence=[1, 2, 1, 3],  # Satellite 1 appears twice
        departure_times=[0.0, 3600.0, 7200.0, 10800.0],
        total_deltav=0.0,
        is_valid=True
    )
    
    constraints = RouteConstraints(
        max_deltav_budget=20.0,
        max_mission_duration=15000.0,
        min_hops=1,
        max_hops=10
    )
    
    result = evaluator.check_constraints(chromosome, constraints)
    
    assert result.is_valid == False
    assert any("duplicate satellites" in violation for violation in result.violations)
    assert any("1" in violation for violation in result.violations)


def test_invalid_satellite_ids():
    """Test handling of invalid satellite IDs."""
    evaluator, mock_propagator = setup_basic_evaluator()
    
    chromosome = RouteChromosome(
        satellite_sequence=[1, 99, 3],  # Satellite 99 doesn't exist
        departure_times=[0.0, 3600.0, 7200.0],
        total_deltav=0.0,
        is_valid=True
    )
    
    constraints = RouteConstraints(
        max_deltav_budget=20.0,
        max_mission_duration=10800.0,
        min_hops=1,
        max_hops=10
    )
    
    result = evaluator.check_constraints(chromosome, constraints)
    
    assert result.is_valid == False
    assert any("not found in satellite database" in violation for violation in result.violations)
    assert any("99" in violation for violation in result.violations)


def test_calculate_route_deltav_invalid_satellite():
    """Test delta-v calculation with invalid satellite ID."""
    evaluator, mock_propagator = setup_basic_evaluator()
    
    satellite_sequence = [1, 99]  # Satellite 99 doesn't exist
    departure_times = [0.0, 3600.0]
    
    try:
        evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Satellite 99 not found" in str(e)


def test_calculate_route_deltav_mismatched_lengths():
    """Test delta-v calculation with mismatched sequence and times lengths."""
    evaluator, mock_propagator = setup_basic_evaluator()
    
    satellite_sequence = [1, 2, 3]
    departure_times = [0.0, 3600.0]  # One less time than satellites
    
    try:
        evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "same length" in str(e)


def test_calculate_route_deltav_invalid_time_order():
    """Test delta-v calculation with non-ascending departure times."""
    evaluator, mock_propagator = setup_basic_evaluator()
    
    satellite_sequence = [1, 2, 3]
    departure_times = [0.0, 7200.0, 3600.0]  # Times not in ascending order
    
    try:
        evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "ascending order" in str(e)


def test_propagator_calculation_failure():
    """Test handling of orbital propagator calculation failures."""
    evaluator, mock_propagator = setup_basic_evaluator()
    
    # Set up propagator to raise exception
    mock_propagator.calculate_transfer_window.side_effect = Exception("Propagation failed")
    
    chromosome = RouteChromosome(
        satellite_sequence=[1, 2],
        departure_times=[0.0, 3600.0],
        total_deltav=0.0,
        is_valid=True
    )
    
    constraints = RouteConstraints(
        max_deltav_budget=20.0,
        max_mission_duration=10800.0,
        min_hops=1,
        max_hops=10
    )
    
    result = evaluator.evaluate_route(chromosome, constraints)
    
    assert result.is_valid == False
    assert result.total_deltav == float('inf')
    assert any("Delta-v calculation failed" in violation for violation in result.constraint_violations)


def test_clear_cache():
    """Test cache clearing functionality."""
    evaluator, mock_propagator = setup_basic_evaluator()
    
    # Add something to cache
    evaluator._deltav_cache[(1, 2, 0.0)] = 2.5
    assert len(evaluator._deltav_cache) == 1
    
    # Clear cache
    evaluator.clear_cache()
    assert len(evaluator._deltav_cache) == 0


def test_get_cache_stats():
    """Test cache statistics functionality."""
    evaluator, mock_propagator = setup_basic_evaluator()
    
    # Add some entries to cache
    evaluator._deltav_cache[(1, 2, 0.0)] = 2.5
    evaluator._deltav_cache[(2, 3, 3600.0)] = 3.0
    evaluator._deltav_cache[(1, 3, 1800.0)] = 2.8
    
    stats = evaluator.get_cache_stats()
    
    assert stats['cache_size'] == 3
    assert stats['unique_source_satellites'] == 2  # Satellites 1 and 2
    assert stats['unique_target_satellites'] == 2  # Satellites 2 and 3
    assert stats['unique_departure_times'] == 3   # Three different times


def test_validate_chromosome_basic_valid():
    """Test basic chromosome validation with valid chromosome."""
    evaluator, mock_propagator = setup_basic_evaluator()
    
    chromosome = RouteChromosome(
        satellite_sequence=[1, 2, 3],
        departure_times=[0.0, 3600.0, 7200.0],
        total_deltav=5.0,
        is_valid=True
    )
    
    errors = evaluator.validate_chromosome_basic(chromosome)
    assert len(errors) == 0


def test_validate_chromosome_basic_empty_sequence():
    """Test basic chromosome validation with empty sequence."""
    evaluator, mock_propagator = setup_basic_evaluator()
    
    chromosome = RouteChromosome(
        satellite_sequence=[],
        departure_times=[],
        total_deltav=0.0,
        is_valid=True
    )
    
    errors = evaluator.validate_chromosome_basic(chromosome)
    assert len(errors) == 1
    assert "cannot be empty" in errors[0]


def test_validate_chromosome_basic_invalid_satellite_id():
    """Test basic chromosome validation with invalid satellite ID."""
    evaluator, mock_propagator = setup_basic_evaluator()
    
    chromosome = RouteChromosome(
        satellite_sequence=[1, -5, 3],  # Negative satellite ID
        departure_times=[0.0, 3600.0, 7200.0],
        total_deltav=5.0,
        is_valid=True
    )
    
    errors = evaluator.validate_chromosome_basic(chromosome)
    assert len(errors) == 1
    assert "Invalid satellite ID: -5" in errors[0]


def test_validate_chromosome_basic_invalid_departure_time():
    """Test basic chromosome validation with invalid departure time."""
    evaluator, mock_propagator = setup_basic_evaluator()
    
    chromosome = RouteChromosome(
        satellite_sequence=[1, 2, 3],
        departure_times=[0.0, float('inf'), 7200.0],  # Invalid time
        total_deltav=5.0,
        is_valid=True
    )
    
    errors = evaluator.validate_chromosome_basic(chromosome)
    assert len(errors) >= 1
    assert any("Invalid departure time at index 1" in error for error in errors)


def test_validate_chromosome_basic_wrong_time_order():
    """Test basic chromosome validation with wrong time order."""
    evaluator, mock_propagator = setup_basic_evaluator()
    
    chromosome = RouteChromosome(
        satellite_sequence=[1, 2, 3],
        departure_times=[0.0, 7200.0, 3600.0],  # Wrong order
        total_deltav=5.0,
        is_valid=True
    )
    
    errors = evaluator.validate_chromosome_basic(chromosome)
    assert len(errors) == 1
    assert "ascending order at index 2" in errors[0]


def run_all_tests():
    """Run all fitness evaluator tests."""
    tests = [
        test_evaluator_initialization_success,
        test_evaluator_initialization_empty_satellites,
        test_evaluator_initialization_none_propagator,
        test_evaluator_initialization_propagator_satellite_mismatch,
        test_calculate_route_deltav_simple_route,
        test_calculate_route_deltav_multi_hop_route,
        test_calculate_route_deltav_caching,
        test_calculate_route_deltav_single_satellite,
        test_calculate_route_deltav_empty_route,
        test_evaluate_route_valid_route,
        test_check_constraints_hop_count_violations,
        test_check_constraints_deltav_budget_violation,
        test_check_constraints_forbidden_satellites,
        test_duplicate_satellites_in_route,
        test_invalid_satellite_ids,
        test_calculate_route_deltav_invalid_satellite,
        test_calculate_route_deltav_mismatched_lengths,
        test_calculate_route_deltav_invalid_time_order,
        test_propagator_calculation_failure,
        test_clear_cache,
        test_get_cache_stats,
        test_validate_chromosome_basic_valid,
        test_validate_chromosome_basic_empty_sequence,
        test_validate_chromosome_basic_invalid_satellite_id,
        test_validate_chromosome_basic_invalid_departure_time,
        test_validate_chromosome_basic_wrong_time_order
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

    
    def test_calculate_route_deltav_simple_route(self):
        """Test delta-v calculation for a simple two-satellite route."""
        # Set up mock transfer window
        self.mock_propagator.calculate_transfer_window.return_value = TransferWindow(
            departure_deltav=1.5,
            arrival_deltav=1.2,
            transfer_time=3600.0,
            optimal_departure_time=0.0,
            transfer_efficiency=0.9
        )
        
        satellite_sequence = [1, 2]
        departure_times = [0.0, 3600.0]
        
        total_deltav = self.evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        
        assert total_deltav == 2.7  # 1.5 + 1.2
        self.mock_propagator.calculate_transfer_window.assert_called_once_with(1, 2, 0.0)
    
    def test_calculate_route_deltav_multi_hop_route(self):
        """Test delta-v calculation for a multi-hop route."""
        # Set up different transfer windows for each hop
        transfer_windows = [
            TransferWindow(1.0, 1.0, 3600.0, 0.0, 0.8),      # 1->2
            TransferWindow(1.5, 1.2, 3600.0, 3600.0, 0.7),   # 2->3
            TransferWindow(0.8, 0.9, 3600.0, 7200.0, 0.9)    # 3->4
        ]
        
        self.mock_propagator.calculate_transfer_window.side_effect = transfer_windows
        
        satellite_sequence = [1, 2, 3, 4]
        departure_times = [0.0, 3600.0, 7200.0, 10800.0]
        
        total_deltav = self.evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        
        expected_deltav = 2.0 + 2.7 + 1.7  # Sum of all hops
        assert total_deltav == expected_deltav
        assert self.mock_propagator.calculate_transfer_window.call_count == 3
    
    def test_calculate_route_deltav_caching(self):
        """Test that delta-v calculations are cached properly."""
        self.mock_propagator.calculate_transfer_window.return_value = TransferWindow(
            departure_deltav=1.0,
            arrival_deltav=1.0,
            transfer_time=3600.0,
            optimal_departure_time=0.0,
            transfer_efficiency=0.8
        )
        
        satellite_sequence = [1, 2]
        departure_times = [0.0, 3600.0]
        
        # First call should use propagator
        deltav1 = self.evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        assert self.mock_propagator.calculate_transfer_window.call_count == 1
        
        # Second call should use cache
        deltav2 = self.evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        assert self.mock_propagator.calculate_transfer_window.call_count == 1  # No additional calls
        assert deltav1 == deltav2
    
    def test_calculate_route_deltav_single_satellite(self):
        """Test delta-v calculation for single satellite (no transfers)."""
        satellite_sequence = [1]
        departure_times = [0.0]
        
        total_deltav = self.evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        
        assert total_deltav == 0.0
        self.mock_propagator.calculate_transfer_window.assert_not_called()
    
    def test_calculate_route_deltav_empty_route(self):
        """Test delta-v calculation for empty route."""
        satellite_sequence = []
        departure_times = []
        
        total_deltav = self.evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        
        assert total_deltav == 0.0
        self.mock_propagator.calculate_transfer_window.assert_not_called()
    
    def test_evaluate_route_valid_route(self):
        """Test comprehensive route evaluation for a valid route."""
        # Set up mock transfer window
        self.mock_propagator.calculate_transfer_window.return_value = TransferWindow(
            departure_deltav=1.0,
            arrival_deltav=1.0,
            transfer_time=3600.0,
            optimal_departure_time=0.0,
            transfer_efficiency=0.8
        )
        
        chromosome = RouteChromosome(
            satellite_sequence=[1, 2, 3],
            departure_times=[0.0, 3600.0, 7200.0],
            total_deltav=0.0,
            is_valid=True,
            constraint_violations=[]
        )
        
        constraints = RouteConstraints(
            max_deltav_budget=10.0,
            max_mission_duration=10800.0,
            min_hops=1,
            max_hops=10
        )
        
        result = self.evaluator.evaluate_route(chromosome, constraints)
        
        assert isinstance(result, FitnessResult)
        assert result.is_valid == True
        assert result.total_deltav == 4.0  # 2 hops * 2.0 deltav each
        assert result.hop_count == 2
        assert result.mission_duration == 7200.0
        assert len(result.constraint_violations) == 0
        assert result.fitness_score > 0



