"""
Tests for RouteFitnessEvaluator Caching Integration

This module tests the integration of the advanced caching system with the
RouteFitnessEvaluator to ensure proper functionality and performance improvements.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from src.route_fitness_evaluator import RouteFitnessEvaluator
from src.fitness_cache import FitnessCacheManager
from src.genetic_algorithm import RouteChromosome, RouteConstraints
from src.orbital_propagator import OrbitalPropagator, TransferWindow
from src.tle_parser import SatelliteData


@pytest.fixture
def mock_satellites():
    """Create mock satellite data for testing."""
    satellites = []
    for i in range(1, 6):  # 5 satellites
        sat = Mock(spec=SatelliteData)
        sat.catalog_number = i
        sat.name = f"SAT-{i}"
        satellites.append(sat)
    return satellites


@pytest.fixture
def mock_orbital_propagator():
    """Create mock orbital propagator for testing."""
    propagator = Mock(spec=OrbitalPropagator)
    propagator.get_satellite_ids.return_value = [1, 2, 3, 4, 5]
    
    # Mock transfer window calculation
    def mock_transfer_window(source_id, target_id, departure_time):
        window = Mock(spec=TransferWindow)
        window.departure_deltav = 0.5
        window.arrival_deltav = 0.3
        return window
    
    propagator.calculate_transfer_window.side_effect = mock_transfer_window
    return propagator


@pytest.fixture
def cache_manager():
    """Create cache manager for testing."""
    return FitnessCacheManager(
        max_deltav_cache_size=100,
        max_fitness_cache_size=50,
        deltav_ttl=3600.0,
        fitness_ttl=1800.0
    )


@pytest.fixture
def fitness_evaluator(mock_satellites, mock_orbital_propagator, cache_manager):
    """Create fitness evaluator with caching for testing."""
    return RouteFitnessEvaluator(
        satellites=mock_satellites,
        orbital_propagator=mock_orbital_propagator,
        cache_manager=cache_manager
    )


class TestRouteFitnessEvaluatorCaching:
    """Test caching functionality in RouteFitnessEvaluator."""
    
    def test_evaluator_initialization_with_cache(self, mock_satellites, mock_orbital_propagator):
        """Test evaluator initialization with custom cache manager."""
        cache_manager = FitnessCacheManager()
        evaluator = RouteFitnessEvaluator(
            satellites=mock_satellites,
            orbital_propagator=mock_orbital_propagator,
            cache_manager=cache_manager
        )
        
        assert evaluator.cache_manager is cache_manager
        assert len(evaluator.satellites) == 5
    
    def test_evaluator_initialization_default_cache(self, mock_satellites, mock_orbital_propagator):
        """Test evaluator initialization with default cache manager."""
        evaluator = RouteFitnessEvaluator(
            satellites=mock_satellites,
            orbital_propagator=mock_orbital_propagator
        )
        
        assert evaluator.cache_manager is not None
        assert isinstance(evaluator.cache_manager, FitnessCacheManager)
    
    def test_deltav_calculation_caching(self, fitness_evaluator):
        """Test that delta-v calculations are properly cached."""
        satellite_sequence = [1, 2, 3]
        departure_times = [1000.0, 2000.0, 3000.0]
        
        # First calculation - should call orbital propagator
        deltav1 = fitness_evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        
        # Verify orbital propagator was called
        assert fitness_evaluator.orbital_propagator.calculate_transfer_window.call_count == 2
        
        # Second calculation - should use cache
        fitness_evaluator.orbital_propagator.calculate_transfer_window.reset_mock()
        deltav2 = fitness_evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        
        # Should not call orbital propagator again
        assert fitness_evaluator.orbital_propagator.calculate_transfer_window.call_count == 0
        assert deltav1 == deltav2
    
    def test_cache_hit_statistics(self, fitness_evaluator):
        """Test that cache statistics are properly tracked."""
        satellite_sequence = [1, 2]
        departure_times = [1000.0, 2000.0]
        
        # Get initial stats
        initial_stats = fitness_evaluator.get_cache_stats()
        initial_requests = initial_stats['advanced_caching']['deltav_cache']['stats'].total_requests
        
        # First calculation - cache miss
        fitness_evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        
        # Second calculation - cache hit
        fitness_evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        
        # Check updated stats
        final_stats = fitness_evaluator.get_cache_stats()
        deltav_stats = final_stats['advanced_caching']['deltav_cache']['stats']
        
        assert deltav_stats.total_requests > initial_requests
        assert deltav_stats.cache_hits > 0
    
    def test_cache_invalidation_satellites(self, fitness_evaluator):
        """Test satellite-based cache invalidation."""
        satellite_sequence = [1, 2, 3]
        departure_times = [1000.0, 2000.0, 3000.0]
        
        # Calculate and cache
        deltav1 = fitness_evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        
        # Invalidate satellite 2
        fitness_evaluator.invalidate_satellite_cache([2])
        
        # Recalculate - should call orbital propagator again for transfers involving satellite 2
        fitness_evaluator.orbital_propagator.calculate_transfer_window.reset_mock()
        deltav2 = fitness_evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        
        # Should call orbital propagator for invalidated transfers
        assert fitness_evaluator.orbital_propagator.calculate_transfer_window.call_count > 0
        assert deltav1 == deltav2  # Values should be the same
    
    def test_cache_invalidation_time_range(self, fitness_evaluator):
        """Test time-based cache invalidation."""
        satellite_sequence = [1, 2, 3]
        departure_times = [1000.0, 2000.0, 3000.0]
        
        # Calculate and cache
        deltav1 = fitness_evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        
        # Invalidate time range covering some transfers
        fitness_evaluator.invalidate_time_range_cache(1500.0, 2500.0)
        
        # Recalculate - should call orbital propagator for invalidated time range
        fitness_evaluator.orbital_propagator.calculate_transfer_window.reset_mock()
        deltav2 = fitness_evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        
        # Should call orbital propagator for transfers in invalidated time range
        assert fitness_evaluator.orbital_propagator.calculate_transfer_window.call_count > 0
        assert deltav1 == deltav2
    
    def test_cache_maintenance(self, fitness_evaluator):
        """Test cache maintenance functionality."""
        # Add some cached values
        satellite_sequence = [1, 2]
        departure_times = [1000.0, 2000.0]
        fitness_evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        
        # Perform maintenance
        fitness_evaluator.perform_cache_maintenance()
        
        # Should not raise any exceptions
        assert True
    
    def test_cache_optimization(self, fitness_evaluator):
        """Test cache optimization for constellation size."""
        initial_size = fitness_evaluator.cache_manager.deltav_cache.max_size
        
        # Optimize for larger constellation
        fitness_evaluator.optimize_cache_for_constellation(1000)
        
        # Cache size should be adjusted
        new_size = fitness_evaluator.cache_manager.deltav_cache.max_size
        assert new_size != initial_size
    
    def test_memory_usage_tracking(self, fitness_evaluator):
        """Test memory usage tracking."""
        # Add some cached values
        satellite_sequence = [1, 2, 3, 4]
        departure_times = [1000.0, 2000.0, 3000.0, 4000.0]
        fitness_evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        
        # Get memory usage
        memory_usage = fitness_evaluator.get_cache_memory_usage()
        
        assert 'advanced_cache_bytes' in memory_usage
        assert 'legacy_cache_bytes' in memory_usage
        assert 'total_bytes' in memory_usage
        assert memory_usage['total_bytes'] > 0
    
    def test_route_and_constraints_hashing(self, fitness_evaluator):
        """Test route and constraints hashing for caching."""
        chromosome = RouteChromosome(
            satellite_sequence=[1, 2, 3],
            departure_times=[1000.0, 2000.0, 3000.0]
        )
        
        constraints = RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400.0,
            min_hops=1,
            max_hops=10
        )
        
        # Test hashing
        route_hash1 = fitness_evaluator.create_route_hash(chromosome)
        route_hash2 = fitness_evaluator.create_route_hash(chromosome)
        constraints_hash1 = fitness_evaluator.create_constraints_hash(constraints)
        constraints_hash2 = fitness_evaluator.create_constraints_hash(constraints)
        
        # Same inputs should produce same hashes
        assert route_hash1 == route_hash2
        assert constraints_hash1 == constraints_hash2
        
        # Different inputs should produce different hashes
        different_chromosome = RouteChromosome(
            satellite_sequence=[1, 3, 2],  # Different order
            departure_times=[1000.0, 2000.0, 3000.0]
        )
        different_route_hash = fitness_evaluator.create_route_hash(different_chromosome)
        assert route_hash1 != different_route_hash
    
    def test_clear_all_caches(self, fitness_evaluator):
        """Test clearing all caches."""
        # Add some cached values
        satellite_sequence = [1, 2]
        departure_times = [1000.0, 2000.0]
        fitness_evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        
        # Verify caches have content
        stats_before = fitness_evaluator.get_cache_stats()
        assert stats_before['advanced_caching']['deltav_cache']['size'] > 0
        
        # Clear caches
        fitness_evaluator.clear_cache()
        
        # Verify caches are empty
        stats_after = fitness_evaluator.get_cache_stats()
        assert stats_after['advanced_caching']['deltav_cache']['size'] == 0
        assert stats_after['legacy_caching']['legacy_cache_size'] == 0
    
    def test_legacy_cache_compatibility(self, fitness_evaluator):
        """Test backward compatibility with legacy cache."""
        # Manually add to legacy cache
        fitness_evaluator._deltav_cache[(1, 2, 1000.0)] = 1.5
        
        # Should use legacy cache if advanced cache misses
        satellite_sequence = [1, 2]
        departure_times = [1000.0, 2000.0]
        
        # Mock advanced cache to return None
        with patch.object(fitness_evaluator.cache_manager, 'get_deltav', return_value=None):
            deltav = fitness_evaluator.calculate_route_deltav(satellite_sequence, departure_times)
            
            # Should have used legacy cache for first hop
            assert deltav > 0
    
    def test_cache_performance_under_load(self, fitness_evaluator):
        """Test cache performance under heavy load."""
        # Generate many different routes
        routes = []
        for i in range(50):
            routes.append(([1, 2, 3], [1000.0 + i, 2000.0 + i, 3000.0 + i]))
        
        # Time first pass (cache misses)
        start_time = time.time()
        for satellite_sequence, departure_times in routes:
            fitness_evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        first_pass_time = time.time() - start_time
        
        # Time second pass (cache hits)
        start_time = time.time()
        for satellite_sequence, departure_times in routes:
            fitness_evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        second_pass_time = time.time() - start_time
        
        # Second pass should be significantly faster
        assert second_pass_time < first_pass_time * 0.5  # At least 50% faster
        
        # Check cache statistics
        stats = fitness_evaluator.get_cache_stats()
        deltav_stats = stats['advanced_caching']['deltav_cache']['stats']
        assert deltav_stats.hit_rate > 0.5  # At least 50% hit rate


class TestCacheErrorHandling:
    """Test error handling in caching system."""
    
    def test_cache_with_orbital_propagator_error(self, fitness_evaluator):
        """Test cache behavior when orbital propagator fails."""
        # Mock orbital propagator to raise exception
        fitness_evaluator.orbital_propagator.calculate_transfer_window.side_effect = Exception("Propagation failed")
        
        satellite_sequence = [1, 2]
        departure_times = [1000.0, 2000.0]
        
        # Should raise exception and not cache invalid result
        with pytest.raises(ValueError, match="Failed to calculate transfer"):
            fitness_evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        
        # Cache should remain empty
        stats = fitness_evaluator.get_cache_stats()
        assert stats['advanced_caching']['deltav_cache']['size'] == 0
    
    def test_cache_with_invalid_satellite_ids(self, fitness_evaluator):
        """Test cache behavior with invalid satellite IDs."""
        satellite_sequence = [1, 999]  # 999 doesn't exist
        departure_times = [1000.0, 2000.0]
        
        # Should raise exception
        with pytest.raises(ValueError, match="Satellite 999 not found"):
            fitness_evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        
        # Cache should remain empty
        stats = fitness_evaluator.get_cache_stats()
        assert stats['advanced_caching']['deltav_cache']['size'] == 0
    
    def test_cache_with_invalid_departure_times(self, fitness_evaluator):
        """Test cache behavior with invalid departure times."""
        satellite_sequence = [1, 2]
        departure_times = [2000.0, 1000.0]  # Not in ascending order
        
        # Should raise exception
        with pytest.raises(ValueError, match="Departure times must be in ascending order"):
            fitness_evaluator.calculate_route_deltav(satellite_sequence, departure_times)
        
        # Cache should remain empty
        stats = fitness_evaluator.get_cache_stats()
        assert stats['advanced_caching']['deltav_cache']['size'] == 0


if __name__ == "__main__":
    pytest.main([__file__])