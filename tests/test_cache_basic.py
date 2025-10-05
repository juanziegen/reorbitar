"""
Basic test for fitness caching system functionality.
"""

import time
from src.fitness_cache import TimeAwareLRUCache, FitnessCacheManager


def test_basic_cache_functionality():
    """Test basic cache put/get functionality."""
    print("Testing basic cache functionality...")
    
    cache = TimeAwareLRUCache(max_size=100, ttl_seconds=3600.0, time_tolerance=60.0)
    
    # Test cache miss
    result = cache.get(1, 2, 1000.0)
    print(f"Cache miss result: {result}")
    assert result is None
    
    # Test cache put
    cache.put(1, 2, 1000.0, 1.5)
    print("Value cached successfully")
    
    # Test cache hit
    result = cache.get(1, 2, 1000.0)
    print(f"Cache hit result: {result}")
    assert result == 1.5
    
    # Test stats
    stats = cache.get_stats()
    print(f"Cache stats: hits={stats.cache_hits}, misses={stats.cache_misses}, hit_rate={stats.hit_rate}")
    
    print("✓ Basic cache functionality works")


def test_cache_manager():
    """Test cache manager functionality."""
    print("\nTesting cache manager...")
    
    manager = FitnessCacheManager()
    
    # Test delta-v caching
    result = manager.get_deltav(1, 2, 1000.0)
    print(f"Manager cache miss: {result}")
    assert result is None
    
    manager.put_deltav(1, 2, 1000.0, 2.5)
    result = manager.get_deltav(1, 2, 1000.0)
    print(f"Manager cache hit: {result}")
    assert result == 2.5
    
    # Test stats
    stats = manager.get_combined_stats()
    print(f"Manager stats: {stats['deltav_cache']['size']} entries")
    
    print("✓ Cache manager works")


def test_integration_with_route_evaluator():
    """Test integration with route fitness evaluator."""
    print("\nTesting integration with route evaluator...")
    
    # Mock the required components
    from unittest.mock import Mock
    from src.route_fitness_evaluator import RouteFitnessEvaluator
    from src.fitness_cache import FitnessCacheManager
    from src.tle_parser import SatelliteData
    from src.orbital_propagator import OrbitalPropagator, TransferWindow
    
    # Create mock satellites
    satellites = []
    for i in range(1, 4):
        sat = Mock(spec=SatelliteData)
        sat.catalog_number = i
        sat.name = f"SAT-{i}"
        satellites.append(sat)
    
    # Create mock orbital propagator
    propagator = Mock(spec=OrbitalPropagator)
    propagator.get_satellite_ids.return_value = [1, 2, 3]
    
    def mock_transfer_window(source_id, target_id, departure_time):
        window = Mock(spec=TransferWindow)
        window.departure_deltav = 0.5
        window.arrival_deltav = 0.3
        return window
    
    propagator.calculate_transfer_window.side_effect = mock_transfer_window
    
    # Create cache manager
    cache_manager = FitnessCacheManager()
    
    # Create evaluator
    evaluator = RouteFitnessEvaluator(
        satellites=satellites,
        orbital_propagator=propagator,
        cache_manager=cache_manager
    )
    
    # Test delta-v calculation with caching
    satellite_sequence = [1, 2, 3]
    departure_times = [1000.0, 2000.0, 3000.0]
    
    # First calculation - should call propagator
    deltav1 = evaluator.calculate_route_deltav(satellite_sequence, departure_times)
    print(f"First calculation: {deltav1} km/s")
    print(f"Propagator calls: {propagator.calculate_transfer_window.call_count}")
    
    # Second calculation - should use cache
    propagator.calculate_transfer_window.reset_mock()
    deltav2 = evaluator.calculate_route_deltav(satellite_sequence, departure_times)
    print(f"Second calculation: {deltav2} km/s")
    print(f"Propagator calls after reset: {propagator.calculate_transfer_window.call_count}")
    
    assert deltav1 == deltav2
    assert propagator.calculate_transfer_window.call_count == 0  # Should use cache
    
    # Check cache stats
    stats = evaluator.get_cache_stats()
    print(f"Cache stats: {stats['advanced_caching']['deltav_cache']['stats'].hit_rate}")
    
    print("✓ Integration with route evaluator works")


if __name__ == "__main__":
    try:
        test_basic_cache_functionality()
        test_cache_manager()
        test_integration_with_route_evaluator()
        print("\n🎉 All basic tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()