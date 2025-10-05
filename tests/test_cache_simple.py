"""
Simple test runner for fitness caching system without pytest dependency.
"""

import sys
import time
import traceback
from src.fitness_cache import CacheEntry, CacheStats, TimeAwareLRUCache, FitnessCacheManager


def test_cache_entry():
    """Test CacheEntry functionality."""
    print("Testing CacheEntry...")
    
    entry = CacheEntry(value=1.5, timestamp=1000.0)
    assert entry.value == 1.5
    assert entry.timestamp == 1000.0
    assert entry.access_count == 0
    
    initial_access = entry.last_access
    time.sleep(0.01)
    entry.update_access()
    assert entry.access_count == 1
    assert entry.last_access > initial_access
    
    print("✓ CacheEntry tests passed")


def test_cache_stats():
    """Test CacheStats functionality."""
    print("Testing CacheStats...")
    
    stats = CacheStats()
    assert stats.hit_rate == 0.0
    assert stats.miss_rate == 1.0
    
    stats = CacheStats(total_requests=100, cache_hits=75, cache_misses=25)
    assert stats.hit_rate == 0.75
    assert stats.miss_rate == 0.25
    
    print("✓ CacheStats tests passed")


def test_time_aware_lru_cache():
    """Test TimeAwareLRUCache functionality."""
    print("Testing TimeAwareLRUCache...")
    
    # Test basic functionality
    cache = TimeAwareLRUCache(max_size=100, ttl_seconds=3600.0)
    
    # Test cache miss
    result = cache.get(1, 2, 1000.0)
    assert result is None
    assert cache._stats.cache_misses == 1
    
    # Test cache put and hit
    cache.put(1, 2, 1000.0, 1.5)
    result = cache.get(1, 2, 1000.0)
    assert result == 1.5
    assert cache._stats.cache_hits == 1
    
    # Test time normalization (within tolerance)
    result = cache.get(1, 2, 1030.0)  # Within 60s tolerance
    assert result == 1.5
    
    # Test cache size limit
    cache = TimeAwareLRUCache(max_size=3, ttl_seconds=3600.0)
    cache.put(1, 2, 1000.0, 1.0)
    cache.put(2, 3, 1000.0, 2.0)
    cache.put(3, 4, 1000.0, 3.0)
    cache.put(4, 5, 1000.0, 4.0)  # Should evict oldest
    
    assert len(cache._cache) == 3
    assert cache._stats.evictions == 1
    assert cache.get(1, 2, 1000.0) is None  # Should be evicted
    
    # Test satellite invalidation
    cache.clear()
    cache.put(1, 2, 1000.0, 1.0)
    cache.put(2, 3, 1000.0, 2.0)
    cache.put(3, 4, 1000.0, 3.0)
    
    cache.invalidate_satellites([2])
    assert cache.get(1, 2, 1000.0) is None  # Contains satellite 2
    assert cache.get(2, 3, 1000.0) is None  # Contains satellite 2
    assert cache.get(3, 4, 1000.0) == 3.0   # Doesn't contain satellite 2
    
    # Test time range invalidation
    cache.clear()
    cache.put(1, 2, 1000.0, 1.0)
    cache.put(2, 3, 2000.0, 2.0)
    cache.put(3, 4, 3000.0, 3.0)
    
    cache.invalidate_time_range(1500.0, 2500.0)
    assert cache.get(1, 2, 1000.0) == 1.0   # Outside range
    assert cache.get(2, 3, 2000.0) is None  # Inside range
    assert cache.get(3, 4, 3000.0) == 3.0   # Outside range
    
    print("✓ TimeAwareLRUCache tests passed")


def test_fitness_cache_manager():
    """Test FitnessCacheManager functionality."""
    print("Testing FitnessCacheManager...")
    
    manager = FitnessCacheManager()
    
    # Test delta-v caching
    result = manager.get_deltav(1, 2, 1000.0)
    assert result is None
    
    manager.put_deltav(1, 2, 1000.0, 1.5)
    result = manager.get_deltav(1, 2, 1000.0)
    assert result == 1.5
    
    # Test satellite invalidation
    manager.put_deltav(1, 2, 1000.0, 1.0)
    manager.put_deltav(2, 3, 1000.0, 2.0)
    manager.put_deltav(3, 4, 1000.0, 3.0)
    
    manager.invalidate_satellite_data([2])
    assert manager.get_deltav(1, 2, 1000.0) is None
    assert manager.get_deltav(2, 3, 1000.0) is None
    assert manager.get_deltav(3, 4, 1000.0) == 3.0
    
    # Test cache optimization
    original_size = manager.deltav_cache.max_size
    manager.optimize_cache_sizes(5000)
    assert manager.deltav_cache.max_size != original_size
    
    # Test combined stats
    stats = manager.get_combined_stats()
    assert 'deltav_cache' in stats
    assert 'fitness_cache' in stats
    assert 'total_memory_estimate' in stats
    
    print("✓ FitnessCacheManager tests passed")


def test_cache_performance():
    """Test cache performance improvement."""
    print("Testing cache performance...")
    
    manager = FitnessCacheManager()
    
    # Simulate multiple calculations
    calculations = [(i, i+1, 1000.0 + i) for i in range(1, 50)]
    
    # First pass - cache misses
    start_time = time.time()
    for source, target, dep_time in calculations:
        result = manager.get_deltav(source, target, dep_time)
        if result is None:
            # Simulate calculation
            time.sleep(0.001)  # 1ms calculation time
            manager.put_deltav(source, target, dep_time, 1.5)
    first_pass_time = time.time() - start_time
    
    # Second pass - cache hits
    start_time = time.time()
    for source, target, dep_time in calculations:
        result = manager.get_deltav(source, target, dep_time)
        assert result == 1.5  # Should hit cache
    second_pass_time = time.time() - start_time
    
    # Second pass should be much faster
    assert second_pass_time < first_pass_time * 0.5
    
    # Check hit rate
    stats = manager.deltav_cache.get_stats()
    assert stats.hit_rate > 0.5
    
    print(f"✓ Cache performance test passed (speedup: {first_pass_time/second_pass_time:.1f}x)")


def run_all_tests():
    """Run all tests."""
    print("Running fitness caching system tests...\n")
    
    tests = [
        test_cache_entry,
        test_cache_stats,
        test_time_aware_lru_cache,
        test_fitness_cache_manager,
        test_cache_performance
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            traceback.print_exc()
            failed += 1
        print()
    
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)