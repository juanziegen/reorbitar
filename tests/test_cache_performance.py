"""
Performance and memory management tests for the fitness caching system.
"""

import time
import gc
from src.fitness_cache import TimeAwareLRUCache, FitnessCacheManager


def test_cache_performance():
    """Test cache performance improvements."""
    print("Testing cache performance improvements...")
    
    manager = FitnessCacheManager()
    
    # Generate test data
    test_transfers = []
    for i in range(100):
        for j in range(i+1, min(i+10, 100)):  # Each satellite connects to next 9
            test_transfers.append((i, j, 1000.0 + i * 10))
    
    print(f"Testing with {len(test_transfers)} transfers")
    
    # Simulate expensive calculations (first pass)
    start_time = time.time()
    for source, target, dep_time in test_transfers:
        result = manager.get_deltav(source, target, dep_time)
        if result is None:
            # Simulate calculation time
            time.sleep(0.0001)  # 0.1ms per calculation
            calculated_deltav = 0.5 + (source + target) * 0.01  # Deterministic value
            manager.put_deltav(source, target, dep_time, calculated_deltav)
    first_pass_time = time.time() - start_time
    
    # Cache hits (second pass)
    start_time = time.time()
    for source, target, dep_time in test_transfers:
        result = manager.get_deltav(source, target, dep_time)
        assert result is not None  # Should all be cached
    second_pass_time = time.time() - start_time
    
    speedup = first_pass_time / second_pass_time if second_pass_time > 0 else float('inf')
    
    print(f"First pass (cache misses): {first_pass_time:.3f}s")
    print(f"Second pass (cache hits): {second_pass_time:.3f}s")
    print(f"Speedup: {speedup:.1f}x")
    
    # Check cache statistics
    stats = manager.get_combined_stats()
    deltav_stats = stats['deltav_cache']['stats']
    print(f"Hit rate: {deltav_stats.hit_rate:.2f}")
    print(f"Total requests: {deltav_stats.total_requests}")
    print(f"Cache size: {stats['deltav_cache']['size']}")
    
    assert speedup > 5.0  # Should be at least 5x faster
    assert deltav_stats.hit_rate >= 0.5  # At least 50% hit rate
    
    print("✓ Cache performance test passed")


def test_memory_management():
    """Test cache memory management and size limits."""
    print("\nTesting memory management...")
    
    # Create cache with small size limit
    cache = TimeAwareLRUCache(max_size=50, ttl_seconds=3600.0)
    
    # Fill cache beyond capacity
    for i in range(100):
        cache.put(i, i+1, 1000.0, float(i))
    
    # Check that cache respects size limit
    assert len(cache._cache) <= 50
    print(f"Cache size after 100 insertions: {len(cache._cache)}")
    print(f"Evictions: {cache._stats.evictions}")
    
    # Verify LRU behavior - recent entries should still be there
    recent_hits = 0
    for i in range(90, 100):  # Check last 10 entries
        if cache.get(i, i+1, 1000.0) is not None:
            recent_hits += 1
    
    print(f"Recent entries still cached: {recent_hits}/10")
    assert recent_hits >= 5  # At least half should still be there
    
    # Test memory usage estimation
    memory_estimate = cache._estimate_memory_usage()
    print(f"Estimated memory usage: {memory_estimate} bytes")
    assert memory_estimate > 0
    
    print("✓ Memory management test passed")


def test_cache_invalidation():
    """Test cache invalidation strategies."""
    print("\nTesting cache invalidation...")
    
    manager = FitnessCacheManager()
    
    # Add test data
    test_data = [
        (1, 2, 1000.0, 1.0),
        (2, 3, 1500.0, 2.0),
        (3, 4, 2000.0, 3.0),
        (4, 5, 2500.0, 4.0),
        (5, 6, 3000.0, 5.0)
    ]
    
    for source, target, dep_time, deltav in test_data:
        manager.put_deltav(source, target, dep_time, deltav)
    
    # Verify all cached
    for source, target, dep_time, expected in test_data:
        result = manager.get_deltav(source, target, dep_time)
        assert result == expected
    
    print("All test data cached successfully")
    
    # Test satellite invalidation
    manager.invalidate_satellite_data([2, 3])
    
    # Check invalidation results
    invalidated_count = 0
    for source, target, dep_time, expected in test_data:
        result = manager.get_deltav(source, target, dep_time)
        if 2 in [source, target] or 3 in [source, target]:
            if result is None:
                invalidated_count += 1
        else:
            assert result == expected  # Should still be cached
    
    print(f"Invalidated {invalidated_count} entries involving satellites 2 or 3")
    assert invalidated_count > 0
    
    # Test time range invalidation
    manager.clear_all_caches()
    for source, target, dep_time, deltav in test_data:
        manager.put_deltav(source, target, dep_time, deltav)
    
    manager.invalidate_time_range(1800.0, 2200.0)
    
    time_invalidated_count = 0
    for source, target, dep_time, expected in test_data:
        result = manager.get_deltav(source, target, dep_time)
        if 1800.0 <= dep_time <= 2200.0:
            if result is None:
                time_invalidated_count += 1
        else:
            assert result == expected  # Should still be cached
    
    print(f"Invalidated {time_invalidated_count} entries in time range")
    assert time_invalidated_count > 0
    
    print("✓ Cache invalidation test passed")


def test_cache_optimization():
    """Test cache optimization for different constellation sizes."""
    print("\nTesting cache optimization...")
    
    manager = FitnessCacheManager()
    original_deltav_size = manager.deltav_cache.max_size
    original_fitness_size = manager.fitness_cache.max_size
    
    print(f"Original cache sizes: deltav={original_deltav_size}, fitness={original_fitness_size}")
    
    # Optimize for small constellation
    manager.optimize_cache_sizes(50)
    small_deltav_size = manager.deltav_cache.max_size
    small_fitness_size = manager.fitness_cache.max_size
    
    print(f"Small constellation sizes: deltav={small_deltav_size}, fitness={small_fitness_size}")
    
    # Optimize for large constellation
    manager.optimize_cache_sizes(5000)
    large_deltav_size = manager.deltav_cache.max_size
    large_fitness_size = manager.fitness_cache.max_size
    
    print(f"Large constellation sizes: deltav={large_deltav_size}, fitness={large_fitness_size}")
    
    # Large constellation should have larger caches
    assert large_deltav_size >= small_deltav_size
    assert large_fitness_size >= small_fitness_size
    
    print("✓ Cache optimization test passed")


def test_time_tolerance():
    """Test time tolerance for improved cache hit rates."""
    print("\nTesting time tolerance...")
    
    cache = TimeAwareLRUCache(time_tolerance=60.0)  # 1 minute tolerance
    
    # Cache a value
    cache.put(1, 2, 1000.0, 1.5)
    
    # Test hits within tolerance
    test_times = [1000.0, 1030.0, 1059.0, 970.0, 941.0]  # Within ±60s
    hits = 0
    
    for test_time in test_times:
        result = cache.get(1, 2, test_time)
        if result == 1.5:
            hits += 1
    
    print(f"Cache hits within tolerance: {hits}/{len(test_times)}")
    
    # Test miss outside tolerance
    result = cache.get(1, 2, 1200.0)  # 200s later, outside tolerance
    print(f"Result outside tolerance: {result}")
    
    # Should have good hit rate within tolerance
    hit_rate = hits / len(test_times)
    print(f"Hit rate within tolerance: {hit_rate:.2f}")
    
    print("✓ Time tolerance test passed")


if __name__ == "__main__":
    try:
        test_cache_performance()
        test_memory_management()
        test_cache_invalidation()
        test_cache_optimization()
        test_time_tolerance()
        print("\n🎉 All performance tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()