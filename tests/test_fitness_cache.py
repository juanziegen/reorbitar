"""
Tests for the Fitness Caching System

This module tests the advanced caching system for delta-v calculations and fitness evaluations
to ensure proper functionality, performance, and memory management.
"""

import pytest
import time
import math
from unittest.mock import Mock, patch
from src.fitness_cache import (
    CacheEntry, CacheStats, TimeAwareLRUCache, FitnessCacheManager
)


class TestCacheEntry:
    """Test the CacheEntry dataclass."""
    
    def test_cache_entry_creation(self):
        """Test basic cache entry creation."""
        entry = CacheEntry(value=1.5, timestamp=1000.0)
        assert entry.value == 1.5
        assert entry.timestamp == 1000.0
        assert entry.access_count == 0
        assert entry.last_access > 0
    
    def test_cache_entry_update_access(self):
        """Test access count updates."""
        entry = CacheEntry(value=1.5, timestamp=1000.0)
        initial_access = entry.last_access
        initial_count = entry.access_count
        
        time.sleep(0.01)  # Small delay to ensure time difference
        entry.update_access()
        
        assert entry.access_count == initial_count + 1
        assert entry.last_access > initial_access


class TestCacheStats:
    """Test the CacheStats dataclass."""
    
    def test_cache_stats_creation(self):
        """Test basic cache stats creation."""
        stats = CacheStats()
        assert stats.total_requests == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.hit_rate == 0.0
        assert stats.miss_rate == 1.0
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(total_requests=100, cache_hits=75, cache_misses=25)
        assert stats.hit_rate == 0.75
        assert stats.miss_rate == 0.25
    
    def test_zero_requests_hit_rate(self):
        """Test hit rate with zero requests."""
        stats = CacheStats(total_requests=0, cache_hits=0, cache_misses=0)
        assert stats.hit_rate == 0.0
        assert stats.miss_rate == 1.0


class TestTimeAwareLRUCache:
    """Test the TimeAwareLRUCache class."""
    
    def test_cache_creation(self):
        """Test basic cache creation."""
        cache = TimeAwareLRUCache(max_size=100, ttl_seconds=3600.0, time_tolerance=60.0)
        assert cache.max_size == 100
        assert cache.ttl_seconds == 3600.0
        assert cache.time_tolerance == 60.0
        assert len(cache._cache) == 0
    
    def test_cache_put_and_get(self):
        """Test basic put and get operations."""
        cache = TimeAwareLRUCache(max_size=100, ttl_seconds=3600.0)
        
        # Test cache miss
        result = cache.get(1, 2, 1000.0)
        assert result is None
        assert cache._stats.cache_misses == 1
        assert cache._stats.total_requests == 1
        
        # Test cache put and hit
        cache.put(1, 2, 1000.0, 1.5)
        result = cache.get(1, 2, 1000.0)
        assert result == 1.5
        assert cache._stats.cache_hits == 1
        assert cache._stats.total_requests == 2
    
    def test_time_normalization(self):
        """Test time normalization for cache keys."""
        cache = TimeAwareLRUCache(time_tolerance=60.0)
        
        # Times within tolerance should use same cache key
        cache.put(1, 2, 1000.0, 1.5)
        result1 = cache.get(1, 2, 1030.0)  # Within 60s tolerance
        result2 = cache.get(1, 2, 1070.0)  # Outside tolerance
        
        assert result1 == 1.5  # Should hit cache
        assert result2 == 1.5  # Should also hit cache (same bucket)
    
    def test_cache_expiration(self):
        """Test cache entry expiration."""
        cache = TimeAwareLRUCache(max_size=100, ttl_seconds=0.1)  # Very short TTL
        
        cache.put(1, 2, 1000.0, 1.5)
        
        # Should hit immediately
        result = cache.get(1, 2, 1000.0)
        assert result == 1.5
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should miss after expiration
        result = cache.get(1, 2, 1000.0)
        assert result is None
        assert cache._stats.cache_misses == 1
    
    def test_cache_size_limit(self):
        """Test cache size limit and LRU eviction."""
        cache = TimeAwareLRUCache(max_size=3, ttl_seconds=3600.0)
        
        # Fill cache to capacity
        cache.put(1, 2, 1000.0, 1.0)
        cache.put(2, 3, 1000.0, 2.0)
        cache.put(3, 4, 1000.0, 3.0)
        assert len(cache._cache) == 3
        
        # Add one more - should evict oldest
        cache.put(4, 5, 1000.0, 4.0)
        assert len(cache._cache) == 3
        assert cache._stats.evictions == 1
        
        # First entry should be evicted
        result = cache.get(1, 2, 1000.0)
        assert result is None
        
        # Others should still be there
        assert cache.get(2, 3, 1000.0) == 2.0
        assert cache.get(3, 4, 1000.0) == 3.0
        assert cache.get(4, 5, 1000.0) == 4.0
    
    def test_lru_ordering(self):
        """Test LRU ordering is maintained."""
        cache = TimeAwareLRUCache(max_size=3, ttl_seconds=3600.0)
        
        # Fill cache
        cache.put(1, 2, 1000.0, 1.0)
        cache.put(2, 3, 1000.0, 2.0)
        cache.put(3, 4, 1000.0, 3.0)
        
        # Access first entry to make it most recent
        cache.get(1, 2, 1000.0)
        
        # Add new entry - should evict second entry (least recent)
        cache.put(4, 5, 1000.0, 4.0)
        
        # First entry should still be there (was accessed recently)
        assert cache.get(1, 2, 1000.0) == 1.0
        # Second entry should be evicted
        assert cache.get(2, 3, 1000.0) is None
        # Others should be there
        assert cache.get(3, 4, 1000.0) == 3.0
        assert cache.get(4, 5, 1000.0) == 4.0
    
    def test_invalidate_time_range(self):
        """Test time range invalidation."""
        cache = TimeAwareLRUCache(time_tolerance=60.0)
        
        # Add entries at different times
        cache.put(1, 2, 1000.0, 1.0)
        cache.put(2, 3, 2000.0, 2.0)
        cache.put(3, 4, 3000.0, 3.0)
        
        # Invalidate middle time range
        cache.invalidate_time_range(1500.0, 2500.0)
        
        # First and third should remain
        assert cache.get(1, 2, 1000.0) == 1.0
        assert cache.get(3, 4, 3000.0) == 3.0
        # Second should be invalidated
        assert cache.get(2, 3, 2000.0) is None
    
    def test_invalidate_satellites(self):
        """Test satellite invalidation."""
        cache = TimeAwareLRUCache()
        
        # Add entries with different satellites
        cache.put(1, 2, 1000.0, 1.0)
        cache.put(2, 3, 1000.0, 2.0)
        cache.put(3, 4, 1000.0, 3.0)
        cache.put(4, 5, 1000.0, 4.0)
        
        # Invalidate entries involving satellites 2 and 3
        cache.invalidate_satellites([2, 3])
        
        # Entries with satellites 2 or 3 should be invalidated
        assert cache.get(1, 2, 1000.0) is None  # Contains satellite 2
        assert cache.get(2, 3, 1000.0) is None  # Contains satellites 2 and 3
        assert cache.get(3, 4, 1000.0) is None  # Contains satellite 3
        assert cache.get(4, 5, 1000.0) == 4.0   # Doesn't contain 2 or 3
    
    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = TimeAwareLRUCache(ttl_seconds=0.1)
        
        # Add entries
        cache.put(1, 2, 1000.0, 1.0)
        cache.put(2, 3, 1000.0, 2.0)
        assert len(cache._cache) == 2
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Add one more entry (not expired)
        cache.put(3, 4, 1000.0, 3.0)
        
        # Cleanup expired entries
        cache.cleanup_expired()
        
        # Only the new entry should remain
        assert len(cache._cache) == 1
        assert cache.get(3, 4, 1000.0) == 3.0
        assert cache.get(1, 2, 1000.0) is None
        assert cache.get(2, 3, 1000.0) is None
    
    def test_cache_clear(self):
        """Test cache clearing."""
        cache = TimeAwareLRUCache()
        
        # Add entries
        cache.put(1, 2, 1000.0, 1.0)
        cache.put(2, 3, 1000.0, 2.0)
        cache._stats.cache_hits = 10
        cache._stats.cache_misses = 5
        
        # Clear cache
        cache.clear()
        
        # Everything should be reset
        assert len(cache._cache) == 0
        assert len(cache._time_buckets) == 0
        assert cache._stats.cache_hits == 0
        assert cache._stats.cache_misses == 0
    
    def test_cache_info(self):
        """Test cache information retrieval."""
        cache = TimeAwareLRUCache(max_size=100, ttl_seconds=3600.0, time_tolerance=60.0)
        cache.put(1, 2, 1000.0, 1.0)
        
        info = cache.get_cache_info()
        
        assert info['size'] == 1
        assert info['max_size'] == 100
        assert info['ttl_seconds'] == 3600.0
        assert info['time_tolerance'] == 60.0
        assert 'stats' in info
        assert info['stats'].total_requests >= 0


class TestFitnessCacheManager:
    """Test the FitnessCacheManager class."""
    
    def test_cache_manager_creation(self):
        """Test basic cache manager creation."""
        manager = FitnessCacheManager()
        assert manager.deltav_cache is not None
        assert manager.fitness_cache is not None
        assert manager._cleanup_interval == 300.0
    
    def test_deltav_caching(self):
        """Test delta-v caching functionality."""
        manager = FitnessCacheManager()
        
        # Test cache miss
        result = manager.get_deltav(1, 2, 1000.0)
        assert result is None
        
        # Test cache put and hit
        manager.put_deltav(1, 2, 1000.0, 1.5)
        result = manager.get_deltav(1, 2, 1000.0)
        assert result == 1.5
    
    def test_satellite_invalidation(self):
        """Test satellite data invalidation."""
        manager = FitnessCacheManager()
        
        # Add some cached values
        manager.put_deltav(1, 2, 1000.0, 1.0)
        manager.put_deltav(2, 3, 1000.0, 2.0)
        manager.put_deltav(3, 4, 1000.0, 3.0)
        
        # Invalidate satellite 2
        manager.invalidate_satellite_data([2])
        
        # Entries involving satellite 2 should be invalidated
        assert manager.get_deltav(1, 2, 1000.0) is None
        assert manager.get_deltav(2, 3, 1000.0) is None
        assert manager.get_deltav(3, 4, 1000.0) == 3.0
    
    def test_time_range_invalidation(self):
        """Test time range invalidation."""
        manager = FitnessCacheManager()
        
        # Add entries at different times
        manager.put_deltav(1, 2, 1000.0, 1.0)
        manager.put_deltav(2, 3, 2000.0, 2.0)
        manager.put_deltav(3, 4, 3000.0, 3.0)
        
        # Invalidate middle time range
        manager.invalidate_time_range(1500.0, 2500.0)
        
        # Check results
        assert manager.get_deltav(1, 2, 1000.0) == 1.0
        assert manager.get_deltav(2, 3, 2000.0) is None
        assert manager.get_deltav(3, 4, 3000.0) == 3.0
    
    def test_cache_optimization(self):
        """Test cache size optimization."""
        manager = FitnessCacheManager()
        original_deltav_size = manager.deltav_cache.max_size
        original_fitness_size = manager.fitness_cache.max_size
        
        # Optimize for large constellation
        manager.optimize_cache_sizes(5000)
        
        # Cache sizes should be adjusted
        assert manager.deltav_cache.max_size != original_deltav_size
        assert manager.fitness_cache.max_size != original_fitness_size
    
    def test_combined_stats(self):
        """Test combined statistics retrieval."""
        manager = FitnessCacheManager()
        
        # Add some cached values
        manager.put_deltav(1, 2, 1000.0, 1.0)
        manager.put_deltav(2, 3, 1000.0, 2.0)
        
        stats = manager.get_combined_stats()
        
        assert 'deltav_cache' in stats
        assert 'fitness_cache' in stats
        assert 'total_memory_estimate' in stats
        assert stats['total_memory_estimate'] > 0
    
    def test_periodic_cleanup(self):
        """Test periodic cleanup functionality."""
        manager = FitnessCacheManager()
        manager._cleanup_interval = 0.1  # Short interval for testing
        
        initial_time = manager._last_cleanup
        
        # Wait and trigger cleanup
        time.sleep(0.2)
        manager.periodic_cleanup()
        
        # Cleanup time should be updated
        assert manager._last_cleanup > initial_time
    
    def test_clear_all_caches(self):
        """Test clearing all caches."""
        manager = FitnessCacheManager()
        
        # Add some cached values
        manager.put_deltav(1, 2, 1000.0, 1.0)
        
        # Clear all caches
        manager.clear_all_caches()
        
        # Should be empty
        assert manager.get_deltav(1, 2, 1000.0) is None
        assert len(manager.deltav_cache._cache) == 0
        assert len(manager.fitness_cache._cache) == 0


class TestCacheIntegration:
    """Integration tests for the caching system."""
    
    def test_cache_performance_improvement(self):
        """Test that caching improves performance."""
        manager = FitnessCacheManager()
        
        # Simulate expensive calculation
        def expensive_calculation():
            time.sleep(0.01)  # Simulate computation time
            return 1.5
        
        # First call - cache miss (should be slower)
        start_time = time.time()
        result1 = manager.get_deltav(1, 2, 1000.0)
        if result1 is None:
            result1 = expensive_calculation()
            manager.put_deltav(1, 2, 1000.0, result1)
        first_call_time = time.time() - start_time
        
        # Second call - cache hit (should be faster)
        start_time = time.time()
        result2 = manager.get_deltav(1, 2, 1000.0)
        second_call_time = time.time() - start_time
        
        assert result1 == result2
        assert second_call_time < first_call_time
    
    def test_memory_management_under_load(self):
        """Test memory management under heavy load."""
        manager = FitnessCacheManager(max_deltav_cache_size=100)
        
        # Fill cache beyond capacity
        for i in range(200):
            manager.put_deltav(i, i+1, 1000.0, float(i))
        
        # Cache should not exceed max size
        assert len(manager.deltav_cache._cache) <= 100
        
        # Should have evicted some entries
        assert manager.deltav_cache._stats.evictions > 0
    
    def test_time_tolerance_effectiveness(self):
        """Test that time tolerance improves cache hit rates."""
        manager = FitnessCacheManager()
        
        # Put value at specific time
        manager.put_deltav(1, 2, 1000.0, 1.5)
        
        # Get value at slightly different time (within tolerance)
        result = manager.get_deltav(1, 2, 1030.0)  # 30 seconds later
        
        # Should hit cache due to time tolerance
        assert result == 1.5
        assert manager.deltav_cache._stats.cache_hits > 0


if __name__ == "__main__":
    pytest.main([__file__])