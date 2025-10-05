"""
Fitness Caching System for Genetic Algorithm Route Optimization

This module provides an advanced caching system for delta-v calculations and fitness evaluations
to improve performance when optimizing routes across large satellite constellations.
"""

import time
import math
from typing import Dict, Tuple, Optional, List, Set
from dataclasses import dataclass, field
from collections import OrderedDict
import weakref


@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata."""
    value: float
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_access = time.time()


@dataclass
class CacheStats:
    """Statistics about cache performance."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    memory_usage_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate


class TimeAwareLRUCache:
    """
    Time-aware LRU cache with automatic invalidation and memory management.
    
    This cache is specifically designed for delta-v calculations that depend on time,
    providing intelligent invalidation strategies and memory management for large
    satellite constellations.
    """
    
    def __init__(self, max_size: int = 10000, ttl_seconds: float = 3600.0, 
                 time_tolerance: float = 60.0):
        """
        Initialize the time-aware cache.
        
        Args:
            max_size: Maximum number of entries to store
            ttl_seconds: Time-to-live for cache entries in seconds
            time_tolerance: Time tolerance for considering entries equivalent (seconds)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.time_tolerance = time_tolerance
        
        # Use OrderedDict for LRU behavior
        self._cache: OrderedDict[Tuple, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        
        # Time-based invalidation tracking
        self._time_buckets: Dict[int, Set[Tuple]] = {}
        
    def _normalize_time_key(self, time_value: float) -> int:
        """
        Normalize time value to a bucket for intelligent key strategies.
        
        Args:
            time_value: Time value to normalize
            
        Returns:
            Normalized time bucket
        """
        return int(time_value // self.time_tolerance)
    
    def _create_cache_key(self, source_id: int, target_id: int, departure_time: float) -> Tuple:
        """
        Create an intelligent cache key with time normalization.
        
        Args:
            source_id: Source satellite ID
            target_id: Target satellite ID
            departure_time: Departure time
            
        Returns:
            Normalized cache key
        """
        # Normalize time to improve cache hit rates for similar times
        normalized_time = self._normalize_time_key(departure_time)
        
        # Create bidirectional key for symmetric transfers (if applicable)
        # For now, keep directional since transfers may not be symmetric
        return (source_id, target_id, normalized_time)
    
    def get(self, source_id: int, target_id: int, departure_time: float) -> Optional[float]:
        """
        Retrieve a cached delta-v value.
        
        Args:
            source_id: Source satellite ID
            target_id: Target satellite ID
            departure_time: Departure time
            
        Returns:
            Cached delta-v value or None if not found/expired
        """
        self._stats.total_requests += 1
        
        key = self._create_cache_key(source_id, target_id, departure_time)
        
        if key in self._cache:
            entry = self._cache[key]
            current_time = time.time()
            
            # Check if entry has expired
            if current_time - entry.timestamp > self.ttl_seconds:
                self._remove_entry(key)
                self._stats.cache_misses += 1
                return None
            
            # Update access statistics and move to end (most recently used)
            entry.update_access()
            self._cache.move_to_end(key)
            self._stats.cache_hits += 1
            
            return entry.value
        
        self._stats.cache_misses += 1
        return None
    
    def put(self, source_id: int, target_id: int, departure_time: float, delta_v: float):
        """
        Store a delta-v value in the cache.
        
        Args:
            source_id: Source satellite ID
            target_id: Target satellite ID
            departure_time: Departure time
            delta_v: Delta-v value to cache
        """
        key = self._create_cache_key(source_id, target_id, departure_time)
        current_time = time.time()
        
        # Create new cache entry
        entry = CacheEntry(value=delta_v, timestamp=current_time)
        
        # Add to time bucket for efficient time-based invalidation
        time_bucket = self._normalize_time_key(departure_time)
        if time_bucket not in self._time_buckets:
            self._time_buckets[time_bucket] = set()
        self._time_buckets[time_bucket].add(key)
        
        # If key already exists, update it
        if key in self._cache:
            self._cache[key] = entry
            self._cache.move_to_end(key)
        else:
            # Add new entry
            self._cache[key] = entry
            
            # Evict oldest entries if cache is full
            while len(self._cache) > self.max_size:
                self._evict_oldest()
    
    def _remove_entry(self, key: Tuple):
        """Remove an entry from cache and time buckets."""
        if key in self._cache:
            # Remove from time buckets
            _, _, time_bucket = key
            if time_bucket in self._time_buckets:
                self._time_buckets[time_bucket].discard(key)
                if not self._time_buckets[time_bucket]:
                    del self._time_buckets[time_bucket]
            
            # Remove from cache
            del self._cache[key]
    
    def _evict_oldest(self):
        """Evict the oldest (least recently used) entry."""
        if self._cache:
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key)
            self._stats.evictions += 1
    
    def invalidate_time_range(self, start_time: float, end_time: float):
        """
        Invalidate all cache entries within a time range.
        
        Args:
            start_time: Start of time range to invalidate
            end_time: End of time range to invalidate
        """
        start_bucket = self._normalize_time_key(start_time)
        end_bucket = self._normalize_time_key(end_time)
        
        keys_to_remove = []
        
        for time_bucket in range(start_bucket, end_bucket + 1):
            if time_bucket in self._time_buckets:
                keys_to_remove.extend(self._time_buckets[time_bucket])
        
        for key in keys_to_remove:
            self._remove_entry(key)
            self._stats.invalidations += 1
    
    def invalidate_satellites(self, satellite_ids: List[int]):
        """
        Invalidate all cache entries involving specific satellites.
        
        Args:
            satellite_ids: List of satellite IDs to invalidate
        """
        satellite_set = set(satellite_ids)
        keys_to_remove = []
        
        for key in self._cache:
            source_id, target_id, _ = key
            if source_id in satellite_set or target_id in satellite_set:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self._remove_entry(key)
            self._stats.invalidations += 1
    
    def cleanup_expired(self):
        """Remove all expired entries from the cache."""
        current_time = time.time()
        keys_to_remove = []
        
        for key, entry in self._cache.items():
            if current_time - entry.timestamp > self.ttl_seconds:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self._remove_entry(key)
            self._stats.invalidations += 1
    
    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._time_buckets.clear()
        self._stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        # Update memory usage estimate
        self._stats.memory_usage_bytes = self._estimate_memory_usage()
        return self._stats
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of the cache in bytes."""
        # Rough estimate: each entry has key (3 ints + 1 float) + CacheEntry
        # This is a simplified calculation
        entry_size = (3 * 8) + (4 * 8) + 64  # Key + CacheEntry + overhead
        return len(self._cache) * entry_size
    
    def get_cache_info(self) -> Dict:
        """Get detailed cache information."""
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds,
            'time_tolerance': self.time_tolerance,
            'time_buckets': len(self._time_buckets),
            'stats': self.get_stats()
        }


class FitnessCacheManager:
    """
    High-level cache manager for fitness evaluation caching.
    
    Manages multiple cache instances and provides intelligent caching strategies
    for different types of fitness calculations.
    """
    
    def __init__(self, max_deltav_cache_size: int = 50000, 
                 max_fitness_cache_size: int = 10000,
                 deltav_ttl: float = 7200.0,  # 2 hours
                 fitness_ttl: float = 1800.0,  # 30 minutes
                 time_tolerance: float = 60.0):  # 1 minute
        """
        Initialize the fitness cache manager.
        
        Args:
            max_deltav_cache_size: Maximum size for delta-v cache
            max_fitness_cache_size: Maximum size for fitness cache
            deltav_ttl: TTL for delta-v calculations
            fitness_ttl: TTL for fitness evaluations
            time_tolerance: Time tolerance for cache keys
        """
        # Delta-v calculation cache (most important for performance)
        self.deltav_cache = TimeAwareLRUCache(
            max_size=max_deltav_cache_size,
            ttl_seconds=deltav_ttl,
            time_tolerance=time_tolerance
        )
        
        # Route fitness cache (for complete route evaluations)
        self.fitness_cache = TimeAwareLRUCache(
            max_size=max_fitness_cache_size,
            ttl_seconds=fitness_ttl,
            time_tolerance=time_tolerance
        )
        
        # Track cache performance
        self._last_cleanup = time.time()
        self._cleanup_interval = 300.0  # 5 minutes
    
    def get_deltav(self, source_id: int, target_id: int, departure_time: float) -> Optional[float]:
        """Get cached delta-v value."""
        return self.deltav_cache.get(source_id, target_id, departure_time)
    
    def put_deltav(self, source_id: int, target_id: int, departure_time: float, delta_v: float):
        """Cache a delta-v value."""
        self.deltav_cache.put(source_id, target_id, departure_time, delta_v)
    
    def get_route_fitness(self, route_hash: str, constraint_hash: str) -> Optional[float]:
        """Get cached route fitness value."""
        # Use string hashes as cache keys for route fitness
        key_parts = (hash(route_hash), hash(constraint_hash), 0)  # 0 for time bucket
        entry = self.fitness_cache._cache.get(key_parts)
        return entry.value if entry else None
    
    def put_route_fitness(self, route_hash: str, constraint_hash: str, fitness: float):
        """Cache a route fitness value."""
        # This is a simplified implementation - in practice, you might want
        # more sophisticated route hashing
        pass
    
    def invalidate_satellite_data(self, satellite_ids: List[int]):
        """Invalidate cache entries for specific satellites."""
        self.deltav_cache.invalidate_satellites(satellite_ids)
        # Fitness cache would need more complex invalidation logic
    
    def invalidate_time_range(self, start_time: float, end_time: float):
        """Invalidate cache entries for a time range."""
        self.deltav_cache.invalidate_time_range(start_time, end_time)
        self.fitness_cache.invalidate_time_range(start_time, end_time)
    
    def periodic_cleanup(self):
        """Perform periodic cache maintenance."""
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            self.deltav_cache.cleanup_expired()
            self.fitness_cache.cleanup_expired()
            self._last_cleanup = current_time
    
    def clear_all_caches(self):
        """Clear all caches."""
        self.deltav_cache.clear()
        self.fitness_cache.clear()
    
    def get_combined_stats(self) -> Dict:
        """Get combined statistics from all caches."""
        return {
            'deltav_cache': self.deltav_cache.get_cache_info(),
            'fitness_cache': self.fitness_cache.get_cache_info(),
            'total_memory_estimate': (
                self.deltav_cache._estimate_memory_usage() + 
                self.fitness_cache._estimate_memory_usage()
            )
        }
    
    def optimize_cache_sizes(self, constellation_size: int):
        """
        Optimize cache sizes based on constellation size.
        
        Args:
            constellation_size: Number of satellites in constellation
        """
        # Heuristic: cache size should scale with constellation size
        # but with diminishing returns for very large constellations
        optimal_deltav_size = min(constellation_size * constellation_size, 100000)
        optimal_fitness_size = min(constellation_size * 50, 20000)
        
        # Update cache sizes if significantly different
        if abs(self.deltav_cache.max_size - optimal_deltav_size) > 1000:
            self.deltav_cache.max_size = optimal_deltav_size
        
        if abs(self.fitness_cache.max_size - optimal_fitness_size) > 500:
            self.fitness_cache.max_size = optimal_fitness_size