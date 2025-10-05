"""
Performance Optimizer for route caching and database optimization.

This module implements route caching for frequently requested satellite combinations,
database optimization for satellite data queries, and performance monitoring
with optimization suggestions.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
import logging
import hashlib
import json
import time
from dataclasses import dataclass, asdict
from collections import defaultdict, OrderedDict
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio

from ..models.satellite import Satellite
from ..models.route import Route
from ..models.cost import MissionCost
from ..models.service_request import ServiceRequest


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry for route optimization results."""
    key: str
    route: Route
    mission_cost: MissionCost
    optimization_metadata: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int
    expiry_time: datetime
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.utcnow() > self.expiry_time
    
    def update_access(self) -> None:
        """Update access statistics."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""
    cache_hit_rate: float
    cache_miss_rate: float
    average_response_time_ms: float
    total_requests: int
    cache_size: int
    memory_usage_mb: float
    database_query_time_ms: float
    optimization_time_ms: float
    last_updated: datetime


class RouteCache:
    """
    High-performance route caching system.
    
    Implements route caching for frequently requested satellite combinations
    with intelligent cache management and performance optimization.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl_hours: int = 24):
        """
        Initialize route cache.
        
        Args:
            max_size: Maximum number of cache entries
            default_ttl_hours: Default time-to-live for cache entries in hours
        """
        self.max_size = max_size
        self.default_ttl = timedelta(hours=default_ttl_hours)
        
        # Thread-safe cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Performance tracking
        self._hits = 0
        self._misses = 0
        self._total_response_time = 0.0
        self._request_count = 0
        
        # Background cleanup
        self._cleanup_interval = 3600  # 1 hour
        self._last_cleanup = datetime.utcnow()
        
        logger.info(f"RouteCache initialized with max_size={max_size}, ttl={default_ttl_hours}h")
    
    def generate_cache_key(self, satellite_ids: List[str], 
                          constraints: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate cache key for satellite combination and constraints.
        
        Args:
            satellite_ids: List of satellite IDs
            constraints: Optional constraints dictionary
            
        Returns:
            Cache key string
        """
        # Sort satellite IDs for consistent key generation
        sorted_ids = sorted(satellite_ids)
        
        # Create key components
        key_data = {
            'satellites': sorted_ids,
            'constraints': constraints or {}
        }
        
        # Generate hash
        key_string = json.dumps(key_data, sort_keys=True)
        cache_key = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        
        return cache_key
    
    def get(self, cache_key: str) -> Optional[Tuple[Route, MissionCost, Dict[str, Any]]]:
        """
        Get cached route optimization result.
        
        Args:
            cache_key: Cache key to lookup
            
        Returns:
            Tuple of (route, mission_cost, metadata) if found, None otherwise
        """
        start_time = time.time()
        
        try:
            with self._lock:
                if cache_key in self._cache:
                    entry = self._cache[cache_key]
                    
                    # Check if expired
                    if entry.is_expired():
                        del self._cache[cache_key]
                        self._misses += 1
                        logger.debug(f"Cache entry expired: {cache_key}")
                        return None
                    
                    # Update access statistics
                    entry.update_access()
                    
                    # Move to end (LRU)
                    self._cache.move_to_end(cache_key)
                    
                    self._hits += 1
                    response_time = (time.time() - start_time) * 1000
                    self._update_response_time(response_time)
                    
                    logger.debug(f"Cache hit: {cache_key}")
                    return entry.route, entry.mission_cost, entry.optimization_metadata
                else:
                    self._misses += 1
                    logger.debug(f"Cache miss: {cache_key}")
                    return None
                    
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            self._misses += 1
            return None
    
    def put(self, cache_key: str, route: Route, mission_cost: MissionCost,
            optimization_metadata: Dict[str, Any], ttl_hours: Optional[int] = None) -> None:
        """
        Store route optimization result in cache.
        
        Args:
            cache_key: Cache key
            route: Optimized route
            mission_cost: Mission cost calculation
            optimization_metadata: Optimization metadata
            ttl_hours: Time-to-live in hours (uses default if None)
        """
        try:
            with self._lock:
                # Calculate expiry time
                ttl = timedelta(hours=ttl_hours) if ttl_hours else self.default_ttl
                expiry_time = datetime.utcnow() + ttl
                
                # Create cache entry
                entry = CacheEntry(
                    key=cache_key,
                    route=route,
                    mission_cost=mission_cost,
                    optimization_metadata=optimization_metadata,
                    created_at=datetime.utcnow(),
                    last_accessed=datetime.utcnow(),
                    access_count=0,
                    expiry_time=expiry_time
                )
                
                # Add to cache
                self._cache[cache_key] = entry
                
                # Enforce size limit
                while len(self._cache) > self.max_size:
                    # Remove least recently used
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    logger.debug(f"Cache evicted LRU entry: {oldest_key}")
                
                logger.debug(f"Cache stored: {cache_key}, expires: {expiry_time}")
                
        except Exception as e:
            logger.error(f"Cache put error: {str(e)}")
    
    def invalidate(self, cache_key: str) -> bool:
        """
        Invalidate specific cache entry.
        
        Args:
            cache_key: Cache key to invalidate
            
        Returns:
            True if entry was found and removed, False otherwise
        """
        try:
            with self._lock:
                if cache_key in self._cache:
                    del self._cache[cache_key]
                    logger.debug(f"Cache invalidated: {cache_key}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Cache invalidate error: {str(e)}")
            return False
    
    def invalidate_pattern(self, satellite_ids: List[str]) -> int:
        """
        Invalidate cache entries containing specific satellites.
        
        Args:
            satellite_ids: List of satellite IDs
            
        Returns:
            Number of entries invalidated
        """
        try:
            with self._lock:
                keys_to_remove = []
                
                for key, entry in self._cache.items():
                    # Check if any of the satellites are in this cached route
                    cached_satellite_ids = [sat.id for sat in entry.route.satellites]
                    if any(sat_id in cached_satellite_ids for sat_id in satellite_ids):
                        keys_to_remove.append(key)
                
                # Remove identified keys
                for key in keys_to_remove:
                    del self._cache[key]
                
                logger.info(f"Cache invalidated {len(keys_to_remove)} entries for satellites: {satellite_ids}")
                return len(keys_to_remove)
                
        except Exception as e:
            logger.error(f"Cache pattern invalidate error: {str(e)}")
            return 0
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of entries cleaned up
        """
        try:
            with self._lock:
                current_time = datetime.utcnow()
                keys_to_remove = []
                
                for key, entry in self._cache.items():
                    if entry.is_expired():
                        keys_to_remove.append(key)
                
                # Remove expired entries
                for key in keys_to_remove:
                    del self._cache[key]
                
                self._last_cleanup = current_time
                logger.debug(f"Cache cleanup removed {len(keys_to_remove)} expired entries")
                return len(keys_to_remove)
                
        except Exception as e:
            logger.error(f"Cache cleanup error: {str(e)}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        try:
            with self._lock:
                total_requests = self._hits + self._misses
                hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0
                miss_rate = (self._misses / total_requests * 100) if total_requests > 0 else 0.0
                avg_response_time = (self._total_response_time / self._request_count) if self._request_count > 0 else 0.0
                
                return {
                    'cache_size': len(self._cache),
                    'max_size': self.max_size,
                    'hit_rate_percent': hit_rate,
                    'miss_rate_percent': miss_rate,
                    'total_hits': self._hits,
                    'total_misses': self._misses,
                    'total_requests': total_requests,
                    'average_response_time_ms': avg_response_time,
                    'last_cleanup': self._last_cleanup.isoformat(),
                    'memory_usage_estimate_mb': len(self._cache) * 0.1  # Rough estimate
                }
                
        except Exception as e:
            logger.error(f"Cache statistics error: {str(e)}")
            return {}
    
    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            with self._lock:
                self._cache.clear()
                logger.info("Cache cleared")
                
        except Exception as e:
            logger.error(f"Cache clear error: {str(e)}")
    
    def _update_response_time(self, response_time_ms: float) -> None:
        """Update response time statistics."""
        self._total_response_time += response_time_ms
        self._request_count += 1


class DatabaseOptimizer:
    """
    Database optimization for satellite data queries.
    
    Implements query optimization, indexing strategies, and performance monitoring
    for satellite database operations.
    """
    
    def __init__(self):
        """Initialize database optimizer."""
        self._query_cache: Dict[str, Any] = {}
        self._query_stats: Dict[str, List[float]] = defaultdict(list)
        self._index_suggestions: List[str] = []
        
        logger.info("DatabaseOptimizer initialized")
    
    def optimize_satellite_query(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize satellite data query parameters.
        
        Args:
            query_params: Original query parameters
            
        Returns:
            Optimized query parameters
        """
        optimized_params = query_params.copy()
        
        # Add query optimizations
        if 'limit' not in optimized_params:
            optimized_params['limit'] = 1000  # Default limit
        
        # Optimize filtering
        if 'filter_valid' not in optimized_params:
            optimized_params['filter_valid'] = True  # Filter invalid satellites by default
        
        # Add sorting for consistent results
        if 'sort_by' not in optimized_params:
            optimized_params['sort_by'] = 'id'
        
        return optimized_params
    
    def track_query_performance(self, query_type: str, execution_time_ms: float) -> None:
        """
        Track query performance for monitoring.
        
        Args:
            query_type: Type of query executed
            execution_time_ms: Execution time in milliseconds
        """
        self._query_stats[query_type].append(execution_time_ms)
        
        # Keep only recent measurements (last 100)
        if len(self._query_stats[query_type]) > 100:
            self._query_stats[query_type] = self._query_stats[query_type][-100:]
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """
        Get database query performance statistics.
        
        Returns:
            Dictionary of query statistics
        """
        stats = {}
        
        for query_type, times in self._query_stats.items():
            if times:
                stats[query_type] = {
                    'count': len(times),
                    'average_time_ms': sum(times) / len(times),
                    'min_time_ms': min(times),
                    'max_time_ms': max(times),
                    'recent_average_ms': sum(times[-10:]) / min(len(times), 10)
                }
        
        return stats
    
    def suggest_optimizations(self) -> List[str]:
        """
        Generate optimization suggestions based on query patterns.
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Analyze query patterns
        for query_type, times in self._query_stats.items():
            if times:
                avg_time = sum(times) / len(times)
                
                if avg_time > 1000:  # Slow queries (>1 second)
                    suggestions.append(f"Consider indexing for {query_type} queries (avg: {avg_time:.1f}ms)")
                
                if len(times) > 50:  # Frequent queries
                    suggestions.append(f"Consider caching for frequent {query_type} queries")
        
        return suggestions


class PerformanceMonitor:
    """
    Performance monitoring and optimization suggestions.
    
    Monitors system performance and provides optimization recommendations
    for route optimization and database operations.
    """
    
    def __init__(self, route_cache: RouteCache, db_optimizer: DatabaseOptimizer):
        """
        Initialize performance monitor.
        
        Args:
            route_cache: Route cache instance
            db_optimizer: Database optimizer instance
        """
        self.route_cache = route_cache
        self.db_optimizer = db_optimizer
        
        # Performance tracking
        self._optimization_times: List[float] = []
        self._request_times: List[float] = []
        self._memory_usage: List[float] = []
        
        logger.info("PerformanceMonitor initialized")
    
    def track_optimization_time(self, time_ms: float) -> None:
        """Track route optimization execution time."""
        self._optimization_times.append(time_ms)
        
        # Keep only recent measurements
        if len(self._optimization_times) > 100:
            self._optimization_times = self._optimization_times[-100:]
    
    def track_request_time(self, time_ms: float) -> None:
        """Track total request processing time."""
        self._request_times.append(time_ms)
        
        # Keep only recent measurements
        if len(self._request_times) > 100:
            self._request_times = self._request_times[-100:]
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Get comprehensive performance metrics.
        
        Returns:
            Performance metrics object
        """
        # Cache statistics
        cache_stats = self.route_cache.get_statistics()
        
        # Database statistics
        db_stats = self.db_optimizer.get_query_statistics()
        avg_db_time = 0.0
        if db_stats:
            db_times = [stats['average_time_ms'] for stats in db_stats.values()]
            avg_db_time = sum(db_times) / len(db_times) if db_times else 0.0
        
        # Optimization statistics
        avg_opt_time = sum(self._optimization_times) / len(self._optimization_times) if self._optimization_times else 0.0
        
        # Request statistics
        avg_req_time = sum(self._request_times) / len(self._request_times) if self._request_times else 0.0
        
        return PerformanceMetrics(
            cache_hit_rate=cache_stats.get('hit_rate_percent', 0.0),
            cache_miss_rate=cache_stats.get('miss_rate_percent', 0.0),
            average_response_time_ms=avg_req_time,
            total_requests=cache_stats.get('total_requests', 0),
            cache_size=cache_stats.get('cache_size', 0),
            memory_usage_mb=cache_stats.get('memory_usage_estimate_mb', 0.0),
            database_query_time_ms=avg_db_time,
            optimization_time_ms=avg_opt_time,
            last_updated=datetime.utcnow()
        )
    
    def generate_optimization_suggestions(self) -> List[str]:
        """
        Generate performance optimization suggestions.
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        metrics = self.get_performance_metrics()
        
        # Cache optimization suggestions
        if metrics.cache_hit_rate < 50.0:
            suggestions.append("Low cache hit rate - consider increasing cache size or TTL")
        
        if metrics.cache_size > 800:  # Near max capacity
            suggestions.append("Cache near capacity - consider increasing max size")
        
        # Response time suggestions
        if metrics.average_response_time_ms > 5000:  # >5 seconds
            suggestions.append("High response times - consider optimizing route algorithms")
        
        if metrics.database_query_time_ms > 1000:  # >1 second
            suggestions.append("Slow database queries - consider adding indexes or query optimization")
        
        if metrics.optimization_time_ms > 10000:  # >10 seconds
            suggestions.append("Slow route optimization - consider reducing population size or generations")
        
        # Memory usage suggestions
        if metrics.memory_usage_mb > 500:  # >500MB
            suggestions.append("High memory usage - consider reducing cache size or implementing cleanup")
        
        # Add database-specific suggestions
        db_suggestions = self.db_optimizer.suggest_optimizations()
        suggestions.extend(db_suggestions)
        
        return suggestions
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Performance report dictionary
        """
        metrics = self.get_performance_metrics()
        suggestions = self.generate_optimization_suggestions()
        
        return {
            'metrics': asdict(metrics),
            'cache_statistics': self.route_cache.get_statistics(),
            'database_statistics': self.db_optimizer.get_query_statistics(),
            'optimization_suggestions': suggestions,
            'performance_grade': self._calculate_performance_grade(metrics),
            'report_generated_at': datetime.utcnow().isoformat()
        }
    
    def _calculate_performance_grade(self, metrics: PerformanceMetrics) -> str:
        """Calculate overall performance grade."""
        score = 0
        
        # Cache performance (30%)
        if metrics.cache_hit_rate > 80:
            score += 30
        elif metrics.cache_hit_rate > 60:
            score += 20
        elif metrics.cache_hit_rate > 40:
            score += 10
        
        # Response time performance (40%)
        if metrics.average_response_time_ms < 1000:
            score += 40
        elif metrics.average_response_time_ms < 3000:
            score += 30
        elif metrics.average_response_time_ms < 5000:
            score += 20
        elif metrics.average_response_time_ms < 10000:
            score += 10
        
        # Database performance (20%)
        if metrics.database_query_time_ms < 100:
            score += 20
        elif metrics.database_query_time_ms < 500:
            score += 15
        elif metrics.database_query_time_ms < 1000:
            score += 10
        elif metrics.database_query_time_ms < 2000:
            score += 5
        
        # Memory efficiency (10%)
        if metrics.memory_usage_mb < 100:
            score += 10
        elif metrics.memory_usage_mb < 250:
            score += 8
        elif metrics.memory_usage_mb < 500:
            score += 5
        
        # Convert to grade
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'


class PerformanceOptimizer:
    """
    Main performance optimization coordinator.
    
    Coordinates route caching, database optimization, and performance monitoring
    to provide comprehensive performance optimization for the debris removal service.
    """
    
    def __init__(self, cache_size: int = 1000, cache_ttl_hours: int = 24):
        """
        Initialize performance optimizer.
        
        Args:
            cache_size: Maximum cache size
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.route_cache = RouteCache(cache_size, cache_ttl_hours)
        self.db_optimizer = DatabaseOptimizer()
        self.performance_monitor = PerformanceMonitor(self.route_cache, self.db_optimizer)
        
        # Background tasks
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._cleanup_task = None
        
        logger.info("PerformanceOptimizer initialized")
    
    async def get_cached_route(self, satellite_ids: List[str], 
                             constraints: Optional[Dict[str, Any]] = None) -> Optional[Tuple[Route, MissionCost, Dict[str, Any]]]:
        """
        Get cached route optimization result.
        
        Args:
            satellite_ids: List of satellite IDs
            constraints: Optional constraints
            
        Returns:
            Cached result if available, None otherwise
        """
        cache_key = self.route_cache.generate_cache_key(satellite_ids, constraints)
        return self.route_cache.get(cache_key)
    
    async def cache_route(self, satellite_ids: List[str], route: Route, 
                         mission_cost: MissionCost, optimization_metadata: Dict[str, Any],
                         constraints: Optional[Dict[str, Any]] = None,
                         ttl_hours: Optional[int] = None) -> None:
        """
        Cache route optimization result.
        
        Args:
            satellite_ids: List of satellite IDs
            route: Optimized route
            mission_cost: Mission cost
            optimization_metadata: Optimization metadata
            constraints: Optional constraints
            ttl_hours: Time-to-live in hours
        """
        cache_key = self.route_cache.generate_cache_key(satellite_ids, constraints)
        self.route_cache.put(cache_key, route, mission_cost, optimization_metadata, ttl_hours)
    
    def optimize_database_query(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize database query parameters."""
        return self.db_optimizer.optimize_satellite_query(query_params)
    
    def track_performance(self, operation_type: str, execution_time_ms: float) -> None:
        """Track performance metrics for different operations."""
        if operation_type == 'route_optimization':
            self.performance_monitor.track_optimization_time(execution_time_ms)
        elif operation_type == 'request_processing':
            self.performance_monitor.track_request_time(execution_time_ms)
        elif operation_type.startswith('db_'):
            self.db_optimizer.track_query_performance(operation_type, execution_time_ms)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return self.performance_monitor.get_performance_report()
    
    def invalidate_satellite_cache(self, satellite_ids: List[str]) -> int:
        """Invalidate cache entries for specific satellites."""
        return self.route_cache.invalidate_pattern(satellite_ids)
    
    async def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries."""
        return self.route_cache.cleanup_expired()
    
    def start_background_cleanup(self) -> None:
        """Start background cache cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._background_cleanup())
            logger.info("Background cache cleanup started")
    
    def stop_background_cleanup(self) -> None:
        """Stop background cache cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None
            logger.info("Background cache cleanup stopped")
    
    async def _background_cleanup(self) -> None:
        """Background task for periodic cache cleanup."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                cleaned = await self.cleanup_expired_cache()
                if cleaned > 0:
                    logger.info(f"Background cleanup removed {cleaned} expired cache entries")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background cleanup error: {str(e)}")
    
    def shutdown(self) -> None:
        """Shutdown performance optimizer and cleanup resources."""
        self.stop_background_cleanup()
        self._executor.shutdown(wait=True)
        logger.info("PerformanceOptimizer shutdown complete")