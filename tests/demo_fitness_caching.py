"""
Demonstration of the Fitness Caching System

This script demonstrates the advanced caching system for delta-v calculations
and shows the performance improvements achieved through intelligent caching.
"""

import time
from unittest.mock import Mock
from src.route_fitness_evaluator import RouteFitnessEvaluator
from src.fitness_cache import FitnessCacheManager
from src.genetic_algorithm import RouteChromosome, RouteConstraints
from src.tle_parser import SatelliteData
from src.orbital_propagator import OrbitalPropagator, TransferWindow


def create_mock_constellation(size=100):
    """Create a mock satellite constellation for testing."""
    satellites = []
    for i in range(1, size + 1):
        sat = Mock(spec=SatelliteData)
        sat.catalog_number = i
        sat.name = f"SAT-{i:03d}"
        satellites.append(sat)
    return satellites


def create_mock_propagator(constellation_size=100):
    """Create a mock orbital propagator."""
    propagator = Mock(spec=OrbitalPropagator)
    propagator.get_satellite_ids.return_value = list(range(1, constellation_size + 1))
    
    def mock_transfer_window(source_id, target_id, departure_time):
        # Simulate realistic delta-v calculations based on satellite IDs and time
        base_deltav = 0.3 + abs(source_id - target_id) * 0.01
        time_factor = 1.0 + (departure_time % 1000) / 10000
        
        window = Mock(spec=TransferWindow)
        window.departure_deltav = base_deltav * time_factor
        window.arrival_deltav = base_deltav * time_factor * 0.8
        return window
    
    propagator.calculate_transfer_window.side_effect = mock_transfer_window
    return propagator


def generate_test_routes(num_routes=50, max_hops=10, constellation_size=100):
    """Generate test routes for performance testing."""
    import random
    
    routes = []
    for i in range(num_routes):
        # Generate random route
        route_length = random.randint(3, max_hops)
        satellites = random.sample(range(1, constellation_size + 1), route_length)
        
        # Generate departure times
        start_time = 1000.0 + i * 100
        departure_times = [start_time + j * 3600 for j in range(route_length)]
        
        routes.append((satellites, departure_times))
    
    return routes


def demonstrate_caching_performance():
    """Demonstrate the performance benefits of caching."""
    print("🚀 Fitness Caching System Demonstration")
    print("=" * 50)
    
    # Create test constellation
    constellation_size = 200
    satellites = create_mock_constellation(constellation_size)
    propagator = create_mock_propagator(constellation_size)
    
    print(f"📡 Created mock constellation with {constellation_size} satellites")
    
    # Create cache manager optimized for constellation size
    cache_manager = FitnessCacheManager()
    cache_manager.optimize_cache_sizes(constellation_size)
    
    # Create fitness evaluator with caching
    evaluator = RouteFitnessEvaluator(
        satellites=satellites,
        orbital_propagator=propagator,
        cache_manager=cache_manager
    )
    
    print(f"💾 Cache optimized for {constellation_size} satellites")
    print(f"   Delta-v cache size: {cache_manager.deltav_cache.max_size}")
    print(f"   Fitness cache size: {cache_manager.fitness_cache.max_size}")
    
    # Generate test routes
    test_routes = generate_test_routes(num_routes=100, max_hops=8, constellation_size=constellation_size)
    print(f"🛰️  Generated {len(test_routes)} test routes")
    
    # Performance test without caching (disable cache)
    print("\n📊 Performance Comparison")
    print("-" * 30)
    
    # First pass - populate cache
    print("🔄 First pass (populating cache)...")
    start_time = time.time()
    
    route_deltavs = []
    for satellites_seq, departure_times in test_routes:
        try:
            deltav = evaluator.calculate_route_deltav(satellites_seq, departure_times)
            route_deltavs.append(deltav)
        except Exception as e:
            print(f"   Warning: Route calculation failed: {e}")
            route_deltavs.append(None)
    
    first_pass_time = time.time() - start_time
    propagator_calls_first = propagator.calculate_transfer_window.call_count
    
    print(f"   Time: {first_pass_time:.3f}s")
    print(f"   Propagator calls: {propagator_calls_first}")
    print(f"   Successful routes: {sum(1 for dv in route_deltavs if dv is not None)}")
    
    # Second pass - use cache
    print("\n🚀 Second pass (using cache)...")
    propagator.calculate_transfer_window.reset_mock()
    start_time = time.time()
    
    cached_deltavs = []
    for satellites_seq, departure_times in test_routes:
        try:
            deltav = evaluator.calculate_route_deltav(satellites_seq, departure_times)
            cached_deltavs.append(deltav)
        except Exception as e:
            cached_deltavs.append(None)
    
    second_pass_time = time.time() - start_time
    propagator_calls_second = propagator.calculate_transfer_window.call_count
    
    print(f"   Time: {second_pass_time:.3f}s")
    print(f"   Propagator calls: {propagator_calls_second}")
    
    # Calculate performance improvement
    if second_pass_time > 0:
        speedup = first_pass_time / second_pass_time
        print(f"   🎯 Speedup: {speedup:.1f}x faster!")
    else:
        print("   🎯 Speedup: Instantaneous (cache hits)!")
    
    # Verify results are identical
    matches = sum(1 for i in range(len(route_deltavs)) 
                 if route_deltavs[i] == cached_deltavs[i])
    print(f"   ✅ Result consistency: {matches}/{len(route_deltavs)} routes match")
    
    # Cache statistics
    print("\n📈 Cache Statistics")
    print("-" * 20)
    
    cache_stats = evaluator.get_cache_stats()
    deltav_stats = cache_stats['advanced_caching']['deltav_cache']['stats']
    
    print(f"   Total requests: {deltav_stats.total_requests}")
    print(f"   Cache hits: {deltav_stats.cache_hits}")
    print(f"   Cache misses: {deltav_stats.cache_misses}")
    print(f"   Hit rate: {deltav_stats.hit_rate:.2%}")
    print(f"   Cache size: {cache_stats['advanced_caching']['deltav_cache']['size']} entries")
    
    # Memory usage
    memory_usage = evaluator.get_cache_memory_usage()
    total_mb = memory_usage['total_bytes'] / (1024 * 1024)
    print(f"   Memory usage: {total_mb:.2f} MB")
    
    return evaluator, cache_stats


def demonstrate_cache_invalidation(evaluator):
    """Demonstrate cache invalidation features."""
    print("\n🔄 Cache Invalidation Demonstration")
    print("-" * 35)
    
    # Get initial cache size
    initial_stats = evaluator.get_cache_stats()
    initial_size = initial_stats['advanced_caching']['deltav_cache']['size']
    print(f"   Initial cache size: {initial_size} entries")
    
    # Invalidate specific satellites
    satellites_to_invalidate = [1, 2, 3, 50, 100]
    print(f"   Invalidating satellites: {satellites_to_invalidate}")
    
    evaluator.invalidate_satellite_cache(satellites_to_invalidate)
    
    after_satellite_stats = evaluator.get_cache_stats()
    after_satellite_size = after_satellite_stats['advanced_caching']['deltav_cache']['size']
    satellite_invalidated = initial_size - after_satellite_size
    
    print(f"   Cache size after satellite invalidation: {after_satellite_size} entries")
    print(f"   Entries invalidated: {satellite_invalidated}")
    
    # Invalidate time range
    print(f"   Invalidating time range: 2000.0 - 4000.0 seconds")
    evaluator.invalidate_time_range_cache(2000.0, 4000.0)
    
    after_time_stats = evaluator.get_cache_stats()
    after_time_size = after_time_stats['advanced_caching']['deltav_cache']['size']
    time_invalidated = after_satellite_size - after_time_size
    
    print(f"   Cache size after time invalidation: {after_time_size} entries")
    print(f"   Additional entries invalidated: {time_invalidated}")
    
    # Perform cache maintenance
    print("   Performing cache maintenance...")
    evaluator.perform_cache_maintenance()
    
    final_stats = evaluator.get_cache_stats()
    final_size = final_stats['advanced_caching']['deltav_cache']['size']
    print(f"   Final cache size: {final_size} entries")


def demonstrate_route_evaluation_caching():
    """Demonstrate route evaluation with caching."""
    print("\n🎯 Route Evaluation with Caching")
    print("-" * 32)
    
    # Create smaller constellation for detailed demonstration
    satellites = create_mock_constellation(20)
    propagator = create_mock_propagator(20)
    cache_manager = FitnessCacheManager()
    
    evaluator = RouteFitnessEvaluator(
        satellites=satellites,
        orbital_propagator=propagator,
        cache_manager=cache_manager
    )
    
    # Create test chromosome
    chromosome = RouteChromosome(
        satellite_sequence=[1, 5, 10, 15, 20],
        departure_times=[1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
    )
    
    constraints = RouteConstraints(
        max_deltav_budget=5.0,
        max_mission_duration=86400.0,
        min_hops=1,
        max_hops=10
    )
    
    print(f"   Route: {chromosome.satellite_sequence}")
    print(f"   Departure times: {[t/1000 for t in chromosome.departure_times]} (ks)")
    
    # Evaluate route multiple times to show caching
    print("\n   Evaluating route (first time)...")
    start_time = time.time()
    result1 = evaluator.evaluate_route(chromosome, constraints)
    first_eval_time = time.time() - start_time
    
    print(f"   Time: {first_eval_time:.4f}s")
    print(f"   Delta-v: {result1.total_deltav:.3f} km/s")
    print(f"   Fitness: {result1.fitness_score:.2f}")
    print(f"   Valid: {result1.is_valid}")
    
    print("\n   Evaluating same route (second time)...")
    start_time = time.time()
    result2 = evaluator.evaluate_route(chromosome, constraints)
    second_eval_time = time.time() - start_time
    
    print(f"   Time: {second_eval_time:.4f}s")
    print(f"   Delta-v: {result2.total_deltav:.3f} km/s")
    print(f"   Fitness: {result2.fitness_score:.2f}")
    
    if second_eval_time > 0:
        eval_speedup = first_eval_time / second_eval_time
        print(f"   🚀 Evaluation speedup: {eval_speedup:.1f}x")
    else:
        print("   🚀 Evaluation speedup: Instantaneous!")
    
    # Show cache effectiveness
    cache_stats = evaluator.get_cache_stats()
    deltav_stats = cache_stats['advanced_caching']['deltav_cache']['stats']
    print(f"   Cache hit rate: {deltav_stats.hit_rate:.2%}")


def main():
    """Run the complete caching demonstration."""
    print("Starting Fitness Caching System Demonstration...\n")
    
    try:
        # Main performance demonstration
        evaluator, cache_stats = demonstrate_caching_performance()
        
        # Cache invalidation demonstration
        demonstrate_cache_invalidation(evaluator)
        
        # Route evaluation demonstration
        demonstrate_route_evaluation_caching()
        
        print("\n" + "=" * 50)
        print("✅ Demonstration completed successfully!")
        print("\nKey Benefits Demonstrated:")
        print("  • Significant performance improvements through caching")
        print("  • Intelligent time-based cache key normalization")
        print("  • Memory-efficient LRU eviction policies")
        print("  • Flexible cache invalidation strategies")
        print("  • Automatic cache optimization for constellation size")
        print("  • Comprehensive cache statistics and monitoring")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()