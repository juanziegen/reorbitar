#!/usr/bin/env python3
"""
Quick performance benchmark tests for validation.

Runs smaller, faster benchmarks to verify the benchmarking system works correctly.
"""

import time
import unittest
from test_performance_benchmarks import PerformanceBenchmark, BenchmarkResult


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test the performance benchmarking system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.benchmark = PerformanceBenchmark()
    
    def test_synthetic_satellite_creation(self):
        """Test creation of synthetic satellites for benchmarking."""
        satellites = self.benchmark._create_synthetic_satellites(10)
        
        self.assertEqual(len(satellites), 10)
        self.assertTrue(all(hasattr(sat, 'catalog_number') for sat in satellites))
        self.assertTrue(all(hasattr(sat, 'inclination') for sat in satellites))
        self.assertTrue(all(hasattr(sat, 'mean_motion') for sat in satellites))
    
    def test_memory_usage_measurement(self):
        """Test memory usage measurement."""
        memory_usage = self.benchmark._get_memory_usage()
        
        self.assertIsInstance(memory_usage, float)
        self.assertGreater(memory_usage, 0)
    
    def test_constraint_creation(self):
        """Test creation of test constraints for different constellation sizes."""
        small_constraints = self.benchmark._create_test_constraints(50)
        large_constraints = self.benchmark._create_test_constraints(2000)
        
        # Larger constellations should have more generous constraints
        self.assertLessEqual(small_constraints.max_deltav_budget, large_constraints.max_deltav_budget)
        self.assertLessEqual(small_constraints.max_hops, large_constraints.max_hops)
    
    def test_ga_config_creation(self):
        """Test GA configuration creation for different sizes."""
        small_config = self.benchmark._create_ga_config(50, optimized=True)
        large_config = self.benchmark._create_ga_config(2000, optimized=True)
        unoptimized_config = self.benchmark._create_ga_config(50, optimized=False)
        
        # Larger constellations should have larger populations
        self.assertLessEqual(small_config.population_size, large_config.population_size)
        
        # Optimized configs should have larger populations than unoptimized
        self.assertGreater(small_config.population_size, unoptimized_config.population_size)
    
    def test_solution_quality_calculation(self):
        """Test solution quality score calculation."""
        from src.genetic_route_optimizer import RouteChromosome, OptimizationResult, OptimizationStats
        
        # Create mock result with valid route
        valid_route = RouteChromosome(
            satellite_sequence=[25544, 25545, 25546],
            departure_times=[0, 3600, 7200],
            total_deltav=3.0,
            is_valid=True,
            constraint_violations=[]
        )
        
        from src.genetic_algorithm import OptimizationStatus
        
        result = OptimizationResult(
            best_route=valid_route,
            optimization_stats=None,
            convergence_history=[],
            execution_time=10.0,
            status=OptimizationStatus.SUCCESS,
            error_message=None
        )
        
        constraints = self.benchmark._create_test_constraints(50)
        quality_score = self.benchmark._calculate_solution_quality(result, constraints)
        
        self.assertGreaterEqual(quality_score, 0)
        self.assertLessEqual(quality_score, 100)
        self.assertGreater(quality_score, 50)  # Should be decent quality for valid route
    
    def test_small_constellation_benchmark(self):
        """Test benchmarking with small constellation (quick test)."""
        # Use very small constellation for quick test
        result = self.benchmark.benchmark_constellation_size(10, optimized=True)
        
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.constellation_size, 10)
        self.assertGreaterEqual(result.execution_time, 0)
        self.assertGreaterEqual(result.memory_usage_mb, 0)
        
        # If successful, should have some generations completed
        if not result.error_message:
            self.assertGreater(result.generations_completed, 0)
    
    def test_benchmark_comparison(self):
        """Test comparison between optimized and unoptimized runs."""
        # Use very small constellation for quick comparison
        optimized = self.benchmark.benchmark_constellation_size(8, optimized=True)
        unoptimized = self.benchmark.benchmark_constellation_size(8, optimized=False)
        
        self.assertIsInstance(optimized, BenchmarkResult)
        self.assertIsInstance(unoptimized, BenchmarkResult)
        
        # Both should complete (though may have errors with synthetic data)
        self.assertEqual(optimized.constellation_size, 8)
        self.assertEqual(unoptimized.constellation_size, 8)
        self.assertTrue(optimized.optimization_enabled)
        self.assertFalse(unoptimized.optimization_enabled)


class QuickBenchmarkRunner:
    """Quick benchmark runner for validation."""
    
    def __init__(self):
        self.benchmark = PerformanceBenchmark()
    
    def run_quick_benchmarks(self):
        """Run quick benchmarks for validation."""
        print("Running quick performance benchmarks...")
        print("=" * 50)
        
        # Test very small constellations
        sizes = [5, 10, 20]
        results = []
        
        for size in sizes:
            print(f"\nTesting constellation size: {size}")
            
            start_time = time.time()
            result = self.benchmark.benchmark_constellation_size(size, optimized=True)
            end_time = time.time()
            
            results.append(result)
            
            print(f"  Execution time: {result.execution_time:.2f}s")
            print(f"  Memory usage: {result.memory_usage_mb:.1f} MB")
            print(f"  Generations: {result.generations_completed}")
            print(f"  Quality score: {result.solution_quality_score:.1f}")
            
            if result.error_message:
                print(f"  Error: {result.error_message}")
            else:
                print(f"  Status: SUCCESS")
        
        # Test convergence analysis with small constellation
        print(f"\nTesting convergence analysis...")
        convergence = self.benchmark.benchmark_convergence_speed(15)
        print(f"  Convergence generation: {convergence.generations_to_convergence}")
        print(f"  Convergence rate: {convergence.convergence_rate:.6f}")
        print(f"  Stagnation periods: {len(convergence.stagnation_periods)}")
        
        print(f"\nQuick benchmarks complete!")
        return results


def main():
    """Run quick benchmarks."""
    runner = QuickBenchmarkRunner()
    runner.run_quick_benchmarks()


if __name__ == "__main__":
    # Run quick benchmarks
    main()
    
    # Also run unit tests
    print("\n" + "=" * 50)
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)