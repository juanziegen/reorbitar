#!/usr/bin/env python3
"""
Validation tests for performance benchmarks.

Ensures all benchmark components work correctly before running full benchmarks.
"""

import unittest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock

from test_performance_benchmarks import PerformanceBenchmark, BenchmarkResult, ConvergenceAnalysis
from memory_profiler import MemoryProfiler, MemorySnapshot, ProfiledOptimization
from src.genetic_algorithm import OptimizationResult, OptimizationStatus


class TestBenchmarkValidation(unittest.TestCase):
    """Validate benchmark functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.benchmark = PerformanceBenchmark()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization with and without satellite data."""
        # Test with synthetic data (when real data unavailable)
        benchmark = PerformanceBenchmark("nonexistent_file.txt")
        self.assertGreater(len(benchmark.satellites), 0)
        self.assertEqual(len(benchmark.satellites), 100)  # Default synthetic count
    
    def test_synthetic_satellite_generation(self):
        """Test synthetic satellite data generation."""
        satellites = self.benchmark._create_synthetic_satellites(50)
        
        self.assertEqual(len(satellites), 50)
        
        # Check all satellites have required attributes
        for sat in satellites:
            self.assertTrue(hasattr(sat, 'catalog_number'))
            self.assertTrue(hasattr(sat, 'name'))
            self.assertTrue(hasattr(sat, 'inclination'))
            self.assertTrue(hasattr(sat, 'raan'))
            self.assertTrue(hasattr(sat, 'eccentricity'))
            self.assertTrue(hasattr(sat, 'mean_motion'))
            
            # Check realistic ranges
            self.assertGreaterEqual(sat.inclination, 0)
            self.assertLessEqual(sat.inclination, 180)
            self.assertGreaterEqual(sat.mean_motion, 10)
            self.assertLessEqual(sat.mean_motion, 16)
    
    def test_constraint_scaling(self):
        """Test constraint scaling for different constellation sizes."""
        small_constraints = self.benchmark._create_test_constraints(50)
        medium_constraints = self.benchmark._create_test_constraints(500)
        large_constraints = self.benchmark._create_test_constraints(2000)
        
        # Constraints should scale with constellation size
        self.assertLessEqual(small_constraints.max_deltav_budget, medium_constraints.max_deltav_budget)
        self.assertLessEqual(medium_constraints.max_deltav_budget, large_constraints.max_deltav_budget)
        
        self.assertLessEqual(small_constraints.max_hops, medium_constraints.max_hops)
        self.assertLessEqual(medium_constraints.max_hops, large_constraints.max_hops)
    
    def test_ga_config_scaling(self):
        """Test GA configuration scaling for different sizes and optimization levels."""
        configs = [
            (50, True),
            (500, True),
            (2000, True),
            (50, False),
            (500, False)
        ]
        
        for size, optimized in configs:
            config = self.benchmark._create_ga_config(size, optimized)
            
            self.assertGreater(config.population_size, 0)
            self.assertGreater(config.max_generations, 0)
            self.assertGreaterEqual(config.mutation_rate, 0)
            self.assertLessEqual(config.mutation_rate, 1)
            self.assertGreaterEqual(config.crossover_rate, 0)
            self.assertLessEqual(config.crossover_rate, 1)
    
    def test_solution_quality_calculation(self):
        """Test solution quality score calculation with various scenarios."""
        from src.genetic_route_optimizer import RouteChromosome, OptimizationResult
        
        constraints = self.benchmark._create_test_constraints(100)
        
        # Test valid high-quality route
        good_route = RouteChromosome(
            satellite_sequence=[25544, 25545, 25546, 25547, 25548],
            departure_times=[0, 3600, 7200, 10800, 14400],
            total_deltav=4.0,  # Within budget
            is_valid=True,
            constraint_violations=[]
        )
        
        from src.genetic_algorithm import OptimizationStatus
        
        good_result = OptimizationResult(
            best_route=good_route,
            optimization_stats=None,
            convergence_history=[],
            execution_time=10.0,
            status=OptimizationStatus.SUCCESS,
            error_message=None
        )
        
        quality_score = self.benchmark._calculate_solution_quality(good_result, constraints)
        self.assertGreater(quality_score, 50)
        self.assertLessEqual(quality_score, 100)
        
        # Test invalid route
        bad_route = RouteChromosome(
            satellite_sequence=[25544],
            departure_times=[0],
            total_deltav=10.0,  # Over budget
            is_valid=False,
            constraint_violations=["Delta-v budget exceeded"]
        )
        
        bad_result = OptimizationResult(
            best_route=bad_route,
            optimization_stats=None,
            convergence_history=[],
            execution_time=10.0,
            status=OptimizationStatus.FAILED,
            error_message="No valid solution found"
        )
        
        bad_quality_score = self.benchmark._calculate_solution_quality(bad_result, constraints)
        self.assertLess(bad_quality_score, quality_score)
    
    @patch('src.genetic_route_optimizer.GeneticRouteOptimizer')
    def test_benchmark_constellation_size_mock(self, mock_optimizer_class):
        """Test constellation size benchmarking with mocked optimizer."""
        # Create mock optimizer and result
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer
        
        # Mock successful optimization result
        from src.genetic_route_optimizer import RouteChromosome, OptimizationResult, OptimizationStats
        
        mock_route = RouteChromosome(
            satellite_sequence=[25544, 25545, 25546],
            departure_times=[0, 3600, 7200],
            total_deltav=3.0,
            is_valid=True,
            constraint_violations=[]
        )
        
        mock_stats = OptimizationStats(
            generations_completed=50,
            best_fitness=0.8,
            average_fitness=0.6,
            population_diversity=0.4,
            constraint_satisfaction_rate=0.9
        )
        
        mock_result = OptimizationResult(
            best_route=mock_route,
            optimization_stats=mock_stats,
            convergence_history=[],
            execution_time=5.0,
            status=OptimizationStatus.SUCCESS,
            error_message=None
        )
        
        mock_optimizer.optimize_route.return_value = mock_result
        
        # Run benchmark
        result = self.benchmark.benchmark_constellation_size(50, optimized=True)
        
        # Verify results
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.constellation_size, 50)
        self.assertGreater(result.execution_time, 0)
        self.assertEqual(result.generations_completed, 50)
        self.assertEqual(result.best_fitness, 0.8)
        self.assertTrue(result.optimization_enabled)
        self.assertIsNone(result.error_message)
    
    def test_benchmark_error_handling(self):
        """Test benchmark error handling when optimization fails."""
        # Create benchmark with no satellites to force error
        benchmark = PerformanceBenchmark()
        benchmark.satellites = []  # Empty satellite list
        
        result = benchmark.benchmark_constellation_size(10, optimized=True)
        
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.constellation_size, 10)
        self.assertIsNotNone(result.error_message)
        self.assertEqual(result.execution_time, 0.0)
        self.assertEqual(result.generations_completed, 0)
    
    def test_convergence_analysis_structure(self):
        """Test convergence analysis data structure."""
        # Create mock convergence analysis
        analysis = ConvergenceAnalysis(
            generations_to_convergence=25,
            convergence_rate=0.05,
            fitness_improvement_per_generation=0.02,
            diversity_decay_rate=0.01,
            stagnation_periods=[(10, 15), (30, 35)]
        )
        
        self.assertEqual(analysis.generations_to_convergence, 25)
        self.assertEqual(analysis.convergence_rate, 0.05)
        self.assertEqual(len(analysis.stagnation_periods), 2)
        self.assertEqual(analysis.stagnation_periods[0], (10, 15))
    
    def test_results_serialization(self):
        """Test benchmark results can be serialized to JSON."""
        result = BenchmarkResult(
            constellation_size=100,
            execution_time=15.5,
            memory_usage_mb=256.7,
            peak_memory_mb=300.2,
            generations_completed=75,
            best_fitness=0.85,
            average_fitness=0.65,
            convergence_generation=60,
            solution_quality_score=78.5,
            cache_hit_rate=0.45,
            optimization_enabled=True
        )
        
        # Convert to dict and serialize
        result_dict = result.__dict__
        json_str = json.dumps(result_dict, default=str)
        
        # Deserialize and verify
        loaded_dict = json.loads(json_str)
        self.assertEqual(loaded_dict['constellation_size'], 100)
        self.assertEqual(loaded_dict['execution_time'], 15.5)
        self.assertEqual(loaded_dict['optimization_enabled'], True)
    
    def test_comprehensive_benchmark_structure(self):
        """Test comprehensive benchmark result structure."""
        # Mock the benchmark methods to avoid long execution
        with patch.object(self.benchmark, 'benchmark_constellation_size') as mock_benchmark, \
             patch.object(self.benchmark, 'benchmark_convergence_speed') as mock_convergence:
            
            # Mock benchmark results
            mock_result = BenchmarkResult(
                constellation_size=50,
                execution_time=10.0,
                memory_usage_mb=100.0,
                peak_memory_mb=120.0,
                generations_completed=50,
                best_fitness=0.8,
                average_fitness=0.6,
                convergence_generation=40,
                solution_quality_score=75.0,
                cache_hit_rate=0.3,
                optimization_enabled=True
            )
            
            mock_benchmark.return_value = mock_result
            
            mock_convergence_result = ConvergenceAnalysis(
                generations_to_convergence=40,
                convergence_rate=0.02,
                fitness_improvement_per_generation=0.01,
                diversity_decay_rate=0.005,
                stagnation_periods=[]
            )
            
            mock_convergence.return_value = mock_convergence_result
            
            # Run comprehensive benchmark
            results = self.benchmark.run_comprehensive_benchmark()
            
            # Verify structure
            self.assertIn('timestamp', results)
            self.assertIn('system_info', results)
            self.assertIn('constellation_benchmarks', results)
            self.assertIn('optimization_comparison', results)
            self.assertIn('convergence_analysis', results)
            self.assertIn('summary', results)
            
            # Verify system info
            system_info = results['system_info']
            self.assertIn('cpu_count', system_info)
            self.assertIn('memory_gb', system_info)
            self.assertIn('python_version', system_info)


class TestMemoryProfiler(unittest.TestCase):
    """Test memory profiler functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profiler = MemoryProfiler(sampling_interval=0.01)
    
    def test_memory_snapshot_creation(self):
        """Test memory snapshot creation."""
        snapshot = MemorySnapshot(
            timestamp=1.5,
            rss_mb=100.0,
            vms_mb=200.0,
            percent=5.0,
            generation=10,
            phase="fitness_evaluation"
        )
        
        self.assertEqual(snapshot.timestamp, 1.5)
        self.assertEqual(snapshot.rss_mb, 100.0)
        self.assertEqual(snapshot.generation, 10)
        self.assertEqual(snapshot.phase, "fitness_evaluation")
    
    def test_profiler_context_manager(self):
        """Test profiler context manager."""
        with ProfiledOptimization(self.profiler) as prof:
            self.assertTrue(prof.monitoring)
            prof.set_phase("test_phase")
            prof.set_generation(5)
            import time
            time.sleep(0.05)  # Brief sleep to collect samples
        
        self.assertFalse(self.profiler.monitoring)
        self.assertGreater(len(self.profiler.snapshots), 0)
    
    def test_memory_analysis_methods(self):
        """Test memory analysis methods."""
        # Add some test snapshots
        self.profiler.snapshots = [
            MemorySnapshot(0.0, 100.0, 200.0, 5.0, 0, "init"),
            MemorySnapshot(1.0, 110.0, 210.0, 5.5, 1, "fitness"),
            MemorySnapshot(2.0, 120.0, 220.0, 6.0, 1, "fitness"),
            MemorySnapshot(3.0, 115.0, 215.0, 5.8, 2, "selection")
        ]
        
        # Test peak memory
        peak = self.profiler.get_peak_memory()
        self.assertEqual(peak, 120.0)
        
        # Test average memory
        avg = self.profiler.get_average_memory()
        self.assertEqual(avg, 111.25)
        
        # Test growth rate
        growth_rate = self.profiler.get_memory_growth_rate()
        self.assertEqual(growth_rate, 5.0)  # (115-100)/3
        
        # Test phase analysis
        phase_analysis = self.profiler.analyze_memory_by_phase()
        self.assertIn("fitness", phase_analysis)
        self.assertIn("init", phase_analysis)
        self.assertIn("selection", phase_analysis)
        
        fitness_stats = phase_analysis["fitness"]
        self.assertEqual(fitness_stats["peak_memory"], 120.0)
        self.assertEqual(fitness_stats["sample_count"], 2)
        
        # Test generation analysis
        gen_analysis = self.profiler.analyze_memory_by_generation()
        self.assertIn(1, gen_analysis)
        self.assertEqual(gen_analysis[1]["sample_count"], 2)
    
    def test_memory_leak_detection(self):
        """Test memory leak detection."""
        # Create snapshots with sustained growth
        snapshots = []
        for i in range(50):
            snapshots.append(MemorySnapshot(
                timestamp=i * 0.1,
                rss_mb=100.0 + i * 2.0,  # 2MB growth per sample
                vms_mb=200.0,
                percent=5.0,
                generation=i // 10
            ))
        
        self.profiler.snapshots = snapshots
        
        leaks = self.profiler.detect_memory_leaks(threshold_mb=5.0)
        self.assertGreater(len(leaks), 0)
        
        # Check leak structure
        leak = leaks[0]
        self.assertIn('memory_growth_mb', leak)
        self.assertIn('growth_rate_mb_per_sec', leak)
        self.assertIn('start_time', leak)
        self.assertIn('end_time', leak)
    
    def test_report_generation(self):
        """Test comprehensive report generation."""
        # Add test data
        self.profiler.snapshots = [
            MemorySnapshot(0.0, 100.0, 200.0, 5.0, 0, "init"),
            MemorySnapshot(1.0, 110.0, 210.0, 5.5, 1, "fitness"),
            MemorySnapshot(2.0, 105.0, 205.0, 5.2, 2, "selection")
        ]
        
        report = self.profiler.generate_report()
        
        # Verify report structure
        self.assertIn('summary', report)
        self.assertIn('phase_analysis', report)
        self.assertIn('generation_analysis', report)
        self.assertIn('memory_leaks', report)
        self.assertIn('recommendations', report)
        
        # Verify summary data
        summary = report['summary']
        self.assertEqual(summary['total_samples'], 3)
        self.assertEqual(summary['peak_memory_mb'], 110.0)
        self.assertGreater(summary['monitoring_duration'], 0)
        
        # Verify recommendations exist
        self.assertIsInstance(report['recommendations'], list)
        self.assertGreater(len(report['recommendations']), 0)


def run_validation_tests():
    """Run all validation tests."""
    print("Running benchmark validation tests...")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestBenchmarkValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryProfiler))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All validation tests passed!")
        print("Benchmark system is ready for use.")
    else:
        print("❌ Some validation tests failed.")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_validation_tests()
    exit(0 if success else 1)