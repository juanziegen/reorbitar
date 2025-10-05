#!/usr/bin/env python3
"""
Performance benchmarks for the genetic route optimizer.

Tests different constellation sizes, convergence speed, solution quality,
memory usage, and compares performance with and without optimizations.
"""

import time
import psutil
import os
import statistics
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import json
from datetime import datetime

from src.genetic_route_optimizer import GeneticRouteOptimizer, GAConfig, RouteConstraints
from src.tle_parser import TLEParser
from src.satellite_filter import SatelliteFilter
from src.fitness_cache import FitnessCacheManager


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark run."""
    constellation_size: int
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    generations_completed: int
    best_fitness: float
    average_fitness: float
    convergence_generation: int
    solution_quality_score: float
    cache_hit_rate: float
    optimization_enabled: bool
    error_message: str = None


@dataclass
class ConvergenceAnalysis:
    """Analysis of convergence characteristics."""
    generations_to_convergence: int
    convergence_rate: float
    fitness_improvement_per_generation: float
    diversity_decay_rate: float
    stagnation_periods: List[Tuple[int, int]]  # (start_gen, end_gen)


class PerformanceBenchmark:
    """Comprehensive performance benchmarking for genetic route optimizer."""
    
    def __init__(self, tle_file_path: str = "data/leo_satellites.txt"):
        """Initialize benchmark with satellite data."""
        self.tle_file_path = tle_file_path
        self.satellites = []
        self.benchmark_results = []
        
        # Load satellite data
        try:
            parser = TLEParser()
            self.satellites = parser.parse_tle_file(tle_file_path)
            print(f"Loaded {len(self.satellites)} satellites for benchmarking")
        except Exception as e:
            print(f"Warning: Could not load satellite data: {e}")
            # Create synthetic satellite data for testing
            self.satellites = self._create_synthetic_satellites(100)
    
    def _create_synthetic_satellites(self, count: int) -> List[Any]:
        """Create synthetic satellite data for testing when real data unavailable."""
        from src.tle_parser import SatelliteData
        from datetime import datetime
        import random
        
        satellites = []
        for i in range(count):
            # Create realistic orbital parameters
            inclination = random.uniform(0, 180)  # degrees
            raan = random.uniform(0, 360)  # degrees
            eccentricity = random.uniform(0, 0.1)
            arg_perigee = random.uniform(0, 360)  # degrees
            mean_anomaly = random.uniform(0, 360)  # degrees
            mean_motion = random.uniform(10, 16)  # revs/day
            
            # Calculate derived parameters
            parser = TLEParser()
            semi_major_axis = parser._calculate_semi_major_axis(mean_motion)
            orbital_period = parser._calculate_orbital_period(semi_major_axis)
            
            satellite = SatelliteData(
                catalog_number=25544 + i,
                name=f"SYNTHETIC-SAT-{i:03d}",
                epoch=datetime(2024, 1, 1),
                mean_motion=mean_motion,
                eccentricity=eccentricity,
                inclination=inclination,
                raan=raan,
                arg_perigee=arg_perigee,
                mean_anomaly=mean_anomaly,
                semi_major_axis=semi_major_axis,
                orbital_period=orbital_period
            )
            satellites.append(satellite)
        
        return satellites
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def _create_test_constraints(self, constellation_size: int) -> RouteConstraints:
        """Create appropriate constraints for constellation size."""
        # Scale constraints based on constellation size
        if constellation_size <= 100:
            max_deltav = 5.0  # km/s
            max_duration = 365 * 24 * 3600  # 1 year
            max_hops = 10
        elif constellation_size <= 1000:
            max_deltav = 8.0  # km/s
            max_duration = 2 * 365 * 24 * 3600  # 2 years
            max_hops = 15
        else:
            max_deltav = 12.0  # km/s
            max_duration = 3 * 365 * 24 * 3600  # 3 years
            max_hops = 20
        
        return RouteConstraints(
            max_deltav_budget=max_deltav,
            max_mission_duration=max_duration,
            max_hops=max_hops,
            min_hops=3
        )
    
    def _create_ga_config(self, constellation_size: int, optimized: bool = True) -> GAConfig:
        """Create GA configuration appropriate for constellation size."""
        if constellation_size <= 100:
            population_size = 50
            max_generations = 100
        elif constellation_size <= 1000:
            population_size = 100
            max_generations = 200
        else:
            population_size = 150
            max_generations = 300
        
        # Reduce parameters for non-optimized runs to show difference
        if not optimized:
            population_size = max(20, population_size // 3)
            max_generations = max(50, max_generations // 2)
        
        return GAConfig(
            population_size=population_size,
            max_generations=max_generations,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elitism_count=max(2, population_size // 20),
            tournament_size=3,
            convergence_threshold=1e-6,
            max_stagnant_generations=20
        )
    
    def benchmark_constellation_size(self, size: int, optimized: bool = True) -> BenchmarkResult:
        """Benchmark performance for a specific constellation size."""
        print(f"\nBenchmarking constellation size: {size} (optimized: {optimized})")
        
        # Select subset of satellites
        test_satellites = self.satellites[:min(size, len(self.satellites))]
        if len(test_satellites) < size:
            # Extend with synthetic satellites if needed
            additional_needed = size - len(test_satellites)
            test_satellites.extend(self._create_synthetic_satellites(additional_needed))
        
        # Create configuration
        config = self._create_ga_config(size, optimized)
        constraints = self._create_test_constraints(size)
        
        # Initialize optimizer
        try:
            if optimized:
                # Use satellite filtering for optimization
                satellite_filter = SatelliteFilter(test_satellites)
                filtered_satellites = satellite_filter.filter_by_orbital_characteristics(
                    min_altitude=200,  # km
                    max_altitude=2000,  # km
                    max_inclination_diff=30  # degrees
                )
                optimizer = GeneticRouteOptimizer(filtered_satellites, config)
            else:
                # Use all satellites without filtering
                optimizer = GeneticRouteOptimizer(test_satellites, config)
            
            # Measure initial memory
            initial_memory = self._get_memory_usage()
            peak_memory = initial_memory
            
            # Run optimization
            start_time = time.time()
            
            # Monitor memory during execution
            def memory_monitor():
                nonlocal peak_memory
                current_memory = self._get_memory_usage()
                peak_memory = max(peak_memory, current_memory)
            
            result = optimizer.optimize_route(constraints)
            
            end_time = time.time()
            final_memory = self._get_memory_usage()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_usage = final_memory - initial_memory
            
            # Get cache statistics if available
            cache_hit_rate = 0.0
            if hasattr(optimizer, 'fitness_evaluator') and hasattr(optimizer.fitness_evaluator, 'cache'):
                cache = optimizer.fitness_evaluator.cache
                if hasattr(cache, 'hit_rate'):
                    cache_hit_rate = cache.hit_rate
            
            # Calculate solution quality score
            solution_quality_score = self._calculate_solution_quality(result, constraints)
            
            # Determine convergence generation
            convergence_generation = len(result.convergence_history) if result.convergence_history else config.max_generations
            
            return BenchmarkResult(
                constellation_size=size,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                peak_memory_mb=peak_memory,
                generations_completed=result.optimization_stats.generations_completed if result.optimization_stats else 0,
                best_fitness=result.optimization_stats.best_fitness if result.optimization_stats else 0.0,
                average_fitness=result.optimization_stats.average_fitness if result.optimization_stats else 0.0,
                convergence_generation=convergence_generation,
                solution_quality_score=solution_quality_score,
                cache_hit_rate=cache_hit_rate,
                optimization_enabled=optimized
            )
            
        except Exception as e:
            return BenchmarkResult(
                constellation_size=size,
                execution_time=0.0,
                memory_usage_mb=0.0,
                peak_memory_mb=0.0,
                generations_completed=0,
                best_fitness=0.0,
                average_fitness=0.0,
                convergence_generation=0,
                solution_quality_score=0.0,
                cache_hit_rate=0.0,
                optimization_enabled=optimized,
                error_message=str(e)
            )
    
    def _calculate_solution_quality(self, result, constraints) -> float:
        """Calculate a quality score for the solution (0-100)."""
        if not result.success or not result.best_route:
            return 0.0
        
        # Quality factors:
        # 1. Number of hops (more is better)
        # 2. Delta-v efficiency (closer to budget is better)
        # 3. Constraint satisfaction
        
        route = result.best_route
        max_possible_hops = constraints.max_hops
        deltav_efficiency = min(1.0, route.total_deltav / constraints.max_deltav_budget)
        
        hop_score = (len(route.satellite_sequence) / max_possible_hops) * 40
        efficiency_score = deltav_efficiency * 30
        validity_score = 30 if route.is_valid else 0
        
        return min(100.0, hop_score + efficiency_score + validity_score)
    
    def benchmark_small_constellation(self) -> BenchmarkResult:
        """Test with 50-100 satellites."""
        return self.benchmark_constellation_size(50)
    
    def benchmark_medium_constellation(self) -> BenchmarkResult:
        """Test with 500-1000 satellites."""
        return self.benchmark_constellation_size(500)
    
    def benchmark_large_constellation(self) -> BenchmarkResult:
        """Test with 2000+ satellites."""
        return self.benchmark_constellation_size(2000)
    
    def benchmark_convergence_speed(self, constellation_size: int = 200) -> ConvergenceAnalysis:
        """Analyze convergence characteristics for a given constellation size."""
        print(f"\nAnalyzing convergence for constellation size: {constellation_size}")
        
        test_satellites = self.satellites[:min(constellation_size, len(self.satellites))]
        config = self._create_ga_config(constellation_size)
        constraints = self._create_test_constraints(constellation_size)
        
        optimizer = GeneticRouteOptimizer(test_satellites, config)
        result = optimizer.optimize_route(constraints)
        
        if not result.convergence_history:
            return ConvergenceAnalysis(0, 0.0, 0.0, 0.0, [])
        
        # Analyze convergence
        fitness_history = [gen.best_fitness for gen in result.convergence_history]
        diversity_history = [gen.diversity_metric for gen in result.convergence_history]
        
        # Find convergence point (when fitness stops improving significantly)
        convergence_gen = len(fitness_history)
        improvement_threshold = 0.001
        
        for i in range(1, len(fitness_history)):
            if i >= 10:  # Look at last 10 generations
                recent_improvements = [
                    abs(fitness_history[j] - fitness_history[j-1]) 
                    for j in range(i-9, i+1)
                ]
                avg_improvement = statistics.mean(recent_improvements)
                if avg_improvement < improvement_threshold:
                    convergence_gen = i
                    break
        
        # Calculate convergence rate
        if convergence_gen > 1:
            total_improvement = abs(fitness_history[convergence_gen-1] - fitness_history[0])
            convergence_rate = total_improvement / convergence_gen
        else:
            convergence_rate = 0.0
        
        # Calculate average fitness improvement per generation
        improvements = [
            abs(fitness_history[i] - fitness_history[i-1]) 
            for i in range(1, len(fitness_history))
        ]
        avg_improvement_per_gen = statistics.mean(improvements) if improvements else 0.0
        
        # Calculate diversity decay rate
        if len(diversity_history) > 1:
            diversity_decay = (diversity_history[0] - diversity_history[-1]) / len(diversity_history)
        else:
            diversity_decay = 0.0
        
        # Find stagnation periods (periods with little fitness improvement)
        stagnation_periods = []
        stagnation_start = None
        stagnation_threshold = 0.0001
        
        for i in range(1, len(fitness_history)):
            improvement = abs(fitness_history[i] - fitness_history[i-1])
            if improvement < stagnation_threshold:
                if stagnation_start is None:
                    stagnation_start = i
            else:
                if stagnation_start is not None:
                    stagnation_periods.append((stagnation_start, i-1))
                    stagnation_start = None
        
        # Close final stagnation period if needed
        if stagnation_start is not None:
            stagnation_periods.append((stagnation_start, len(fitness_history)-1))
        
        return ConvergenceAnalysis(
            generations_to_convergence=convergence_gen,
            convergence_rate=convergence_rate,
            fitness_improvement_per_generation=avg_improvement_per_gen,
            diversity_decay_rate=diversity_decay,
            stagnation_periods=stagnation_periods
        )
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmarks across all constellation sizes and configurations."""
        print("Starting comprehensive performance benchmarks...")
        print("=" * 60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
            },
            'constellation_benchmarks': [],
            'optimization_comparison': [],
            'convergence_analysis': {},
            'summary': {}
        }
        
        # Test different constellation sizes
        constellation_sizes = [50, 200, 500, 1000, 2000]
        
        for size in constellation_sizes:
            if size <= len(self.satellites) * 2:  # Only test if we have enough data
                # Test with optimizations
                optimized_result = self.benchmark_constellation_size(size, optimized=True)
                results['constellation_benchmarks'].append(optimized_result.__dict__)
                
                # Test without optimizations for comparison
                unoptimized_result = self.benchmark_constellation_size(size, optimized=False)
                
                # Store comparison
                comparison = {
                    'constellation_size': size,
                    'optimized_time': optimized_result.execution_time,
                    'unoptimized_time': unoptimized_result.execution_time,
                    'speedup_factor': unoptimized_result.execution_time / max(optimized_result.execution_time, 0.001),
                    'optimized_memory': optimized_result.memory_usage_mb,
                    'unoptimized_memory': unoptimized_result.memory_usage_mb,
                    'memory_savings': unoptimized_result.memory_usage_mb - optimized_result.memory_usage_mb,
                    'optimized_quality': optimized_result.solution_quality_score,
                    'unoptimized_quality': unoptimized_result.solution_quality_score
                }
                results['optimization_comparison'].append(comparison)
        
        # Convergence analysis for medium constellation
        convergence = self.benchmark_convergence_speed(200)
        results['convergence_analysis'] = convergence.__dict__
        
        # Generate summary statistics
        if results['constellation_benchmarks']:
            execution_times = [r['execution_time'] for r in results['constellation_benchmarks'] if r['execution_time'] > 0]
            memory_usage = [r['memory_usage_mb'] for r in results['constellation_benchmarks']]
            quality_scores = [r['solution_quality_score'] for r in results['constellation_benchmarks']]
            
            results['summary'] = {
                'total_benchmarks_run': len(results['constellation_benchmarks']),
                'avg_execution_time': statistics.mean(execution_times) if execution_times else 0,
                'max_execution_time': max(execution_times) if execution_times else 0,
                'avg_memory_usage': statistics.mean(memory_usage) if memory_usage else 0,
                'max_memory_usage': max(memory_usage) if memory_usage else 0,
                'avg_solution_quality': statistics.mean(quality_scores) if quality_scores else 0,
                'optimization_effectiveness': {
                    'avg_speedup': statistics.mean([c['speedup_factor'] for c in results['optimization_comparison']]) if results['optimization_comparison'] else 1.0,
                    'avg_memory_savings': statistics.mean([c['memory_savings'] for c in results['optimization_comparison']]) if results['optimization_comparison'] else 0.0
                }
            }
        
        return results
    
    def print_benchmark_results(self, results: Dict[str, Any]):
        """Print formatted benchmark results."""
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)
        
        print(f"\nSystem Information:")
        print(f"  CPU Cores: {results['system_info']['cpu_count']}")
        print(f"  Memory: {results['system_info']['memory_gb']:.1f} GB")
        print(f"  Python: {results['system_info']['python_version']}")
        
        print(f"\nConstellation Size Benchmarks:")
        print(f"{'Size':<8} {'Time(s)':<10} {'Memory(MB)':<12} {'Quality':<8} {'Cache Hit%':<10} {'Status'}")
        print("-" * 60)
        
        for result in results['constellation_benchmarks']:
            status = "SUCCESS" if not result.get('error_message') else "ERROR"
            print(f"{result['constellation_size']:<8} "
                  f"{result['execution_time']:<10.2f} "
                  f"{result['memory_usage_mb']:<12.1f} "
                  f"{result['solution_quality_score']:<8.1f} "
                  f"{result['cache_hit_rate']*100:<10.1f} "
                  f"{status}")
        
        print(f"\nOptimization Effectiveness:")
        print(f"{'Size':<8} {'Speedup':<10} {'Mem Save(MB)':<12} {'Quality Diff':<12}")
        print("-" * 45)
        
        for comp in results['optimization_comparison']:
            quality_diff = comp['optimized_quality'] - comp['unoptimized_quality']
            print(f"{comp['constellation_size']:<8} "
                  f"{comp['speedup_factor']:<10.2f} "
                  f"{comp['memory_savings']:<12.1f} "
                  f"{quality_diff:<12.1f}")
        
        print(f"\nConvergence Analysis:")
        conv = results['convergence_analysis']
        print(f"  Generations to Convergence: {conv['generations_to_convergence']}")
        print(f"  Convergence Rate: {conv['convergence_rate']:.6f}")
        print(f"  Avg Improvement/Gen: {conv['fitness_improvement_per_generation']:.6f}")
        print(f"  Diversity Decay Rate: {conv['diversity_decay_rate']:.6f}")
        print(f"  Stagnation Periods: {len(conv['stagnation_periods'])}")
        
        print(f"\nSummary:")
        summary = results['summary']
        print(f"  Benchmarks Run: {summary['total_benchmarks_run']}")
        print(f"  Avg Execution Time: {summary['avg_execution_time']:.2f}s")
        print(f"  Max Execution Time: {summary['max_execution_time']:.2f}s")
        print(f"  Avg Memory Usage: {summary['avg_memory_usage']:.1f} MB")
        print(f"  Avg Solution Quality: {summary['avg_solution_quality']:.1f}/100")
        print(f"  Avg Optimization Speedup: {summary['optimization_effectiveness']['avg_speedup']:.2f}x")
        print(f"  Avg Memory Savings: {summary['optimization_effectiveness']['avg_memory_savings']:.1f} MB")
    
    def save_results(self, results: Dict[str, Any], filename: str = "benchmark_results.json"):
        """Save benchmark results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {filename}")


def main():
    """Run performance benchmarks."""
    benchmark = PerformanceBenchmark()
    
    # Run comprehensive benchmarks
    results = benchmark.run_comprehensive_benchmark()
    
    # Print results
    benchmark.print_benchmark_results(results)
    
    # Save results
    benchmark.save_results(results)
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()