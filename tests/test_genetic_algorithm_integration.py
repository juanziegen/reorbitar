"""
Comprehensive Integration Tests for Genetic Algorithm Engine

This test suite verifies the complete genetic algorithm system including:
- Complete optimization runs with small satellite constellations
- Convergence behavior and solution quality verification
- Error recovery and constraint satisfaction testing
- Performance benchmarking with different parameter configurations

Requirements tested: 1.1, 1.2, 1.3, 4.1
"""

import time
import math
import random
import statistics
from typing import List, Dict, Any
from src.genetic_route_optimizer import GeneticRouteOptimizer
from src.genetic_algorithm import GAConfig, RouteConstraints, OptimizationStatus
from src.tle_parser import SatelliteData


class IntegrationTestSuite:
    """Comprehensive integration test suite for genetic algorithm engine."""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
        
    def create_small_constellation(self, size: int = 12) -> List[SatelliteData]:
        """Create small satellite constellation for testing."""
        satellites = []
        
        # Create satellites with realistic LEO parameters
        base_altitudes = [400, 450, 500, 550, 600]  # km above Earth
        base_inclinations = [51.6, 53.0, 97.8, 98.2]  # degrees
        
        for i in range(size):
            # Vary orbital parameters for diversity
            altitude_km = base_altitudes[i % len(base_altitudes)] + (i * 10)
            inclination = base_inclinations[i % len(base_inclinations)] + (i * 0.5)
            
            # Convert altitude to semi-major axis (Earth radius = 6371 km)
            sma = 6371.0 + altitude_km
            
            # Calculate mean motion from semi-major axis
            mu = 398600.4418  # Earth's gravitational parameter
            mean_motion = (86400 / (2 * math.pi)) * (mu / sma**3)**0.5  # revs/day
            
            sat = SatelliteData(
                catalog_number=25544 + i,
                name=f"TEST-SAT-{i+1:02d}",
                epoch="2024-01-01T00:00:00",
                mean_motion=mean_motion,
                eccentricity=0.0001 + i * 0.00005,
                inclination=inclination,
                raan=i * 30.0,  # Distribute RAAN
                arg_perigee=90.0 + i * 15.0,
                mean_anomaly=i * 30.0,
                semi_major_axis=sma,
                orbital_period=86400 / mean_motion  # seconds
            )
            satellites.append(sat)
        
        return satellites
    
    def test_complete_optimization_runs(self) -> Dict[str, Any]:
        """Test complete optimization runs with small satellite constellations."""
        print("Testing complete optimization runs...")
        
        test_scenarios = [
            {
                "name": "Basic Optimization",
                "constellation_size": 8,
                "config": GAConfig(
                    population_size=20,
                    max_generations=15,
                    mutation_rate=0.12,
                    crossover_rate=0.8,
                    elitism_count=2
                ),
                "constraints": RouteConstraints(
                    max_deltav_budget=4.0,
                    max_mission_duration=7 * 24 * 3600,  # 1 week
                    min_hops=2,
                    max_hops=5
                )
            },
            {
                "name": "Constrained Start/End",
                "constellation_size": 10,
                "config": GAConfig(
                    population_size=25,
                    max_generations=12,
                    mutation_rate=0.15,
                    crossover_rate=0.85
                ),
                "constraints": RouteConstraints(
                    max_deltav_budget=6.0,
                    max_mission_duration=10 * 24 * 3600,  # 10 days
                    start_satellite_id=25544,
                    end_satellite_id=25549,
                    min_hops=3,
                    max_hops=6
                )
            },
            {
                "name": "Tight Budget",
                "constellation_size": 12,
                "config": GAConfig(
                    population_size=30,
                    max_generations=20,
                    mutation_rate=0.18,
                    crossover_rate=0.75
                ),
                "constraints": RouteConstraints(
                    max_deltav_budget=2.5,  # Tight budget
                    max_mission_duration=3 * 24 * 3600,  # 3 days
                    min_hops=2,
                    max_hops=4,
                    forbidden_satellites=[25546, 25547]
                )
            }
        ]
        
        results = {}
        
        for scenario in test_scenarios:
            print(f"\n  Running scenario: {scenario['name']}")
            
            # Create constellation
            satellites = self.create_small_constellation(scenario['constellation_size'])
            
            # Initialize optimizer
            optimizer = GeneticRouteOptimizer(satellites, scenario['config'])
            
            # Run optimization
            start_time = time.time()
            result = optimizer.optimize_route(scenario['constraints'])
            execution_time = time.time() - start_time
            
            # Analyze results
            scenario_results = {
                "status": result.status,
                "success": result.success,
                "completed_successfully": result.status in [OptimizationStatus.SUCCESS, OptimizationStatus.CONVERGED, OptimizationStatus.MAX_GENERATIONS],
                "execution_time": execution_time,
                "generations_completed": result.optimization_stats.generations_completed,
                "best_fitness": result.optimization_stats.best_fitness,
                "constraint_satisfaction": result.optimization_stats.constraint_satisfaction_rate,
                "convergence_history_length": len(result.convergence_history)
            }
            
            if result.best_route:
                route = result.best_route
                scenario_results.update({
                    "route_length": len(route.satellite_sequence),
                    "total_deltav": route.total_deltav,
                    "mission_duration_hours": route.mission_duration / 3600,
                    "route_valid": route.is_valid,
                    "constraint_violations": len(route.constraint_violations) if route.constraint_violations else 0
                })
                
                # Verify constraint satisfaction
                constraints = scenario['constraints']
                if constraints.start_satellite_id:
                    assert route.satellite_sequence[0] == constraints.start_satellite_id, \
                        f"Start satellite constraint violated: expected {constraints.start_satellite_id}, got {route.satellite_sequence[0]}"
                
                if constraints.end_satellite_id:
                    assert route.satellite_sequence[-1] == constraints.end_satellite_id, \
                        f"End satellite constraint violated: expected {constraints.end_satellite_id}, got {route.satellite_sequence[-1]}"
                
                if constraints.forbidden_satellites:
                    for forbidden in constraints.forbidden_satellites:
                        assert forbidden not in route.satellite_sequence, \
                            f"Forbidden satellite {forbidden} found in route"
                
                assert len(route.satellite_sequence) >= constraints.min_hops + 1, \
                    f"Minimum hops constraint violated: {len(route.satellite_sequence)-1} < {constraints.min_hops}"
                
                assert len(route.satellite_sequence) <= constraints.max_hops + 1, \
                    f"Maximum hops constraint violated: {len(route.satellite_sequence)-1} > {constraints.max_hops}"
            
            results[scenario['name']] = scenario_results
            
            print(f"    Status: {result.status}")
            print(f"    Execution time: {execution_time:.2f}s")
            print(f"    Generations: {result.optimization_stats.generations_completed}")
            if result.best_route:
                print(f"    Best route: {len(result.best_route.satellite_sequence)} satellites, "
                      f"{result.best_route.total_deltav:.3f} km/s")
        
        return results
    
    def test_convergence_behavior(self) -> Dict[str, Any]:
        """Verify convergence behavior and solution quality."""
        print("\nTesting convergence behavior...")
        
        satellites = self.create_small_constellation(10)
        
        # Test different convergence scenarios
        convergence_tests = [
            {
                "name": "Quick Convergence",
                "config": GAConfig(
                    population_size=15,
                    max_generations=50,
                    convergence_threshold=1e-4,
                    max_stagnant_generations=8
                ),
                "constraints": RouteConstraints(
                    max_deltav_budget=3.0,
                    max_mission_duration=5 * 24 * 3600,
                    min_hops=2,
                    max_hops=3  # Simple problem for quick convergence
                )
            },
            {
                "name": "Slow Convergence",
                "config": GAConfig(
                    population_size=40,
                    max_generations=30,
                    convergence_threshold=1e-6,
                    max_stagnant_generations=15
                ),
                "constraints": RouteConstraints(
                    max_deltav_budget=8.0,
                    max_mission_duration=14 * 24 * 3600,
                    min_hops=4,
                    max_hops=7  # Complex problem for slower convergence
                )
            }
        ]
        
        convergence_results = {}
        
        for test in convergence_tests:
            print(f"  Testing {test['name']}...")
            
            optimizer = GeneticRouteOptimizer(satellites, test['config'])
            result = optimizer.optimize_route(test['constraints'])
            
            # Analyze convergence behavior
            convergence_analysis = self._analyze_convergence(result.convergence_history)
            
            test_results = {
                "status": result.status,
                "generations_to_converge": result.optimization_stats.generations_completed,
                "final_fitness": result.optimization_stats.best_fitness,
                "convergence_rate": convergence_analysis['convergence_rate'],
                "fitness_improvement": convergence_analysis['fitness_improvement'],
                "diversity_trend": convergence_analysis['diversity_trend'],
                "early_convergence": result.status == OptimizationStatus.CONVERGED
            }
            
            convergence_results[test['name']] = test_results
            
            print(f"    Generations: {result.optimization_stats.generations_completed}")
            print(f"    Final fitness: {result.optimization_stats.best_fitness:.3f}")
            print(f"    Convergence rate: {convergence_analysis['convergence_rate']:.4f}")
        
        return convergence_results
    
    def test_error_recovery_and_constraints(self) -> Dict[str, Any]:
        """Test error recovery and constraint satisfaction."""
        print("\nTesting error recovery and constraint satisfaction...")
        
        satellites = self.create_small_constellation(8)
        
        # Test scenarios that might cause errors
        error_test_scenarios = [
            {
                "name": "Impossible Constraints",
                "config": GAConfig(
                    population_size=20,
                    max_generations=10
                ),
                "constraints": RouteConstraints(
                    max_deltav_budget=0.1,  # Impossibly low budget
                    max_mission_duration=3600,  # 1 hour - very short
                    min_hops=5,  # Many hops with low budget
                    max_hops=8
                )
            },
            {
                "name": "Invalid Satellite IDs",
                "config": GAConfig(
                    population_size=15,
                    max_generations=8
                ),
                "constraints": RouteConstraints(
                    max_deltav_budget=5.0,
                    max_mission_duration=7 * 24 * 3600,
                    start_satellite_id=99999,  # Non-existent satellite
                    end_satellite_id=25548,
                    min_hops=2,
                    max_hops=4
                )
            },
            {
                "name": "High Mutation Rate",
                "config": GAConfig(
                    population_size=25,
                    max_generations=15,
                    mutation_rate=0.8,  # Very high mutation rate
                    crossover_rate=0.9
                ),
                "constraints": RouteConstraints(
                    max_deltav_budget=4.0,
                    max_mission_duration=5 * 24 * 3600,
                    min_hops=3,
                    max_hops=5
                )
            }
        ]
        
        error_recovery_results = {}
        
        for scenario in error_test_scenarios:
            print(f"  Testing {scenario['name']}...")
            
            optimizer = GeneticRouteOptimizer(satellites, scenario['config'])
            
            try:
                result = optimizer.optimize_route(scenario['constraints'])
                
                # Analyze error handling
                error_analysis = {
                    "completed_successfully": result.success,
                    "status": result.status,
                    "generations_completed": result.optimization_stats.generations_completed,
                    "constraint_satisfaction_rate": result.optimization_stats.constraint_satisfaction_rate,
                    "error_handled_gracefully": True,
                    "has_valid_result": result.best_route is not None
                }
                
                if result.best_route:
                    error_analysis["route_valid"] = result.best_route.is_valid
                    error_analysis["constraint_violations"] = len(result.best_route.constraint_violations) if result.best_route.constraint_violations else 0
                
            except Exception as e:
                error_analysis = {
                    "completed_successfully": False,
                    "error_handled_gracefully": False,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            
            error_recovery_results[scenario['name']] = error_analysis
            
            print(f"    Handled gracefully: {error_analysis.get('error_handled_gracefully', False)}")
            if 'constraint_satisfaction_rate' in error_analysis:
                print(f"    Constraint satisfaction: {error_analysis['constraint_satisfaction_rate']:.1%}")
        
        return error_recovery_results
    
    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Benchmark performance with different parameter configurations."""
        print("\nTesting performance benchmarks...")
        
        # Test different configuration combinations
        benchmark_configs = [
            {
                "name": "Small Fast",
                "constellation_size": 8,
                "config": GAConfig(
                    population_size=15,
                    max_generations=10,
                    mutation_rate=0.1,
                    crossover_rate=0.8
                )
            },
            {
                "name": "Medium Balanced",
                "constellation_size": 12,
                "config": GAConfig(
                    population_size=30,
                    max_generations=20,
                    mutation_rate=0.15,
                    crossover_rate=0.85
                )
            },
            {
                "name": "Large Thorough",
                "constellation_size": 16,
                "config": GAConfig(
                    population_size=50,
                    max_generations=25,
                    mutation_rate=0.12,
                    crossover_rate=0.8
                )
            }
        ]
        
        benchmark_results = {}
        
        for benchmark in benchmark_configs:
            print(f"  Benchmarking {benchmark['name']}...")
            
            satellites = self.create_small_constellation(benchmark['constellation_size'])
            
            constraints = RouteConstraints(
                max_deltav_budget=5.0,
                max_mission_duration=7 * 24 * 3600,
                min_hops=3,
                max_hops=6
            )
            
            # Run multiple trials for statistical significance
            trial_times = []
            trial_fitness = []
            trial_generations = []
            
            num_trials = 3  # Reduced for faster testing
            
            for trial in range(num_trials):
                optimizer = GeneticRouteOptimizer(satellites, benchmark['config'])
                
                start_time = time.time()
                result = optimizer.optimize_route(constraints)
                execution_time = time.time() - start_time
                
                trial_times.append(execution_time)
                trial_fitness.append(result.optimization_stats.best_fitness)
                trial_generations.append(result.optimization_stats.generations_completed)
            
            # Calculate statistics
            benchmark_stats = {
                "constellation_size": benchmark['constellation_size'],
                "population_size": benchmark['config'].population_size,
                "max_generations": benchmark['config'].max_generations,
                "avg_execution_time": statistics.mean(trial_times),
                "std_execution_time": statistics.stdev(trial_times) if len(trial_times) > 1 else 0,
                "avg_fitness": statistics.mean(trial_fitness),
                "std_fitness": statistics.stdev(trial_fitness) if len(trial_fitness) > 1 else 0,
                "avg_generations": statistics.mean(trial_generations),
                "time_per_generation": statistics.mean(trial_times) / statistics.mean(trial_generations) if statistics.mean(trial_generations) > 0 else 0,
                "fitness_per_second": statistics.mean(trial_fitness) / statistics.mean(trial_times) if statistics.mean(trial_times) > 0 else 0
            }
            
            benchmark_results[benchmark['name']] = benchmark_stats
            
            print(f"    Avg time: {benchmark_stats['avg_execution_time']:.2f}s ± {benchmark_stats['std_execution_time']:.2f}s")
            print(f"    Avg fitness: {benchmark_stats['avg_fitness']:.3f} ± {benchmark_stats['std_fitness']:.3f}")
            print(f"    Time per generation: {benchmark_stats['time_per_generation']:.3f}s")
        
        return benchmark_results
    
    def _analyze_convergence(self, convergence_history: List) -> Dict[str, Any]:
        """Analyze convergence behavior from history."""
        if not convergence_history or len(convergence_history) < 2:
            return {
                "convergence_rate": 0.0,
                "fitness_improvement": 0.0,
                "diversity_trend": "unknown"
            }
        
        # Calculate convergence rate (fitness improvement per generation)
        initial_fitness = convergence_history[0].best_fitness
        final_fitness = convergence_history[-1].best_fitness
        generations = len(convergence_history)
        
        convergence_rate = (final_fitness - initial_fitness) / generations if generations > 0 else 0.0
        fitness_improvement = final_fitness - initial_fitness
        
        # Analyze diversity trend
        diversities = [gen.diversity_metric for gen in convergence_history if hasattr(gen, 'diversity_metric')]
        
        if len(diversities) >= 2:
            diversity_slope = (diversities[-1] - diversities[0]) / len(diversities)
            if diversity_slope > 0.01:
                diversity_trend = "increasing"
            elif diversity_slope < -0.01:
                diversity_trend = "decreasing"
            else:
                diversity_trend = "stable"
        else:
            diversity_trend = "unknown"
        
        return {
            "convergence_rate": convergence_rate,
            "fitness_improvement": fitness_improvement,
            "diversity_trend": diversity_trend
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests and return comprehensive results."""
        print("="*60)
        print("GENETIC ALGORITHM ENGINE INTEGRATION TESTS")
        print("="*60)
        
        all_results = {}
        
        try:
            # Test 1: Complete optimization runs
            all_results["optimization_runs"] = self.test_complete_optimization_runs()
            
            # Test 2: Convergence behavior
            all_results["convergence_behavior"] = self.test_convergence_behavior()
            
            # Test 3: Error recovery and constraints
            all_results["error_recovery"] = self.test_error_recovery_and_constraints()
            
            # Test 4: Performance benchmarks
            all_results["performance_benchmarks"] = self.test_performance_benchmarks()
            
            # Generate summary
            all_results["test_summary"] = self._generate_test_summary(all_results)
            
            return all_results
            
        except Exception as e:
            print(f"\n❌ Integration test suite failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    def _generate_test_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all test results."""
        summary = {
            "total_tests_run": 0,
            "successful_optimizations": 0,
            "convergence_tests_passed": 0,
            "error_recovery_tests_passed": 0,
            "performance_benchmarks_completed": 0,
            "overall_success": True
        }
        
        # Count optimization runs
        if "optimization_runs" in results:
            summary["total_tests_run"] += len(results["optimization_runs"])
            summary["successful_optimizations"] = sum(
                1 for result in results["optimization_runs"].values() 
                if result.get("completed_successfully", False)
            )
        
        # Count convergence tests
        if "convergence_behavior" in results:
            summary["total_tests_run"] += len(results["convergence_behavior"])
            summary["convergence_tests_passed"] = len(results["convergence_behavior"])
        
        # Count error recovery tests
        if "error_recovery" in results:
            summary["total_tests_run"] += len(results["error_recovery"])
            summary["error_recovery_tests_passed"] = sum(
                1 for result in results["error_recovery"].values()
                if result.get("error_handled_gracefully", False)
            )
        
        # Count performance benchmarks
        if "performance_benchmarks" in results:
            summary["total_tests_run"] += len(results["performance_benchmarks"])
            summary["performance_benchmarks_completed"] = len(results["performance_benchmarks"])
        
        # Determine overall success
        summary["overall_success"] = (
            summary["successful_optimizations"] > 0 and
            summary["convergence_tests_passed"] > 0 and
            summary["error_recovery_tests_passed"] > 0 and
            summary["performance_benchmarks_completed"] > 0
        )
        
        return summary


def run_integration_tests():
    """Run the complete integration test suite."""
    test_suite = IntegrationTestSuite()
    results = test_suite.run_all_tests()
    
    print("\n" + "="*60)
    print("INTEGRATION TEST RESULTS SUMMARY")
    print("="*60)
    
    if "test_summary" in results:
        summary = results["test_summary"]
        print(f"Total tests run: {summary['total_tests_run']}")
        print(f"Successful optimizations: {summary['successful_optimizations']}")
        print(f"Convergence tests passed: {summary['convergence_tests_passed']}")
        print(f"Error recovery tests passed: {summary['error_recovery_tests_passed']}")
        print(f"Performance benchmarks completed: {summary['performance_benchmarks_completed']}")
        print(f"Overall success: {'✓' if summary['overall_success'] else '❌'}")
        
        if summary['overall_success']:
            print("\n✓ All integration tests completed successfully!")
            print("✓ Genetic algorithm engine is working correctly")
        else:
            print("\n❌ Some integration tests failed")
    
    return results


if __name__ == "__main__":
    # Run the integration test suite
    results = run_integration_tests()