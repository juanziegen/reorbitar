"""
Demonstration of Progress Tracking and Statistics Features

This script demonstrates the enhanced progress tracking, statistics calculation,
and convergence monitoring functionality of the genetic route optimizer.
"""

import sys
import os
sys.path.insert(0, 'src')

from genetic_route_optimizer import GeneticRouteOptimizer
from genetic_algorithm import GAConfig, RouteConstraints
from tle_parser import SatelliteData


def create_demo_satellites():
    """Create a constellation of test satellites for demonstration."""
    satellites = []
    for i in range(15):
        sat = SatelliteData(
            catalog_number=25544 + i,
            name=f"DEMO-SAT-{i:02d}",
            epoch="2024-01-01T00:00:00",
            mean_motion=15.5 + i * 0.05,  # Varied orbital periods
            eccentricity=0.001 + i * 0.0002,
            inclination=51.6 + (i % 5) * 3.0,  # Different inclination groups
            raan=45.0 + i * 24.0,
            arg_perigee=90.0 + i * 15.0,
            mean_anomaly=0.0 + i * 24.0,
            semi_major_axis=6800.0 + i * 30.0,  # Different altitudes
            orbital_period=90.0 + i * 1.5
        )
        satellites.append(sat)
    return satellites


def demo_progress_callback():
    """Demonstrate real-time progress tracking with callback."""
    print("=" * 70)
    print("DEMONSTRATION: Real-time Progress Tracking")
    print("=" * 70)
    
    satellites = create_demo_satellites()
    config = GAConfig(
        population_size=20,
        max_generations=15,
        mutation_rate=0.12,
        crossover_rate=0.85,
        elitism_count=3,
        max_stagnant_generations=8
    )
    
    constraints = RouteConstraints(
        max_deltav_budget=8.0,
        max_mission_duration=172800.0,  # 2 days
        min_hops=3,
        max_hops=6
    )
    
    optimizer = GeneticRouteOptimizer(satellites, config)
    
    # Set up progress callback
    def progress_callback(info):
        print(f"Gen {info['generation']:2d} | "
              f"Best: {info['best_fitness']:6.3f} | "
              f"Avg: {info['average_fitness']:6.3f} | "
              f"Diversity: {info['population_diversity']:5.3f} | "
              f"Stagnant: {info['stagnant_generations']:2d} | "
              f"Trend: {info['convergence_trend']:12s} | "
              f"Progress: {info['progress_percentage']:5.1f}%")
        
        # Show adaptive parameters every few generations
        if info['generation'] % 5 == 0 and info['generation'] > 0:
            print(f"     Adaptive Mutation: {info['adaptive_mutation_rate']:.4f}, "
                  f"Crossover: {info['adaptive_crossover_rate']:.4f}")
    
    optimizer.set_progress_callback(progress_callback)
    
    print("Starting optimization with real-time progress tracking...")
    print("Gen    | Best Fit | Avg Fit  | Diversity | Stagnant | Trend        | Progress")
    print("-" * 70)
    
    result = optimizer.optimize_route(constraints)
    
    print("-" * 70)
    print(f"Optimization completed with status: {result.status.value}")
    if result.best_route:
        print(f"Best route found: {result.best_route.hop_count} hops, "
              f"{result.best_route.total_deltav:.3f} km/s delta-v")
    
    return optimizer


def demo_detailed_statistics(optimizer):
    """Demonstrate detailed statistics compilation and analysis."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Detailed Statistics and Analysis")
    print("=" * 70)
    
    stats = optimizer.get_detailed_statistics()
    
    # Optimization Summary
    print("\n📊 OPTIMIZATION SUMMARY")
    print("-" * 30)
    summary = stats['optimization_summary']
    print(f"Total Generations:     {summary['total_generations']}")
    print(f"Convergence Generation: {summary.get('convergence_generation', 'N/A')}")
    print(f"Peak Fitness:          {summary['peak_fitness']:.6f}")
    print(f"Final Fitness:         {summary['final_fitness']:.6f}")
    print(f"Fitness Improvement:   {summary['fitness_improvement']:.6f}")
    print(f"Improvement %:         {summary['improvement_percentage']:.2f}%")
    
    # Fitness Statistics
    print("\n📈 FITNESS STATISTICS")
    print("-" * 30)
    fitness_stats = stats['fitness_statistics']
    print(f"Fitness Variance:      {fitness_stats['fitness_variance']:.6f}")
    print(f"Fitness Trend:         {fitness_stats['fitness_trend']}")
    print(f"History Length:        {len(fitness_stats['best_fitness_history'])}")
    
    # Diversity Analysis
    print("\n🔄 DIVERSITY ANALYSIS")
    print("-" * 30)
    diversity = stats['diversity_analysis']
    print(f"Average Diversity:     {diversity['average_diversity']:.6f}")
    print(f"Min Diversity:         {diversity['min_diversity']:.6f}")
    print(f"Max Diversity:         {diversity['max_diversity']:.6f}")
    print(f"Diversity Trend:       {diversity['diversity_trend']}")
    
    # Performance Metrics
    print("\n⚡ PERFORMANCE METRICS")
    print("-" * 30)
    performance = stats['performance_metrics']
    print(f"Stagnation Periods:    {performance['stagnation_periods']}")
    print(f"Longest Stagnation:    {performance['longest_stagnation']} generations")
    print(f"Improvement Rate:      {performance['improvement_rate']:.4f}")
    print(f"Convergence Efficiency: {performance['convergence_efficiency']:.6f}")
    
    # Population Statistics
    print("\n👥 POPULATION STATISTICS")
    print("-" * 30)
    population = stats['population_statistics']
    print(f"Final Constraint Satisfaction: {population['final_constraint_satisfaction']:.2%}")
    print(f"Avg Constraint Satisfaction:   {population['average_constraint_satisfaction']:.2%}")
    print(f"Final Valid Solutions:         {population['final_valid_solutions']}")
    
    return stats


def demo_convergence_analysis(optimizer, stats):
    """Demonstrate convergence analysis and trends."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Convergence Analysis")
    print("=" * 70)
    
    # Show fitness progression
    print("\n📊 FITNESS PROGRESSION")
    print("-" * 40)
    fitness_history = stats['fitness_statistics']['best_fitness_history']
    
    # Show every few generations to avoid clutter
    step = max(1, len(fitness_history) // 10)
    for i in range(0, len(fitness_history), step):
        gen = i
        fitness = fitness_history[i]
        improvement = ""
        if i > 0:
            prev_fitness = fitness_history[i-step] if i-step >= 0 else fitness_history[0]
            change = fitness - prev_fitness
            improvement = f" ({change:+.4f})"
        print(f"Generation {gen:2d}: {fitness:.6f}{improvement}")
    
    # Show diversity progression
    print("\n🔄 DIVERSITY PROGRESSION")
    print("-" * 40)
    diversity_history = stats['diversity_analysis']['diversity_history']
    
    for i in range(0, len(diversity_history), step):
        gen = i
        diversity = diversity_history[i]
        print(f"Generation {gen:2d}: {diversity:.6f}")
    
    # Convergence metrics
    print("\n📈 CONVERGENCE METRICS")
    print("-" * 40)
    metrics = optimizer.convergence_metrics
    print(f"Improvement Rate:      {metrics['improvement_rate']:.4f}")
    print(f"Diversity Trend:       {metrics['diversity_trend']}")
    print(f"Stagnation Periods:    {len(metrics['stagnation_periods'])}")
    
    if metrics['stagnation_periods']:
        print("\nStagnation Period Details:")
        for i, period in enumerate(metrics['stagnation_periods']):
            start = period['start']
            end = period.get('end', 'ongoing')
            print(f"  Period {i+1}: Generation {start} to {end}")


def main():
    """Run the complete progress tracking demonstration."""
    print("🚀 GENETIC ROUTE OPTIMIZER - PROGRESS TRACKING DEMONSTRATION")
    print("=" * 70)
    print("This demonstration shows the enhanced progress tracking and")
    print("statistics capabilities of the genetic route optimizer.")
    
    # Run optimization with progress tracking
    optimizer = demo_progress_callback()
    
    # Show detailed statistics
    stats = demo_detailed_statistics(optimizer)
    
    # Show convergence analysis
    demo_convergence_analysis(optimizer, stats)
    
    print("\n" + "=" * 70)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("The genetic route optimizer now includes:")
    print("• Real-time progress tracking with customizable callbacks")
    print("• Generation-by-generation detailed logging")
    print("• Comprehensive fitness and diversity statistics")
    print("• Advanced convergence monitoring and analysis")
    print("• Detailed optimization result compilation")
    print("• Adaptive parameter tracking")
    print("• Performance metrics and trend analysis")


if __name__ == "__main__":
    main()