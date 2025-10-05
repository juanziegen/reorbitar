"""
Genetic Route Optimizer Module

This module implements the main genetic algorithm engine for satellite route optimization.
It orchestrates population management, genetic operations, fitness evaluation, and convergence detection.
"""

import time
import random
import math
import logging
from typing import List, Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass

from src.genetic_algorithm import (
    RouteChromosome, 
    GAConfig, 
    RouteConstraints, 
    OptimizationResult,
    OptimizationStats,
    GenerationStats,
    OptimizationStatus
)
from src.route_fitness_evaluator import RouteFitnessEvaluator
from src.genetic_operators import CrossoverOperator, MutationOperator, SelectionOperator
from src.chromosome_initializer import ChromosomeInitializer, InitializationConfig
from src.orbital_propagator import OrbitalPropagator
from src.tle_parser import SatelliteData
from src.ga_error_handler import GAErrorHandler, ErrorType, ErrorSeverity


class GeneticRouteOptimizer:
    """
    Main genetic algorithm engine for satellite route optimization.
    
    This class orchestrates the entire genetic algorithm process, including
    population initialization, fitness evaluation, genetic operations,
    and convergence detection.
    """
    
    def __init__(self, satellites: List[SatelliteData], config: GAConfig = None):
        """
        Initialize the genetic route optimizer.
        
        Args:
            satellites: List of satellite data for route optimization
            config: Genetic algorithm configuration parameters
            
        Raises:
            ValueError: If satellites list is empty or contains invalid data
        """
        if not satellites:
            raise ValueError("Satellites list cannot be empty")
        
        self.satellites = satellites
        self.config = config or GAConfig()
        
        # Initialize components
        self.orbital_propagator = OrbitalPropagator(satellites)
        self.fitness_evaluator = RouteFitnessEvaluator(satellites, self.orbital_propagator)
        self.chromosome_initializer = ChromosomeInitializer(satellites)
        
        # Initialize genetic operators
        self.crossover_operator = CrossoverOperator(self.config)
        self.mutation_operator = MutationOperator(self.config)
        self.selection_operator = SelectionOperator(self.config)
        
        # Initialize error handler
        self.error_handler = GAErrorHandler(satellites, self.config)
        
        # Optimization state
        self.current_generation = 0
        self.best_chromosome = None
        self.best_fitness = float('-inf')
        self.stagnant_generations = 0
        self.convergence_history = []
        
        # Adaptive parameters
        self.adaptive_mutation_rate = self.config.mutation_rate
        self.adaptive_crossover_rate = self.config.crossover_rate
        
        # Progress tracking
        self.progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.logger = logging.getLogger(__name__)
        self.detailed_logging = False
        
        # Statistics tracking
        self.fitness_history = []
        self.diversity_history = []
        self.convergence_metrics = {
            'improvement_rate': 0.0,
            'stagnation_periods': [],
            'diversity_trend': 'stable'
        }
    
    def optimize_route(self, constraints: RouteConstraints) -> OptimizationResult:
        """
        Run genetic algorithm to find optimal satellite route.
        
        This is the main method that orchestrates the entire optimization process,
        including population initialization, evolution, and convergence detection.
        
        Args:
            constraints: Route constraints and optimization parameters
            
        Returns:
            OptimizationResult with best route found and optimization statistics
        """
        start_time = time.time()
        self._start_time = start_time  # Store for progress tracking
        
        try:
            # Reset optimization state
            self._reset_optimization_state()
            
            # Initialize population with error handling
            population = self._initialize_population_with_recovery(constraints)
            if not population:
                return self.error_handler.handle_system_error(
                    ValueError("Failed to initialize population after recovery attempts"),
                    {'generation': 0, 'phase': 'initialization'}
                )
            
            # Evaluate initial population
            fitness_scores = self._evaluate_population_with_recovery(population, constraints, 0)
            self._update_best_solution(population, fitness_scores)
            
            # Record initial generation statistics
            self._record_generation_stats(population, fitness_scores, 0)
            self._log_generation_progress(0, fitness_scores, population)
            self._notify_progress_callback(0, fitness_scores, population)
            
            # Main evolution loop with comprehensive error handling
            for generation in range(1, self.config.max_generations + 1):
                self.current_generation = generation
                
                try:
                    # Create next generation with error recovery
                    new_population = self._create_next_generation_with_recovery(
                        population, fitness_scores, constraints, generation
                    )
                    
                    # Evaluate new population with error handling
                    new_fitness_scores = self._evaluate_population_with_recovery(
                        new_population, constraints, generation
                    )
                    
                    # Update best solution
                    improved = self._update_best_solution(new_population, new_fitness_scores)
                    
                    # Record generation statistics
                    self._record_generation_stats(new_population, new_fitness_scores, generation)
                    
                    # Log progress and notify callback
                    self._log_generation_progress(generation, new_fitness_scores, new_population)
                    self._notify_progress_callback(generation, new_fitness_scores, new_population)
                    
                    # Update convergence metrics
                    self._update_convergence_metrics(generation, new_fitness_scores, new_population)
                    
                    # Check for premature convergence and inject diversity if needed
                    if self._detect_premature_convergence(new_population, generation):
                        new_population = self.error_handler.handle_population_convergence(
                            new_population, constraints, generation
                        )
                        new_fitness_scores = self._evaluate_population_with_recovery(
                            new_population, constraints, generation
                        )
                    
                    # Check for convergence
                    if self._check_convergence(improved):
                        status = OptimizationStatus.CONVERGED
                        self._log_convergence(status, generation)
                        break
                    
                    # Update population for next iteration
                    population = new_population
                    fitness_scores = new_fitness_scores
                    
                    # Adapt parameters based on progress
                    self._adapt_parameters(population, fitness_scores)
                    
                except Exception as generation_error:
                    # Handle generation-level errors
                    self.logger.error(f"Error in generation {generation}: {generation_error}")
                    
                    # Attempt recovery
                    if self._attempt_generation_recovery(generation_error, generation):
                        continue
                    else:
                        # Critical error - abort optimization
                        return self.error_handler.handle_system_error(
                            generation_error,
                            {'generation': generation, 'phase': 'evolution'}
                        )
            
            else:
                # Reached maximum generations
                status = OptimizationStatus.MAX_GENERATIONS
            
            # Create final result with error summary
            execution_time = time.time() - start_time
            result = self._create_success_result(status, execution_time)
            
            # Add error summary to result
            error_summary = self.error_handler.get_error_summary()
            if hasattr(result, 'additional_info'):
                result.additional_info = error_summary
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            return self.error_handler.handle_system_error(
                e, {'generation': self.current_generation, 'phase': 'optimization'}
            )
    
    def _reset_optimization_state(self):
        """Reset optimization state for new run."""
        self.current_generation = 0
        self.best_chromosome = None
        self.best_fitness = float('-inf')
        self.stagnant_generations = 0
        self.convergence_history = []
        self.adaptive_mutation_rate = self.config.mutation_rate
        self.adaptive_crossover_rate = self.config.crossover_rate
        
        # Reset progress tracking
        self.fitness_history = []
        self.diversity_history = []
        self.convergence_metrics = {
            'improvement_rate': 0.0,
            'stagnation_periods': [],
            'diversity_trend': 'stable'
        }
        
        # Clear fitness evaluator cache
        self.fitness_evaluator.clear_cache()
        
        # Log optimization start
        if self.detailed_logging:
            self.logger.info("Starting genetic algorithm optimization")
            self.logger.info(f"Population size: {self.config.population_size}")
            self.logger.info(f"Max generations: {self.config.max_generations}")
            self.logger.info(f"Mutation rate: {self.config.mutation_rate}")
            self.logger.info(f"Crossover rate: {self.config.crossover_rate}")
    
    def _initialize_population_with_recovery(self, constraints: RouteConstraints) -> List[RouteChromosome]:
        """Initialize population using mixed strategies with error recovery."""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                population = self.chromosome_initializer.initialize_population(
                    self.config.population_size, constraints
                )
                
                # Validate and repair population
                valid_population = []
                for chromosome in population:
                    if self._is_chromosome_valid_basic(chromosome):
                        valid_population.append(chromosome)
                    else:
                        # Attempt to repair invalid chromosome
                        repaired = self.error_handler.repair_invalid_chromosome(
                            chromosome, constraints, 0
                        )
                        if self._is_chromosome_valid_basic(repaired):
                            valid_population.append(repaired)
                
                # Check if we have enough valid chromosomes
                min_required = max(2, self.config.population_size // 4)  # At least 25%
                
                if len(valid_population) >= min_required:
                    # Pad population if needed using fallback chromosomes
                    while len(valid_population) < self.config.population_size:
                        fallback = self.error_handler._create_fallback_chromosome(constraints, 0)
                        valid_population.append(fallback)
                    
                    return valid_population[:self.config.population_size]
                else:
                    raise ValueError(f"Only {len(valid_population)} valid chromosomes created, need at least {min_required}")
                    
            except Exception as e:
                self.logger.warning(f"Population initialization attempt {attempt + 1} failed: {e}")
                
                if attempt == max_attempts - 1:
                    # Final attempt - create minimal population
                    return self._create_minimal_population(constraints)
        
        return None
    
    def _create_minimal_population(self, constraints: RouteConstraints) -> List[RouteChromosome]:
        """Create minimal valid population as last resort."""
        population = []
        
        try:
            # Create simple chromosomes with different characteristics
            available_satellites = [sat.catalog_number for sat in self.satellites]
            
            for i in range(self.config.population_size):
                # Create chromosome with 2-4 satellites
                route_length = random.randint(2, min(4, len(available_satellites)))
                sequence = random.sample(available_satellites, route_length)
                
                # Simple timing - 1 hour intervals
                times = [i * 3600.0 for i in range(route_length)]
                
                chromosome = RouteChromosome(
                    satellite_sequence=sequence,
                    departure_times=times
                )
                
                population.append(chromosome)
            
            return population
            
        except Exception as e:
            self.logger.error(f"Failed to create minimal population: {e}")
            return []
    
    def _evaluate_population_with_recovery(self, population: List[RouteChromosome], 
                                          constraints: RouteConstraints,
                                          generation: int) -> List[float]:
        """Evaluate fitness for entire population with comprehensive error handling."""
        fitness_scores = []
        failed_evaluations = 0
        
        for i, chromosome in enumerate(population):
            try:
                fitness_result = self.fitness_evaluator.evaluate_route(chromosome, constraints)
                fitness_scores.append(fitness_result.fitness_score)
                
                # Update chromosome with evaluation results
                chromosome.total_deltav = fitness_result.total_deltav
                chromosome.is_valid = fitness_result.is_valid
                chromosome.constraint_violations = fitness_result.constraint_violations
                
            except Exception as e:
                failed_evaluations += 1
                
                # Use error handler for graceful failure handling
                penalty_fitness = self.error_handler.handle_fitness_evaluation_error(
                    chromosome, e, generation
                )
                fitness_scores.append(penalty_fitness)
                
                # If too many evaluations fail, attempt chromosome repair
                if failed_evaluations > len(population) * 0.5:  # More than 50% failed
                    self.logger.warning(f"High fitness evaluation failure rate: {failed_evaluations}/{len(population)}")
                    
                    # Attempt to repair the chromosome
                    repaired_chromosome = self.error_handler.repair_invalid_chromosome(
                        chromosome, constraints, generation
                    )
                    
                    # Replace the failed chromosome
                    population[i] = repaired_chromosome
                    
                    # Try evaluating the repaired chromosome
                    try:
                        fitness_result = self.fitness_evaluator.evaluate_route(repaired_chromosome, constraints)
                        fitness_scores[-1] = fitness_result.fitness_score
                        repaired_chromosome.total_deltav = fitness_result.total_deltav
                        repaired_chromosome.is_valid = fitness_result.is_valid
                        repaired_chromosome.constraint_violations = fitness_result.constraint_violations
                    except Exception as repair_error:
                        self.logger.error(f"Repaired chromosome also failed evaluation: {repair_error}")
        
        # Log evaluation statistics
        if failed_evaluations > 0:
            failure_rate = (failed_evaluations / len(population)) * 100
            self.logger.info(f"Generation {generation}: {failed_evaluations}/{len(population)} "
                           f"fitness evaluations failed ({failure_rate:.1f}%)")
        
        return fitness_scores
    
    def _create_next_generation_with_recovery(self, population: List[RouteChromosome],
                                            fitness_scores: List[float],
                                            constraints: RouteConstraints,
                                            generation: int) -> List[RouteChromosome]:
        """Create next generation through selection, crossover, and mutation with error recovery."""
        next_generation = []
        operation_failures = 0
        
        try:
            # Preserve elite chromosomes
            elites = self.selection_operator.elitism_selection(population, fitness_scores)
            next_generation.extend(elites)
            
            # Fill remaining slots through reproduction
            while len(next_generation) < self.config.population_size:
                try:
                    # Select parents with error handling
                    parents = self._select_parents_with_recovery(population, fitness_scores)
                    
                    if len(parents) < 2:
                        # Fallback: create new chromosome
                        fallback = self.error_handler._create_fallback_chromosome(constraints, generation)
                        next_generation.append(fallback)
                        continue
                    
                    # Apply crossover with error handling
                    offspring1, offspring2 = self._apply_crossover_with_recovery(
                        parents[0], parents[1], constraints, generation
                    )
                    
                    # Apply mutation with error handling
                    population_diversity = self.selection_operator.calculate_population_diversity(population)
                    
                    offspring1 = self._apply_mutation_with_recovery(
                        offspring1, constraints, population_diversity, generation
                    )
                    offspring2 = self._apply_mutation_with_recovery(
                        offspring2, constraints, population_diversity, generation
                    )
                    
                    # Validate offspring before adding
                    if self._is_chromosome_valid_basic(offspring1):
                        if len(next_generation) < self.config.population_size:
                            next_generation.append(offspring1)
                    else:
                        # Repair invalid offspring
                        repaired1 = self.error_handler.repair_invalid_chromosome(
                            offspring1, constraints, generation
                        )
                        if len(next_generation) < self.config.population_size:
                            next_generation.append(repaired1)
                    
                    if self._is_chromosome_valid_basic(offspring2):
                        if len(next_generation) < self.config.population_size:
                            next_generation.append(offspring2)
                    else:
                        # Repair invalid offspring
                        repaired2 = self.error_handler.repair_invalid_chromosome(
                            offspring2, constraints, generation
                        )
                        if len(next_generation) < self.config.population_size:
                            next_generation.append(repaired2)
                    
                except Exception as operation_error:
                    operation_failures += 1
                    self.logger.warning(f"Genetic operation failed: {operation_error}")
                    
                    # Add fallback chromosome
                    fallback = self.error_handler._create_fallback_chromosome(constraints, generation)
                    if len(next_generation) < self.config.population_size:
                        next_generation.append(fallback)
                    
                    # If too many operations fail, break and fill with fallbacks
                    if operation_failures > self.config.population_size // 4:
                        break
            
            # Fill any remaining slots with fallback chromosomes
            while len(next_generation) < self.config.population_size:
                fallback = self.error_handler._create_fallback_chromosome(constraints, generation)
                next_generation.append(fallback)
            
            # Log operation statistics
            if operation_failures > 0:
                failure_rate = (operation_failures / self.config.population_size) * 100
                self.logger.info(f"Generation {generation}: {operation_failures} genetic operations failed "
                               f"({failure_rate:.1f}% failure rate)")
            
            return next_generation[:self.config.population_size]
            
        except Exception as e:
            self.logger.error(f"Critical error in generation creation: {e}")
            # Emergency fallback - return previous population with some modifications
            return self._create_emergency_population(population, constraints, generation)
    
    def _select_parents_with_recovery(self, population: List[RouteChromosome], 
                                    fitness_scores: List[float]) -> List[RouteChromosome]:
        """Select parents with error recovery."""
        try:
            return self.selection_operator.tournament_selection(population, fitness_scores, 2)
        except Exception as e:
            self.logger.warning(f"Parent selection failed: {e}")
            # Fallback: random selection
            if len(population) >= 2:
                return random.sample(population, 2)
            elif len(population) == 1:
                return [population[0], population[0]]
            else:
                return []
    
    def _apply_crossover_with_recovery(self, parent1: RouteChromosome, parent2: RouteChromosome,
                                     constraints: RouteConstraints, generation: int) -> Tuple[RouteChromosome, RouteChromosome]:
        """Apply crossover with error recovery."""
        try:
            if random.random() < self.adaptive_crossover_rate:
                return self.crossover_operator.order_crossover(parent1, parent2, constraints)
            else:
                return parent1, parent2
        except Exception as e:
            self.logger.warning(f"Crossover operation failed: {e}")
            # Fallback: return parents unchanged
            return parent1, parent2
    
    def _apply_mutation_with_recovery(self, chromosome: RouteChromosome, constraints: RouteConstraints,
                                    population_diversity: float, generation: int) -> RouteChromosome:
        """Apply mutation with error recovery."""
        try:
            return self.mutation_operator.adaptive_mutation(chromosome, constraints, population_diversity)
        except Exception as e:
            self.logger.warning(f"Mutation operation failed: {e}")
            # Fallback: return chromosome unchanged or create new one
            if self._is_chromosome_valid_basic(chromosome):
                return chromosome
            else:
                return self.error_handler._create_fallback_chromosome(constraints, generation)
    
    def _create_emergency_population(self, original_population: List[RouteChromosome],
                                   constraints: RouteConstraints, generation: int) -> List[RouteChromosome]:
        """Create emergency population when generation creation fails."""
        emergency_population = []
        
        # Keep valid chromosomes from original population
        for chromosome in original_population:
            if self._is_chromosome_valid_basic(chromosome):
                emergency_population.append(chromosome)
        
        # Fill remaining slots with fallback chromosomes
        while len(emergency_population) < self.config.population_size:
            fallback = self.error_handler._create_fallback_chromosome(constraints, generation)
            emergency_population.append(fallback)
        
        self.logger.warning(f"Emergency population created for generation {generation}")
        return emergency_population[:self.config.population_size]
    
    def _update_best_solution(self, population: List[RouteChromosome],
                            fitness_scores: List[float]) -> bool:
        """Update best solution found so far."""
        if not fitness_scores:
            return False
        
        current_best_fitness = max(fitness_scores)
        current_best_idx = fitness_scores.index(current_best_fitness)
        current_best_chromosome = population[current_best_idx]
        
        improved = False
        if current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_chromosome = current_best_chromosome
            self.stagnant_generations = 0
            improved = True
        else:
            self.stagnant_generations += 1
        
        return improved
    
    def _record_generation_stats(self, population: List[RouteChromosome],
                               fitness_scores: List[float], generation: int):
        """Record statistics for current generation."""
        if not fitness_scores:
            return
        
        # Calculate diversity
        diversity = self.selection_operator.calculate_population_diversity(population)
        
        # Count valid solutions
        valid_count = sum(1 for chrom in population if chrom.is_valid)
        constraint_satisfaction_rate = valid_count / len(population) if population else 0.0
        
        # Create generation statistics
        gen_stats = GenerationStats(
            generation=generation,
            best_fitness=max(fitness_scores),
            average_fitness=sum(fitness_scores) / len(fitness_scores),
            worst_fitness=min(fitness_scores),
            diversity_metric=diversity,
            valid_solutions_count=valid_count,
            constraint_satisfaction_rate=constraint_satisfaction_rate
        )
        
        self.convergence_history.append(gen_stats)
    
    def _check_convergence(self, improved: bool) -> bool:
        """Check if algorithm has converged."""
        # Check stagnation
        if self.stagnant_generations >= self.config.max_stagnant_generations:
            return True
        
        # Check fitness convergence
        if len(self.convergence_history) >= 10:
            recent_best = [stats.best_fitness for stats in self.convergence_history[-10:]]
            fitness_variance = self._calculate_variance(recent_best)
            
            if fitness_variance < self.config.convergence_threshold:
                return True
        
        return False
    
    def _adapt_parameters(self, population: List[RouteChromosome], fitness_scores: List[float]):
        """Adapt genetic algorithm parameters based on current progress."""
        if not fitness_scores:
            return
        
        # Calculate population diversity
        diversity = self.selection_operator.calculate_population_diversity(population)
        
        # Adapt mutation rate based on diversity and stagnation
        if diversity < 0.3 or self.stagnant_generations > 10:
            # Low diversity or stagnation - increase mutation
            self.adaptive_mutation_rate = min(0.3, self.config.mutation_rate * 1.5)
        elif diversity > 0.7:
            # High diversity - decrease mutation
            self.adaptive_mutation_rate = max(0.05, self.config.mutation_rate * 0.8)
        else:
            # Normal diversity - use base rate
            self.adaptive_mutation_rate = self.config.mutation_rate
        
        # Adapt crossover rate based on improvement rate
        if self.stagnant_generations > 5:
            # Increase exploration through more crossover
            self.adaptive_crossover_rate = min(0.9, self.config.crossover_rate * 1.2)
        else:
            # Use base crossover rate
            self.adaptive_crossover_rate = self.config.crossover_rate
        
        # Update operator parameters
        self.mutation_operator.base_mutation_rate = self.adaptive_mutation_rate
    
    def _is_chromosome_valid_basic(self, chromosome: RouteChromosome) -> bool:
        """Perform basic validation of chromosome structure."""
        if not chromosome.satellite_sequence:
            return False
        
        if len(chromosome.satellite_sequence) != len(chromosome.departure_times):
            return False
        
        # Check for valid satellite IDs
        for sat_id in chromosome.satellite_sequence:
            if sat_id not in {sat.catalog_number for sat in self.satellites}:
                return False
        
        # Check departure times are ascending
        if len(chromosome.departure_times) > 1:
            for i in range(1, len(chromosome.departure_times)):
                if chromosome.departure_times[i] <= chromosome.departure_times[i-1]:
                    return False
        
        return True
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _create_success_result(self, status: OptimizationStatus, execution_time: float) -> OptimizationResult:
        """Create successful optimization result."""
        # Calculate final statistics
        final_stats = OptimizationStats(
            generations_completed=self.current_generation,
            best_fitness=self.best_fitness,
            average_fitness=self.convergence_history[-1].average_fitness if self.convergence_history else 0.0,
            population_diversity=self.convergence_history[-1].diversity_metric if self.convergence_history else 0.0,
            constraint_satisfaction_rate=self.convergence_history[-1].constraint_satisfaction_rate if self.convergence_history else 0.0,
            convergence_generation=self.current_generation if status == OptimizationStatus.CONVERGED else None,
            stagnant_generations=self.stagnant_generations
        )
        
        return OptimizationResult(
            best_route=self.best_chromosome,
            optimization_stats=final_stats,
            convergence_history=self.convergence_history.copy(),
            execution_time=execution_time,
            status=status,
            error_message=None
        )
    
    def _create_failed_result(self, error_message: str, execution_time: float) -> OptimizationResult:
        """Create failed optimization result."""
        final_stats = OptimizationStats(
            generations_completed=self.current_generation,
            best_fitness=self.best_fitness,
            average_fitness=0.0,
            population_diversity=0.0,
            constraint_satisfaction_rate=0.0,
            stagnant_generations=self.stagnant_generations
        )
        
        return OptimizationResult(
            best_route=self.best_chromosome,
            optimization_stats=final_stats,
            convergence_history=self.convergence_history.copy(),
            execution_time=execution_time,
            status=OptimizationStatus.FAILED,
            error_message=error_message
        )
    
    def get_optimization_progress(self) -> Dict[str, Any]:
        """Get current optimization progress information."""
        if not self.convergence_history:
            return {
                'current_generation': 0,
                'best_fitness': 0.0,
                'stagnant_generations': 0,
                'population_diversity': 0.0
            }
        
        latest_stats = self.convergence_history[-1]
        
        return {
            'current_generation': self.current_generation,
            'best_fitness': self.best_fitness,
            'average_fitness': latest_stats.average_fitness,
            'stagnant_generations': self.stagnant_generations,
            'population_diversity': latest_stats.diversity_metric,
            'constraint_satisfaction_rate': latest_stats.constraint_satisfaction_rate,
            'adaptive_mutation_rate': self.adaptive_mutation_rate,
            'adaptive_crossover_rate': self.adaptive_crossover_rate
        }
    
    def set_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Set callback function for real-time progress updates.
        
        Args:
            callback: Function that receives progress information dictionary
        """
        self.progress_callback = callback
    
    def enable_detailed_logging(self, enable: bool = True):
        """
        Enable or disable detailed generation-by-generation logging.
        
        Args:
            enable: Whether to enable detailed logging
        """
        self.detailed_logging = enable
        if enable:
            logging.basicConfig(level=logging.INFO)
    
    def _log_generation_progress(self, generation: int, fitness_scores: List[float], 
                               population: List[RouteChromosome]):
        """Log detailed progress information for current generation."""
        if not self.detailed_logging or not fitness_scores:
            return
        
        best_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        worst_fitness = min(fitness_scores)
        
        diversity = self.selection_operator.calculate_population_diversity(population)
        valid_count = sum(1 for chrom in population if chrom.is_valid)
        
        self.logger.info(f"Generation {generation:4d}: "
                        f"Best={best_fitness:8.3f}, "
                        f"Avg={avg_fitness:8.3f}, "
                        f"Worst={worst_fitness:8.3f}, "
                        f"Diversity={diversity:6.3f}, "
                        f"Valid={valid_count:3d}/{len(population):3d}")
        
        # Log additional details every 10 generations
        if generation % 10 == 0:
            self.logger.info(f"  Stagnant generations: {self.stagnant_generations}")
            self.logger.info(f"  Adaptive mutation rate: {self.adaptive_mutation_rate:.4f}")
            self.logger.info(f"  Adaptive crossover rate: {self.adaptive_crossover_rate:.4f}")
            
            if self.best_chromosome:
                self.logger.info(f"  Best route hops: {self.best_chromosome.hop_count}")
                self.logger.info(f"  Best route delta-v: {self.best_chromosome.total_deltav:.3f} km/s")
    
    def _notify_progress_callback(self, generation: int, fitness_scores: List[float],
                                population: List[RouteChromosome]):
        """Notify progress callback with current optimization status."""
        if not self.progress_callback or not fitness_scores:
            return
        
        try:
            progress_info = self._compile_progress_info(generation, fitness_scores, population)
            self.progress_callback(progress_info)
        except Exception as e:
            self.logger.warning(f"Progress callback failed: {e}")
    
    def _compile_progress_info(self, generation: int, fitness_scores: List[float],
                             population: List[RouteChromosome]) -> Dict[str, Any]:
        """Compile comprehensive progress information."""
        if not fitness_scores:
            return {}
        
        diversity = self.selection_operator.calculate_population_diversity(population)
        valid_count = sum(1 for chrom in population if chrom.is_valid)
        
        # Calculate improvement rate over last 10 generations
        improvement_rate = 0.0
        if len(self.convergence_history) >= 10:
            recent_best = [stats.best_fitness for stats in self.convergence_history[-10:]]
            if len(recent_best) > 1:
                improvement_rate = (recent_best[-1] - recent_best[0]) / len(recent_best)
        
        progress_info = {
            'generation': generation,
            'max_generations': self.config.max_generations,
            'progress_percentage': (generation / self.config.max_generations) * 100,
            
            # Fitness statistics
            'best_fitness': max(fitness_scores),
            'average_fitness': sum(fitness_scores) / len(fitness_scores),
            'worst_fitness': min(fitness_scores),
            'fitness_std': self._calculate_standard_deviation(fitness_scores),
            
            # Population statistics
            'population_size': len(population),
            'population_diversity': diversity,
            'valid_solutions_count': valid_count,
            'constraint_satisfaction_rate': valid_count / len(population),
            
            # Convergence metrics
            'stagnant_generations': self.stagnant_generations,
            'improvement_rate': improvement_rate,
            'convergence_trend': self._analyze_convergence_trend(),
            
            # Adaptive parameters
            'adaptive_mutation_rate': self.adaptive_mutation_rate,
            'adaptive_crossover_rate': self.adaptive_crossover_rate,
            
            # Best solution info
            'best_route_hops': self.best_chromosome.hop_count if self.best_chromosome else 0,
            'best_route_deltav': self.best_chromosome.total_deltav if self.best_chromosome else 0.0,
            'best_route_valid': self.best_chromosome.is_valid if self.best_chromosome else False,
            
            # Time information
            'elapsed_time': time.time() - getattr(self, '_start_time', time.time()),
            'estimated_remaining_time': self._estimate_remaining_time(generation)
        }
        
        return progress_info
    
    def _update_convergence_metrics(self, generation: int, fitness_scores: List[float],
                                  population: List[RouteChromosome]):
        """Update detailed convergence metrics for analysis."""
        if not fitness_scores:
            return
        
        # Track fitness history
        best_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        self.fitness_history.append({
            'generation': generation,
            'best': best_fitness,
            'average': avg_fitness,
            'improvement': best_fitness > self.best_fitness
        })
        
        # Track diversity history
        diversity = self.selection_operator.calculate_population_diversity(population)
        self.diversity_history.append({
            'generation': generation,
            'diversity': diversity
        })
        
        # Update improvement rate
        if len(self.fitness_history) >= 10:
            recent_improvements = sum(1 for entry in self.fitness_history[-10:] if entry['improvement'])
            self.convergence_metrics['improvement_rate'] = recent_improvements / 10.0
        
        # Track stagnation periods
        if self.stagnant_generations == 0 and len(self.convergence_metrics['stagnation_periods']) > 0:
            # End of stagnation period
            self.convergence_metrics['stagnation_periods'][-1]['end'] = generation - 1
        elif self.stagnant_generations == 1:
            # Start of new stagnation period
            self.convergence_metrics['stagnation_periods'].append({
                'start': generation,
                'end': None
            })
        
        # Analyze diversity trend
        if len(self.diversity_history) >= 5:
            recent_diversity = [entry['diversity'] for entry in self.diversity_history[-5:]]
            diversity_slope = self._calculate_trend_slope(recent_diversity)
            
            if diversity_slope > 0.01:
                self.convergence_metrics['diversity_trend'] = 'increasing'
            elif diversity_slope < -0.01:
                self.convergence_metrics['diversity_trend'] = 'decreasing'
            else:
                self.convergence_metrics['diversity_trend'] = 'stable'
    
    def _analyze_convergence_trend(self) -> str:
        """Analyze current convergence trend."""
        if len(self.fitness_history) < 5:
            return 'insufficient_data'
        
        recent_best = [entry['best'] for entry in self.fitness_history[-5:]]
        trend_slope = self._calculate_trend_slope(recent_best)
        
        if trend_slope > 0.001:
            return 'improving'
        elif trend_slope < -0.001:
            return 'degrading'
        elif self.stagnant_generations > self.config.max_stagnant_generations // 2:
            return 'stagnating'
        else:
            return 'stable'
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope using simple linear regression."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_values = list(range(n))
        
        # Calculate means
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        # Calculate slope
        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_standard_deviation(self, values: List[float]) -> float:
        """Calculate standard deviation of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def _estimate_remaining_time(self, current_generation: int) -> float:
        """Estimate remaining optimization time based on current progress."""
        if not hasattr(self, '_start_time') or current_generation == 0:
            return 0.0
        
        elapsed_time = time.time() - self._start_time
        time_per_generation = elapsed_time / current_generation
        remaining_generations = self.config.max_generations - current_generation
        
        return time_per_generation * remaining_generations
    
    def _log_convergence(self, status: OptimizationStatus, generation: int):
        """Log convergence information."""
        if not self.detailed_logging:
            return
        
        self.logger.info(f"Optimization completed with status: {status.value}")
        self.logger.info(f"Converged at generation: {generation}")
        self.logger.info(f"Best fitness achieved: {self.best_fitness:.6f}")
        
        if self.best_chromosome:
            self.logger.info(f"Best route details:")
            self.logger.info(f"  Hops: {self.best_chromosome.hop_count}")
            self.logger.info(f"  Delta-v: {self.best_chromosome.total_deltav:.3f} km/s")
            self.logger.info(f"  Duration: {self.best_chromosome.mission_duration:.1f} seconds")
            self.logger.info(f"  Valid: {self.best_chromosome.is_valid}")
    
    def get_detailed_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization statistics and analysis.
        
        Returns:
            Dictionary containing detailed statistics and convergence analysis
        """
        if not self.convergence_history:
            return {}
        
        # Basic statistics
        all_best_fitness = [stats.best_fitness for stats in self.convergence_history]
        all_avg_fitness = [stats.average_fitness for stats in self.convergence_history]
        all_diversity = [stats.diversity_metric for stats in self.convergence_history]
        
        # Convergence analysis
        convergence_generation = None
        for i, stats in enumerate(self.convergence_history):
            if i > 0 and abs(stats.best_fitness - self.convergence_history[i-1].best_fitness) < self.config.convergence_threshold:
                convergence_generation = i
                break
        
        # Performance metrics
        peak_fitness = max(all_best_fitness) if all_best_fitness else 0.0
        final_fitness = all_best_fitness[-1] if all_best_fitness else 0.0
        fitness_improvement = final_fitness - all_best_fitness[0] if len(all_best_fitness) > 1 else 0.0
        
        # Diversity analysis
        avg_diversity = sum(all_diversity) / len(all_diversity) if all_diversity else 0.0
        min_diversity = min(all_diversity) if all_diversity else 0.0
        max_diversity = max(all_diversity) if all_diversity else 0.0
        
        return {
            'optimization_summary': {
                'total_generations': len(self.convergence_history),
                'convergence_generation': convergence_generation,
                'peak_fitness': peak_fitness,
                'final_fitness': final_fitness,
                'fitness_improvement': fitness_improvement,
                'improvement_percentage': (fitness_improvement / abs(all_best_fitness[0])) * 100 if all_best_fitness and all_best_fitness[0] != 0 else 0.0
            },
            
            'fitness_statistics': {
                'best_fitness_history': all_best_fitness,
                'average_fitness_history': all_avg_fitness,
                'fitness_variance': self._calculate_variance(all_best_fitness),
                'fitness_trend': self._analyze_fitness_trend(all_best_fitness)
            },
            
            'diversity_analysis': {
                'diversity_history': all_diversity,
                'average_diversity': avg_diversity,
                'min_diversity': min_diversity,
                'max_diversity': max_diversity,
                'diversity_trend': self.convergence_metrics['diversity_trend']
            },
            
            'convergence_metrics': self.convergence_metrics.copy(),
            
            'performance_metrics': {
                'stagnation_periods': len(self.convergence_metrics['stagnation_periods']),
                'longest_stagnation': self._get_longest_stagnation_period(),
                'improvement_rate': self.convergence_metrics['improvement_rate'],
                'convergence_efficiency': final_fitness / len(self.convergence_history) if self.convergence_history else 0.0
            },
            
            'population_statistics': {
                'final_constraint_satisfaction': self.convergence_history[-1].constraint_satisfaction_rate,
                'average_constraint_satisfaction': sum(stats.constraint_satisfaction_rate for stats in self.convergence_history) / len(self.convergence_history),
                'final_valid_solutions': self.convergence_history[-1].valid_solutions_count
            }
        }
    
    def _analyze_fitness_trend(self, fitness_values: List[float]) -> str:
        """Analyze overall fitness trend throughout optimization."""
        if len(fitness_values) < 3:
            return 'insufficient_data'
        
        # Calculate trend over entire optimization
        trend_slope = self._calculate_trend_slope(fitness_values)
        
        # Calculate improvement phases
        improvement_phases = 0
        for i in range(1, len(fitness_values)):
            if fitness_values[i] > fitness_values[i-1]:
                improvement_phases += 1
        
        improvement_ratio = improvement_phases / (len(fitness_values) - 1)
        
        if trend_slope > 0.01 and improvement_ratio > 0.3:
            return 'strong_improvement'
        elif trend_slope > 0.001 and improvement_ratio > 0.2:
            return 'moderate_improvement'
        elif trend_slope > -0.001 and improvement_ratio > 0.1:
            return 'slow_improvement'
        else:
            return 'minimal_improvement'
    
    def _get_longest_stagnation_period(self) -> int:
        """Get the length of the longest stagnation period."""
        if not self.convergence_metrics['stagnation_periods']:
            return 0
        
        longest = 0
        for period in self.convergence_metrics['stagnation_periods']:
            if period['end'] is not None:
                length = period['end'] - period['start'] + 1
                longest = max(longest, length)
            else:
                # Current ongoing stagnation
                length = self.current_generation - period['start'] + 1
                longest = max(longest, length)
        
        return longest
    
    def _detect_premature_convergence(self, population: List[RouteChromosome], generation: int) -> bool:
        """Detect if population has converged prematurely and needs diversity injection."""
        if generation < 10:  # Don't check too early
            return False
        
        # Calculate population diversity
        diversity = self.selection_operator.calculate_population_diversity(population)
        
        # Check if diversity is too low
        if diversity < 0.1:  # Very low diversity threshold
            return True
        
        # Check if fitness variance is too low
        if hasattr(self, 'convergence_history') and len(self.convergence_history) >= 5:
            recent_best = [stats.best_fitness for stats in self.convergence_history[-5:]]
            fitness_variance = self._calculate_variance(recent_best)
            
            if fitness_variance < 1e-6:  # Very low fitness variance
                return True
        
        # Check stagnation
        if self.stagnant_generations > self.config.max_stagnant_generations // 2:
            return True
        
        return False
    
    def _attempt_generation_recovery(self, error: Exception, generation: int) -> bool:
        """Attempt to recover from generation-level errors."""
        error_str = str(error).lower()
        
        # Check if this is a recoverable error
        recoverable_errors = ['memory', 'timeout', 'fitness', 'chromosome']
        
        if not any(err_type in error_str for err_type in recoverable_errors):
            return False  # Not recoverable
        
        # Limit recovery attempts
        if self.error_handler.recovery_attempts >= self.error_handler.max_recovery_attempts:
            self.logger.error(f"Maximum recovery attempts ({self.error_handler.max_recovery_attempts}) exceeded")
            return False
        
        self.error_handler.recovery_attempts += 1
        
        try:
            # Clear caches to free memory
            if hasattr(self.fitness_evaluator, 'clear_cache'):
                self.fitness_evaluator.clear_cache()
            
            # Reduce population size temporarily if memory error
            if 'memory' in error_str:
                original_size = self.config.population_size
                self.config.population_size = max(10, original_size // 2)
                self.logger.info(f"Reduced population size from {original_size} to {self.config.population_size}")
            
            # Reduce complexity if timeout error
            if 'timeout' in error_str:
                # Reduce mutation rate to speed up operations
                self.adaptive_mutation_rate *= 0.5
                self.logger.info(f"Reduced mutation rate to {self.adaptive_mutation_rate}")
            
            return True
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery attempt failed: {recovery_error}")
            return False