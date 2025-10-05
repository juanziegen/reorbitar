"""
Genetic Algorithm Core Data Structures for Satellite Route Optimization

This module contains the core data structures used by the genetic algorithm
to optimize satellite hopping routes within delta-v and time constraints.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


@dataclass
class RouteChromosome:
    """
    Represents a satellite hopping route as a genetic algorithm chromosome.
    
    This is the fundamental unit of evolution in the genetic algorithm,
    representing a complete route through multiple satellites with timing information.
    """
    satellite_sequence: List[int]  # Catalog numbers of satellites in route
    departure_times: List[float]   # Departure time at each satellite (seconds from epoch)
    total_deltav: float = 0.0     # Cached fitness value (km/s)
    is_valid: bool = True         # Whether route satisfies constraints
    constraint_violations: List[str] = field(default_factory=list)  # Details of violations
    
    def __post_init__(self):
        """Validate chromosome data after initialization."""
        if len(self.satellite_sequence) != len(self.departure_times):
            self.is_valid = False
            self.constraint_violations.append(
                "Satellite sequence and departure times must have same length"
            )
    
    @property
    def hop_count(self) -> int:
        """Number of satellite hops in this route."""
        return max(0, len(self.satellite_sequence) - 1)
    
    @property
    def mission_duration(self) -> float:
        """Total mission duration in seconds."""
        if len(self.departure_times) < 2:
            return 0.0
        return self.departure_times[-1] - self.departure_times[0]


@dataclass
class GAConfig:
    """
    Genetic algorithm configuration parameters.
    
    Controls the behavior and performance characteristics of the genetic algorithm
    optimization process.
    """
    population_size: int = 100
    max_generations: int = 500
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism_count: int = 5
    tournament_size: int = 3
    convergence_threshold: float = 1e-6
    max_stagnant_generations: int = 50
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.population_size < 2:
            raise ValueError("Population size must be at least 2")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("Mutation rate must be between 0.0 and 1.0")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("Crossover rate must be between 0.0 and 1.0")
        if self.elitism_count >= self.population_size:
            raise ValueError("Elitism count must be less than population size")
        if self.tournament_size < 1:
            raise ValueError("Tournament size must be at least 1")


@dataclass
class RouteConstraints:
    """
    Constraints for route optimization.
    
    Defines the mission requirements and limitations that valid routes must satisfy.
    """
    max_deltav_budget: float          # Maximum total delta-v (km/s)
    max_mission_duration: float       # Maximum mission time (seconds)
    start_satellite_id: Optional[int] = None  # Fixed starting satellite (None = any)
    end_satellite_id: Optional[int] = None    # Fixed ending satellite (None = any)
    min_hops: int = 1                # Minimum number of hops
    max_hops: int = 50               # Maximum number of hops
    forbidden_satellites: List[int] = field(default_factory=list)  # Satellites to avoid
    
    def __post_init__(self):
        """Validate constraint parameters."""
        if self.max_deltav_budget <= 0:
            raise ValueError("Delta-v budget must be positive")
        if self.max_mission_duration <= 0:
            raise ValueError("Mission duration must be positive")
        if self.min_hops < 1:
            raise ValueError("Minimum hops must be at least 1")
        if self.max_hops < self.min_hops:
            raise ValueError("Maximum hops must be >= minimum hops")


class OptimizationStatus(Enum):
    """Status of optimization run."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CONVERGED = "converged"
    MAX_GENERATIONS = "max_generations"


@dataclass
class GenerationStats:
    """
    Statistics for a single generation.
    
    Tracks the performance and characteristics of the population
    during a specific generation of the genetic algorithm.
    """
    generation: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    diversity_metric: float
    valid_solutions_count: int
    constraint_satisfaction_rate: float
    
    @property
    def fitness_range(self) -> float:
        """Range between best and worst fitness."""
        return self.best_fitness - self.worst_fitness


@dataclass
class OptimizationStats:
    """
    Statistics from optimization run.
    
    Provides comprehensive metrics about the genetic algorithm's
    performance and convergence characteristics.
    """
    generations_completed: int
    best_fitness: float
    average_fitness: float
    population_diversity: float
    constraint_satisfaction_rate: float
    convergence_generation: Optional[int] = None
    stagnant_generations: int = 0
    
    @property
    def convergence_efficiency(self) -> float:
        """Efficiency metric: best fitness per generation."""
        if self.generations_completed == 0:
            return 0.0
        return self.best_fitness / self.generations_completed


@dataclass
class OptimizationResult:
    """
    Complete result of genetic algorithm optimization.
    
    Contains the best route found, comprehensive statistics,
    and detailed information about the optimization process.
    """
    best_route: Optional[RouteChromosome]
    optimization_stats: OptimizationStats
    convergence_history: List[GenerationStats]
    execution_time: float
    status: OptimizationStatus
    error_message: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Whether optimization completed successfully."""
        return self.status in [OptimizationStatus.SUCCESS, OptimizationStatus.CONVERGED]
    
    @property
    def total_hops(self) -> int:
        """Total number of hops in best route."""
        if self.best_route is None:
            return 0
        return self.best_route.hop_count
    
    @property
    def total_deltav(self) -> float:
        """Total delta-v of best route."""
        if self.best_route is None:
            return 0.0
        return self.best_route.total_deltav


@dataclass
class FitnessResult:
    """
    Result of fitness evaluation for a route chromosome.
    
    Contains detailed information about route quality and constraint satisfaction.
    """
    fitness_score: float
    total_deltav: float
    hop_count: int
    mission_duration: float
    constraint_violations: List[str]
    is_valid: bool
    
    @property
    def penalty_score(self) -> float:
        """Penalty applied for constraint violations."""
        return len(self.constraint_violations) * 1000.0  # Large penalty per violation


@dataclass
class ConstraintResult:
    """
    Result of constraint checking for a route.
    
    Provides detailed information about which constraints are satisfied
    and which are violated.
    """
    is_valid: bool
    violations: List[str]
    deltav_usage: float
    deltav_budget: float
    duration_usage: float
    duration_budget: float
    hop_count: int
    min_hops: int
    max_hops: int
    
    @property
    def deltav_utilization(self) -> float:
        """Percentage of delta-v budget used."""
        if self.deltav_budget == 0:
            return 0.0
        return (self.deltav_usage / self.deltav_budget) * 100.0
    
    @property
    def duration_utilization(self) -> float:
        """Percentage of time budget used."""
        if self.duration_budget == 0:
            return 0.0
        return (self.duration_usage / self.duration_budget) * 100.0