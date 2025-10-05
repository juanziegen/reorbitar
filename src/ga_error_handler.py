"""
Genetic Algorithm Error Handler

This module provides comprehensive error handling and recovery mechanisms
for the genetic algorithm route optimization system.
"""

import logging
import random
import traceback
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from src.genetic_algorithm import (
    RouteChromosome, 
    RouteConstraints, 
    GAConfig,
    OptimizationResult,
    OptimizationStatus,
    OptimizationStats
)
from src.tle_parser import SatelliteData


class ErrorSeverity(Enum):
    """Severity levels for genetic algorithm errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorType(Enum):
    """Types of errors that can occur during genetic algorithm execution."""
    FITNESS_EVALUATION = "fitness_evaluation"
    CHROMOSOME_INVALID = "chromosome_invalid"
    POPULATION_CONVERGENCE = "population_convergence"
    CONSTRAINT_VIOLATION = "constraint_violation"
    ORBITAL_CALCULATION = "orbital_calculation"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    TIMEOUT = "timeout"
    CONFIGURATION_ERROR = "configuration_error"


@dataclass
class ErrorReport:
    """Detailed error report for genetic algorithm failures."""
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    generation: int
    chromosome_id: Optional[str] = None
    stack_trace: Optional[str] = None
    recovery_action: Optional[str] = None
    additional_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}


class GAErrorHandler:
    """
    Comprehensive error handling and recovery system for genetic algorithms.
    
    This class provides graceful handling of various error conditions that can
    occur during genetic algorithm execution, including fitness evaluation failures,
    invalid chromosomes, population convergence issues, and system-level errors.
    """
    
    def __init__(self, satellites: List[SatelliteData], config: GAConfig):
        """
        Initialize error handler with satellite data and configuration.
        
        Args:
            satellites: List of satellite data for validation
            config: Genetic algorithm configuration
        """
        self.satellites = satellites
        self.config = config
        self.satellite_ids = {sat.catalog_number for sat in satellites}
        
        # Error tracking
        self.error_history: List[ErrorReport] = []
        self.recovery_attempts = 0
        self.max_recovery_attempts = 10
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        
        # Recovery strategies
        self.recovery_strategies = {
            ErrorType.FITNESS_EVALUATION: self._handle_fitness_evaluation_error,
            ErrorType.CHROMOSOME_INVALID: self._handle_invalid_chromosome,
            ErrorType.POPULATION_CONVERGENCE: self._handle_population_convergence,
            ErrorType.CONSTRAINT_VIOLATION: self._handle_constraint_violation,
            ErrorType.ORBITAL_CALCULATION: self._handle_orbital_calculation_error,
            ErrorType.MEMORY_EXHAUSTION: self._handle_memory_exhaustion,
            ErrorType.TIMEOUT: self._handle_timeout_error,
            ErrorType.CONFIGURATION_ERROR: self._handle_configuration_error
        }
    
    def handle_fitness_evaluation_error(self, chromosome: RouteChromosome, 
                                      error: Exception, generation: int) -> float:
        """
        Handle fitness evaluation failures gracefully.
        
        Args:
            chromosome: Chromosome that failed fitness evaluation
            error: Exception that occurred during evaluation
            generation: Current generation number
            
        Returns:
            Penalty fitness score for the failed chromosome
        """
        error_report = ErrorReport(
            error_type=ErrorType.FITNESS_EVALUATION,
            severity=ErrorSeverity.MEDIUM,
            message=f"Fitness evaluation failed: {str(error)}",
            generation=generation,
            chromosome_id=self._get_chromosome_id(chromosome),
            stack_trace=traceback.format_exc(),
            recovery_action="Applied penalty fitness score"
        )
        
        self._log_error(error_report)
        self.error_history.append(error_report)
        
        # Apply penalty fitness based on error severity
        penalty_score = self._calculate_penalty_fitness(chromosome, error)
        
        # Mark chromosome as invalid
        chromosome.is_valid = False
        chromosome.constraint_violations.append(f"Fitness evaluation failed: {str(error)}")
        
        return penalty_score
    
    def repair_invalid_chromosome(self, chromosome: RouteChromosome, 
                                constraints: RouteConstraints, 
                                generation: int) -> RouteChromosome:
        """
        Attempt to repair chromosomes with invalid satellite sequences.
        
        Args:
            chromosome: Invalid chromosome to repair
            constraints: Route constraints for validation
            generation: Current generation number
            
        Returns:
            Repaired chromosome or new valid chromosome if repair fails
        """
        error_report = ErrorReport(
            error_type=ErrorType.CHROMOSOME_INVALID,
            severity=ErrorSeverity.MEDIUM,
            message="Attempting to repair invalid chromosome",
            generation=generation,
            chromosome_id=self._get_chromosome_id(chromosome),
            recovery_action="Chromosome repair attempted"
        )
        
        self._log_error(error_report)
        
        try:
            # Attempt various repair strategies
            repaired = self._attempt_chromosome_repair(chromosome, constraints)
            
            if self._validate_chromosome_basic(repaired):
                error_report.recovery_action = "Chromosome successfully repaired"
                self.error_history.append(error_report)
                return repaired
            else:
                # Repair failed, create new valid chromosome
                return self._create_fallback_chromosome(constraints, generation)
                
        except Exception as repair_error:
            error_report.severity = ErrorSeverity.HIGH
            error_report.message += f" Repair failed: {str(repair_error)}"
            error_report.recovery_action = "Created new fallback chromosome"
            self.error_history.append(error_report)
            
            return self._create_fallback_chromosome(constraints, generation)
    
    def handle_population_convergence(self, population: List[RouteChromosome], 
                                    constraints: RouteConstraints,
                                    generation: int) -> List[RouteChromosome]:
        """
        Inject diversity when population converges prematurely.
        
        Args:
            population: Current population showing convergence
            constraints: Route constraints
            generation: Current generation number
            
        Returns:
            Population with injected diversity
        """
        error_report = ErrorReport(
            error_type=ErrorType.POPULATION_CONVERGENCE,
            severity=ErrorSeverity.MEDIUM,
            message="Population convergence detected, injecting diversity",
            generation=generation,
            recovery_action="Diversity injection performed",
            additional_info={
                'population_size': len(population),
                'diversity_injection_rate': 0.3
            }
        )
        
        self._log_error(error_report)
        self.error_history.append(error_report)
        
        try:
            # Calculate how many chromosomes to replace
            replacement_count = max(1, int(len(population) * 0.3))  # Replace 30%
            
            # Keep best chromosomes
            sorted_population = sorted(population, 
                                     key=lambda x: getattr(x, 'fitness_score', 0.0), 
                                     reverse=True)
            
            preserved_count = len(population) - replacement_count
            new_population = sorted_population[:preserved_count]
            
            # Generate diverse new chromosomes
            for _ in range(replacement_count):
                new_chromosome = self._create_diverse_chromosome(constraints, population)
                new_population.append(new_chromosome)
            
            return new_population
            
        except Exception as diversity_error:
            error_report.severity = ErrorSeverity.HIGH
            error_report.message += f" Diversity injection failed: {str(diversity_error)}"
            error_report.recovery_action = "Returned original population"
            
            return population
    
    def handle_system_error(self, error: Exception, context: Dict[str, Any]) -> OptimizationResult:
        """
        Handle critical system-level errors with comprehensive fallback strategies.
        
        Args:
            error: System error that occurred
            context: Context information about the error
            
        Returns:
            Failed optimization result with error details
        """
        error_type = self._classify_system_error(error)
        
        error_report = ErrorReport(
            error_type=error_type,
            severity=ErrorSeverity.CRITICAL,
            message=f"Critical system error: {str(error)}",
            generation=context.get('generation', 0),
            stack_trace=traceback.format_exc(),
            recovery_action="System-level fallback applied",
            additional_info=context
        )
        
        self._log_error(error_report)
        self.error_history.append(error_report)
        
        # Apply appropriate recovery strategy
        if error_type in self.recovery_strategies:
            try:
                return self.recovery_strategies[error_type](error, context)
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy failed: {recovery_error}")
        
        # Final fallback - return failed result
        return self._create_critical_failure_result(error_report)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all errors encountered.
        
        Returns:
            Dictionary containing error statistics and analysis
        """
        if not self.error_history:
            return {'total_errors': 0, 'error_types': {}, 'severity_distribution': {}}
        
        # Count errors by type
        error_type_counts = {}
        severity_counts = {}
        
        for error in self.error_history:
            error_type_counts[error.error_type.value] = error_type_counts.get(error.error_type.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        # Calculate error rates by generation
        generation_errors = {}
        for error in self.error_history:
            gen = error.generation
            generation_errors[gen] = generation_errors.get(gen, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'error_types': error_type_counts,
            'severity_distribution': severity_counts,
            'generation_error_counts': generation_errors,
            'recovery_attempts': self.recovery_attempts,
            'most_common_error': max(error_type_counts.items(), key=lambda x: x[1])[0] if error_type_counts else None,
            'critical_errors': sum(1 for e in self.error_history if e.severity == ErrorSeverity.CRITICAL),
            'successful_recoveries': sum(1 for e in self.error_history if 'successfully' in e.recovery_action.lower())
        }
    
    def _attempt_chromosome_repair(self, chromosome: RouteChromosome, 
                                 constraints: RouteConstraints) -> RouteChromosome:
        """Attempt to repair an invalid chromosome using various strategies."""
        # Strategy 1: Remove invalid satellite IDs
        valid_sequence = [sat_id for sat_id in chromosome.satellite_sequence 
                         if sat_id in self.satellite_ids]
        
        # Strategy 2: Remove duplicates while preserving order
        seen = set()
        unique_sequence = []
        for sat_id in valid_sequence:
            if sat_id not in seen:
                unique_sequence.append(sat_id)
                seen.add(sat_id)
        
        # Strategy 3: Ensure minimum length
        if len(unique_sequence) < constraints.min_hops + 1:
            # Add random valid satellites
            available_satellites = list(self.satellite_ids - seen)
            needed = (constraints.min_hops + 1) - len(unique_sequence)
            
            if len(available_satellites) >= needed:
                additional = random.sample(available_satellites, needed)
                unique_sequence.extend(additional)
        
        # Strategy 4: Trim to maximum length
        if len(unique_sequence) > constraints.max_hops + 1:
            unique_sequence = unique_sequence[:constraints.max_hops + 1]
        
        # Strategy 5: Fix timing sequence
        repaired_times = self._repair_timing_sequence(unique_sequence, chromosome.departure_times)
        
        # Create repaired chromosome
        repaired = RouteChromosome(
            satellite_sequence=unique_sequence,
            departure_times=repaired_times
        )
        
        return repaired
    
    def _repair_timing_sequence(self, satellite_sequence: List[int], 
                              original_times: List[float]) -> List[float]:
        """Repair timing sequence to match satellite sequence length."""
        seq_length = len(satellite_sequence)
        
        if len(original_times) == seq_length:
            # Check if times are ascending
            if all(original_times[i] < original_times[i+1] for i in range(len(original_times)-1)):
                return original_times.copy()
        
        # Generate new ascending timing sequence
        if len(original_times) >= 2:
            start_time = original_times[0]
            end_time = original_times[-1]
        else:
            start_time = 0.0
            end_time = 86400.0  # 24 hours default
        
        if seq_length == 1:
            return [start_time]
        
        # Generate evenly spaced times
        time_step = (end_time - start_time) / (seq_length - 1)
        return [start_time + i * time_step for i in range(seq_length)]
    
    def _create_fallback_chromosome(self, constraints: RouteConstraints, 
                                  generation: int) -> RouteChromosome:
        """Create a simple valid chromosome as fallback."""
        try:
            # Create minimal valid route
            available_satellites = list(self.satellite_ids)
            
            if constraints.start_satellite_id and constraints.start_satellite_id in available_satellites:
                sequence = [constraints.start_satellite_id]
                available_satellites.remove(constraints.start_satellite_id)
            else:
                sequence = [random.choice(available_satellites)]
                available_satellites.remove(sequence[0])
            
            # Add minimum required hops
            hops_needed = max(1, constraints.min_hops)
            for _ in range(hops_needed):
                if available_satellites:
                    next_sat = random.choice(available_satellites)
                    sequence.append(next_sat)
                    available_satellites.remove(next_sat)
                else:
                    break
            
            # Add end satellite if specified
            if (constraints.end_satellite_id and 
                constraints.end_satellite_id in available_satellites and
                constraints.end_satellite_id != sequence[-1]):
                sequence.append(constraints.end_satellite_id)
            
            # Generate timing
            times = [i * 3600.0 for i in range(len(sequence))]  # 1 hour intervals
            
            return RouteChromosome(
                satellite_sequence=sequence,
                departure_times=times
            )
            
        except Exception as fallback_error:
            self.logger.error(f"Fallback chromosome creation failed: {fallback_error}")
            # Ultimate fallback - single satellite
            sat_id = list(self.satellite_ids)[0]
            return RouteChromosome(
                satellite_sequence=[sat_id],
                departure_times=[0.0]
            )
    
    def _create_diverse_chromosome(self, constraints: RouteConstraints, 
                                 existing_population: List[RouteChromosome]) -> RouteChromosome:
        """Create a chromosome that adds diversity to the population."""
        # Analyze existing population to identify underrepresented characteristics
        used_satellites = set()
        route_lengths = []
        
        for chromosome in existing_population:
            used_satellites.update(chromosome.satellite_sequence)
            route_lengths.append(len(chromosome.satellite_sequence))
        
        # Create diverse route
        available_satellites = list(self.satellite_ids - used_satellites)
        if not available_satellites:
            available_satellites = list(self.satellite_ids)
        
        # Choose diverse route length
        avg_length = sum(route_lengths) / len(route_lengths) if route_lengths else 3
        diverse_length = random.choice([
            max(2, int(avg_length * 0.5)),  # Shorter route
            int(avg_length * 1.5),          # Longer route
            random.randint(constraints.min_hops + 1, min(constraints.max_hops + 1, len(available_satellites)))
        ])
        
        # Build diverse sequence
        sequence = random.sample(available_satellites, min(diverse_length, len(available_satellites)))
        
        # Generate timing with some randomness
        base_interval = random.uniform(1800, 7200)  # 30 minutes to 2 hours
        times = [i * base_interval + random.uniform(-300, 300) for i in range(len(sequence))]
        times.sort()  # Ensure ascending order
        
        return RouteChromosome(
            satellite_sequence=sequence,
            departure_times=times
        )
    
    def _calculate_penalty_fitness(self, chromosome: RouteChromosome, error: Exception) -> float:
        """Calculate appropriate penalty fitness based on error type and chromosome."""
        base_penalty = -1000.0
        
        # Adjust penalty based on error type
        if "timeout" in str(error).lower():
            return base_penalty * 2.0
        elif "memory" in str(error).lower():
            return base_penalty * 1.5
        elif "orbital" in str(error).lower():
            return base_penalty * 1.2
        
        # Adjust based on chromosome characteristics
        if chromosome.satellite_sequence:
            # Less penalty for chromosomes with more hops (they were trying to be good)
            hop_bonus = min(100, len(chromosome.satellite_sequence) * 10)
            return base_penalty + hop_bonus
        
        return base_penalty
    
    def _validate_chromosome_basic(self, chromosome: RouteChromosome) -> bool:
        """Perform basic validation of chromosome structure."""
        if not chromosome.satellite_sequence:
            return False
        
        if len(chromosome.satellite_sequence) != len(chromosome.departure_times):
            return False
        
        # Check for valid satellite IDs
        for sat_id in chromosome.satellite_sequence:
            if sat_id not in self.satellite_ids:
                return False
        
        # Check departure times are ascending
        if len(chromosome.departure_times) > 1:
            for i in range(1, len(chromosome.departure_times)):
                if chromosome.departure_times[i] <= chromosome.departure_times[i-1]:
                    return False
        
        return True
    
    def _get_chromosome_id(self, chromosome: RouteChromosome) -> str:
        """Generate unique identifier for chromosome for tracking."""
        sequence_str = "-".join(map(str, chromosome.satellite_sequence[:3]))  # First 3 satellites
        return f"chr_{sequence_str}_{len(chromosome.satellite_sequence)}"
    
    def _classify_system_error(self, error: Exception) -> ErrorType:
        """Classify system error into appropriate error type."""
        error_str = str(error).lower()
        
        if "memory" in error_str or "out of memory" in error_str:
            return ErrorType.MEMORY_EXHAUSTION
        elif "timeout" in error_str or "time" in error_str:
            return ErrorType.TIMEOUT
        elif "orbital" in error_str or "propagat" in error_str:
            return ErrorType.ORBITAL_CALCULATION
        elif "config" in error_str or "parameter" in error_str:
            return ErrorType.CONFIGURATION_ERROR
        else:
            return ErrorType.FITNESS_EVALUATION
    
    def _log_error(self, error_report: ErrorReport):
        """Log error report with appropriate severity level."""
        log_message = f"GA Error [{error_report.error_type.value}]: {error_report.message}"
        
        if error_report.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_report.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_report.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        if error_report.recovery_action:
            self.logger.info(f"Recovery action: {error_report.recovery_action}")
    
    def _create_critical_failure_result(self, error_report: ErrorReport) -> OptimizationResult:
        """Create optimization result for critical system failures."""
        stats = OptimizationStats(
            generations_completed=error_report.generation,
            best_fitness=0.0,
            average_fitness=0.0,
            population_diversity=0.0,
            constraint_satisfaction_rate=0.0,
            stagnant_generations=0
        )
        
        return OptimizationResult(
            best_route=None,
            optimization_stats=stats,
            convergence_history=[],
            execution_time=0.0,
            status=OptimizationStatus.FAILED,
            error_message=f"{error_report.error_type.value}: {error_report.message}"
        )
    
    # Recovery strategy implementations
    def _handle_fitness_evaluation_error(self, error: Exception, context: Dict[str, Any]) -> OptimizationResult:
        """Handle fitness evaluation errors."""
        return self._create_critical_failure_result(ErrorReport(
            error_type=ErrorType.FITNESS_EVALUATION,
            severity=ErrorSeverity.CRITICAL,
            message=f"Persistent fitness evaluation failures: {str(error)}",
            generation=context.get('generation', 0)
        ))
    
    def _handle_invalid_chromosome(self, error: Exception, context: Dict[str, Any]) -> OptimizationResult:
        """Handle invalid chromosome errors."""
        return self._create_critical_failure_result(ErrorReport(
            error_type=ErrorType.CHROMOSOME_INVALID,
            severity=ErrorSeverity.CRITICAL,
            message=f"Unable to create valid chromosomes: {str(error)}",
            generation=context.get('generation', 0)
        ))
    
    def _handle_population_convergence(self, error: Exception, context: Dict[str, Any]) -> OptimizationResult:
        """Handle population convergence errors."""
        return self._create_critical_failure_result(ErrorReport(
            error_type=ErrorType.POPULATION_CONVERGENCE,
            severity=ErrorSeverity.HIGH,
            message=f"Population diversity could not be restored: {str(error)}",
            generation=context.get('generation', 0)
        ))
    
    def _handle_constraint_violation(self, error: Exception, context: Dict[str, Any]) -> OptimizationResult:
        """Handle constraint violation errors."""
        return self._create_critical_failure_result(ErrorReport(
            error_type=ErrorType.CONSTRAINT_VIOLATION,
            severity=ErrorSeverity.HIGH,
            message=f"Constraint violations could not be resolved: {str(error)}",
            generation=context.get('generation', 0)
        ))
    
    def _handle_orbital_calculation_error(self, error: Exception, context: Dict[str, Any]) -> OptimizationResult:
        """Handle orbital calculation errors."""
        return self._create_critical_failure_result(ErrorReport(
            error_type=ErrorType.ORBITAL_CALCULATION,
            severity=ErrorSeverity.CRITICAL,
            message=f"Orbital calculations failed: {str(error)}",
            generation=context.get('generation', 0)
        ))
    
    def _handle_memory_exhaustion(self, error: Exception, context: Dict[str, Any]) -> OptimizationResult:
        """Handle memory exhaustion errors."""
        return self._create_critical_failure_result(ErrorReport(
            error_type=ErrorType.MEMORY_EXHAUSTION,
            severity=ErrorSeverity.CRITICAL,
            message=f"Memory exhaustion: {str(error)}",
            generation=context.get('generation', 0)
        ))
    
    def _handle_timeout_error(self, error: Exception, context: Dict[str, Any]) -> OptimizationResult:
        """Handle timeout errors."""
        return self._create_critical_failure_result(ErrorReport(
            error_type=ErrorType.TIMEOUT,
            severity=ErrorSeverity.HIGH,
            message=f"Operation timeout: {str(error)}",
            generation=context.get('generation', 0)
        ))
    
    def _handle_configuration_error(self, error: Exception, context: Dict[str, Any]) -> OptimizationResult:
        """Handle configuration errors."""
        return self._create_critical_failure_result(ErrorReport(
            error_type=ErrorType.CONFIGURATION_ERROR,
            severity=ErrorSeverity.CRITICAL,
            message=f"Configuration error: {str(error)}",
            generation=context.get('generation', 0)
        ))