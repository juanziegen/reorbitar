"""
Route Fitness Evaluator Module

This module provides comprehensive fitness evaluation for genetic algorithm route optimization.
It evaluates route chromosomes based on delta-v costs, constraint satisfaction, and optimization objectives.
"""

import math
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .genetic_algorithm import (
    RouteChromosome, 
    RouteConstraints, 
    FitnessResult, 
    ConstraintResult
)
from .orbital_propagator import OrbitalPropagator, TransferWindow
from .tle_parser import SatelliteData
from .fitness_cache import FitnessCacheManager


class RouteFitnessEvaluator:
    """Evaluates fitness of route chromosomes for genetic algorithm optimization."""
    
    def __init__(self, satellites: List[SatelliteData], orbital_propagator: OrbitalPropagator,
                 cache_manager: Optional[FitnessCacheManager] = None):
        """Initialize with satellite data and orbital propagation capability.
        
        Args:
            satellites: List of satellite data
            orbital_propagator: Orbital propagator for time-based calculations
            cache_manager: Optional cache manager for performance optimization
            
        Raises:
            ValueError: If satellites list is empty or propagator is None
        """
        if not satellites:
            raise ValueError("Satellites list cannot be empty")
        if orbital_propagator is None:
            raise ValueError("Orbital propagator cannot be None")
        
        self.satellites = {sat.catalog_number: sat for sat in satellites}
        self.orbital_propagator = orbital_propagator
        
        # Initialize advanced caching system
        if cache_manager is None:
            # Create default cache manager optimized for constellation size
            constellation_size = len(satellites)
            self.cache_manager = FitnessCacheManager()
            self.cache_manager.optimize_cache_sizes(constellation_size)
        else:
            self.cache_manager = cache_manager
        
        # Legacy cache for backward compatibility (deprecated)
        self._deltav_cache: Dict[Tuple[int, int, float], float] = {}
        
        # Validate that all satellites in propagator are in our satellite list
        propagator_ids = set(self.orbital_propagator.get_satellite_ids())
        satellite_ids = set(self.satellites.keys())
        if not propagator_ids.issubset(satellite_ids):
            missing_ids = propagator_ids - satellite_ids
            raise ValueError(f"Propagator contains satellites not in satellite list: {missing_ids}")
    
    def evaluate_route(self, chromosome: RouteChromosome, constraints: RouteConstraints) -> FitnessResult:
        """Comprehensive route evaluation including delta-v, timing, and constraints.
        
        Args:
            chromosome: Route chromosome to evaluate
            constraints: Route constraints to check against
            
        Returns:
            FitnessResult with comprehensive evaluation metrics
        """
        # Initialize result with basic chromosome information
        violations = []
        is_valid = chromosome.is_valid
        
        # Add any existing chromosome violations
        violations.extend(chromosome.constraint_violations)
        
        # Calculate route delta-v
        try:
            total_deltav = self.calculate_route_deltav(
                chromosome.satellite_sequence, 
                chromosome.departure_times
            )
        except Exception as e:
            violations.append(f"Delta-v calculation failed: {str(e)}")
            total_deltav = float('inf')
            is_valid = False
        
        # Check constraints
        constraint_result = self.check_constraints(chromosome, constraints)
        violations.extend(constraint_result.violations)
        is_valid = is_valid and constraint_result.is_valid
        
        # Calculate fitness score
        fitness_score = self._calculate_fitness_score(
            chromosome, total_deltav, constraint_result, constraints
        )
        
        return FitnessResult(
            fitness_score=fitness_score,
            total_deltav=total_deltav,
            hop_count=chromosome.hop_count,
            mission_duration=chromosome.mission_duration,
            constraint_violations=violations,
            is_valid=is_valid
        )
    
    def calculate_route_deltav(self, satellite_sequence: List[int], departure_times: List[float]) -> float:
        """Calculate total delta-v for a route considering orbital positions at departure times.
        
        Args:
            satellite_sequence: List of satellite catalog numbers in route order
            departure_times: List of departure times corresponding to each satellite
            
        Returns:
            Total delta-v required for the route in km/s
            
        Raises:
            ValueError: If sequence and times don't match or contain invalid data
        """
        if len(satellite_sequence) != len(departure_times):
            raise ValueError("Satellite sequence and departure times must have same length")
        
        if len(satellite_sequence) < 2:
            return 0.0  # No transfers needed for single satellite or empty route
        
        # Validate all satellites exist
        for sat_id in satellite_sequence:
            if sat_id not in self.satellites:
                raise ValueError(f"Satellite {sat_id} not found in satellite database")
        
        # Validate departure times are in ascending order
        for i in range(1, len(departure_times)):
            if departure_times[i] <= departure_times[i-1]:
                raise ValueError(f"Departure times must be in ascending order: {departure_times}")
        
        total_deltav = 0.0
        
        # Calculate delta-v for each hop in the route
        for i in range(len(satellite_sequence) - 1):
            source_id = satellite_sequence[i]
            target_id = satellite_sequence[i + 1]
            departure_time = departure_times[i]
            
            # Check advanced cache first
            hop_deltav = self.cache_manager.get_deltav(source_id, target_id, departure_time)
            
            if hop_deltav is None:
                # Check legacy cache for backward compatibility
                cache_key = (source_id, target_id, departure_time)
                if cache_key in self._deltav_cache:
                    hop_deltav = self._deltav_cache[cache_key]
                else:
                    # Calculate transfer delta-v using orbital propagator
                    try:
                        transfer_window = self.orbital_propagator.calculate_transfer_window(
                            source_id, target_id, departure_time
                        )
                        hop_deltav = transfer_window.departure_deltav + transfer_window.arrival_deltav
                        
                        # Cache the result in both caches
                        self.cache_manager.put_deltav(source_id, target_id, departure_time, hop_deltav)
                        self._deltav_cache[cache_key] = hop_deltav
                        
                    except Exception as e:
                        raise ValueError(f"Failed to calculate transfer from {source_id} to {target_id} at time {departure_time}: {str(e)}")
            
            total_deltav += hop_deltav
        
        return total_deltav
    
    def check_constraints(self, chromosome: RouteChromosome, constraints: RouteConstraints) -> ConstraintResult:
        """Verify route satisfies all constraints.
        
        Args:
            chromosome: Route chromosome to check
            constraints: Constraints to validate against
            
        Returns:
            ConstraintResult with detailed constraint validation information
        """
        violations = []
        
        # Check hop count constraints
        hop_count = chromosome.hop_count
        if hop_count < constraints.min_hops:
            violations.append(f"Route has {hop_count} hops, minimum required: {constraints.min_hops}")
        if hop_count > constraints.max_hops:
            violations.append(f"Route has {hop_count} hops, maximum allowed: {constraints.max_hops}")
        
        # Check delta-v budget constraint
        try:
            total_deltav = self.calculate_route_deltav(
                chromosome.satellite_sequence, 
                chromosome.departure_times
            )
        except Exception as e:
            violations.append(f"Cannot validate delta-v constraint: {str(e)}")
            total_deltav = float('inf')
        
        if total_deltav > constraints.max_deltav_budget:
            violations.append(
                f"Route delta-v {total_deltav:.3f} km/s exceeds budget {constraints.max_deltav_budget:.3f} km/s"
            )
        
        # Check mission duration constraint
        mission_duration = chromosome.mission_duration
        if mission_duration > constraints.max_mission_duration:
            violations.append(
                f"Mission duration {mission_duration:.0f}s exceeds limit {constraints.max_mission_duration:.0f}s"
            )
        
        # Check starting satellite constraint
        if constraints.start_satellite_id is not None:
            if not chromosome.satellite_sequence:
                violations.append("Route is empty but starting satellite is required")
            elif chromosome.satellite_sequence[0] != constraints.start_satellite_id:
                violations.append(
                    f"Route starts with satellite {chromosome.satellite_sequence[0]}, "
                    f"required: {constraints.start_satellite_id}"
                )
        
        # Check ending satellite constraint
        if constraints.end_satellite_id is not None:
            if not chromosome.satellite_sequence:
                violations.append("Route is empty but ending satellite is required")
            elif chromosome.satellite_sequence[-1] != constraints.end_satellite_id:
                violations.append(
                    f"Route ends with satellite {chromosome.satellite_sequence[-1]}, "
                    f"required: {constraints.end_satellite_id}"
                )
        
        # Check forbidden satellites constraint
        if constraints.forbidden_satellites:
            forbidden_in_route = set(chromosome.satellite_sequence) & set(constraints.forbidden_satellites)
            if forbidden_in_route:
                violations.append(f"Route contains forbidden satellites: {list(forbidden_in_route)}")
        
        # Check for duplicate satellites in route (not allowed)
        if len(chromosome.satellite_sequence) != len(set(chromosome.satellite_sequence)):
            duplicates = []
            seen = set()
            for sat_id in chromosome.satellite_sequence:
                if sat_id in seen:
                    duplicates.append(sat_id)
                seen.add(sat_id)
            violations.append(f"Route contains duplicate satellites: {list(set(duplicates))}")
        
        # Check that all satellites in route exist
        for sat_id in chromosome.satellite_sequence:
            if sat_id not in self.satellites:
                violations.append(f"Satellite {sat_id} not found in satellite database")
        
        is_valid = len(violations) == 0
        
        return ConstraintResult(
            is_valid=is_valid,
            violations=violations,
            deltav_usage=total_deltav if math.isfinite(total_deltav) else 0.0,
            deltav_budget=constraints.max_deltav_budget,
            duration_usage=mission_duration,
            duration_budget=constraints.max_mission_duration,
            hop_count=hop_count,
            min_hops=constraints.min_hops,
            max_hops=constraints.max_hops
        )
    
    def _calculate_fitness_score(self, chromosome: RouteChromosome, total_deltav: float,
                               constraint_result: ConstraintResult, constraints: RouteConstraints) -> float:
        """Calculate fitness score for a route chromosome.
        
        The fitness function aims to maximize the number of hops while minimizing delta-v usage
        and penalizing constraint violations.
        
        Args:
            chromosome: Route chromosome
            total_deltav: Total delta-v for the route
            constraint_result: Result of constraint checking
            constraints: Route constraints
            
        Returns:
            Fitness score (higher is better)
        """
        # Base fitness is the number of hops (we want to maximize this)
        base_fitness = float(chromosome.hop_count)
        
        # Apply penalty for constraint violations
        violation_penalty = len(constraint_result.violations) * 1000.0
        
        # Apply penalty for delta-v usage (encourage efficient routes)
        if math.isfinite(total_deltav) and constraints.max_deltav_budget > 0:
            deltav_utilization = total_deltav / constraints.max_deltav_budget
            # Penalty increases quadratically with delta-v usage
            deltav_penalty = deltav_utilization * deltav_utilization * 10.0
        else:
            deltav_penalty = 1000.0  # Large penalty for invalid delta-v
        
        # Apply penalty for mission duration (encourage shorter missions)
        if constraints.max_mission_duration > 0:
            duration_utilization = chromosome.mission_duration / constraints.max_mission_duration
            # Small penalty for duration usage
            duration_penalty = duration_utilization * 1.0
        else:
            duration_penalty = 0.0
        
        # Calculate final fitness score
        fitness_score = base_fitness - violation_penalty - deltav_penalty - duration_penalty
        
        # Ensure fitness is never negative for valid routes
        if constraint_result.is_valid and fitness_score < 0:
            fitness_score = 0.1  # Small positive value for valid but inefficient routes
        
        return fitness_score
    
    def clear_cache(self):
        """Clear all caches (both legacy and advanced)."""
        self._deltav_cache.clear()
        self.cache_manager.clear_all_caches()
    
    def get_cache_stats(self) -> Dict:
        """Get comprehensive statistics about all caches.
        
        Returns:
            Dictionary with detailed cache statistics
        """
        # Get advanced cache stats
        advanced_stats = self.cache_manager.get_combined_stats()
        
        # Get legacy cache stats for backward compatibility
        legacy_stats = {
            'legacy_cache_size': len(self._deltav_cache),
            'legacy_unique_sources': len(set(key[0] for key in self._deltav_cache.keys())) if self._deltav_cache else 0,
            'legacy_unique_targets': len(set(key[1] for key in self._deltav_cache.keys())) if self._deltav_cache else 0,
            'legacy_unique_times': len(set(key[2] for key in self._deltav_cache.keys())) if self._deltav_cache else 0
        }
        
        return {
            'advanced_caching': advanced_stats,
            'legacy_caching': legacy_stats,
            'total_satellites': len(self.satellites)
        }
    
    def invalidate_satellite_cache(self, satellite_ids: List[int]):
        """
        Invalidate cache entries for specific satellites.
        
        This is useful when satellite orbital data is updated or satellites
        are removed from the constellation.
        
        Args:
            satellite_ids: List of satellite IDs to invalidate from cache
        """
        # Invalidate advanced cache
        self.cache_manager.invalidate_satellite_data(satellite_ids)
        
        # Invalidate legacy cache
        keys_to_remove = []
        for key in self._deltav_cache:
            source_id, target_id, _ = key
            if source_id in satellite_ids or target_id in satellite_ids:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._deltav_cache[key]
    
    def invalidate_time_range_cache(self, start_time: float, end_time: float):
        """
        Invalidate cache entries for a specific time range.
        
        This is useful when orbital propagation models are updated or
        when you want to force recalculation for a specific time period.
        
        Args:
            start_time: Start time for invalidation (seconds from epoch)
            end_time: End time for invalidation (seconds from epoch)
        """
        # Invalidate advanced cache
        self.cache_manager.invalidate_time_range(start_time, end_time)
        
        # Invalidate legacy cache
        keys_to_remove = []
        for key in self._deltav_cache:
            _, _, departure_time = key
            if start_time <= departure_time <= end_time:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._deltav_cache[key]
    
    def perform_cache_maintenance(self):
        """
        Perform periodic cache maintenance.
        
        This should be called periodically during long optimization runs
        to clean up expired entries and optimize memory usage.
        """
        self.cache_manager.periodic_cleanup()
    
    def optimize_cache_for_constellation(self, constellation_size: Optional[int] = None):
        """
        Optimize cache settings for the current constellation size.
        
        Args:
            constellation_size: Override constellation size (uses current if None)
        """
        if constellation_size is None:
            constellation_size = len(self.satellites)
        
        self.cache_manager.optimize_cache_sizes(constellation_size)
    
    def get_cache_memory_usage(self) -> Dict[str, int]:
        """
        Get estimated memory usage of all caches.
        
        Returns:
            Dictionary with memory usage estimates in bytes
        """
        advanced_memory = self.cache_manager.get_combined_stats()['total_memory_estimate']
        
        # Estimate legacy cache memory usage
        legacy_memory = len(self._deltav_cache) * (3 * 8 + 8 + 64)  # Rough estimate
        
        return {
            'advanced_cache_bytes': advanced_memory,
            'legacy_cache_bytes': legacy_memory,
            'total_bytes': advanced_memory + legacy_memory
        }
    
    def create_route_hash(self, chromosome: RouteChromosome) -> str:
        """
        Create a hash for a route chromosome for caching purposes.
        
        Args:
            chromosome: Route chromosome to hash
            
        Returns:
            Hash string representing the route
        """
        # Create a deterministic hash of the route
        route_data = f"{chromosome.satellite_sequence}_{chromosome.departure_times}"
        return hashlib.md5(route_data.encode()).hexdigest()
    
    def create_constraints_hash(self, constraints: RouteConstraints) -> str:
        """
        Create a hash for route constraints for caching purposes.
        
        Args:
            constraints: Route constraints to hash
            
        Returns:
            Hash string representing the constraints
        """
        # Create a deterministic hash of the constraints
        constraints_data = (
            f"{constraints.max_deltav_budget}_{constraints.max_mission_duration}_"
            f"{constraints.start_satellite_id}_{constraints.end_satellite_id}_"
            f"{constraints.min_hops}_{constraints.max_hops}_{constraints.forbidden_satellites}"
        )
        return hashlib.md5(constraints_data.encode()).hexdigest()
    
    def validate_chromosome_basic(self, chromosome: RouteChromosome) -> List[str]:
        """Perform basic validation of chromosome structure.
        
        Args:
            chromosome: Route chromosome to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not chromosome.satellite_sequence:
            errors.append("Satellite sequence cannot be empty")
            return errors
        
        if not chromosome.departure_times:
            errors.append("Departure times cannot be empty")
            return errors
        
        if len(chromosome.satellite_sequence) != len(chromosome.departure_times):
            errors.append("Satellite sequence and departure times must have same length")
        
        # Check for valid satellite IDs
        for sat_id in chromosome.satellite_sequence:
            if not isinstance(sat_id, int) or sat_id <= 0:
                errors.append(f"Invalid satellite ID: {sat_id}")
        
        # Check for valid departure times
        for i, time in enumerate(chromosome.departure_times):
            if not isinstance(time, (int, float)) or not math.isfinite(time):
                errors.append(f"Invalid departure time at index {i}: {time}")
        
        # Check departure times are in ascending order
        if len(chromosome.departure_times) > 1:
            for i in range(1, len(chromosome.departure_times)):
                if chromosome.departure_times[i] <= chromosome.departure_times[i-1]:
                    errors.append(f"Departure times must be in ascending order at index {i}")
        
        return errors