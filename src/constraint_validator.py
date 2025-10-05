"""
Constraint Validation System

This module provides comprehensive constraint validation for genetic algorithm route optimization.
It implements detailed checking for all route constraints including delta-v budget, time limits,
starting/ending satellites, and forbidden satellites.
"""

import math
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from genetic_algorithm import RouteChromosome, RouteConstraints, ConstraintResult
from tle_parser import SatelliteData


@dataclass
class DetailedConstraintViolation:
    """Detailed information about a specific constraint violation."""
    constraint_type: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    actual_value: Optional[float] = None
    expected_value: Optional[float] = None
    satellite_ids: Optional[List[int]] = None


@dataclass
class ConstraintValidationResult:
    """Comprehensive result of constraint validation."""
    is_valid: bool
    violations: List[DetailedConstraintViolation]
    warnings: List[str]
    constraint_satisfaction_score: float  # 0.0 to 1.0
    detailed_metrics: Dict[str, float]


class ConstraintValidator:
    """Comprehensive constraint validation system for route optimization."""
    
    def __init__(self, satellites: Dict[int, SatelliteData]):
        """Initialize with satellite database.
        
        Args:
            satellites: Dictionary mapping satellite IDs to SatelliteData
        """
        self.satellites = satellites
        self.satellite_ids = set(satellites.keys())
    
    def validate_route_constraints(self, chromosome: RouteChromosome, 
                                 constraints: RouteConstraints,
                                 total_deltav: Optional[float] = None) -> ConstraintValidationResult:
        """Perform comprehensive constraint validation on a route.
        
        Args:
            chromosome: Route chromosome to validate
            constraints: Route constraints to check against
            total_deltav: Pre-calculated total delta-v (optional, for performance)
            
        Returns:
            ConstraintValidationResult with detailed validation information
        """
        violations = []
        warnings = []
        detailed_metrics = {}
        
        # Use provided delta-v or chromosome's cached value
        route_deltav = total_deltav if total_deltav is not None else chromosome.total_deltav
        
        # 1. Delta-v budget constraint checking
        deltav_violation = self._check_deltav_constraint(route_deltav, constraints)
        if deltav_violation:
            violations.append(deltav_violation)
        detailed_metrics['deltav_usage'] = route_deltav
        detailed_metrics['deltav_budget'] = constraints.max_deltav_budget
        detailed_metrics['deltav_utilization'] = (route_deltav / constraints.max_deltav_budget) * 100.0
        
        # 2. Time constraint validation
        time_violation = self._check_time_constraint(chromosome, constraints)
        if time_violation:
            violations.append(time_violation)
        detailed_metrics['mission_duration'] = chromosome.mission_duration
        detailed_metrics['duration_budget'] = constraints.max_mission_duration
        detailed_metrics['duration_utilization'] = (chromosome.mission_duration / constraints.max_mission_duration) * 100.0
        
        # 3. Starting satellite constraint enforcement
        start_violation = self._check_start_satellite_constraint(chromosome, constraints)
        if start_violation:
            violations.append(start_violation)
        
        # 4. Ending satellite constraint enforcement
        end_violation = self._check_end_satellite_constraint(chromosome, constraints)
        if end_violation:
            violations.append(end_violation)
        
        # 5. Hop count constraints
        hop_violations = self._check_hop_constraints(chromosome, constraints)
        violations.extend(hop_violations)
        detailed_metrics['hop_count'] = chromosome.hop_count
        detailed_metrics['min_hops'] = constraints.min_hops
        detailed_metrics['max_hops'] = constraints.max_hops
        
        # 6. Forbidden satellite filtering
        forbidden_violations = self._check_forbidden_satellites(chromosome, constraints)
        violations.extend(forbidden_violations)
        
        # 7. Satellite existence validation
        existence_violations = self._check_satellite_existence(chromosome)
        violations.extend(existence_violations)
        
        # 8. Route validity checks
        validity_violations = self._check_route_validity(chromosome)
        violations.extend(validity_violations)
        
        # Calculate constraint satisfaction score
        total_constraints = 8  # Number of constraint categories
        violated_constraints = len([v for v in violations if v.severity == 'error'])
        constraint_satisfaction_score = max(0.0, (total_constraints - violated_constraints) / total_constraints)
        
        # Add warnings for near-limit conditions
        warnings.extend(self._generate_warnings(chromosome, constraints, detailed_metrics))
        
        is_valid = len([v for v in violations if v.severity == 'error']) == 0
        
        return ConstraintValidationResult(
            is_valid=is_valid,
            violations=violations,
            warnings=warnings,
            constraint_satisfaction_score=constraint_satisfaction_score,
            detailed_metrics=detailed_metrics
        )
    
    def _check_deltav_constraint(self, route_deltav: float, 
                               constraints: RouteConstraints) -> Optional[DetailedConstraintViolation]:
        """Check delta-v budget constraint."""
        if route_deltav > constraints.max_deltav_budget:
            excess = route_deltav - constraints.max_deltav_budget
            return DetailedConstraintViolation(
                constraint_type="deltav_budget",
                severity="error",
                message=f"Route delta-v {route_deltav:.3f} km/s exceeds budget {constraints.max_deltav_budget:.3f} km/s by {excess:.3f} km/s",
                actual_value=route_deltav,
                expected_value=constraints.max_deltav_budget
            )
        return None
    
    def _check_time_constraint(self, chromosome: RouteChromosome, 
                             constraints: RouteConstraints) -> Optional[DetailedConstraintViolation]:
        """Check time constraint validation."""
        mission_duration = chromosome.mission_duration
        if mission_duration > constraints.max_mission_duration:
            excess = mission_duration - constraints.max_mission_duration
            return DetailedConstraintViolation(
                constraint_type="mission_duration",
                severity="error",
                message=f"Mission duration {mission_duration:.0f}s exceeds limit {constraints.max_mission_duration:.0f}s by {excess:.0f}s",
                actual_value=mission_duration,
                expected_value=constraints.max_mission_duration
            )
        return None
    
    def _check_start_satellite_constraint(self, chromosome: RouteChromosome, 
                                        constraints: RouteConstraints) -> Optional[DetailedConstraintViolation]:
        """Check starting satellite constraint enforcement."""
        if constraints.start_satellite_id is not None:
            if not chromosome.satellite_sequence:
                return DetailedConstraintViolation(
                    constraint_type="start_satellite",
                    severity="error",
                    message="Route has no satellites but start satellite is required",
                    expected_value=float(constraints.start_satellite_id)
                )
            
            actual_start = chromosome.satellite_sequence[0]
            if actual_start != constraints.start_satellite_id:
                return DetailedConstraintViolation(
                    constraint_type="start_satellite",
                    severity="error",
                    message=f"Route starts at satellite {actual_start} but must start at {constraints.start_satellite_id}",
                    actual_value=float(actual_start),
                    expected_value=float(constraints.start_satellite_id),
                    satellite_ids=[actual_start]
                )
        return None
    
    def _check_end_satellite_constraint(self, chromosome: RouteChromosome, 
                                      constraints: RouteConstraints) -> Optional[DetailedConstraintViolation]:
        """Check ending satellite constraint enforcement."""
        if constraints.end_satellite_id is not None:
            if not chromosome.satellite_sequence:
                return DetailedConstraintViolation(
                    constraint_type="end_satellite",
                    severity="error",
                    message="Route has no satellites but end satellite is required",
                    expected_value=float(constraints.end_satellite_id)
                )
            
            actual_end = chromosome.satellite_sequence[-1]
            if actual_end != constraints.end_satellite_id:
                return DetailedConstraintViolation(
                    constraint_type="end_satellite",
                    severity="error",
                    message=f"Route ends at satellite {actual_end} but must end at {constraints.end_satellite_id}",
                    actual_value=float(actual_end),
                    expected_value=float(constraints.end_satellite_id),
                    satellite_ids=[actual_end]
                )
        return None
    
    def _check_hop_constraints(self, chromosome: RouteChromosome, 
                             constraints: RouteConstraints) -> List[DetailedConstraintViolation]:
        """Check hop count constraints."""
        violations = []
        hop_count = chromosome.hop_count
        
        if hop_count < constraints.min_hops:
            violations.append(DetailedConstraintViolation(
                constraint_type="min_hops",
                severity="error",
                message=f"Route has {hop_count} hops but minimum is {constraints.min_hops}",
                actual_value=float(hop_count),
                expected_value=float(constraints.min_hops)
            ))
        
        if hop_count > constraints.max_hops:
            violations.append(DetailedConstraintViolation(
                constraint_type="max_hops",
                severity="error",
                message=f"Route has {hop_count} hops but maximum is {constraints.max_hops}",
                actual_value=float(hop_count),
                expected_value=float(constraints.max_hops)
            ))
        
        return violations
    
    def _check_forbidden_satellites(self, chromosome: RouteChromosome, 
                                  constraints: RouteConstraints) -> List[DetailedConstraintViolation]:
        """Check forbidden satellite filtering."""
        violations = []
        if not constraints.forbidden_satellites:
            return violations
        
        forbidden_set = set(constraints.forbidden_satellites)
        route_satellites = set(chromosome.satellite_sequence)
        forbidden_in_route = route_satellites.intersection(forbidden_set)
        
        if forbidden_in_route:
            violations.append(DetailedConstraintViolation(
                constraint_type="forbidden_satellites",
                severity="error",
                message=f"Route contains forbidden satellites: {sorted(forbidden_in_route)}",
                satellite_ids=list(forbidden_in_route)
            ))
        
        return violations
    
    def _check_satellite_existence(self, chromosome: RouteChromosome) -> List[DetailedConstraintViolation]:
        """Check that all satellites in route exist in database."""
        violations = []
        missing_satellites = []
        
        for sat_id in chromosome.satellite_sequence:
            if sat_id not in self.satellite_ids:
                missing_satellites.append(sat_id)
        
        if missing_satellites:
            violations.append(DetailedConstraintViolation(
                constraint_type="satellite_existence",
                severity="error",
                message=f"Route contains non-existent satellites: {missing_satellites}",
                satellite_ids=missing_satellites
            ))
        
        return violations
    
    def _check_route_validity(self, chromosome: RouteChromosome) -> List[DetailedConstraintViolation]:
        """Check basic route validity."""
        violations = []
        
        # Check for empty route
        if not chromosome.satellite_sequence:
            violations.append(DetailedConstraintViolation(
                constraint_type="route_validity",
                severity="error",
                message="Route contains no satellites"
            ))
            return violations
        
        # Check for mismatched sequence and timing lengths
        if len(chromosome.satellite_sequence) != len(chromosome.departure_times):
            violations.append(DetailedConstraintViolation(
                constraint_type="route_validity",
                severity="error",
                message=f"Satellite sequence length {len(chromosome.satellite_sequence)} does not match departure times length {len(chromosome.departure_times)}"
            ))
        
        # Check for non-increasing departure times
        for i in range(1, len(chromosome.departure_times)):
            if chromosome.departure_times[i] <= chromosome.departure_times[i-1]:
                violations.append(DetailedConstraintViolation(
                    constraint_type="route_validity",
                    severity="error",
                    message=f"Departure times must be strictly increasing: time[{i-1}]={chromosome.departure_times[i-1]:.0f}s >= time[{i}]={chromosome.departure_times[i]:.0f}s"
                ))
                break
        
        # Check for duplicate consecutive satellites
        for i in range(1, len(chromosome.satellite_sequence)):
            if chromosome.satellite_sequence[i] == chromosome.satellite_sequence[i-1]:
                violations.append(DetailedConstraintViolation(
                    constraint_type="route_validity",
                    severity="warning",
                    message=f"Route contains consecutive duplicate satellite {chromosome.satellite_sequence[i]} at positions {i-1} and {i}",
                    satellite_ids=[chromosome.satellite_sequence[i]]
                ))
        
        return violations
    
    def _generate_warnings(self, chromosome: RouteChromosome, constraints: RouteConstraints, 
                         metrics: Dict[str, float]) -> List[str]:
        """Generate warnings for near-limit conditions."""
        warnings = []
        
        # Delta-v usage warnings
        deltav_utilization = metrics.get('deltav_utilization', 0.0)
        if deltav_utilization > 90.0:
            warnings.append(f"High delta-v usage: {deltav_utilization:.1f}% of budget")
        elif deltav_utilization > 75.0:
            warnings.append(f"Moderate delta-v usage: {deltav_utilization:.1f}% of budget")
        
        # Duration usage warnings
        duration_utilization = metrics.get('duration_utilization', 0.0)
        if duration_utilization > 90.0:
            warnings.append(f"High duration usage: {duration_utilization:.1f}% of time budget")
        elif duration_utilization > 75.0:
            warnings.append(f"Moderate duration usage: {duration_utilization:.1f}% of time budget")
        
        # Hop count warnings
        hop_count = metrics.get('hop_count', 0)
        max_hops = metrics.get('max_hops', 0)
        if hop_count > max_hops * 0.9:
            warnings.append(f"High hop count: {hop_count} of {max_hops} maximum")
        
        return warnings
    
    def create_constraint_result(self, validation_result: ConstraintValidationResult) -> ConstraintResult:
        """Convert detailed validation result to simple ConstraintResult for compatibility."""
        error_violations = [v.message for v in validation_result.violations if v.severity == 'error']
        
        return ConstraintResult(
            is_valid=validation_result.is_valid,
            violations=error_violations,
            deltav_usage=validation_result.detailed_metrics.get('deltav_usage', 0.0),
            deltav_budget=validation_result.detailed_metrics.get('deltav_budget', 0.0),
            duration_usage=validation_result.detailed_metrics.get('mission_duration', 0.0),
            duration_budget=validation_result.detailed_metrics.get('duration_budget', 0.0),
            hop_count=int(validation_result.detailed_metrics.get('hop_count', 0)),
            min_hops=int(validation_result.detailed_metrics.get('min_hops', 0)),
            max_hops=int(validation_result.detailed_metrics.get('max_hops', 0))
        )
    
    def validate_constraints_simple(self, chromosome: RouteChromosome, 
                                  constraints: RouteConstraints,
                                  total_deltav: Optional[float] = None) -> ConstraintResult:
        """Simple constraint validation returning ConstraintResult for backward compatibility."""
        detailed_result = self.validate_route_constraints(chromosome, constraints, total_deltav)
        return self.create_constraint_result(detailed_result)
    
    def filter_forbidden_satellites(self, satellite_ids: List[int], 
                                  forbidden_satellites: List[int]) -> List[int]:
        """Filter out forbidden satellites from a list of satellite IDs.
        
        Args:
            satellite_ids: List of satellite IDs to filter
            forbidden_satellites: List of forbidden satellite IDs
            
        Returns:
            Filtered list with forbidden satellites removed
        """
        if not forbidden_satellites:
            return satellite_ids.copy()
        
        forbidden_set = set(forbidden_satellites)
        return [sat_id for sat_id in satellite_ids if sat_id not in forbidden_set]
    
    def get_valid_satellites(self, constraints: RouteConstraints) -> List[int]:
        """Get list of valid satellites considering all constraints.
        
        Args:
            constraints: Route constraints to apply
            
        Returns:
            List of satellite IDs that satisfy constraints
        """
        valid_satellites = list(self.satellite_ids)
        
        # Filter out forbidden satellites
        valid_satellites = self.filter_forbidden_satellites(valid_satellites, constraints.forbidden_satellites)
        
        return valid_satellites
    
    def is_satellite_valid(self, satellite_id: int, constraints: RouteConstraints) -> bool:
        """Check if a specific satellite is valid under constraints.
        
        Args:
            satellite_id: Satellite ID to check
            constraints: Route constraints to apply
            
        Returns:
            True if satellite is valid, False otherwise
        """
        # Check existence
        if satellite_id not in self.satellite_ids:
            return False
        
        # Check forbidden list
        if satellite_id in constraints.forbidden_satellites:
            return False
        
        return True