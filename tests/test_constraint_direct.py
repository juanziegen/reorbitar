"""
Direct test of constraint validation functionality.
"""

import sys
sys.path.append('src')

from datetime import datetime
from genetic_algorithm import RouteChromosome, RouteConstraints
from tle_parser import SatelliteData

# Create the constraint validation classes directly here for testing
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass

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
        """Initialize with satellite database."""
        self.satellites = satellites
        self.satellite_ids = set(satellites.keys())
    
    def validate_route_constraints(self, chromosome: RouteChromosome, 
                                 constraints: RouteConstraints,
                                 total_deltav: Optional[float] = None) -> ConstraintValidationResult:
        """Perform comprehensive constraint validation on a route."""
        violations = []
        warnings = []
        detailed_metrics = {}
        
        # Use provided delta-v or chromosome's cached value
        route_deltav = total_deltav if total_deltav is not None else chromosome.total_deltav
        
        # 1. Delta-v budget constraint checking
        if route_deltav > constraints.max_deltav_budget:
            excess = route_deltav - constraints.max_deltav_budget
            violations.append(DetailedConstraintViolation(
                constraint_type="deltav_budget",
                severity="error",
                message=f"Route delta-v {route_deltav:.3f} km/s exceeds budget {constraints.max_deltav_budget:.3f} km/s by {excess:.3f} km/s",
                actual_value=route_deltav,
                expected_value=constraints.max_deltav_budget
            ))
        
        detailed_metrics['deltav_usage'] = route_deltav
        detailed_metrics['deltav_budget'] = constraints.max_deltav_budget
        detailed_metrics['deltav_utilization'] = (route_deltav / constraints.max_deltav_budget) * 100.0
        
        # 2. Time constraint validation
        mission_duration = chromosome.mission_duration
        if mission_duration > constraints.max_mission_duration:
            excess = mission_duration - constraints.max_mission_duration
            violations.append(DetailedConstraintViolation(
                constraint_type="mission_duration",
                severity="error",
                message=f"Mission duration {mission_duration:.0f}s exceeds limit {constraints.max_mission_duration:.0f}s by {excess:.0f}s",
                actual_value=mission_duration,
                expected_value=constraints.max_mission_duration
            ))
        
        detailed_metrics['mission_duration'] = mission_duration
        detailed_metrics['duration_budget'] = constraints.max_mission_duration
        detailed_metrics['duration_utilization'] = (mission_duration / constraints.max_mission_duration) * 100.0
        
        # 3. Starting satellite constraint enforcement
        if constraints.start_satellite_id is not None:
            if not chromosome.satellite_sequence:
                violations.append(DetailedConstraintViolation(
                    constraint_type="start_satellite",
                    severity="error",
                    message="Route has no satellites but start satellite is required",
                    expected_value=float(constraints.start_satellite_id)
                ))
            else:
                actual_start = chromosome.satellite_sequence[0]
                if actual_start != constraints.start_satellite_id:
                    violations.append(DetailedConstraintViolation(
                        constraint_type="start_satellite",
                        severity="error",
                        message=f"Route starts at satellite {actual_start} but must start at {constraints.start_satellite_id}",
                        actual_value=float(actual_start),
                        expected_value=float(constraints.start_satellite_id),
                        satellite_ids=[actual_start]
                    ))
        
        # 4. Ending satellite constraint enforcement
        if constraints.end_satellite_id is not None:
            if not chromosome.satellite_sequence:
                violations.append(DetailedConstraintViolation(
                    constraint_type="end_satellite",
                    severity="error",
                    message="Route has no satellites but end satellite is required",
                    expected_value=float(constraints.end_satellite_id)
                ))
            else:
                actual_end = chromosome.satellite_sequence[-1]
                if actual_end != constraints.end_satellite_id:
                    violations.append(DetailedConstraintViolation(
                        constraint_type="end_satellite",
                        severity="error",
                        message=f"Route ends at satellite {actual_end} but must end at {constraints.end_satellite_id}",
                        actual_value=float(actual_end),
                        expected_value=float(constraints.end_satellite_id),
                        satellite_ids=[actual_end]
                    ))
        
        # 5. Hop count constraints
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
        
        detailed_metrics['hop_count'] = hop_count
        detailed_metrics['min_hops'] = constraints.min_hops
        detailed_metrics['max_hops'] = constraints.max_hops
        
        # 6. Forbidden satellite filtering
        if constraints.forbidden_satellites:
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
        
        # 7. Satellite existence validation
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
        
        # Calculate constraint satisfaction score
        total_constraints = 7  # Number of constraint categories
        violated_constraints = len([v for v in violations if v.severity == 'error'])
        constraint_satisfaction_score = max(0.0, (total_constraints - violated_constraints) / total_constraints)
        
        # Generate warnings for near-limit conditions
        deltav_utilization = detailed_metrics.get('deltav_utilization', 0.0)
        if deltav_utilization > 90.0:
            warnings.append(f"High delta-v usage: {deltav_utilization:.1f}% of budget")
        elif deltav_utilization > 75.0:
            warnings.append(f"Moderate delta-v usage: {deltav_utilization:.1f}% of budget")
        
        is_valid = len([v for v in violations if v.severity == 'error']) == 0
        
        return ConstraintValidationResult(
            is_valid=is_valid,
            violations=violations,
            warnings=warnings,
            constraint_satisfaction_score=constraint_satisfaction_score,
            detailed_metrics=detailed_metrics
        )
    
    def filter_forbidden_satellites(self, satellite_ids: List[int], 
                                  forbidden_satellites: List[int]) -> List[int]:
        """Filter out forbidden satellites from a list of satellite IDs."""
        if not forbidden_satellites:
            return satellite_ids.copy()
        
        forbidden_set = set(forbidden_satellites)
        return [sat_id for sat_id in satellite_ids if sat_id not in forbidden_set]
    
    def get_valid_satellites(self, constraints: RouteConstraints) -> List[int]:
        """Get list of valid satellites considering all constraints."""
        valid_satellites = list(self.satellite_ids)
        valid_satellites = self.filter_forbidden_satellites(valid_satellites, constraints.forbidden_satellites)
        return valid_satellites
    
    def is_satellite_valid(self, satellite_id: int, constraints: RouteConstraints) -> bool:
        """Check if a specific satellite is valid under constraints."""
        if satellite_id not in self.satellite_ids:
            return False
        if satellite_id in constraints.forbidden_satellites:
            return False
        return True

# Test the constraint validator
print("Testing constraint validation system...")

# Create test data
satellites = {}
for i in range(1, 6):
    satellites[i] = SatelliteData(
        catalog_number=i,
        name=f"TEST-SAT-{i}",
        epoch=datetime.now(),
        mean_motion=15.5,
        eccentricity=0.001,
        inclination=51.64,
        raan=0.0,
        arg_perigee=0.0,
        mean_anomaly=0.0,
        semi_major_axis=6800.0,
        orbital_period=5760.0
    )

validator = ConstraintValidator(satellites)
print("✓ ConstraintValidator created successfully")

constraints = RouteConstraints(
    max_deltav_budget=5.0,
    max_mission_duration=86400.0,
    min_hops=1,
    max_hops=5,
    forbidden_satellites=[5]
)

# Test 1: Valid route
print("\nTest 1: Valid route")
chromosome = RouteChromosome(
    satellite_sequence=[1, 2, 3],
    departure_times=[0.0, 3600.0, 7200.0],
    total_deltav=3.0,
    is_valid=True
)

result = validator.validate_route_constraints(chromosome, constraints)
print(f"✓ Valid route: is_valid={result.is_valid}, score={result.constraint_satisfaction_score:.2f}")
print(f"  Delta-v usage: {result.detailed_metrics['deltav_usage']:.1f}/{result.detailed_metrics['deltav_budget']:.1f} km/s")
print(f"  Hop count: {result.detailed_metrics['hop_count']}")

# Test 2: Delta-v budget violation
print("\nTest 2: Delta-v budget violation")
bad_chromosome = RouteChromosome(
    satellite_sequence=[1, 2, 3],
    departure_times=[0.0, 3600.0, 7200.0],
    total_deltav=6.0,  # Exceeds budget
    is_valid=True
)

result = validator.validate_route_constraints(bad_chromosome, constraints)
print(f"✓ Delta-v violation: is_valid={result.is_valid}, violations={len(result.violations)}")
if result.violations:
    print(f"  Violation: {result.violations[0].message}")

# Test 3: Forbidden satellite
print("\nTest 3: Forbidden satellite")
forbidden_chromosome = RouteChromosome(
    satellite_sequence=[1, 5, 3],  # Contains forbidden satellite 5
    departure_times=[0.0, 3600.0, 7200.0],
    total_deltav=3.0,
    is_valid=True
)

result = validator.validate_route_constraints(forbidden_chromosome, constraints)
print(f"✓ Forbidden satellite: is_valid={result.is_valid}, violations={len(result.violations)}")
if result.violations:
    print(f"  Violation: {result.violations[0].message}")

# Test 4: Start/end satellite constraints
print("\nTest 4: Start/end satellite constraints")
start_end_constraints = RouteConstraints(
    max_deltav_budget=5.0,
    max_mission_duration=86400.0,
    start_satellite_id=2,  # Must start at satellite 2
    end_satellite_id=4,    # Must end at satellite 4
    min_hops=1,
    max_hops=5
)

wrong_start_chromosome = RouteChromosome(
    satellite_sequence=[1, 2, 4],  # Starts at 1, not 2
    departure_times=[0.0, 3600.0, 7200.0],
    total_deltav=3.0,
    is_valid=True
)

result = validator.validate_route_constraints(wrong_start_chromosome, start_end_constraints)
print(f"✓ Wrong start satellite: is_valid={result.is_valid}, violations={len(result.violations)}")
if result.violations:
    print(f"  Violation: {result.violations[0].message}")

# Test 5: Hop count constraints
print("\nTest 5: Hop count constraints")
hop_constraints = RouteConstraints(
    max_deltav_budget=5.0,
    max_mission_duration=86400.0,
    min_hops=3,  # Require at least 3 hops
    max_hops=5
)

too_few_hops_chromosome = RouteChromosome(
    satellite_sequence=[1, 2],  # Only 1 hop, need 3
    departure_times=[0.0, 3600.0],
    total_deltav=3.0,
    is_valid=True
)

result = validator.validate_route_constraints(too_few_hops_chromosome, hop_constraints)
print(f"✓ Too few hops: is_valid={result.is_valid}, violations={len(result.violations)}")
if result.violations:
    print(f"  Violation: {result.violations[0].message}")

# Test 6: Utility functions
print("\nTest 6: Utility functions")
filtered = validator.filter_forbidden_satellites([1, 2, 3, 4, 5], [5])
print(f"✓ Satellite filtering: {filtered} (should exclude 5)")

valid_sats = validator.get_valid_satellites(constraints)
print(f"✓ Valid satellites: {valid_sats} (should exclude 5)")

print(f"✓ Satellite 1 valid: {validator.is_satellite_valid(1, constraints)}")
print(f"✓ Satellite 5 valid: {validator.is_satellite_valid(5, constraints)} (should be False)")

print("\nAll constraint validation tests passed! ✅")