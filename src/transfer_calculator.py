"""
Transfer Calculator Module

This module provides high-level delta-v calculations and transfer optimization
for satellite-to-satellite transfers.
"""

from dataclasses import dataclass
from typing import List, Tuple
import math
from .tle_parser import SatelliteData
from .orbital_mechanics import (
    OrbitalElements, 
    tle_to_orbital_elements, 
    calculate_hohmann_transfer,
    calculate_combined_maneuver_dv,
    optimize_transfer_timing,
    EARTH_MU,
    EARTH_RADIUS
)


@dataclass
class DeltaVBreakdown:
    """Detailed delta-v components."""
    departure_dv: float
    arrival_dv: float
    plane_change_dv: float
    total_dv: float


@dataclass
class TransferResult:
    """Complete transfer solution."""
    source_satellite: SatelliteData
    target_satellite: SatelliteData
    total_deltav: float
    departure_deltav: float
    arrival_deltav: float
    plane_change_deltav: float
    transfer_time: float
    transfer_orbit_apogee: float
    transfer_orbit_perigee: float
    complexity_assessment: str
    warnings: List[str]


def calculate_transfer_deltav(source_sat: SatelliteData, target_sat: SatelliteData) -> TransferResult:
    """Calculate complete transfer delta-v requirements.
    
    Args:
        source_sat: Source satellite TLE data
        target_sat: Target satellite TLE data
        
    Returns:
        TransferResult with complete transfer analysis
    """
    # Validate inputs
    if source_sat is None:
        raise ValueError("Source satellite data cannot be None")
    if target_sat is None:
        raise ValueError("Target satellite data cannot be None")
    
    # Check for same satellite
    if hasattr(source_sat, 'catalog_number') and hasattr(target_sat, 'catalog_number'):
        if source_sat.catalog_number == target_sat.catalog_number:
            raise ValueError("Source and target satellites cannot be the same")
    
    try:
        # Convert TLE data to orbital elements
        source_elements = tle_to_orbital_elements(source_sat)
        target_elements = tle_to_orbital_elements(target_sat)
        
        # Calculate Hohmann transfer
        hohmann_transfer = calculate_hohmann_transfer(source_elements, target_elements)
        
        # Calculate combined maneuver (includes plane change optimization)
        combined_maneuver = calculate_combined_maneuver_dv(source_elements, target_elements)
        
    except Exception as e:
        raise ValueError(f"Error in orbital calculations: {e}")
    
    # Use the combined maneuver results for the most efficient transfer
    total_deltav = combined_maneuver.total_dv
    departure_deltav = combined_maneuver.hohmann_departure_dv
    arrival_deltav = combined_maneuver.hohmann_arrival_dv
    plane_change_deltav = combined_maneuver.plane_change_dv
    
    # Transfer orbit parameters from Hohmann calculation
    transfer_orbit_apogee = hohmann_transfer.apogee
    transfer_orbit_perigee = hohmann_transfer.perigee
    transfer_time = hohmann_transfer.transfer_time
    
    # Assess transfer complexity
    complexity_assessment = assess_transfer_complexity(source_elements, target_elements)
    
    # Generate warnings for challenging transfers
    warnings = _generate_transfer_warnings(source_elements, target_elements, combined_maneuver)
    
    return TransferResult(
        source_satellite=source_sat,
        target_satellite=target_sat,
        total_deltav=total_deltav,
        departure_deltav=departure_deltav,
        arrival_deltav=arrival_deltav,
        plane_change_deltav=plane_change_deltav,
        transfer_time=transfer_time,
        transfer_orbit_apogee=transfer_orbit_apogee,
        transfer_orbit_perigee=transfer_orbit_perigee,
        complexity_assessment=complexity_assessment,
        warnings=warnings
    )


def calculate_hohmann_deltav(r1: float, r2: float) -> Tuple[float, float]:
    """Calculate Hohmann transfer delta-v for circular orbits.
    
    Args:
        r1: Initial orbit radius (km)
        r2: Final orbit radius (km)
        
    Returns:
        Tuple of (departure_dv, arrival_dv) in km/s
    """
    # Validate inputs
    if r1 is None or r2 is None:
        raise ValueError("Orbit radii cannot be None")
    if not isinstance(r1, (int, float)) or not isinstance(r2, (int, float)):
        raise TypeError("Orbit radii must be numeric values")
    if r1 <= EARTH_RADIUS or r2 <= EARTH_RADIUS:
        raise ValueError(f"Orbit radii must be above Earth's surface ({EARTH_RADIUS:.1f} km)")
    if r1 <= 0 or r2 <= 0:
        raise ValueError("Orbit radii must be positive")
    if not math.isfinite(r1) or not math.isfinite(r2):
        raise ValueError("Orbit radii must be finite values")
    
    # Check for same orbit
    if abs(r1 - r2) < 0.001:  # Within 1 meter
        return (0.0, 0.0)
    
    try:
        # Transfer orbit semi-major axis
        a_transfer = (r1 + r2) / 2
        
        # Velocities in circular orbits
        v1 = math.sqrt(EARTH_MU / r1)  # Initial circular velocity
        v2 = math.sqrt(EARTH_MU / r2)  # Final circular velocity
        
        # Transfer orbit velocities
        if r1 < r2:  # Ascending transfer
            v_transfer_perigee = math.sqrt(EARTH_MU * (2/r1 - 1/a_transfer))
            v_transfer_apogee = math.sqrt(EARTH_MU * (2/r2 - 1/a_transfer))
            departure_dv = v_transfer_perigee - v1
            arrival_dv = v2 - v_transfer_apogee
        else:  # Descending transfer
            v_transfer_apogee = math.sqrt(EARTH_MU * (2/r1 - 1/a_transfer))
            v_transfer_perigee = math.sqrt(EARTH_MU * (2/r2 - 1/a_transfer))
            departure_dv = v1 - v_transfer_apogee
            arrival_dv = v_transfer_perigee - v2
    
        return (abs(departure_dv), abs(arrival_dv))
        
    except (ValueError, ArithmeticError, OverflowError) as e:
        raise ValueError(f"Error calculating Hohmann transfer delta-v: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error in Hohmann transfer calculation: {e}")


def calculate_inclination_change_deltav(source: OrbitalElements, target: OrbitalElements) -> float:
    """Calculate inclination change delta-v.
    
    Args:
        source: Source orbital elements
        target: Target orbital elements
        
    Returns:
        Delta-v required for inclination change in km/s
    """
    inclination_change = abs(target.inclination - source.inclination)
    
    if inclination_change < 1e-6:
        return 0.0
    
    # Perform plane change at higher altitude for efficiency
    maneuver_radius = max(source.semi_major_axis, target.semi_major_axis)
    velocity = math.sqrt(EARTH_MU / maneuver_radius)
    
    # Delta-v for plane change: 2 * v * sin(Δi/2)
    delta_v = 2 * velocity * math.sin(inclination_change / 2)
    
    return delta_v


def find_optimal_transfer_window(source: OrbitalElements, target: OrbitalElements) -> float:
    """Find optimal transfer window using existing optimize_transfer_timing.
    
    Args:
        source: Source orbital elements
        target: Target orbital elements
        
    Returns:
        Optimal phase angle in radians for transfer initiation
    """
    return optimize_transfer_timing(source, target)


def calculate_phase_angle(source: OrbitalElements, target: OrbitalElements, 
                         current_time: float = 0.0) -> float:
    """Calculate current phase angle between two satellites.
    
    Args:
        source: Source orbital elements
        target: Target orbital elements
        current_time: Current time offset from epoch (seconds)
        
    Returns:
        Phase angle in radians
    """
    # Calculate mean motions (angular velocities)
    n_source = math.sqrt(EARTH_MU / source.semi_major_axis**3)
    n_target = math.sqrt(EARTH_MU / target.semi_major_axis**3)
    
    # Calculate current mean anomalies
    M_source = source.mean_anomaly + n_source * current_time
    M_target = target.mean_anomaly + n_target * current_time
    
    # For simplicity, assume circular orbits and use mean anomaly as position
    # Phase angle is the angular separation
    phase_angle = M_target - M_source
    
    # Normalize to [0, 2π]
    while phase_angle < 0:
        phase_angle += 2 * math.pi
    while phase_angle > 2 * math.pi:
        phase_angle -= 2 * math.pi
    
    return phase_angle


def calculate_synodic_period(source: OrbitalElements, target: OrbitalElements) -> float:
    """Calculate synodic period between two satellites.
    
    The synodic period is the time between successive alignments of the satellites.
    
    Args:
        source: Source orbital elements
        target: Target orbital elements
        
    Returns:
        Synodic period in seconds
    """
    # Calculate orbital periods
    T_source = 2 * math.pi * math.sqrt(source.semi_major_axis**3 / EARTH_MU)
    T_target = 2 * math.pi * math.sqrt(target.semi_major_axis**3 / EARTH_MU)
    
    # Synodic period formula: 1/T_syn = |1/T1 - 1/T2|
    if abs(T_source - T_target) < 1e-6:
        return float('inf')  # Same period - no relative motion
    
    synodic_period = 1 / abs(1/T_source - 1/T_target)
    
    return synodic_period


def calculate_wait_time_for_optimal_phase(source: OrbitalElements, target: OrbitalElements,
                                        current_phase_angle: float = None) -> float:
    """Calculate wait time until optimal transfer phase angle.
    
    Args:
        source: Source orbital elements
        target: Target orbital elements
        current_phase_angle: Current phase angle in radians (if None, calculated)
        
    Returns:
        Wait time in seconds until optimal transfer window
    """
    # Get optimal phase angle
    optimal_phase = find_optimal_transfer_window(source, target)
    
    # Get current phase angle if not provided
    if current_phase_angle is None:
        current_phase_angle = calculate_phase_angle(source, target)
    
    # Calculate angular difference to optimal phase
    phase_diff = optimal_phase - current_phase_angle
    
    # Normalize to [0, 2π]
    while phase_diff < 0:
        phase_diff += 2 * math.pi
    while phase_diff > 2 * math.pi:
        phase_diff -= 2 * math.pi
    
    # Calculate relative angular velocity
    n_source = math.sqrt(EARTH_MU / source.semi_major_axis**3)
    n_target = math.sqrt(EARTH_MU / target.semi_major_axis**3)
    relative_angular_velocity = abs(n_source - n_target)
    
    if relative_angular_velocity < 1e-12:
        return float('inf')  # No relative motion
    
    # Time to reach optimal phase
    wait_time = phase_diff / relative_angular_velocity
    
    return wait_time


def _generate_transfer_warnings(source: OrbitalElements, target: OrbitalElements, 
                              combined_maneuver) -> List[str]:
    """Generate warnings for challenging transfers.
    
    Args:
        source: Source orbital elements
        target: Target orbital elements
        combined_maneuver: Combined maneuver result
        
    Returns:
        List of warning messages
    """
    warnings = []
    
    # Calculate altitude difference
    altitude_diff = abs(target.semi_major_axis - source.semi_major_axis)
    altitude_diff_km = altitude_diff
    
    # Calculate inclination change in degrees
    inclination_change_deg = math.degrees(abs(target.inclination - source.inclination))
    
    # High delta-v warning
    if combined_maneuver.total_dv > 2.0:
        warnings.append(f"High total delta-v requirement: {combined_maneuver.total_dv:.2f} km/s")
    
    # Large altitude change warning
    if altitude_diff_km > 1000:
        warnings.append(f"Large altitude change: {altitude_diff_km:.0f} km difference")
    
    # Significant inclination change warning
    if inclination_change_deg > 10.0:
        warnings.append(f"Significant inclination change: {inclination_change_deg:.1f}° - very expensive")
    elif inclination_change_deg > 5.0:
        warnings.append(f"Moderate inclination change: {inclination_change_deg:.1f}° - expensive")
    
    # Extreme inclination change warning
    if inclination_change_deg > 30.0:
        warnings.append("Extreme inclination change - may be mission-prohibitive")
    
    # Plane change efficiency warning
    if combined_maneuver.plane_change_dv > combined_maneuver.hohmann_departure_dv + combined_maneuver.hohmann_arrival_dv:
        warnings.append("Plane change dominates transfer cost - consider alternative mission profile")
    
    return warnings


def assess_transfer_complexity(source: OrbitalElements, target: OrbitalElements) -> str:
    """Assess transfer complexity and difficulty rating.
    
    Args:
        source: Source orbital elements
        target: Target orbital elements
        
    Returns:
        String describing transfer complexity level
    """
    # Calculate key parameters for complexity assessment
    altitude_diff = abs(target.semi_major_axis - source.semi_major_axis)
    inclination_change_deg = math.degrees(abs(target.inclination - source.inclination))
    
    # Calculate approximate total delta-v for quick assessment
    combined_maneuver = calculate_combined_maneuver_dv(source, target)
    total_dv = combined_maneuver.total_dv
    
    # Complexity scoring based on multiple factors
    complexity_score = 0
    
    # Delta-v contribution (0-4 points)
    if total_dv > 3.0:
        complexity_score += 4
    elif total_dv > 2.0:
        complexity_score += 3
    elif total_dv > 1.0:
        complexity_score += 2
    elif total_dv > 0.5:
        complexity_score += 1
    
    # Inclination change contribution (0-4 points)
    if inclination_change_deg > 30.0:
        complexity_score += 4
    elif inclination_change_deg > 15.0:
        complexity_score += 3
    elif inclination_change_deg > 5.0:
        complexity_score += 2
    elif inclination_change_deg > 1.0:
        complexity_score += 1
    
    # Altitude change contribution (0-2 points)
    if altitude_diff > 2000:
        complexity_score += 2
    elif altitude_diff > 500:
        complexity_score += 1
    
    # Determine complexity level based on total score
    if complexity_score <= 1:
        return "Simple"
    elif complexity_score <= 3:
        return "Moderate"
    elif complexity_score <= 6:
        return "Complex"
    elif complexity_score <= 8:
        return "Very Complex"
    else:
        return "Extremely Complex"