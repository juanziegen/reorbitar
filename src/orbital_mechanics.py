"""
Orbital Mechanics Module

This module contains core orbital mechanics calculations and coordinate
transformations for satellite transfer calculations.
"""

from dataclasses import dataclass
from typing import Tuple
import math


# Constants
EARTH_MU = 398600.4418  # km³/s² - Earth gravitational parameter
EARTH_RADIUS = 6378.137  # km - Earth radius


@dataclass
class OrbitalElements:
    """Container for orbital parameters."""
    semi_major_axis: float
    eccentricity: float
    inclination: float
    raan: float
    arg_perigee: float
    mean_anomaly: float


@dataclass
class StateVector:
    """Position and velocity vectors."""
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]


@dataclass
class TransferOrbit:
    """Transfer trajectory parameters."""
    apogee: float
    perigee: float
    transfer_time: float
    departure_dv: float
    arrival_dv: float


def tle_to_orbital_elements(tle_data) -> OrbitalElements:
    """Convert TLE data to orbital elements."""
    if tle_data is None:
        raise ValueError("TLE data cannot be None")
    
    try:
        # Validate input data exists
        required_attrs = ['semi_major_axis', 'eccentricity', 'inclination', 'raan', 'arg_perigee', 'mean_anomaly']
        for attr in required_attrs:
            if not hasattr(tle_data, attr):
                raise ValueError(f"TLE data missing required attribute: {attr}")
            value = getattr(tle_data, attr)
            if value is None:
                raise ValueError(f"TLE data attribute {attr} cannot be None")
        
        # Additional validation for orbital mechanics calculations
        if tle_data.semi_major_axis <= EARTH_RADIUS:
            raise ValueError(f"Semi-major axis {tle_data.semi_major_axis:.1f} km is below Earth's surface")
        
        return OrbitalElements(
            semi_major_axis=tle_data.semi_major_axis,
            eccentricity=tle_data.eccentricity,
            inclination=math.radians(tle_data.inclination),  # Convert to radians
            raan=math.radians(tle_data.raan),  # Convert to radians
            arg_perigee=math.radians(tle_data.arg_perigee),  # Convert to radians
            mean_anomaly=math.radians(tle_data.mean_anomaly)  # Convert to radians
        )
    except AttributeError as e:
        raise ValueError(f"Invalid TLE data structure: {e}")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Error converting TLE data to orbital elements: {e}")


def orbital_elements_to_state_vector(elements: OrbitalElements, time: float) -> StateVector:
    """Convert orbital elements to state vector at given time."""
    # For this implementation, we'll use the mean anomaly directly
    # In a more complete implementation, we would propagate the orbit to the given time
    
    # Calculate true anomaly from mean anomaly (simplified for circular/near-circular orbits)
    true_anomaly = _mean_to_true_anomaly(elements.mean_anomaly, elements.eccentricity)
    
    # Calculate position and velocity in orbital plane
    a = elements.semi_major_axis
    e = elements.eccentricity
    nu = true_anomaly
    
    # Distance from focus
    r = a * (1 - e * e) / (1 + e * math.cos(nu))
    
    # Position in orbital plane (perifocal coordinates)
    x_pqw = r * math.cos(nu)
    y_pqw = r * math.sin(nu)
    z_pqw = 0.0
    
    # Velocity in orbital plane
    h = math.sqrt(EARTH_MU * a * (1 - e * e))  # Specific angular momentum
    vx_pqw = -(EARTH_MU / h) * math.sin(nu)
    vy_pqw = (EARTH_MU / h) * (e + math.cos(nu))
    vz_pqw = 0.0
    
    # Transform from perifocal to Earth-centered inertial (ECI) coordinates
    # Rotation matrices for RAAN, inclination, and argument of perigee
    cos_raan = math.cos(elements.raan)
    sin_raan = math.sin(elements.raan)
    cos_inc = math.cos(elements.inclination)
    sin_inc = math.sin(elements.inclination)
    cos_argp = math.cos(elements.arg_perigee)
    sin_argp = math.sin(elements.arg_perigee)
    
    # Combined rotation matrix elements
    r11 = cos_raan * cos_argp - sin_raan * sin_argp * cos_inc
    r12 = -cos_raan * sin_argp - sin_raan * cos_argp * cos_inc
    r13 = sin_raan * sin_inc
    
    r21 = sin_raan * cos_argp + cos_raan * sin_argp * cos_inc
    r22 = -sin_raan * sin_argp + cos_raan * cos_argp * cos_inc
    r23 = -cos_raan * sin_inc
    
    r31 = sin_argp * sin_inc
    r32 = cos_argp * sin_inc
    r33 = cos_inc
    
    # Transform position
    x_eci = r11 * x_pqw + r12 * y_pqw + r13 * z_pqw
    y_eci = r21 * x_pqw + r22 * y_pqw + r23 * z_pqw
    z_eci = r31 * x_pqw + r32 * y_pqw + r33 * z_pqw
    
    # Transform velocity
    vx_eci = r11 * vx_pqw + r12 * vy_pqw + r13 * vz_pqw
    vy_eci = r21 * vx_pqw + r22 * vy_pqw + r23 * vz_pqw
    vz_eci = r31 * vx_pqw + r32 * vy_pqw + r33 * vz_pqw
    
    return StateVector(
        position=(x_eci, y_eci, z_eci),
        velocity=(vx_eci, vy_eci, vz_eci)
    )


def _mean_to_true_anomaly(mean_anomaly: float, eccentricity: float) -> float:
    """Convert mean anomaly to true anomaly using iterative solution of Kepler's equation."""
    # For circular orbits, mean anomaly equals true anomaly
    if eccentricity < 1e-6:
        return mean_anomaly
    
    # Solve Kepler's equation: M = E - e*sin(E) for eccentric anomaly E
    # Using Newton-Raphson iteration
    E = mean_anomaly  # Initial guess
    for _ in range(10):  # Usually converges in 3-4 iterations
        f = E - eccentricity * math.sin(E) - mean_anomaly
        df = 1 - eccentricity * math.cos(E)
        E_new = E - f / df
        if abs(E_new - E) < 1e-12:
            break
        E = E_new
    
    # Convert eccentric anomaly to true anomaly
    true_anomaly = 2 * math.atan2(
        math.sqrt(1 + eccentricity) * math.sin(E / 2),
        math.sqrt(1 - eccentricity) * math.cos(E / 2)
    )
    
    return true_anomaly


def calculate_hohmann_transfer(source: OrbitalElements, target: OrbitalElements) -> TransferOrbit:
    """Calculate Hohmann transfer orbit between two circular orbits."""
    # Validate input parameters
    if source is None or target is None:
        raise ValueError("Source and target orbital elements cannot be None")
    
    # Validate orbital elements
    _validate_orbital_elements_for_calculations(source, "source")
    _validate_orbital_elements_for_calculations(target, "target")
    
    # For Hohmann transfer, we assume circular orbits at the semi-major axis
    r1 = source.semi_major_axis  # Initial orbit radius
    r2 = target.semi_major_axis  # Final orbit radius
    
    # Check for same orbit (no transfer needed)
    if abs(r1 - r2) < 0.1:  # Within 100m
        # Even for identical orbits, calculate the theoretical transfer time
        # (half the orbital period of the common orbit)
        common_radius = (r1 + r2) / 2
        transfer_period = 2 * math.pi * math.sqrt(common_radius**3 / EARTH_MU)
        transfer_time = transfer_period / 2
        
        return TransferOrbit(
            apogee=max(r1, r2),
            perigee=min(r1, r2),
            transfer_time=transfer_time,
            departure_dv=0.0,
            arrival_dv=0.0
        )
    
    # Transfer orbit semi-major axis
    a_transfer = (r1 + r2) / 2
    
    # Transfer orbit apogee and perigee
    if r1 < r2:  # Ascending transfer
        perigee = r1
        apogee = r2
    else:  # Descending transfer
        perigee = r2
        apogee = r1
    
    # Velocities in circular orbits
    v1 = math.sqrt(EARTH_MU / r1)  # Initial circular velocity
    v2 = math.sqrt(EARTH_MU / r2)  # Final circular velocity
    
    # Transfer orbit velocities at perigee and apogee
    v_transfer_perigee = math.sqrt(EARTH_MU * (2/perigee - 1/a_transfer))
    v_transfer_apogee = math.sqrt(EARTH_MU * (2/apogee - 1/a_transfer))
    
    # Calculate delta-v requirements
    if r1 < r2:  # Ascending transfer (r1 is perigee, r2 is apogee)
        departure_dv = v_transfer_perigee - v1  # Accelerate at perigee
        arrival_dv = v2 - v_transfer_apogee     # Accelerate at apogee
    else:  # Descending transfer (r1 is apogee, r2 is perigee)
        departure_dv = v1 - v_transfer_apogee   # Decelerate at apogee
        arrival_dv = v_transfer_perigee - v2    # Decelerate at perigee
    
    # Transfer time (half the orbital period of transfer orbit)
    transfer_period = 2 * math.pi * math.sqrt(a_transfer**3 / EARTH_MU)
    transfer_time = transfer_period / 2  # Half orbit for Hohmann transfer
    
    return TransferOrbit(
        apogee=apogee,
        perigee=perigee,
        transfer_time=transfer_time,
        departure_dv=abs(departure_dv),
        arrival_dv=abs(arrival_dv)
    )


def calculate_plane_change_dv(inclination_change: float, velocity: float) -> float:
    """Calculate delta-v for pure plane change maneuver.
    
    Args:
        inclination_change: Change in inclination in radians
        velocity: Orbital velocity at the maneuver point in km/s
        
    Returns:
        Delta-v required for plane change in km/s
    """
    # For a simple plane change, delta-v = 2 * v * sin(Δi/2)
    # where Δi is the inclination change and v is the orbital velocity
    if abs(inclination_change) < 1e-6:
        return 0.0
    
    delta_v = 2 * velocity * math.sin(abs(inclination_change) / 2)
    return delta_v


def calculate_inclination_change_dv(source: OrbitalElements, target: OrbitalElements, 
                                   maneuver_radius: float = None) -> float:
    """Calculate delta-v for inclination change between two orbits.
    
    Args:
        source: Source orbital elements
        target: Target orbital elements  
        maneuver_radius: Radius at which to perform maneuver (km). If None, uses higher altitude.
        
    Returns:
        Delta-v required for inclination change in km/s
    """
    inclination_change = target.inclination - source.inclination
    
    if abs(inclination_change) < 1e-6:
        return 0.0
    
    # If no specific radius given, perform at higher altitude for efficiency
    if maneuver_radius is None:
        maneuver_radius = max(source.semi_major_axis, target.semi_major_axis)
    
    # Calculate velocity at maneuver point
    velocity = math.sqrt(EARTH_MU / maneuver_radius)
    
    return calculate_plane_change_dv(inclination_change, velocity)


@dataclass
class CombinedManeuverResult:
    """Result of combined altitude and plane change maneuver calculation."""
    hohmann_departure_dv: float
    hohmann_arrival_dv: float
    plane_change_dv: float
    total_dv: float
    maneuver_strategy: str
    optimal_maneuver_radius: float


def calculate_combined_maneuver_dv(source: OrbitalElements, target: OrbitalElements) -> CombinedManeuverResult:
    """Calculate delta-v for combined altitude and plane change maneuvers.
    
    This function determines the most efficient strategy for combining Hohmann transfers
    with plane changes, considering different timing options.
    
    Args:
        source: Source orbital elements
        target: Target orbital elements
        
    Returns:
        CombinedManeuverResult with detailed breakdown of maneuver requirements
    """
    # Calculate Hohmann transfer
    hohmann_transfer = calculate_hohmann_transfer(source, target)
    
    # Calculate inclination change
    inclination_change = target.inclination - source.inclination
    
    if abs(inclination_change) < 1e-6:
        # No plane change needed
        return CombinedManeuverResult(
            hohmann_departure_dv=hohmann_transfer.departure_dv,
            hohmann_arrival_dv=hohmann_transfer.arrival_dv,
            plane_change_dv=0.0,
            total_dv=hohmann_transfer.departure_dv + hohmann_transfer.arrival_dv,
            maneuver_strategy="hohmann_only",
            optimal_maneuver_radius=0.0
        )
    
    # Evaluate different strategies for combining maneuvers
    strategies = _evaluate_combined_maneuver_strategies(source, target, hohmann_transfer)
    
    # Select the strategy with minimum total delta-v
    best_strategy = min(strategies, key=lambda s: s.total_dv)
    
    return best_strategy


def _evaluate_combined_maneuver_strategies(source: OrbitalElements, target: OrbitalElements, 
                                         hohmann_transfer: TransferOrbit) -> list:
    """Evaluate different strategies for combining altitude and plane changes.
    
    Returns:
        List of CombinedManeuverResult objects for different strategies
    """
    strategies = []
    inclination_change = target.inclination - source.inclination
    
    # Strategy 1: Plane change at source altitude (departure)
    v_source = math.sqrt(EARTH_MU / source.semi_major_axis)
    plane_change_at_source = calculate_plane_change_dv(inclination_change, v_source)
    strategies.append(CombinedManeuverResult(
        hohmann_departure_dv=hohmann_transfer.departure_dv,
        hohmann_arrival_dv=hohmann_transfer.arrival_dv,
        plane_change_dv=plane_change_at_source,
        total_dv=hohmann_transfer.departure_dv + hohmann_transfer.arrival_dv + plane_change_at_source,
        maneuver_strategy="plane_change_at_departure",
        optimal_maneuver_radius=source.semi_major_axis
    ))
    
    # Strategy 2: Plane change at target altitude (arrival)
    v_target = math.sqrt(EARTH_MU / target.semi_major_axis)
    plane_change_at_target = calculate_plane_change_dv(inclination_change, v_target)
    strategies.append(CombinedManeuverResult(
        hohmann_departure_dv=hohmann_transfer.departure_dv,
        hohmann_arrival_dv=hohmann_transfer.arrival_dv,
        plane_change_dv=plane_change_at_target,
        total_dv=hohmann_transfer.departure_dv + hohmann_transfer.arrival_dv + plane_change_at_target,
        maneuver_strategy="plane_change_at_arrival",
        optimal_maneuver_radius=target.semi_major_axis
    ))
    
    # Strategy 3: Plane change at apogee of transfer orbit (most efficient for large changes)
    transfer_apogee = hohmann_transfer.apogee
    v_apogee = math.sqrt(EARTH_MU * (2/transfer_apogee - 2/(source.semi_major_axis + target.semi_major_axis)))
    plane_change_at_apogee = calculate_plane_change_dv(inclination_change, v_apogee)
    strategies.append(CombinedManeuverResult(
        hohmann_departure_dv=hohmann_transfer.departure_dv,
        hohmann_arrival_dv=hohmann_transfer.arrival_dv,
        plane_change_dv=plane_change_at_apogee,
        total_dv=hohmann_transfer.departure_dv + hohmann_transfer.arrival_dv + plane_change_at_apogee,
        maneuver_strategy="plane_change_at_apogee",
        optimal_maneuver_radius=transfer_apogee
    ))
    
    return strategies


@dataclass
class PlaneChangeOptimization:
    """Result of plane change timing optimization."""
    optimal_strategy: str
    optimal_radius: float
    delta_v_savings: float
    efficiency_rating: str
    recommendations: list


def optimize_plane_change_timing(source: OrbitalElements, target: OrbitalElements) -> PlaneChangeOptimization:
    """Determine optimal timing and strategy for plane change maneuver.
    
    Args:
        source: Source orbital elements
        target: Target orbital elements
        
    Returns:
        PlaneChangeOptimization with detailed analysis and recommendations
    """
    inclination_change = abs(target.inclination - source.inclination)
    inclination_change_deg = math.degrees(inclination_change)
    
    # Calculate delta-v for different strategies
    combined_result = calculate_combined_maneuver_dv(source, target)
    
    # Determine efficiency and provide recommendations
    recommendations = []
    
    if inclination_change_deg < 1.0:
        efficiency_rating = "excellent"
        optimal_strategy = "minimal_plane_change"
        recommendations.append("Inclination change is minimal - negligible impact on mission")
        
    elif inclination_change_deg < 5.0:
        efficiency_rating = "very_good"
        optimal_strategy = "small_plane_change_at_optimal_point"
        recommendations.append("Small inclination change - perform at higher altitude for efficiency")
        
    elif inclination_change_deg < 15.0:
        efficiency_rating = "good"
        optimal_strategy = "moderate_plane_change_at_apogee"
        recommendations.append("Moderate inclination change - strongly recommend performing at apogee")
        recommendations.append("Consider mission timing to minimize plane change requirements")
        
    elif inclination_change_deg < 30.0:
        efficiency_rating = "challenging"
        optimal_strategy = "significant_plane_change_expensive"
        recommendations.append("Large inclination change - very expensive maneuver")
        recommendations.append("Consider alternative mission profiles or launch timing")
        recommendations.append("Evaluate if bi-propellant system is adequate")
        
    else:
        efficiency_rating = "prohibitive"
        optimal_strategy = "major_plane_change_prohibitive"
        recommendations.append("Extreme inclination change - likely mission-prohibitive")
        recommendations.append("Strongly recommend alternative approach")
        recommendations.append("Consider dedicated launch or different target selection")
    
    # Calculate potential savings from optimal timing
    # Compare worst case (at lowest altitude) vs best case (at highest altitude)
    r_min = min(source.semi_major_axis, target.semi_major_axis)
    r_max = max(source.semi_major_axis, target.semi_major_axis)
    
    v_min_altitude = math.sqrt(EARTH_MU / r_min)
    v_max_altitude = math.sqrt(EARTH_MU / r_max)
    
    dv_worst = calculate_plane_change_dv(inclination_change, v_min_altitude)
    dv_best = calculate_plane_change_dv(inclination_change, v_max_altitude)
    
    delta_v_savings = dv_worst - dv_best
    
    return PlaneChangeOptimization(
        optimal_strategy=combined_result.maneuver_strategy,
        optimal_radius=combined_result.optimal_maneuver_radius,
        delta_v_savings=delta_v_savings,
        efficiency_rating=efficiency_rating,
        recommendations=recommendations
    )


def calculate_comprehensive_plane_change(source: OrbitalElements, target: OrbitalElements) -> dict:
    """Calculate comprehensive plane change analysis including all strategies and optimizations.
    
    Args:
        source: Source orbital elements
        target: Target orbital elements
        
    Returns:
        Dictionary containing complete plane change analysis
    """
    inclination_change = target.inclination - source.inclination
    inclination_change_deg = math.degrees(abs(inclination_change))
    
    # Get combined maneuver analysis
    combined_result = calculate_combined_maneuver_dv(source, target)
    
    # Get timing optimization
    timing_optimization = optimize_plane_change_timing(source, target)
    
    # Calculate individual components for reference
    pure_inclination_dv = calculate_inclination_change_dv(source, target)
    hohmann_transfer = calculate_hohmann_transfer(source, target)
    
    # Determine if plane change is significant
    is_significant_plane_change = inclination_change_deg > 5.0
    
    return {
        'inclination_change_deg': inclination_change_deg,
        'inclination_change_rad': abs(inclination_change),
        'is_significant_plane_change': is_significant_plane_change,
        'pure_plane_change_dv': pure_inclination_dv,
        'combined_maneuver': combined_result,
        'timing_optimization': timing_optimization,
        'hohmann_only_dv': hohmann_transfer.departure_dv + hohmann_transfer.arrival_dv,
        'plane_change_penalty': combined_result.total_dv - (hohmann_transfer.departure_dv + hohmann_transfer.arrival_dv)
    }


def optimize_transfer_timing(source: OrbitalElements, target: OrbitalElements) -> float:
    """Calculate optimal phase angle for transfer initiation.
    
    Returns:
        Phase angle in radians for optimal transfer timing
    """
    # Validate inputs
    if source is None or target is None:
        raise ValueError("Source and target orbital elements cannot be None")
    
    _validate_orbital_elements_for_calculations(source, "source")
    _validate_orbital_elements_for_calculations(target, "target")
    
    try:
        # Calculate the angular velocities
        n1 = math.sqrt(EARTH_MU / source.semi_major_axis**3)  # Source orbit angular velocity
        n2 = math.sqrt(EARTH_MU / target.semi_major_axis**3)  # Target orbit angular velocity
        
        # Calculate transfer orbit parameters
        a_transfer = (source.semi_major_axis + target.semi_major_axis) / 2
        transfer_time = math.pi * math.sqrt(a_transfer**3 / EARTH_MU)  # Half period
        
        # Calculate how much the target satellite moves during transfer
        target_angular_displacement = n2 * transfer_time
        
        # The required phase angle is π minus the target's angular displacement
        # This ensures the spacecraft arrives when the target is at the rendezvous point
        optimal_phase_angle = math.pi - target_angular_displacement
        
        # Normalize to [0, 2π]
        while optimal_phase_angle < 0:
            optimal_phase_angle += 2 * math.pi
        while optimal_phase_angle > 2 * math.pi:
            optimal_phase_angle -= 2 * math.pi
        
        return optimal_phase_angle
        
    except (ValueError, ArithmeticError, OverflowError) as e:
        raise ValueError(f"Error calculating optimal transfer timing: {e}")


def _validate_orbital_elements_for_calculations(elements: OrbitalElements, name: str) -> None:
    """Validate orbital elements for use in calculations."""
    if elements is None:
        raise ValueError(f"{name} orbital elements cannot be None")
    
    # Check for required attributes
    required_attrs = ['semi_major_axis', 'eccentricity', 'inclination']
    for attr in required_attrs:
        if not hasattr(elements, attr):
            raise ValueError(f"{name} orbital elements missing required attribute: {attr}")
        value = getattr(elements, attr)
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            raise ValueError(f"{name} orbital elements attribute {attr} is invalid: {value}")
    
    # Validate semi-major axis
    if elements.semi_major_axis <= EARTH_RADIUS:
        raise ValueError(f"{name} semi-major axis {elements.semi_major_axis:.1f} km is below Earth's surface")
    if elements.semi_major_axis > 100000:  # 100,000 km is beyond reasonable LEO/MEO range
        raise ValueError(f"{name} semi-major axis {elements.semi_major_axis:.1f} km is unreasonably large")
    
    # Validate eccentricity
    if elements.eccentricity < 0 or elements.eccentricity >= 1:
        raise ValueError(f"{name} eccentricity {elements.eccentricity:.6f} is invalid (must be 0 ≤ e < 1)")
    
    # Validate inclination (should be in radians for calculations)
    if elements.inclination < 0 or elements.inclination > math.pi:
        raise ValueError(f"{name} inclination {elements.inclination:.6f} rad is invalid (must be 0 ≤ i ≤ π)")
    
    # Check for physically reasonable perigee
    perigee = elements.semi_major_axis * (1 - elements.eccentricity)
    if perigee < EARTH_RADIUS + 100:  # 100 km minimum altitude
        raise ValueError(f"{name} orbit perigee {perigee - EARTH_RADIUS:.1f} km altitude is too low for stable orbit")