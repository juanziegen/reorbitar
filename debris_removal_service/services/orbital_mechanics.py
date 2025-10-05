"""
Orbital Mechanics Service

This module provides orbital mechanics calculations specifically for the debris removal service,
integrating with the existing orbital mechanics system and providing high-level interfaces
for delta-v calculations, transfer planning, and orbital analysis.
"""

import math
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta
import logging

from ..models.satellite import Satellite, OrbitalElements
from ..models.route import ManeuverDetails

# Import existing orbital mechanics components
import sys
import os
src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from orbital_mechanics import (
        tle_to_orbital_elements,
        orbital_elements_to_state_vector,
        EARTH_MU,
        EARTH_RADIUS
    )
    from orbital_propagator import OrbitalPropagator, TransferWindow
    from tle_parser import SatelliteData
    
    # Check if advanced functions are available
    try:
        from orbital_mechanics import calculate_combined_maneuver_dv, optimize_transfer_timing
        ADVANCED_FUNCTIONS_AVAILABLE = True
    except ImportError:
        ADVANCED_FUNCTIONS_AVAILABLE = False
        
except ImportError as e:
    # Fallback for when genetic algorithm modules are not available
    print(f"Warning: Could not import genetic algorithm modules: {e}")
    EARTH_MU = 398600.4418
    EARTH_RADIUS = 6378.137
    ADVANCED_FUNCTIONS_AVAILABLE = False
    
    # Create dummy classes for type hints
    class TransferWindow:
        def __init__(self):
            pass
    
    class SatelliteData:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)


class OrbitalMechanicsService:
    """
    High-level orbital mechanics service for debris removal operations.
    
    This service provides orbital mechanics calculations tailored for satellite
    debris removal missions, including delta-v calculations, transfer planning,
    and orbital analysis.
    """
    
    def __init__(self):
        """Initialize the orbital mechanics service."""
        self.logger = logging.getLogger(__name__)
        self._propagator_cache: Dict[str, OrbitalPropagator] = {}
    
    def calculate_transfer_deltav(self, from_satellite: Satellite, to_satellite: Satellite,
                                departure_time: Optional[datetime] = None) -> Tuple[float, ManeuverDetails]:
        """
        Calculate delta-v required for transfer between two satellites.
        
        Args:
            from_satellite: Origin satellite
            to_satellite: Destination satellite
            departure_time: Optional departure time (defaults to current time)
            
        Returns:
            Tuple of (total_deltav_ms, maneuver_details)
            
        Raises:
            ValueError: If satellites have invalid orbital data
        """
        try:
            # Validate satellite orbital data
            if not from_satellite.orbital_elements or not to_satellite.orbital_elements:
                raise ValueError("Both satellites must have valid orbital elements")
            
            # Convert to orbital mechanics format
            from_elements = self._convert_to_orbital_elements(from_satellite.orbital_elements)
            to_elements = self._convert_to_orbital_elements(to_satellite.orbital_elements)
            
            # Calculate basic transfer parameters
            deltav_components = self._calculate_hohmann_transfer(from_elements, to_elements)
            
            # Calculate plane change requirements
            plane_change_dv = self._calculate_plane_change_deltav(from_elements, to_elements)
            
            # Combine maneuvers for optimal efficiency
            total_dv = deltav_components['departure'] + deltav_components['arrival'] + plane_change_dv
            
            # Create detailed maneuver information
            maneuver_details = ManeuverDetails(
                departure_burn_dv=deltav_components['departure'],
                arrival_burn_dv=deltav_components['arrival'],
                plane_change_dv=plane_change_dv,
                total_dv=total_dv,
                transfer_type=deltav_components['transfer_type'],
                phase_angle=deltav_components['phase_angle'],
                wait_time=timedelta(seconds=deltav_components['wait_time'])
            )
            
            return total_dv, maneuver_details
            
        except Exception as e:
            self.logger.error(f"Failed to calculate transfer delta-v: {str(e)}")
            raise
    
    def calculate_optimal_transfer_window(self, from_satellite: Satellite, to_satellite: Satellite,
                                       time_window_hours: float = 24.0) -> TransferWindow:
        """
        Find optimal transfer window between two satellites.
        
        Args:
            from_satellite: Origin satellite
            to_satellite: Destination satellite
            time_window_hours: Time window to search for optimal transfer
            
        Returns:
            TransferWindow with optimal transfer parameters
        """
        try:
            # Create satellite data for propagator
            satellites = [
                self._convert_to_satellite_data(from_satellite),
                self._convert_to_satellite_data(to_satellite)
            ]
            
            # Create or get cached propagator
            cache_key = f"{from_satellite.id}_{to_satellite.id}"
            if cache_key not in self._propagator_cache:
                self._propagator_cache[cache_key] = OrbitalPropagator(satellites)
            
            propagator = self._propagator_cache[cache_key]
            
            # Calculate transfer window
            current_time = datetime.utcnow().timestamp()
            window_seconds = time_window_hours * 3600
            
            transfer_window = propagator.calculate_transfer_window(
                int(from_satellite.id),
                int(to_satellite.id),
                current_time,
                window_seconds
            )
            
            return transfer_window
            
        except Exception as e:
            self.logger.error(f"Failed to calculate optimal transfer window: {str(e)}")
            raise
    
    def analyze_orbital_characteristics(self, satellite: Satellite) -> Dict[str, float]:
        """
        Analyze orbital characteristics of a satellite.
        
        Args:
            satellite: Satellite to analyze
            
        Returns:
            Dictionary with orbital analysis results
        """
        try:
            elements = satellite.orbital_elements
            if not elements:
                raise ValueError("Satellite must have valid orbital elements")
            
            # Calculate orbital parameters
            perigee_alt, apogee_alt = satellite.get_altitude_km()
            
            # Calculate orbital period
            period_seconds = 2 * math.pi * math.sqrt(
                (elements.semi_major_axis ** 3) / EARTH_MU
            )
            
            # Calculate orbital velocity at perigee and apogee
            perigee_radius = elements.semi_major_axis * (1 - elements.eccentricity)
            apogee_radius = elements.semi_major_axis * (1 + elements.eccentricity)
            
            perigee_velocity = math.sqrt(EARTH_MU * (2/perigee_radius - 1/elements.semi_major_axis))
            apogee_velocity = math.sqrt(EARTH_MU * (2/apogee_radius - 1/elements.semi_major_axis))
            
            # Calculate inclination category
            inclination_category = self._categorize_inclination(elements.inclination)
            
            return {
                'perigee_altitude_km': perigee_alt,
                'apogee_altitude_km': apogee_alt,
                'orbital_period_minutes': period_seconds / 60,
                'perigee_velocity_kms': perigee_velocity,
                'apogee_velocity_kms': apogee_velocity,
                'inclination_deg': elements.inclination,
                'inclination_category': inclination_category,
                'eccentricity': elements.eccentricity,
                'semi_major_axis_km': elements.semi_major_axis
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze orbital characteristics: {str(e)}")
            raise
    
    def estimate_collection_feasibility(self, satellites: List[Satellite]) -> Dict[str, any]:
        """
        Estimate feasibility of collecting a group of satellites.
        
        Args:
            satellites: List of satellites to analyze for collection
            
        Returns:
            Dictionary with feasibility analysis
        """
        try:
            if len(satellites) < 2:
                return {
                    'feasible': False,
                    'reason': 'Need at least 2 satellites for collection analysis'
                }
            
            # Analyze orbital characteristics of all satellites
            orbital_analyses = [self.analyze_orbital_characteristics(sat) for sat in satellites]
            
            # Calculate altitude spread
            altitudes = [(analysis['perigee_altitude_km'] + analysis['apogee_altitude_km']) / 2 
                        for analysis in orbital_analyses]
            altitude_spread = max(altitudes) - min(altitudes)
            
            # Calculate inclination spread
            inclinations = [analysis['inclination_deg'] for analysis in orbital_analyses]
            inclination_spread = max(inclinations) - min(inclinations)
            
            # Estimate total delta-v requirement
            estimated_deltav = self._estimate_total_collection_deltav(satellites)
            
            # Determine feasibility
            feasible = True
            reasons = []
            
            if altitude_spread > 2000:  # More than 2000 km altitude difference
                feasible = False
                reasons.append(f"Large altitude spread: {altitude_spread:.1f} km")
            
            if inclination_spread > 30:  # More than 30 degrees inclination difference
                feasible = False
                reasons.append(f"Large inclination spread: {inclination_spread:.1f} degrees")
            
            if estimated_deltav > 5000:  # More than 5 km/s total delta-v
                feasible = False
                reasons.append(f"High delta-v requirement: {estimated_deltav:.1f} m/s")
            
            return {
                'feasible': feasible,
                'reasons': reasons if not feasible else ['Collection appears feasible'],
                'estimated_deltav_ms': estimated_deltav,
                'altitude_spread_km': altitude_spread,
                'inclination_spread_deg': inclination_spread,
                'satellite_count': len(satellites)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to estimate collection feasibility: {str(e)}")
            return {
                'feasible': False,
                'reason': f'Analysis failed: {str(e)}'
            }
    
    def _convert_to_orbital_elements(self, elements: OrbitalElements):
        """Convert debris removal service orbital elements to orbital mechanics format."""
        from orbital_mechanics import OrbitalElements as OMOrbitalElements
        
        return OMOrbitalElements(
            semi_major_axis=elements.semi_major_axis,
            eccentricity=elements.eccentricity,
            inclination=math.radians(elements.inclination),
            raan=math.radians(elements.raan),
            arg_perigee=math.radians(elements.argument_of_perigee),
            mean_anomaly=math.radians(elements.mean_anomaly)
        )
    
    def _convert_to_satellite_data(self, satellite: Satellite) -> SatelliteData:
        """Convert debris removal service satellite to orbital mechanics format."""
        elements = satellite.orbital_elements
        
        return SatelliteData(
            catalog_number=int(satellite.id),
            name=satellite.name,
            tle_line1=satellite.tle_line1,
            tle_line2=satellite.tle_line2,
            mass_kg=satellite.mass,
            semi_major_axis=elements.semi_major_axis,
            eccentricity=elements.eccentricity,
            inclination=elements.inclination,
            raan=elements.raan,
            arg_perigee=elements.argument_of_perigee,
            mean_anomaly=elements.mean_anomaly,
            mean_motion=elements.mean_motion,
            epoch=elements.epoch
        )
    
    def _calculate_hohmann_transfer(self, from_elements, to_elements) -> Dict[str, float]:
        """Calculate Hohmann transfer parameters between two orbits."""
        # Simplified Hohmann transfer calculation
        r1 = from_elements.semi_major_axis
        r2 = to_elements.semi_major_axis
        
        # Transfer orbit semi-major axis
        a_transfer = (r1 + r2) / 2
        
        # Velocities
        v1 = math.sqrt(EARTH_MU / r1)
        v2 = math.sqrt(EARTH_MU / r2)
        v_transfer_1 = math.sqrt(EARTH_MU * (2/r1 - 1/a_transfer))
        v_transfer_2 = math.sqrt(EARTH_MU * (2/r2 - 1/a_transfer))
        
        # Delta-v requirements
        departure_dv = abs(v_transfer_1 - v1) * 1000  # Convert to m/s
        arrival_dv = abs(v2 - v_transfer_2) * 1000    # Convert to m/s
        
        # Transfer time (half orbital period of transfer orbit)
        transfer_time = math.pi * math.sqrt((a_transfer ** 3) / EARTH_MU)
        
        # Phase angle calculation (simplified)
        phase_angle = 180 * (1 - math.sqrt((r1/r2)**3))
        
        return {
            'departure': departure_dv,
            'arrival': arrival_dv,
            'transfer_type': 'hohmann',
            'phase_angle': phase_angle,
            'wait_time': transfer_time
        }
    
    def _calculate_plane_change_deltav(self, from_elements, to_elements) -> float:
        """Calculate delta-v required for plane change maneuver."""
        # Inclination difference
        delta_i = abs(to_elements.inclination - from_elements.inclination)
        
        # RAAN difference
        delta_raan = abs(to_elements.raan - from_elements.raan)
        
        # Combined plane change angle
        plane_change_angle = math.acos(
            math.cos(delta_i) * math.cos(delta_raan)
        )
        
        # Velocity at maneuver point (assume at perigee of from orbit)
        r_maneuver = from_elements.semi_major_axis * (1 - from_elements.eccentricity)
        v_maneuver = math.sqrt(EARTH_MU / r_maneuver)
        
        # Plane change delta-v
        plane_change_dv = 2 * v_maneuver * math.sin(plane_change_angle / 2) * 1000  # Convert to m/s
        
        return plane_change_dv
    
    def _categorize_inclination(self, inclination_deg: float) -> str:
        """Categorize orbit based on inclination."""
        if inclination_deg < 10:
            return "equatorial"
        elif inclination_deg < 30:
            return "low_inclination"
        elif inclination_deg < 60:
            return "medium_inclination"
        elif inclination_deg < 120:
            return "high_inclination"
        elif inclination_deg < 150:
            return "retrograde"
        else:
            return "polar"
    
    def _estimate_total_collection_deltav(self, satellites: List[Satellite]) -> float:
        """Estimate total delta-v for collecting all satellites in sequence."""
        if len(satellites) < 2:
            return 0.0
        
        total_deltav = 0.0
        
        # Calculate delta-v for each hop
        for i in range(len(satellites) - 1):
            try:
                deltav, _ = self.calculate_transfer_deltav(satellites[i], satellites[i + 1])
                total_deltav += deltav
            except Exception as e:
                self.logger.warning(f"Failed to calculate delta-v for hop {i}: {str(e)}")
                # Use a conservative estimate
                total_deltav += 1000.0  # 1 km/s per hop
        
        return total_deltav