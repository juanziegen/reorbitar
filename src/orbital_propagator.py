"""
Orbital Propagator Module

This module provides time-based orbital propagation capabilities for the genetic
algorithm route optimizer. It extends the existing orbital mechanics functionality
to handle satellite position calculations at specific times.
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Optional
from .tle_parser import SatelliteData
from .orbital_mechanics import (
    OrbitalElements, 
    StateVector,
    tle_to_orbital_elements,
    orbital_elements_to_state_vector,
    calculate_combined_maneuver_dv,
    optimize_transfer_timing,
    EARTH_MU
)
from .transfer_calculator import calculate_transfer_deltav


@dataclass
class TransferWindow:
    """Optimal transfer parameters between two satellites at a specific time."""
    departure_deltav: float
    arrival_deltav: float
    transfer_time: float
    optimal_departure_time: float
    transfer_efficiency: float  # Ratio of actual to theoretical minimum delta-v


class OrbitalPropagator:
    """Propagates satellite orbits to specific times for accurate delta-v calculations."""
    
    def __init__(self, satellites: List[SatelliteData]):
        """Initialize with satellite orbital data.
        
        Args:
            satellites: List of satellite TLE data
        """
        if not satellites:
            raise ValueError("Satellites list cannot be empty")
        
        self.satellites = {}
        self.orbital_elements = {}
        
        # Store satellite data and convert to orbital elements
        for sat in satellites:
            if sat.catalog_number in self.satellites:
                raise ValueError(f"Duplicate satellite catalog number: {sat.catalog_number}")
            
            self.satellites[sat.catalog_number] = sat
            try:
                self.orbital_elements[sat.catalog_number] = tle_to_orbital_elements(sat)
            except Exception as e:
                raise ValueError(f"Failed to convert TLE data for satellite {sat.catalog_number}: {e}")
    
    def propagate_to_time(self, satellite_id: int, target_time: float) -> OrbitalElements:
        """Propagate satellite orbit to specific time.
        
        Args:
            satellite_id: Catalog number of the satellite
            target_time: Target time in seconds from epoch
            
        Returns:
            OrbitalElements at the target time
            
        Raises:
            ValueError: If satellite_id not found or invalid time
        """
        if satellite_id not in self.satellites:
            raise ValueError(f"Satellite {satellite_id} not found in propagator")
        
        if not math.isfinite(target_time):
            raise ValueError(f"Invalid target time: {target_time}")
        
        # Get original orbital elements
        original_elements = self.orbital_elements[satellite_id]
        
        # For this implementation, we'll use mean motion propagation
        # In a more sophisticated implementation, we would use SGP4/SDP4
        
        # Calculate mean motion (radians per second)
        n = math.sqrt(EARTH_MU / original_elements.semi_major_axis**3)
        
        # Propagate mean anomaly
        propagated_mean_anomaly = original_elements.mean_anomaly + n * target_time
        
        # Normalize mean anomaly to [0, 2π]
        propagated_mean_anomaly = propagated_mean_anomaly % (2 * math.pi)
        
        # Create new orbital elements with propagated mean anomaly
        # For this simplified propagation, other elements remain constant
        propagated_elements = OrbitalElements(
            semi_major_axis=original_elements.semi_major_axis,
            eccentricity=original_elements.eccentricity,
            inclination=original_elements.inclination,
            raan=original_elements.raan,
            arg_perigee=original_elements.arg_perigee,
            mean_anomaly=propagated_mean_anomaly
        )
        
        return propagated_elements
    
    def get_satellite_position(self, satellite_id: int, time: float) -> StateVector:
        """Get satellite position and velocity at specific time.
        
        Args:
            satellite_id: Catalog number of the satellite
            time: Time in seconds from epoch
            
        Returns:
            StateVector with position and velocity at the specified time
            
        Raises:
            ValueError: If satellite_id not found or invalid time
        """
        # Propagate orbital elements to the target time
        propagated_elements = self.propagate_to_time(satellite_id, time)
        
        # Convert to state vector
        state_vector = orbital_elements_to_state_vector(propagated_elements, time)
        
        return state_vector
    
    def calculate_transfer_window(self, source_id: int, target_id: int, 
                                departure_time: float) -> TransferWindow:
        """Calculate optimal transfer parameters between satellites at specific time.
        
        This method integrates with the existing transfer calculator to provide
        comprehensive transfer analysis including plane changes and timing optimization.
        
        Args:
            source_id: Source satellite catalog number
            target_id: Target satellite catalog number
            departure_time: Departure time in seconds from epoch
            
        Returns:
            TransferWindow with optimal transfer parameters
            
        Raises:
            ValueError: If satellite IDs not found or invalid departure time
        """
        if source_id not in self.satellites:
            raise ValueError(f"Source satellite {source_id} not found")
        if target_id not in self.satellites:
            raise ValueError(f"Target satellite {target_id} not found")
        if source_id == target_id:
            raise ValueError("Source and target satellites cannot be the same")
        if not math.isfinite(departure_time):
            raise ValueError(f"Invalid departure time: {departure_time}")
        
        # Get satellite data at departure time
        source_sat = self.satellites[source_id]
        target_sat = self.satellites[target_id]
        
        # Create temporary satellite data with propagated orbital elements
        # This allows us to use the existing transfer calculator with time-specific positions
        source_elements = self.propagate_to_time(source_id, departure_time)
        
        # Estimate transfer time for target propagation
        initial_transfer_time = self._estimate_transfer_time(source_elements, 
                                                           self.orbital_elements[target_id])
        
        # Get target orbital elements at estimated arrival time
        arrival_time = departure_time + initial_transfer_time
        target_elements = self.propagate_to_time(target_id, arrival_time)
        
        # Use existing transfer calculator with propagated elements
        # We need to create temporary satellite data objects with updated orbital elements
        temp_source_sat = self._create_temp_satellite_data(source_sat, source_elements)
        temp_target_sat = self._create_temp_satellite_data(target_sat, target_elements)
        
        try:
            # Calculate comprehensive transfer using existing calculator
            transfer_result = calculate_transfer_deltav(temp_source_sat, temp_target_sat)
            
            # Calculate transfer efficiency
            transfer_efficiency = self._calculate_transfer_efficiency(
                source_elements, target_elements, transfer_result.total_deltav
            )
            
            return TransferWindow(
                departure_deltav=transfer_result.departure_deltav,
                arrival_deltav=transfer_result.arrival_deltav,
                transfer_time=transfer_result.transfer_time,
                optimal_departure_time=departure_time,
                transfer_efficiency=transfer_efficiency
            )
            
        except Exception as e:
            # Fallback to simplified calculation if comprehensive method fails
            departure_deltav, arrival_deltav = self._calculate_hohmann_deltav(
                source_elements.semi_major_axis, 
                target_elements.semi_major_axis
            )
            
            transfer_efficiency = self._calculate_transfer_efficiency(
                source_elements, target_elements, departure_deltav + arrival_deltav
            )
            
            return TransferWindow(
                departure_deltav=departure_deltav,
                arrival_deltav=arrival_deltav,
                transfer_time=initial_transfer_time,
                optimal_departure_time=departure_time,
                transfer_efficiency=transfer_efficiency
            )
    
    def find_optimal_departure_time(self, source_id: int, target_id: int, 
                                  time_window_start: float, time_window_end: float,
                                  time_step: float = 3600.0) -> TransferWindow:
        """Find optimal departure time within a given time window.
        
        Args:
            source_id: Source satellite catalog number
            target_id: Target satellite catalog number
            time_window_start: Start of search window (seconds from epoch)
            time_window_end: End of search window (seconds from epoch)
            time_step: Time step for search (seconds, default 1 hour)
            
        Returns:
            TransferWindow with optimal departure time and parameters
            
        Raises:
            ValueError: If invalid parameters or no valid transfers found
        """
        if time_window_start >= time_window_end:
            raise ValueError("Time window start must be before end")
        if time_step <= 0:
            raise ValueError("Time step must be positive")
        
        best_transfer = None
        best_total_dv = float('inf')
        
        current_time = time_window_start
        while current_time <= time_window_end:
            try:
                transfer_window = self.calculate_transfer_window(source_id, target_id, current_time)
                total_dv = transfer_window.departure_deltav + transfer_window.arrival_deltav
                
                if total_dv < best_total_dv:
                    best_total_dv = total_dv
                    best_transfer = transfer_window
                    
            except Exception:
                # Skip invalid transfer times
                pass
            
            current_time += time_step
        
        if best_transfer is None:
            raise ValueError("No valid transfers found in the specified time window")
        
        return best_transfer
    
    def _estimate_transfer_time(self, source_elements: OrbitalElements, 
                              target_elements: OrbitalElements) -> float:
        """Estimate transfer time for initial calculations.
        
        Args:
            source_elements: Source orbital elements
            target_elements: Target orbital elements
            
        Returns:
            Estimated transfer time in seconds
        """
        # Use Hohmann transfer time as estimate
        transfer_semi_major_axis = (source_elements.semi_major_axis + 
                                  target_elements.semi_major_axis) / 2
        transfer_time = math.pi * math.sqrt(transfer_semi_major_axis**3 / EARTH_MU)
        return transfer_time
    
    def _create_temp_satellite_data(self, original_sat: SatelliteData, 
                                  new_elements: OrbitalElements) -> SatelliteData:
        """Create temporary satellite data with updated orbital elements.
        
        Args:
            original_sat: Original satellite data
            new_elements: New orbital elements (in radians)
            
        Returns:
            SatelliteData with updated orbital parameters
        """
        # Convert radians back to degrees for SatelliteData
        return SatelliteData(
            catalog_number=original_sat.catalog_number,
            name=original_sat.name,
            epoch=original_sat.epoch,
            mean_motion=original_sat.mean_motion,  # Keep original mean motion
            eccentricity=new_elements.eccentricity,
            inclination=math.degrees(new_elements.inclination),
            raan=math.degrees(new_elements.raan),
            arg_perigee=math.degrees(new_elements.arg_perigee),
            mean_anomaly=math.degrees(new_elements.mean_anomaly),
            semi_major_axis=new_elements.semi_major_axis,
            orbital_period=original_sat.orbital_period  # Keep original period
        )
    
    def _calculate_transfer_efficiency(self, source_elements: OrbitalElements,
                                     target_elements: OrbitalElements,
                                     actual_total_dv: float) -> float:
        """Calculate transfer efficiency compared to theoretical minimum.
        
        Args:
            source_elements: Source orbital elements
            target_elements: Target orbital elements
            actual_total_dv: Actual total delta-v required
            
        Returns:
            Transfer efficiency ratio (0-1, where 1 is perfect efficiency)
        """
        # Theoretical minimum is the difference in circular orbital velocities
        v_source = math.sqrt(EARTH_MU / source_elements.semi_major_axis)
        v_target = math.sqrt(EARTH_MU / target_elements.semi_major_axis)
        theoretical_min_dv = abs(v_target - v_source)
        
        if theoretical_min_dv > 0 and actual_total_dv > 0:
            efficiency = theoretical_min_dv / actual_total_dv
            return min(efficiency, 1.0)  # Cap at 1.0
        else:
            return 1.0
    
    def _calculate_hohmann_deltav(self, r1: float, r2: float) -> tuple:
        """Calculate Hohmann transfer delta-v for circular orbits.
        
        Args:
            r1: Initial orbit radius (km)
            r2: Final orbit radius (km)
            
        Returns:
            Tuple of (departure_dv, arrival_dv) in km/s
        """
        # Check for same orbit
        if abs(r1 - r2) < 0.001:  # Within 1 meter
            return (0.0, 0.0)
        
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
    
    def get_satellite_ids(self) -> List[int]:
        """Get list of all satellite catalog numbers in the propagator.
        
        Returns:
            List of satellite catalog numbers
        """
        return list(self.satellites.keys())
    
    def get_satellite_data(self, satellite_id: int) -> SatelliteData:
        """Get satellite data for a given catalog number.
        
        Args:
            satellite_id: Catalog number of the satellite
            
        Returns:
            SatelliteData for the requested satellite
            
        Raises:
            ValueError: If satellite_id not found
        """
        if satellite_id not in self.satellites:
            raise ValueError(f"Satellite {satellite_id} not found")
        
        return self.satellites[satellite_id]