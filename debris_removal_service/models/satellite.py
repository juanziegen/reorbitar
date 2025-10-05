"""
Satellite data models and TLE parsing functionality.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
import re


@dataclass
class OrbitalElements:
    """Orbital elements extracted from TLE data."""
    semi_major_axis: float  # km
    eccentricity: float
    inclination: float  # degrees
    raan: float  # Right Ascension of Ascending Node, degrees
    argument_of_perigee: float  # degrees
    mean_anomaly: float  # degrees
    mean_motion: float  # revolutions per day
    epoch: datetime


@dataclass
class Satellite:
    """Satellite data model with TLE parsing and validation."""
    id: str
    name: str
    tle_line1: str
    tle_line2: str
    mass: float  # kg
    material_composition: Dict[str, float]  # material_type -> percentage
    decommission_date: datetime
    orbital_elements: Optional[OrbitalElements] = None
    
    def __post_init__(self):
        """Parse TLE data and extract orbital elements after initialization."""
        if self.orbital_elements is None:
            self.orbital_elements = self._parse_tle()
    
    def _parse_tle(self) -> OrbitalElements:
        """Parse Two-Line Element data and extract orbital parameters."""
        try:
            # Validate TLE format
            if not self._validate_tle_format():
                raise ValueError(f"Invalid TLE format for satellite {self.id}")
            
            # Parse line 1
            line1 = self.tle_line1.strip()
            epoch_year = int(line1[18:20])
            epoch_day = float(line1[20:32])
            
            # Convert 2-digit year to 4-digit year
            if epoch_year < 57:  # Assuming years 00-56 are 2000-2056
                epoch_year += 2000
            else:  # Years 57-99 are 1957-1999
                epoch_year += 1900
            
            # Calculate epoch datetime
            epoch = datetime(epoch_year, 1, 1) + \
                   datetime.timedelta(days=epoch_day - 1)
            
            # Parse line 2
            line2 = self.tle_line2.strip()
            inclination = float(line2[8:16])
            raan = float(line2[17:25])
            eccentricity = float('0.' + line2[26:33])
            argument_of_perigee = float(line2[34:42])
            mean_anomaly = float(line2[43:51])
            mean_motion = float(line2[52:63])
            
            # Calculate semi-major axis from mean motion
            # Using Kepler's third law: n = sqrt(GM/a^3)
            # where n is mean motion in rad/s, GM = 398600.4418 km^3/s^2 for Earth
            GM = 398600.4418  # km^3/s^2
            n_rad_per_sec = mean_motion * 2 * 3.14159265359 / 86400  # convert rev/day to rad/s
            semi_major_axis = (GM / (n_rad_per_sec ** 2)) ** (1/3)
            
            return OrbitalElements(
                semi_major_axis=semi_major_axis,
                eccentricity=eccentricity,
                inclination=inclination,
                raan=raan,
                argument_of_perigee=argument_of_perigee,
                mean_anomaly=mean_anomaly,
                mean_motion=mean_motion,
                epoch=epoch
            )
            
        except (ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse TLE for satellite {self.id}: {str(e)}")
    
    def _validate_tle_format(self) -> bool:
        """Validate TLE format according to standard specifications."""
        # Check line lengths
        if len(self.tle_line1) != 69 or len(self.tle_line2) != 69:
            return False
        
        # Check line numbers
        if not (self.tle_line1[0] == '1' and self.tle_line2[0] == '2'):
            return False
        
        # Check satellite numbers match
        sat_num_1 = self.tle_line1[2:7]
        sat_num_2 = self.tle_line2[2:7]
        if sat_num_1 != sat_num_2:
            return False
        
        # Validate checksum (simple validation)
        for line in [self.tle_line1, self.tle_line2]:
            checksum = 0
            for char in line[:-1]:
                if char.isdigit():
                    checksum += int(char)
                elif char == '-':
                    checksum += 1
            if checksum % 10 != int(line[-1]):
                return False
        
        return True
    
    def get_altitude_km(self) -> tuple[float, float]:
        """Calculate perigee and apogee altitudes in kilometers."""
        if not self.orbital_elements:
            raise ValueError("Orbital elements not available")
        
        earth_radius = 6371.0  # km
        a = self.orbital_elements.semi_major_axis
        e = self.orbital_elements.eccentricity
        
        perigee_altitude = a * (1 - e) - earth_radius
        apogee_altitude = a * (1 + e) - earth_radius
        
        return perigee_altitude, apogee_altitude
    
    def is_valid(self) -> bool:
        """Check if satellite data is valid and complete."""
        try:
            # Check required fields
            if not all([self.id, self.name, self.tle_line1, self.tle_line2]):
                return False
            
            # Check mass is positive
            if self.mass <= 0:
                return False
            
            # Check material composition sums to approximately 100%
            if self.material_composition:
                total_composition = sum(self.material_composition.values())
                if not (0.95 <= total_composition <= 1.05):  # Allow 5% tolerance
                    return False
            
            # Check TLE format
            if not self._validate_tle_format():
                return False
            
            # Check orbital elements are reasonable
            if self.orbital_elements:
                elements = self.orbital_elements
                if not (0 <= elements.eccentricity < 1):
                    return False
                if not (0 <= elements.inclination <= 180):
                    return False
                if elements.semi_major_axis < 6371:  # Below Earth's surface
                    return False
            
            return True
            
        except Exception:
            return False