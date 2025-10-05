"""
TLE Parser Module

This module handles parsing and validation of Two-Line Element (TLE) data
from satellite files.
"""

import math
import warnings
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

# Set up logging for TLE parsing errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SatelliteData:
    """Data structure for satellite orbital elements."""
    catalog_number: int
    name: str
    epoch: datetime
    mean_motion: float
    eccentricity: float
    inclination: float
    raan: float
    arg_perigee: float
    mean_anomaly: float
    semi_major_axis: float
    orbital_period: float


class TLEParser:
    """Main parsing class for TLE data."""
    
    # Earth's gravitational parameter (km³/s²)
    MU_EARTH = 398600.4418
    
    def parse_tle_file(self, filepath: str) -> List[SatelliteData]:
        """Parse TLE file and return list of satellite data."""
        satellites = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"TLE file not found: {filepath}. Please ensure the file exists in the current directory.")
        except PermissionError:
            raise PermissionError(f"Permission denied accessing TLE file: {filepath}. Check file permissions.")
        except UnicodeDecodeError as e:
            raise ValueError(f"Unable to decode TLE file {filepath}. File may be corrupted or in wrong encoding: {e}")
        except IOError as e:
            raise IOError(f"Error reading TLE file {filepath}: {e}")
        
        if not lines:
            raise ValueError(f"TLE file {filepath} is empty or contains no readable data.")
        
        # Process lines, looking for TLE pairs
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Look for line 1 of TLE
            if line.startswith('1 '):
                # Look for the corresponding line 2
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line:  # Skip empty lines
                        j += 1
                        continue
                    
                    if next_line.startswith('2 '):
                        # Found a TLE pair
                        try:
                            satellite = self.parse_tle_entry(line, next_line)
                            if satellite:
                                satellites.append(satellite)
                        except ValueError as e:
                            logger.warning(f"Skipping invalid TLE entry (lines {i+1}, {j+1}): {e}")
                            warnings.warn(f"Skipping invalid TLE entry: {e}")
                        except Exception as e:
                            logger.error(f"Unexpected error parsing TLE entry (lines {i+1}, {j+1}): {e}")
                            warnings.warn(f"Unexpected error parsing TLE entry: {e}")
                        i = j + 1  # Move past this TLE pair
                        break
                    else:
                        # Found a non-line-2, this line 1 doesn't have a pair
                        logger.info(f"Line 1 at line {i+1} has no matching line 2")
                        i = j  # Continue from this position
                        break
                else:
                    # Reached end of file without finding line 2
                    logger.info(f"Line 1 at line {i+1} has no matching line 2 (end of file)")
                    break
            else:
                # Not a line 1, skip it
                if line:  # Only log if line isn't empty
                    logger.debug(f"Skipping non-TLE line {i+1}: {line[:20]}...")
                i += 1
        
        return satellites
    
    def parse_tle_entry(self, line1: str, line2: str) -> Optional[SatelliteData]:
        """Parse individual TLE entry from two lines."""
        # Validate line lengths
        if len(line1) < 69 or len(line2) < 69:
            raise ValueError(f"TLE lines too short (line1: {len(line1)}, line2: {len(line2)})")
        
        # Validate checksums
        if not self.validate_tle_checksum(line1):
            raise ValueError("Invalid checksum in line 1")
        if not self.validate_tle_checksum(line2):
            raise ValueError("Invalid checksum in line 2")
        
        try:
            # Extract catalog number from both lines and verify they match
            catalog_num_1 = int(line1[2:7])
            catalog_num_2 = int(line2[2:7])
            
            if catalog_num_1 != catalog_num_2:
                raise ValueError(f"Catalog numbers don't match: {catalog_num_1} vs {catalog_num_2}")
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError("Invalid catalog number format")
            raise
        
        try:
            # Parse Line 1
            catalog_number = catalog_num_1
            classification = line1[7]
            intl_designator = line1[9:17].strip()
            
            # Parse epoch
            epoch_year = int(line1[18:20])
            # Convert 2-digit year to 4-digit year
            if epoch_year < 57:  # Assuming years 00-56 are 2000-2056, 57-99 are 1957-1999
                epoch_year += 2000
            else:
                epoch_year += 1900
            
            epoch_day = float(line1[20:32])
            if epoch_day < 1 or epoch_day > 366:
                raise ValueError(f"Invalid epoch day: {epoch_day} (must be between 1 and 366)")
            
            epoch = self._epoch_from_year_day(epoch_year, epoch_day)
            
            # Parse Line 2
            inclination = float(line2[8:16])  # degrees
            raan = float(line2[17:25])  # degrees (Right Ascension of Ascending Node)
            eccentricity = float('0.' + line2[26:33])  # decimal point assumed
            arg_perigee = float(line2[34:42])  # degrees (Argument of Perigee)
            mean_anomaly = float(line2[43:51])  # degrees
            mean_motion = float(line2[52:63])  # revolutions per day
            
            # Comprehensive orbital parameter validation
            self._validate_orbital_parameters(inclination, eccentricity, mean_motion, raan, arg_perigee, mean_anomaly)
                
        except ValueError as e:
            if "could not convert" in str(e) or "invalid literal" in str(e):
                raise ValueError("Invalid numeric format in TLE data - check for corrupted or malformed TLE lines")
            raise
        except IndexError as e:
            raise ValueError(f"TLE line format error - insufficient data in TLE lines: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error parsing TLE data: {e}")
        
        # Calculate derived parameters
        semi_major_axis = self._calculate_semi_major_axis(mean_motion)
        orbital_period = self._calculate_orbital_period(semi_major_axis)
        
        # Create satellite name from catalog number (since names aren't in this file)
        name = f"SATELLITE-{catalog_number:05d}"
        
        return SatelliteData(
            catalog_number=catalog_number,
            name=name,
            epoch=epoch,
            mean_motion=mean_motion,
            eccentricity=eccentricity,
            inclination=inclination,
            raan=raan,
            arg_perigee=arg_perigee,
            mean_anomaly=mean_anomaly,
            semi_major_axis=semi_major_axis,
            orbital_period=orbital_period
        )
    
    def validate_tle_checksum(self, line: str) -> bool:
        """Validate TLE checksum."""
        if len(line) < 69:
            return False
        
        # Calculate checksum for first 68 characters
        checksum = 0
        for char in line[:68]:
            if char.isdigit():
                checksum += int(char)
            elif char == '-':
                checksum += 1
        
        # Get the last digit as modulo 10
        calculated_checksum = checksum % 10
        expected_checksum = int(line[68])
        
        return calculated_checksum == expected_checksum
    
    def _epoch_from_year_day(self, year: int, day_of_year: float) -> datetime:
        """Convert year and fractional day of year to datetime."""
        # Start with January 1st of the given year
        base_date = datetime(year, 1, 1)
        
        # Add the fractional days (subtract 1 because day 1 is January 1st)
        days_to_add = day_of_year - 1
        full_days = int(days_to_add)
        fractional_day = days_to_add - full_days
        
        # Add full days
        result_date = base_date + timedelta(days=full_days)
        
        # Add fractional day as hours, minutes, seconds
        seconds_in_fractional_day = fractional_day * 24 * 3600
        result_date += timedelta(seconds=seconds_in_fractional_day)
        
        return result_date
    
    def _calculate_semi_major_axis(self, mean_motion: float) -> float:
        """Calculate semi-major axis from mean motion."""
        # mean_motion is in revolutions per day, convert to radians per second
        n = mean_motion * 2 * math.pi / 86400  # rad/s
        
        # From Kepler's third law: n² = μ/a³, so a = (μ/n²)^(1/3)
        a = (self.MU_EARTH / (n * n)) ** (1/3)
        
        return a  # km
    
    def _calculate_orbital_period(self, semi_major_axis: float) -> float:
        """Calculate orbital period from semi-major axis."""
        if semi_major_axis <= 0:
            raise ValueError(f"Invalid semi-major axis: {semi_major_axis} (must be positive)")
        
        # T = 2π * sqrt(a³/μ)
        period_seconds = 2 * math.pi * math.sqrt((semi_major_axis ** 3) / self.MU_EARTH)
        
        # Convert to minutes
        return period_seconds / 60
    
    def _validate_orbital_parameters(self, inclination: float, eccentricity: float, 
                                   mean_motion: float, raan: float, arg_perigee: float, 
                                   mean_anomaly: float) -> None:
        """Validate orbital parameters for physical reasonableness."""
        # Inclination validation
        if inclination < 0 or inclination > 180:
            raise ValueError(f"Invalid inclination: {inclination}° (must be between 0° and 180°)")
        
        # Eccentricity validation
        if eccentricity < 0:
            raise ValueError(f"Invalid eccentricity: {eccentricity} (must be non-negative)")
        if eccentricity >= 1:
            raise ValueError(f"Invalid eccentricity: {eccentricity} (must be less than 1 for elliptical orbits)")
        if eccentricity > 0.9:
            logger.warning(f"Very high eccentricity: {eccentricity:.3f} - orbit may be highly elliptical")
        
        # Mean motion validation
        if mean_motion <= 0:
            raise ValueError(f"Invalid mean motion: {mean_motion} (must be positive)")
        if mean_motion > 20:  # More than 20 revolutions per day is extremely high
            raise ValueError(f"Invalid mean motion: {mean_motion} rev/day (unrealistically high for LEO)")
        if mean_motion < 0.5:  # Less than 0.5 revolutions per day is very low for LEO
            logger.warning(f"Very low mean motion: {mean_motion:.3f} rev/day - may not be LEO orbit")
        
        # RAAN validation
        if raan < 0 or raan >= 360:
            raise ValueError(f"Invalid RAAN: {raan}° (must be between 0° and 360°)")
        
        # Argument of perigee validation
        if arg_perigee < 0 or arg_perigee >= 360:
            raise ValueError(f"Invalid argument of perigee: {arg_perigee}° (must be between 0° and 360°)")
        
        # Mean anomaly validation
        if mean_anomaly < 0 or mean_anomaly >= 360:
            raise ValueError(f"Invalid mean anomaly: {mean_anomaly}° (must be between 0° and 360°)")
        
        # Calculate and validate derived parameters
        try:
            semi_major_axis = self._calculate_semi_major_axis(mean_motion)
            
            # Check for reasonable LEO altitudes
            earth_radius = 6378.137  # km
            altitude = semi_major_axis - earth_radius
            
            if altitude < 150:  # Below typical atmospheric drag limit
                raise ValueError(f"Invalid altitude: {altitude:.1f} km (too low for stable orbit)")
            if altitude > 2000:  # Above typical LEO range
                logger.warning(f"High altitude: {altitude:.1f} km - may not be LEO orbit")
            
            # Check for physically reasonable perigee with eccentricity
            perigee = semi_major_axis * (1 - eccentricity)
            perigee_altitude = perigee - earth_radius
            
            if perigee_altitude < 100:  # Below atmospheric limit
                raise ValueError(f"Invalid perigee altitude: {perigee_altitude:.1f} km (orbit would decay rapidly)")
                
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Error validating derived orbital parameters: {e}")