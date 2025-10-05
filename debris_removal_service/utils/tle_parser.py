"""
TLE (Two-Line Element) parsing utilities.
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from ..models.satellite import Satellite, OrbitalElements


class TLEParser:
    """Parser for Two-Line Element (TLE) data format."""
    
    @staticmethod
    def parse_tle_file(file_path: str) -> List[Satellite]:
        """
        Parse a TLE file and return a list of Satellite objects.
        
        Args:
            file_path: Path to the TLE file
            
        Returns:
            List of Satellite objects
        """
        satellites = []
        
        try:
            with open(file_path, 'r') as file:
                lines = [line.strip() for line in file.readlines()]
            
            # Process lines in groups of 3 (name, line1, line2)
            for i in range(0, len(lines), 3):
                if i + 2 < len(lines):
                    name = lines[i]
                    line1 = lines[i + 1]
                    line2 = lines[i + 2]
                    
                    # Skip empty lines or comments
                    if not name or name.startswith('#'):
                        continue
                    
                    try:
                        satellite = TLEParser.parse_tle_lines(name, line1, line2)
                        if satellite:
                            satellites.append(satellite)
                    except Exception as e:
                        print(f"Warning: Failed to parse satellite {name}: {str(e)}")
                        continue
        
        except FileNotFoundError:
            raise FileNotFoundError(f"TLE file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error reading TLE file: {str(e)}")
        
        return satellites
    
    @staticmethod
    def parse_tle_lines(name: str, line1: str, line2: str, 
                       default_mass: float = 100.0,
                       default_composition: Optional[Dict[str, float]] = None) -> Optional[Satellite]:
        """
        Parse individual TLE lines into a Satellite object.
        
        Args:
            name: Satellite name
            line1: TLE line 1
            line2: TLE line 2
            default_mass: Default mass if not specified
            default_composition: Default material composition
            
        Returns:
            Satellite object or None if parsing fails
        """
        try:
            # Extract satellite ID from TLE
            satellite_id = line1[2:7].strip()
            
            # Use default composition if not provided
            if default_composition is None:
                default_composition = {
                    'aluminum': 0.4,
                    'steel': 0.2,
                    'electronics': 0.15,
                    'plastic': 0.1,
                    'other': 0.15
                }
            
            # Create satellite with default decommission date (current time)
            satellite = Satellite(
                id=satellite_id,
                name=name.strip(),
                tle_line1=line1,
                tle_line2=line2,
                mass=default_mass,
                material_composition=default_composition,
                decommission_date=datetime.utcnow()
            )
            
            return satellite
            
        except Exception as e:
            print(f"Error parsing TLE for {name}: {str(e)}")
            return None
    
    @staticmethod
    def extract_orbital_info(line1: str, line2: str) -> Dict[str, float]:
        """
        Extract orbital information from TLE lines without creating a full Satellite object.
        
        Args:
            line1: TLE line 1
            line2: TLE line 2
            
        Returns:
            Dictionary with orbital parameters
        """
        try:
            # Parse epoch from line 1
            epoch_year = int(line1[18:20])
            epoch_day = float(line1[20:32])
            
            # Convert 2-digit year to 4-digit year
            if epoch_year < 57:
                epoch_year += 2000
            else:
                epoch_year += 1900
            
            # Parse orbital elements from line 2
            inclination = float(line2[8:16])
            raan = float(line2[17:25])
            eccentricity = float('0.' + line2[26:33])
            argument_of_perigee = float(line2[34:42])
            mean_anomaly = float(line2[43:51])
            mean_motion = float(line2[52:63])
            
            # Calculate semi-major axis
            GM = 398600.4418  # km^3/s^2
            n_rad_per_sec = mean_motion * 2 * 3.14159265359 / 86400
            semi_major_axis = (GM / (n_rad_per_sec ** 2)) ** (1/3)
            
            # Calculate altitudes
            earth_radius = 6371.0  # km
            perigee_altitude = semi_major_axis * (1 - eccentricity) - earth_radius
            apogee_altitude = semi_major_axis * (1 + eccentricity) - earth_radius
            
            return {
                'epoch_year': epoch_year,
                'epoch_day': epoch_day,
                'inclination': inclination,
                'raan': raan,
                'eccentricity': eccentricity,
                'argument_of_perigee': argument_of_perigee,
                'mean_anomaly': mean_anomaly,
                'mean_motion': mean_motion,
                'semi_major_axis': semi_major_axis,
                'perigee_altitude': perigee_altitude,
                'apogee_altitude': apogee_altitude
            }
            
        except Exception as e:
            raise ValueError(f"Failed to extract orbital info: {str(e)}")
    
    @staticmethod
    def validate_tle_format(line1: str, line2: str) -> Tuple[bool, List[str]]:
        """
        Validate TLE format without parsing.
        
        Args:
            line1: TLE line 1
            line2: TLE line 2
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check line lengths
        if len(line1) != 69:
            errors.append(f"Line 1 must be 69 characters, got {len(line1)}")
        
        if len(line2) != 69:
            errors.append(f"Line 2 must be 69 characters, got {len(line2)}")
        
        # Check line numbers
        if not line1.startswith('1 '):
            errors.append("Line 1 must start with '1 '")
        
        if not line2.startswith('2 '):
            errors.append("Line 2 must start with '2 '")
        
        # Check satellite numbers match
        if len(line1) >= 7 and len(line2) >= 7:
            sat_num_1 = line1[2:7]
            sat_num_2 = line2[2:7]
            if sat_num_1 != sat_num_2:
                errors.append("Satellite numbers don't match between lines")
        
        # Validate checksums
        for i, line in enumerate([line1, line2], 1):
            if len(line) == 69:
                checksum = 0
                for char in line[:-1]:
                    if char.isdigit():
                        checksum += int(char)
                    elif char == '-':
                        checksum += 1
                
                expected_checksum = int(line[-1])
                if checksum % 10 != expected_checksum:
                    errors.append(f"Line {i} checksum invalid")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def get_satellite_catalog_number(line1: str) -> str:
        """Extract satellite catalog number from TLE line 1."""
        return line1[2:7].strip()
    
    @staticmethod
    def get_classification(line1: str) -> str:
        """Extract classification from TLE line 1."""
        return line1[7]
    
    @staticmethod
    def get_launch_info(line1: str) -> Tuple[str, str, str]:
        """Extract launch year, launch number, and piece from TLE line 1."""
        launch_year = line1[9:11]
        launch_number = line1[11:14]
        piece = line1[14:17]
        return launch_year, launch_number, piece
    
    @staticmethod
    def format_tle_output(satellite: Satellite) -> str:
        """Format a Satellite object back to TLE format."""
        return f"{satellite.name}\n{satellite.tle_line1}\n{satellite.tle_line2}"