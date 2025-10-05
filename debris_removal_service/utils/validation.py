"""
Satellite data validation utilities.
"""

import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from ..models.satellite import Satellite


class SatelliteDataValidator:
    """Comprehensive satellite data validation."""
    
    # TLE format validation patterns
    TLE_LINE1_PATTERN = re.compile(r'^1 \d{5}[A-Z] \d{2}\d{3}[A-Z]{3} \d{5}\.\d{8} [+-]\.\d{8} [+-]\d{5}[+-]\d [+-]\d{5}[+-]\d \d \d{4}$')
    TLE_LINE2_PATTERN = re.compile(r'^2 \d{5} \d{3}\.\d{4} \d{3}\.\d{4} \d{7} \d{3}\.\d{4} \d{3}\.\d{4} \d{2}\.\d{8}\d{5}$')
    
    @staticmethod
    def validate_satellite(satellite: Satellite) -> Tuple[bool, List[str]]:
        """
        Comprehensive validation of satellite data.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Basic field validation
        if not satellite.id or not satellite.id.strip():
            errors.append("Satellite ID is required")
        
        if not satellite.name or not satellite.name.strip():
            errors.append("Satellite name is required")
        
        if satellite.mass <= 0:
            errors.append("Satellite mass must be positive")
        
        # TLE validation
        tle_errors = SatelliteDataValidator._validate_tle(satellite.tle_line1, satellite.tle_line2)
        errors.extend(tle_errors)
        
        # Material composition validation
        composition_errors = SatelliteDataValidator._validate_material_composition(satellite.material_composition)
        errors.extend(composition_errors)
        
        # Orbital elements validation
        if satellite.orbital_elements:
            orbital_errors = SatelliteDataValidator._validate_orbital_elements(satellite.orbital_elements)
            errors.extend(orbital_errors)
        
        # Decommission date validation
        if satellite.decommission_date > datetime.utcnow():
            # This is actually valid - future decommission dates are allowed
            pass
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _validate_tle(line1: str, line2: str) -> List[str]:
        """Validate TLE format and checksums."""
        errors = []
        
        # Check line lengths
        if len(line1) != 69:
            errors.append(f"TLE line 1 must be 69 characters, got {len(line1)}")
        
        if len(line2) != 69:
            errors.append(f"TLE line 2 must be 69 characters, got {len(line2)}")
        
        # Check line numbers
        if not line1.startswith('1 '):
            errors.append("TLE line 1 must start with '1 '")
        
        if not line2.startswith('2 '):
            errors.append("TLE line 2 must start with '2 '")
        
        # Check satellite numbers match
        if len(line1) >= 7 and len(line2) >= 7:
            sat_num_1 = line1[2:7]
            sat_num_2 = line2[2:7]
            if sat_num_1 != sat_num_2:
                errors.append("Satellite numbers in TLE lines don't match")
        
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
                    errors.append(f"TLE line {i} checksum invalid: expected {expected_checksum}, calculated {checksum % 10}")
        
        return errors
    
    @staticmethod
    def _validate_material_composition(composition: Dict[str, float]) -> List[str]:
        """Validate material composition data."""
        errors = []
        
        if not composition:
            errors.append("Material composition is required")
            return errors
        
        # Check percentages are valid
        for material, percentage in composition.items():
            if not (0.0 <= percentage <= 1.0):
                errors.append(f"Material percentage for {material} must be between 0 and 1, got {percentage}")
        
        # Check total composition
        total = sum(composition.values())
        if not (0.95 <= total <= 1.05):  # Allow 5% tolerance
            errors.append(f"Total material composition must sum to approximately 1.0, got {total}")
        
        # Check for reasonable material types
        valid_materials = {
            'aluminum', 'steel', 'titanium', 'carbon_fiber', 'copper', 'silicon',
            'plastic', 'glass', 'electronics', 'fuel_residue', 'other'
        }
        
        for material in composition.keys():
            if material.lower() not in valid_materials:
                # This is a warning, not an error
                pass
        
        return errors
    
    @staticmethod
    def _validate_orbital_elements(elements) -> List[str]:
        """Validate orbital elements for physical reasonableness."""
        errors = []
        
        # Eccentricity must be between 0 and 1 for elliptical orbits
        if not (0.0 <= elements.eccentricity < 1.0):
            errors.append(f"Eccentricity must be between 0 and 1, got {elements.eccentricity}")
        
        # Inclination must be between 0 and 180 degrees
        if not (0.0 <= elements.inclination <= 180.0):
            errors.append(f"Inclination must be between 0 and 180 degrees, got {elements.inclination}")
        
        # RAAN must be between 0 and 360 degrees
        if not (0.0 <= elements.raan < 360.0):
            errors.append(f"RAAN must be between 0 and 360 degrees, got {elements.raan}")
        
        # Argument of perigee must be between 0 and 360 degrees
        if not (0.0 <= elements.argument_of_perigee < 360.0):
            errors.append(f"Argument of perigee must be between 0 and 360 degrees, got {elements.argument_of_perigee}")
        
        # Mean anomaly must be between 0 and 360 degrees
        if not (0.0 <= elements.mean_anomaly < 360.0):
            errors.append(f"Mean anomaly must be between 0 and 360 degrees, got {elements.mean_anomaly}")
        
        # Semi-major axis must be reasonable for Earth orbit
        earth_radius = 6371.0  # km
        if elements.semi_major_axis < earth_radius:
            errors.append(f"Semi-major axis {elements.semi_major_axis} km is below Earth's surface")
        
        if elements.semi_major_axis > 100000.0:  # 100,000 km is very high
            errors.append(f"Semi-major axis {elements.semi_major_axis} km is unusually high")
        
        # Mean motion must be positive
        if elements.mean_motion <= 0:
            errors.append(f"Mean motion must be positive, got {elements.mean_motion}")
        
        # Check altitude ranges
        try:
            perigee = elements.semi_major_axis * (1 - elements.eccentricity) - earth_radius
            apogee = elements.semi_major_axis * (1 + elements.eccentricity) - earth_radius
            
            if perigee < 100:  # Below typical LEO
                errors.append(f"Perigee altitude {perigee:.1f} km is very low")
            
            if apogee > 35786:  # Above GEO
                errors.append(f"Apogee altitude {apogee:.1f} km is above GEO")
                
        except Exception as e:
            errors.append(f"Error calculating orbital altitudes: {str(e)}")
        
        return errors
    
    @staticmethod
    def validate_satellite_list(satellites: List[Satellite]) -> Tuple[List[Satellite], List[Tuple[str, List[str]]]]:
        """
        Validate a list of satellites and return valid ones plus errors.
        
        Returns:
            Tuple of (valid_satellites, list_of_(satellite_id, errors))
        """
        valid_satellites = []
        validation_errors = []
        
        for satellite in satellites:
            is_valid, errors = SatelliteDataValidator.validate_satellite(satellite)
            
            if is_valid:
                valid_satellites.append(satellite)
            else:
                validation_errors.append((satellite.id, errors))
        
        return valid_satellites, validation_errors
    
    @staticmethod
    def get_validation_summary(satellites: List[Satellite]) -> Dict[str, any]:
        """Get a summary of validation results for a list of satellites."""
        valid_satellites, validation_errors = SatelliteDataValidator.validate_satellite_list(satellites)
        
        return {
            'total_satellites': len(satellites),
            'valid_satellites': len(valid_satellites),
            'invalid_satellites': len(validation_errors),
            'validation_rate': len(valid_satellites) / len(satellites) if satellites else 0.0,
            'error_summary': validation_errors
        }