"""
Satellite Data Validator Module

Provides comprehensive validation for satellite data quality and completeness
to ensure genetic algorithm optimization can run reliably.
"""

import math
import warnings
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime, timedelta

from .tle_parser import SatelliteData


class SatelliteDataValidator:
    """Validates satellite data for genetic algorithm compatibility."""
    
    def __init__(self):
        """Initialize validator with default thresholds."""
        # Validation thresholds
        self.min_leo_altitude = 150.0  # km
        self.max_leo_altitude = 2000.0  # km
        self.max_eccentricity = 0.25  # Reasonable for LEO
        self.min_mean_motion = 0.5  # rev/day
        self.max_mean_motion = 20.0  # rev/day
        self.max_epoch_age_days = 30  # Days since epoch
        
        # Earth radius for calculations
        self.earth_radius = 6378.137  # km
    
    def validate_satellite_dataset(self, satellites: List[SatelliteData]) -> Dict[str, any]:
        """
        Comprehensive validation of entire satellite dataset.
        
        Args:
            satellites: List of satellite data to validate
            
        Returns:
            Dictionary containing validation results and statistics
        """
        if not satellites:
            return {
                'valid': False,
                'error': 'Empty satellite dataset',
                'total_satellites': 0,
                'valid_satellites': 0,
                'warnings': [],
                'errors': [],
                'statistics': {},
                'genetic_algorithm_ready': False
            }
        
        valid_satellites = []
        warnings_list = []
        errors_list = []
        
        # Individual satellite validation
        for i, satellite in enumerate(satellites):
            try:
                validation_result = self.validate_individual_satellite(satellite)
                
                if validation_result['valid']:
                    valid_satellites.append(satellite)
                else:
                    errors_list.append(f"Satellite {satellite.catalog_number}: {validation_result['error']}")
                
                # Collect warnings
                warnings_list.extend([
                    f"Satellite {satellite.catalog_number}: {warning}" 
                    for warning in validation_result.get('warnings', [])
                ])
                
            except Exception as e:
                errors_list.append(f"Satellite {satellite.catalog_number}: Validation failed - {e}")
        
        # Dataset-level validation
        dataset_warnings, dataset_errors = self._validate_dataset_consistency(satellites)
        warnings_list.extend(dataset_warnings)
        errors_list.extend(dataset_errors)
        
        # Genetic algorithm specific validation
        ga_warnings, ga_errors = self._validate_genetic_algorithm_compatibility(valid_satellites)
        warnings_list.extend(ga_warnings)
        errors_list.extend(ga_errors)
        
        # Calculate statistics
        statistics = self._calculate_dataset_statistics(valid_satellites)
        
        # Determine overall validity
        valid_count = len(valid_satellites)
        total_count = len(satellites)
        
        # Dataset is valid if we have at least some valid satellites and no critical errors
        is_valid = (valid_count > 0 and 
                   valid_count >= total_count * 0.5 and  # At least 50% valid
                   not any('critical' in error.lower() for error in errors_list))
        
        # Check genetic algorithm readiness
        ga_ready = (valid_count >= 5 and  # Minimum satellites for GA
                   not any('genetic algorithm' in error.lower() for error in errors_list))
        
        return {
            'valid': is_valid,
            'total_satellites': total_count,
            'valid_satellites': valid_count,
            'invalid_satellites': total_count - valid_count,
            'validity_rate': valid_count / total_count if total_count > 0 else 0.0,
            'warnings': warnings_list,
            'errors': errors_list,
            'statistics': statistics,
            'valid_satellite_list': valid_satellites,
            'genetic_algorithm_ready': ga_ready
        }
    
    def validate_individual_satellite(self, satellite: SatelliteData) -> Dict[str, any]:
        """
        Validate individual satellite data.
        
        Args:
            satellite: Satellite data to validate
            
        Returns:
            Dictionary containing validation results
        """
        warnings_list = []
        errors_list = []
        
        try:
            # Basic data presence validation
            if not self._validate_basic_data_presence(satellite):
                return {
                    'valid': False,
                    'error': 'Missing required satellite data fields',
                    'warnings': warnings_list
                }
            
            # Orbital parameter validation
            orbital_validation = self._validate_orbital_parameters(satellite)
            if not orbital_validation['valid']:
                errors_list.append(orbital_validation['error'])
            warnings_list.extend(orbital_validation.get('warnings', []))
            
            # LEO classification validation
            leo_validation = self._validate_leo_classification(satellite)
            if not leo_validation['valid']:
                errors_list.append(leo_validation['error'])
            warnings_list.extend(leo_validation.get('warnings', []))
            
            # Epoch freshness validation
            epoch_validation = self._validate_epoch_freshness(satellite)
            warnings_list.extend(epoch_validation.get('warnings', []))
            
            # Physical reasonableness validation
            physics_validation = self._validate_physical_reasonableness(satellite)
            if not physics_validation['valid']:
                errors_list.append(physics_validation['error'])
            warnings_list.extend(physics_validation.get('warnings', []))
            
            # Overall validity
            is_valid = len(errors_list) == 0
            
            return {
                'valid': is_valid,
                'error': '; '.join(errors_list) if errors_list else None,
                'warnings': warnings_list
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Validation exception: {e}',
                'warnings': warnings_list
            }
    
    def _validate_basic_data_presence(self, satellite: SatelliteData) -> bool:
        """Validate that all required data fields are present."""
        required_fields = [
            'catalog_number', 'name', 'epoch', 'mean_motion', 'eccentricity',
            'inclination', 'raan', 'arg_perigee', 'mean_anomaly',
            'semi_major_axis', 'orbital_period'
        ]
        
        for field in required_fields:
            if not hasattr(satellite, field) or getattr(satellite, field) is None:
                return False
        
        return True
    
    def _validate_orbital_parameters(self, satellite: SatelliteData) -> Dict[str, any]:
        """Validate orbital parameters for physical correctness."""
        warnings_list = []
        errors_list = []
        
        # Inclination validation
        if not (0 <= satellite.inclination <= 180):
            errors_list.append(f"Invalid inclination: {satellite.inclination}° (must be 0-180°)")
        
        # Eccentricity validation
        if satellite.eccentricity < 0:
            errors_list.append(f"Invalid eccentricity: {satellite.eccentricity} (must be non-negative)")
        elif satellite.eccentricity >= 1:
            errors_list.append(f"Invalid eccentricity: {satellite.eccentricity} (must be < 1 for elliptical orbits)")
        elif satellite.eccentricity > self.max_eccentricity:
            warnings_list.append(f"High eccentricity: {satellite.eccentricity:.3f} (may affect transfer calculations)")
        
        # Mean motion validation
        if satellite.mean_motion <= 0:
            errors_list.append(f"Invalid mean motion: {satellite.mean_motion} (must be positive)")
        elif satellite.mean_motion < self.min_mean_motion:
            warnings_list.append(f"Very low mean motion: {satellite.mean_motion:.3f} rev/day")
        elif satellite.mean_motion > self.max_mean_motion:
            errors_list.append(f"Unrealistic mean motion: {satellite.mean_motion} rev/day (too high for LEO)")
        
        # Angular parameter validation
        angular_params = [
            ('RAAN', satellite.raan),
            ('Argument of Perigee', satellite.arg_perigee),
            ('Mean Anomaly', satellite.mean_anomaly)
        ]
        
        for param_name, param_value in angular_params:
            if not (0 <= param_value < 360):
                errors_list.append(f"Invalid {param_name}: {param_value}° (must be 0-360°)")
        
        return {
            'valid': len(errors_list) == 0,
            'error': '; '.join(errors_list) if errors_list else None,
            'warnings': warnings_list
        }
    
    def _validate_leo_classification(self, satellite: SatelliteData) -> Dict[str, any]:
        """Validate that satellite is in LEO range."""
        warnings_list = []
        errors_list = []
        
        # Calculate altitude
        altitude = satellite.semi_major_axis - self.earth_radius
        
        # Check LEO range
        if altitude < self.min_leo_altitude:
            errors_list.append(f"Altitude too low: {altitude:.1f} km (below {self.min_leo_altitude} km)")
        elif altitude > self.max_leo_altitude:
            warnings_list.append(f"High altitude: {altitude:.1f} km (above typical LEO range)")
        
        # Check perigee altitude with eccentricity
        perigee = satellite.semi_major_axis * (1 - satellite.eccentricity)
        perigee_altitude = perigee - self.earth_radius
        
        if perigee_altitude < 100:  # Below atmospheric limit
            errors_list.append(f"Perigee too low: {perigee_altitude:.1f} km (orbit would decay rapidly)")
        
        return {
            'valid': len(errors_list) == 0,
            'error': '; '.join(errors_list) if errors_list else None,
            'warnings': warnings_list
        }
    
    def _validate_epoch_freshness(self, satellite: SatelliteData) -> Dict[str, any]:
        """Validate epoch freshness for accurate orbital propagation."""
        warnings_list = []
        
        if satellite.epoch:
            age = datetime.utcnow() - satellite.epoch
            age_days = age.total_seconds() / 86400
            
            if age_days > self.max_epoch_age_days:
                warnings_list.append(f"Old epoch data: {age_days:.1f} days old (may affect accuracy)")
            elif age_days > 7:
                warnings_list.append(f"Epoch data is {age_days:.1f} days old")
        
        return {'warnings': warnings_list}
    
    def _validate_physical_reasonableness(self, satellite: SatelliteData) -> Dict[str, any]:
        """Validate physical reasonableness of derived parameters."""
        warnings_list = []
        errors_list = []
        
        # Validate semi-major axis consistency with mean motion
        try:
            # Calculate expected semi-major axis from mean motion
            mu_earth = 398600.4418  # km³/s²
            n = satellite.mean_motion * 2 * math.pi / 86400  # rad/s
            expected_sma = (mu_earth / (n * n)) ** (1/3)
            
            # Check consistency (allow 1% tolerance)
            sma_diff = abs(satellite.semi_major_axis - expected_sma)
            if sma_diff > expected_sma * 0.01:
                warnings_list.append(f"Semi-major axis inconsistent with mean motion (diff: {sma_diff:.1f} km)")
        
        except (ValueError, ZeroDivisionError):
            errors_list.append("Cannot validate semi-major axis consistency")
        
        # Validate orbital period consistency
        try:
            expected_period = 2 * math.pi * math.sqrt((satellite.semi_major_axis ** 3) / mu_earth) / 60
            period_diff = abs(satellite.orbital_period - expected_period)
            
            if period_diff > expected_period * 0.01:
                warnings_list.append(f"Orbital period inconsistent (diff: {period_diff:.1f} min)")
        
        except (ValueError, ZeroDivisionError):
            errors_list.append("Cannot validate orbital period consistency")
        
        return {
            'valid': len(errors_list) == 0,
            'error': '; '.join(errors_list) if errors_list else None,
            'warnings': warnings_list
        }
    
    def _validate_dataset_consistency(self, satellites: List[SatelliteData]) -> Tuple[List[str], List[str]]:
        """Validate consistency across the entire dataset."""
        warnings_list = []
        errors_list = []
        
        if not satellites:
            return warnings_list, errors_list
        
        # Check for duplicate catalog numbers
        catalog_numbers = [sat.catalog_number for sat in satellites]
        duplicates = set([x for x in catalog_numbers if catalog_numbers.count(x) > 1])
        
        if duplicates:
            errors_list.append(f"Duplicate catalog numbers found: {sorted(duplicates)}")
        
        # Check epoch spread
        epochs = [sat.epoch for sat in satellites if sat.epoch]
        if epochs:
            epoch_range = max(epochs) - min(epochs)
            if epoch_range.total_seconds() > 86400 * 7:  # More than 7 days
                warnings_list.append(f"Large epoch spread: {epoch_range.days} days (may affect accuracy)")
        
        # Check altitude distribution
        altitudes = [sat.semi_major_axis - self.earth_radius for sat in satellites]
        if altitudes:
            altitude_range = max(altitudes) - min(altitudes)
            if altitude_range > 1500:  # Very wide altitude range
                warnings_list.append(f"Wide altitude range: {altitude_range:.1f} km")
        
        return warnings_list, errors_list
    
    def _validate_genetic_algorithm_compatibility(self, satellites: List[SatelliteData]) -> Tuple[List[str], List[str]]:
        """Validate satellite dataset for genetic algorithm compatibility."""
        warnings_list = []
        errors_list = []
        
        if not satellites:
            errors_list.append("No valid satellites available for genetic algorithm")
            return warnings_list, errors_list
        
        # Check minimum constellation size
        if len(satellites) < 5:
            errors_list.append(f"Genetic algorithm requires at least 5 satellites (found {len(satellites)})")
        elif len(satellites) < 20:
            warnings_list.append(f"Small constellation size ({len(satellites)} satellites) may limit optimization quality")
        
        # Check orbital diversity for meaningful route optimization
        altitudes = [sat.semi_major_axis - self.earth_radius for sat in satellites]
        altitude_range = max(altitudes) - min(altitudes)
        
        if altitude_range < 10:  # Less than 10km altitude range
            warnings_list.append(f"Limited altitude diversity ({altitude_range:.1f} km range) may reduce route options")
        
        inclinations = [sat.inclination for sat in satellites]
        inclination_range = max(inclinations) - min(inclinations)
        
        if inclination_range < 5:  # Less than 5 degree inclination range
            warnings_list.append(f"Limited inclination diversity ({inclination_range:.1f}° range) may reduce route options")
        
        # Check for orbital parameter completeness required for transfer calculations
        incomplete_satellites = []
        for sat in satellites:
            required_params = ['semi_major_axis', 'eccentricity', 'inclination', 
                             'raan', 'arg_perigee', 'mean_anomaly', 'mean_motion']
            
            missing_params = []
            for param in required_params:
                if not hasattr(sat, param) or getattr(sat, param) is None:
                    missing_params.append(param)
            
            if missing_params:
                incomplete_satellites.append(f"{sat.catalog_number}: missing {', '.join(missing_params)}")
        
        if incomplete_satellites:
            if len(incomplete_satellites) > 5:
                errors_list.append(f"{len(incomplete_satellites)} satellites missing orbital parameters")
            else:
                errors_list.extend([f"Incomplete orbital data: {sat}" for sat in incomplete_satellites])
        
        # Check epoch freshness for accurate orbital propagation
        current_time = datetime.utcnow()
        old_epochs = [sat for sat in satellites 
                     if sat.epoch and (current_time - sat.epoch).days > 30]
        
        if old_epochs:
            old_count = len(old_epochs)
            if old_count > len(satellites) * 0.5:  # More than 50% old
                warnings_list.append(f"Many satellites ({old_count}) have old epoch data (>30 days)")
            else:
                warnings_list.append(f"{old_count} satellites have old epoch data (>30 days)")
        
        # Check for reasonable orbital parameters for transfer calculations
        extreme_eccentricity = [sat for sat in satellites if sat.eccentricity > 0.2]
        if extreme_eccentricity:
            warnings_list.append(f"{len(extreme_eccentricity)} satellites have high eccentricity (>0.2)")
        
        # Check for satellites with very different orbital characteristics
        # that might cause numerical issues in transfer calculations
        very_low_altitude = [sat for sat in satellites if (sat.semi_major_axis - self.earth_radius) < 200]
        if very_low_altitude:
            warnings_list.append(f"{len(very_low_altitude)} satellites at very low altitude (<200 km)")
        
        very_high_altitude = [sat for sat in satellites if (sat.semi_major_axis - self.earth_radius) > 1500]
        if very_high_altitude:
            warnings_list.append(f"{len(very_high_altitude)} satellites at high altitude (>1500 km)")
        
        return warnings_list, errors_list
    
    def _calculate_dataset_statistics(self, satellites: List[SatelliteData]) -> Dict[str, any]:
        """Calculate statistics for the satellite dataset."""
        if not satellites:
            return {}
        
        altitudes = [sat.semi_major_axis - self.earth_radius for sat in satellites]
        inclinations = [sat.inclination for sat in satellites]
        eccentricities = [sat.eccentricity for sat in satellites]
        mean_motions = [sat.mean_motion for sat in satellites]
        
        return {
            'count': len(satellites),
            'altitude_stats': {
                'min': min(altitudes),
                'max': max(altitudes),
                'mean': sum(altitudes) / len(altitudes),
                'range': max(altitudes) - min(altitudes)
            },
            'inclination_stats': {
                'min': min(inclinations),
                'max': max(inclinations),
                'mean': sum(inclinations) / len(inclinations),
                'range': max(inclinations) - min(inclinations)
            },
            'eccentricity_stats': {
                'min': min(eccentricities),
                'max': max(eccentricities),
                'mean': sum(eccentricities) / len(eccentricities)
            },
            'mean_motion_stats': {
                'min': min(mean_motions),
                'max': max(mean_motions),
                'mean': sum(mean_motions) / len(mean_motions)
            },
            'catalog_number_range': {
                'min': min(sat.catalog_number for sat in satellites),
                'max': max(sat.catalog_number for sat in satellites)
            }
        }
    
    def filter_valid_satellites(self, satellites: List[SatelliteData]) -> List[SatelliteData]:
        """
        Filter satellite list to return only valid satellites.
        
        Args:
            satellites: List of satellite data to filter
            
        Returns:
            List of valid satellites suitable for genetic algorithm
        """
        valid_satellites = []
        
        for satellite in satellites:
            validation_result = self.validate_individual_satellite(satellite)
            if validation_result['valid']:
                valid_satellites.append(satellite)
        
        return valid_satellites
    
    def validate_for_genetic_algorithm(self, satellites: List[SatelliteData]) -> Dict[str, any]:
        """
        Specialized validation for genetic algorithm compatibility.
        
        Args:
            satellites: List of satellite data to validate
            
        Returns:
            Dictionary containing GA-specific validation results
        """
        # First run standard validation
        standard_result = self.validate_satellite_dataset(satellites)
        
        if not standard_result['valid']:
            return {
                'ready': False,
                'error': 'Standard satellite data validation failed',
                'recommendations': [
                    'Fix satellite data quality issues first',
                    'Use option 6 for detailed validation report'
                ],
                'valid_satellites': standard_result['valid_satellites'],
                'total_satellites': standard_result['total_satellites']
            }
        
        valid_satellites = standard_result['valid_satellite_list']
        recommendations = []
        warnings = []
        
        # GA-specific checks
        if len(valid_satellites) < 5:
            return {
                'ready': False,
                'error': f'Insufficient satellites for genetic algorithm (need ≥5, have {len(valid_satellites)})',
                'recommendations': [
                    'Load a larger satellite dataset',
                    'Check TLE file for more satellite entries'
                ],
                'valid_satellites': len(valid_satellites),
                'total_satellites': standard_result['total_satellites']
            }
        
        # Check constellation characteristics
        altitudes = [sat.semi_major_axis - 6378.137 for sat in valid_satellites]
        inclinations = [sat.inclination for sat in valid_satellites]
        
        altitude_range = max(altitudes) - min(altitudes)
        inclination_range = max(inclinations) - min(inclinations)
        
        if len(valid_satellites) < 20:
            warnings.append(f"Small constellation ({len(valid_satellites)} satellites)")
            recommendations.append("Consider using a larger satellite dataset for better optimization")
        
        if altitude_range < 50:
            warnings.append(f"Limited altitude diversity ({altitude_range:.1f} km range)")
            recommendations.append("Routes may be limited due to similar satellite altitudes")
        
        if inclination_range < 10:
            warnings.append(f"Limited inclination diversity ({inclination_range:.1f}° range)")
            recommendations.append("Routes may be limited due to similar orbital planes")
        
        # Check for transfer calculation compatibility
        transfer_issues = 0
        for sat in valid_satellites:
            if sat.eccentricity > 0.25:
                transfer_issues += 1
        
        if transfer_issues > 0:
            warnings.append(f"{transfer_issues} satellites have high eccentricity")
            recommendations.append("High eccentricity orbits may affect transfer accuracy")
        
        # Overall readiness assessment
        ready = len(valid_satellites) >= 5 and len(warnings) < 5
        
        return {
            'ready': ready,
            'valid_satellites': len(valid_satellites),
            'total_satellites': standard_result['total_satellites'],
            'warnings': warnings,
            'recommendations': recommendations,
            'constellation_stats': {
                'altitude_range': altitude_range,
                'inclination_range': inclination_range,
                'mean_altitude': sum(altitudes) / len(altitudes),
                'mean_inclination': sum(inclinations) / len(inclinations)
            }
        }
    
    def print_validation_report(self, validation_result: Dict[str, any]) -> None:
        """Print a formatted validation report."""
        print("\n" + "="*60)
        print("SATELLITE DATA VALIDATION REPORT")
        print("="*60)
        
        # Overall status
        status = "VALID" if validation_result['valid'] else "INVALID"
        print(f"\nDataset Status: {status}")
        print(f"Total Satellites: {validation_result['total_satellites']}")
        print(f"Valid Satellites: {validation_result['valid_satellites']}")
        print(f"Invalid Satellites: {validation_result.get('invalid_satellites', 0)}")
        print(f"Validity Rate: {validation_result.get('validity_rate', 0):.1%}")
        
        # Genetic algorithm readiness
        if 'genetic_algorithm_ready' in validation_result:
            ga_status = "READY" if validation_result['genetic_algorithm_ready'] else "NOT READY"
            print(f"Genetic Algorithm: {ga_status}")
        
        # Statistics
        if 'statistics' in validation_result and validation_result['statistics']:
            stats = validation_result['statistics']
            print(f"\nDataset Statistics:")
            
            if 'altitude_stats' in stats:
                alt_stats = stats['altitude_stats']
                print(f"  Altitude Range: {alt_stats['min']:.1f} - {alt_stats['max']:.1f} km")
                print(f"  Mean Altitude: {alt_stats['mean']:.1f} km")
            
            if 'inclination_stats' in stats:
                inc_stats = stats['inclination_stats']
                print(f"  Inclination Range: {inc_stats['min']:.1f} - {inc_stats['max']:.1f}°")
                print(f"  Mean Inclination: {inc_stats['mean']:.1f}°")
        
        # Warnings
        if validation_result.get('warnings'):
            print(f"\nWarnings ({len(validation_result['warnings'])}):")
            for warning in validation_result['warnings'][:10]:  # Show first 10
                print(f"  ⚠️  {warning}")
            if len(validation_result['warnings']) > 10:
                print(f"  ... and {len(validation_result['warnings']) - 10} more warnings")
        
        # Errors
        if validation_result.get('errors'):
            print(f"\nErrors ({len(validation_result['errors'])}):")
            for error in validation_result['errors'][:10]:  # Show first 10
                print(f"  ❌ {error}")
            if len(validation_result['errors']) > 10:
                print(f"  ... and {len(validation_result['errors']) - 10} more errors")
        
        print("="*60)