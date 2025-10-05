"""
Unit tests for transfer_calculator module.

Tests the high-level delta-v calculations, complexity assessment,
and transfer optimization functions.
"""

import unittest
import math
from datetime import datetime
from src.transfer_calculator import (
    calculate_transfer_deltav,
    calculate_hohmann_deltav,
    calculate_inclination_change_deltav,
    find_optimal_transfer_window,
    assess_transfer_complexity,
    calculate_phase_angle,
    calculate_synodic_period,
    calculate_wait_time_for_optimal_phase,
    TransferResult,
    DeltaVBreakdown
)
from src.tle_parser import SatelliteData
from src.orbital_mechanics import OrbitalElements, EARTH_MU


class TestTransferCalculator(unittest.TestCase):
    """Test cases for transfer calculator functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test satellite data for LEO satellites
        self.source_sat = SatelliteData(
            catalog_number=25544,
            name="ISS (ZARYA)",
            epoch=datetime(2024, 1, 1),
            mean_motion=15.49,  # revolutions per day
            eccentricity=0.0001,
            inclination=51.6,  # degrees
            raan=0.0,
            arg_perigee=0.0,
            mean_anomaly=0.0,
            semi_major_axis=6793.0,  # km (approximately 415 km altitude)
            orbital_period=5568.0  # seconds
        )
        
        self.target_sat = SatelliteData(
            catalog_number=43013,
            name="STARLINK-1007",
            epoch=datetime(2024, 1, 1),
            mean_motion=15.06,  # revolutions per day
            eccentricity=0.0001,
            inclination=53.0,  # degrees
            raan=0.0,
            arg_perigee=0.0,
            mean_anomaly=0.0,
            semi_major_axis=6928.0,  # km (approximately 550 km altitude)
            orbital_period=5760.0  # seconds
        )
        
        # Create orbital elements for direct testing
        self.source_elements = OrbitalElements(
            semi_major_axis=6793.0,
            eccentricity=0.0001,
            inclination=math.radians(51.6),
            raan=0.0,
            arg_perigee=0.0,
            mean_anomaly=0.0
        )
        
        self.target_elements = OrbitalElements(
            semi_major_axis=6928.0,
            eccentricity=0.0001,
            inclination=math.radians(53.0),
            raan=0.0,
            arg_perigee=0.0,
            mean_anomaly=0.0
        )
    
    def test_calculate_hohmann_deltav(self):
        """Test Hohmann transfer delta-v calculation."""
        r1 = 6793.0  # km
        r2 = 6928.0  # km
        
        departure_dv, arrival_dv = calculate_hohmann_deltav(r1, r2)
        
        # Verify results are positive
        self.assertGreater(departure_dv, 0)
        self.assertGreater(arrival_dv, 0)
        
        # For this small altitude change, delta-v should be relatively small
        self.assertLess(departure_dv, 0.1)  # Less than 100 m/s
        self.assertLess(arrival_dv, 0.1)    # Less than 100 m/s
        
        # Test descending transfer
        departure_dv_desc, arrival_dv_desc = calculate_hohmann_deltav(r2, r1)
        
        # Delta-v magnitudes should be the same for opposite transfers
        self.assertAlmostEqual(departure_dv, arrival_dv_desc, places=6)
        self.assertAlmostEqual(arrival_dv, departure_dv_desc, places=6)
    
    def test_calculate_inclination_change_deltav(self):
        """Test inclination change delta-v calculation."""
        # Test with small inclination change
        dv = calculate_inclination_change_deltav(self.source_elements, self.target_elements)
        
        # Should be positive for different inclinations
        self.assertGreater(dv, 0)
        
        # For 1.4 degree change at ~550 km altitude, should be significant
        self.assertGreater(dv, 0.1)  # More than 100 m/s
        self.assertLess(dv, 1.0)     # Less than 1 km/s
        
        # Test with same inclination
        same_inclination_elements = OrbitalElements(
            semi_major_axis=6928.0,
            eccentricity=0.0001,
            inclination=math.radians(51.6),  # Same as source
            raan=0.0,
            arg_perigee=0.0,
            mean_anomaly=0.0
        )
        
        dv_same = calculate_inclination_change_deltav(self.source_elements, same_inclination_elements)
        self.assertAlmostEqual(dv_same, 0.0, places=6)
    
    def test_calculate_transfer_deltav(self):
        """Test end-to-end transfer calculation."""
        result = calculate_transfer_deltav(self.source_sat, self.target_sat)
        
        # Verify result structure
        self.assertIsInstance(result, TransferResult)
        self.assertEqual(result.source_satellite, self.source_sat)
        self.assertEqual(result.target_satellite, self.target_sat)
        
        # Verify delta-v components are positive
        self.assertGreater(result.total_deltav, 0)
        self.assertGreater(result.departure_deltav, 0)
        self.assertGreater(result.arrival_deltav, 0)
        self.assertGreater(result.plane_change_deltav, 0)
        
        # Total should be sum of components (approximately)
        component_sum = result.departure_deltav + result.arrival_deltav + result.plane_change_deltav
        self.assertAlmostEqual(result.total_deltav, component_sum, places=3)
        
        # Transfer time should be positive
        self.assertGreater(result.transfer_time, 0)
        
        # Orbit parameters should be reasonable
        self.assertGreater(result.transfer_orbit_apogee, result.transfer_orbit_perigee)
        
        # Should have complexity assessment
        self.assertIsInstance(result.complexity_assessment, str)
        self.assertIn(result.complexity_assessment, 
                     ["Simple", "Moderate", "Complex", "Very Complex", "Extremely Complex"])
        
        # Should have warnings list
        self.assertIsInstance(result.warnings, list)
    
    def test_assess_transfer_complexity(self):
        """Test transfer complexity assessment."""
        complexity = assess_transfer_complexity(self.source_elements, self.target_elements)
        
        # Should return valid complexity level
        valid_levels = ["Simple", "Moderate", "Complex", "Very Complex", "Extremely Complex"]
        self.assertIn(complexity, valid_levels)
        
        # Test with same orbit (should be simple)
        same_complexity = assess_transfer_complexity(self.source_elements, self.source_elements)
        self.assertEqual(same_complexity, "Simple")
        
        # Test with extreme inclination change
        extreme_inclination_elements = OrbitalElements(
            semi_major_axis=6928.0,
            eccentricity=0.0001,
            inclination=math.radians(90.0),  # Polar orbit
            raan=0.0,
            arg_perigee=0.0,
            mean_anomaly=0.0
        )
        
        extreme_complexity = assess_transfer_complexity(self.source_elements, extreme_inclination_elements)
        self.assertIn(extreme_complexity, ["Complex", "Very Complex", "Extremely Complex"])
    
    def test_find_optimal_transfer_window(self):
        """Test optimal transfer window calculation."""
        optimal_phase = find_optimal_transfer_window(self.source_elements, self.target_elements)
        
        # Should return angle in radians
        self.assertIsInstance(optimal_phase, float)
        self.assertGreaterEqual(optimal_phase, 0)
        self.assertLessEqual(optimal_phase, 2 * math.pi)
    
    def test_calculate_phase_angle(self):
        """Test phase angle calculation."""
        phase_angle = calculate_phase_angle(self.source_elements, self.target_elements)
        
        # Should return angle in radians
        self.assertIsInstance(phase_angle, float)
        self.assertGreaterEqual(phase_angle, 0)
        self.assertLessEqual(phase_angle, 2 * math.pi)
        
        # Test with time offset
        phase_angle_later = calculate_phase_angle(self.source_elements, self.target_elements, 3600.0)
        
        # Should be different after 1 hour
        self.assertNotAlmostEqual(phase_angle, phase_angle_later, places=3)
    
    def test_calculate_synodic_period(self):
        """Test synodic period calculation."""
        synodic_period = calculate_synodic_period(self.source_elements, self.target_elements)
        
        # Should be positive and finite
        self.assertGreater(synodic_period, 0)
        self.assertNotEqual(synodic_period, float('inf'))
        
        # Should be longer than individual orbital periods
        T_source = 2 * math.pi * math.sqrt(self.source_elements.semi_major_axis**3 / EARTH_MU)
        T_target = 2 * math.pi * math.sqrt(self.target_elements.semi_major_axis**3 / EARTH_MU)
        
        self.assertGreater(synodic_period, T_source)
        self.assertGreater(synodic_period, T_target)
        
        # Test with same period (should return infinity)
        synodic_same = calculate_synodic_period(self.source_elements, self.source_elements)
        self.assertEqual(synodic_same, float('inf'))
    
    def test_calculate_wait_time_for_optimal_phase(self):
        """Test wait time calculation for optimal phase."""
        wait_time = calculate_wait_time_for_optimal_phase(self.source_elements, self.target_elements)
        
        # Should be non-negative
        self.assertGreaterEqual(wait_time, 0)
        
        # Should be finite for different orbits
        self.assertNotEqual(wait_time, float('inf'))
        
        # Test with specific current phase angle
        current_phase = math.pi  # 180 degrees
        wait_time_specific = calculate_wait_time_for_optimal_phase(
            self.source_elements, self.target_elements, current_phase
        )
        
        self.assertGreaterEqual(wait_time_specific, 0)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with very similar orbits
        similar_elements = OrbitalElements(
            semi_major_axis=6793.1,  # Very close altitude
            eccentricity=0.0001,
            inclination=math.radians(51.61),  # Very close inclination
            raan=0.0,
            arg_perigee=0.0,
            mean_anomaly=0.0
        )
        
        similar_sat = SatelliteData(
            catalog_number=99999,
            name="SIMILAR SAT",
            epoch=datetime(2024, 1, 1),
            mean_motion=15.489,
            eccentricity=0.0001,
            inclination=51.61,
            raan=0.0,
            arg_perigee=0.0,
            mean_anomaly=0.0,
            semi_major_axis=6793.1,
            orbital_period=5569.0
        )
        
        result = calculate_transfer_deltav(self.source_sat, similar_sat)
        
        # Should still work but with very small delta-v
        self.assertGreater(result.total_deltav, 0)
        self.assertLess(result.total_deltav, 0.01)  # Less than 10 m/s
        
        # Complexity should be simple
        self.assertEqual(result.complexity_assessment, "Simple")


if __name__ == '__main__':
    unittest.main()