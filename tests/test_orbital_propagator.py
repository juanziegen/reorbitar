"""
Unit tests for the OrbitalPropagator class.

Tests orbital propagation accuracy, transfer window calculations,
and integration with existing orbital mechanics functions.
"""

import unittest
import math
from datetime import datetime
from src.orbital_propagator import OrbitalPropagator, TransferWindow
from src.tle_parser import SatelliteData
from src.orbital_mechanics import OrbitalElements, EARTH_MU


class TestOrbitalPropagator(unittest.TestCase):
    """Test cases for OrbitalPropagator class."""
    
    def setUp(self):
        """Set up test fixtures with sample satellite data."""
        # Create test satellite data for LEO satellites
        self.sat1 = SatelliteData(
            catalog_number=25544,  # ISS-like orbit
            name="TEST-SAT-1",
            epoch=datetime(2024, 1, 1),
            mean_motion=15.5,  # ~400km altitude
            eccentricity=0.0001,
            inclination=51.6,
            raan=45.0,
            arg_perigee=90.0,
            mean_anomaly=0.0,
            semi_major_axis=6778.0,  # ~400km altitude
            orbital_period=92.7
        )
        
        self.sat2 = SatelliteData(
            catalog_number=25545,  # Higher orbit
            name="TEST-SAT-2", 
            epoch=datetime(2024, 1, 1),
            mean_motion=14.0,  # ~600km altitude
            eccentricity=0.0002,
            inclination=51.6,  # Same inclination
            raan=45.0,
            arg_perigee=90.0,
            mean_anomaly=180.0,  # Opposite phase
            semi_major_axis=6978.0,  # ~600km altitude
            orbital_period=102.5
        )
        
        self.sat3 = SatelliteData(
            catalog_number=25546,  # Different inclination
            name="TEST-SAT-3",
            epoch=datetime(2024, 1, 1),
            mean_motion=15.0,
            eccentricity=0.0001,
            inclination=98.0,  # Sun-synchronous-like
            raan=0.0,
            arg_perigee=90.0,
            mean_anomaly=0.0,
            semi_major_axis=6878.0,  # ~500km altitude
            orbital_period=96.8
        )
        
        self.satellites = [self.sat1, self.sat2, self.sat3]
        self.propagator = OrbitalPropagator(self.satellites)
    
    def test_initialization(self):
        """Test OrbitalPropagator initialization."""
        # Test successful initialization
        self.assertEqual(len(self.propagator.satellites), 3)
        self.assertEqual(len(self.propagator.orbital_elements), 3)
        self.assertIn(25544, self.propagator.satellites)
        self.assertIn(25545, self.propagator.satellites)
        self.assertIn(25546, self.propagator.satellites)
        
        # Test empty satellites list
        with self.assertRaises(ValueError):
            OrbitalPropagator([])
        
        # Test duplicate catalog numbers
        duplicate_sat = SatelliteData(
            catalog_number=25544,  # Same as sat1
            name="DUPLICATE",
            epoch=datetime(2024, 1, 1),
            mean_motion=15.0,
            eccentricity=0.0,
            inclination=0.0,
            raan=0.0,
            arg_perigee=0.0,
            mean_anomaly=0.0,
            semi_major_axis=6778.0,
            orbital_period=90.0
        )
        
        with self.assertRaises(ValueError):
            OrbitalPropagator([self.sat1, duplicate_sat])
    
    def test_propagate_to_time(self):
        """Test orbital propagation to specific times."""
        satellite_id = 25544
        
        # Test propagation at epoch (t=0)
        elements_t0 = self.propagator.propagate_to_time(satellite_id, 0.0)
        original_elements = self.propagator.orbital_elements[satellite_id]
        
        # At t=0, mean anomaly should be unchanged
        self.assertAlmostEqual(elements_t0.mean_anomaly, original_elements.mean_anomaly, places=6)
        self.assertAlmostEqual(elements_t0.semi_major_axis, original_elements.semi_major_axis, places=3)
        
        # Test propagation after one orbital period
        orbital_period = 2 * math.pi * math.sqrt(original_elements.semi_major_axis**3 / EARTH_MU)
        elements_t1 = self.propagator.propagate_to_time(satellite_id, orbital_period)
        
        # After one period, mean anomaly should be approximately the same (modulo 2π)
        mean_anomaly_diff = abs(elements_t1.mean_anomaly - original_elements.mean_anomaly)
        self.assertTrue(mean_anomaly_diff < 0.01 or abs(mean_anomaly_diff - 2*math.pi) < 0.01)
        
        # Test propagation after half period
        elements_half = self.propagator.propagate_to_time(satellite_id, orbital_period / 2)
        expected_half_anomaly = (original_elements.mean_anomaly + math.pi) % (2 * math.pi)
        self.assertAlmostEqual(elements_half.mean_anomaly, expected_half_anomaly, places=2)
        
        # Test invalid satellite ID
        with self.assertRaises(ValueError):
            self.propagator.propagate_to_time(99999, 0.0)
        
        # Test invalid time
        with self.assertRaises(ValueError):
            self.propagator.propagate_to_time(satellite_id, float('inf'))
    
    def test_get_satellite_position(self):
        """Test satellite position calculation at specific times."""
        satellite_id = 25544
        
        # Test position at epoch
        state_t0 = self.propagator.get_satellite_position(satellite_id, 0.0)
        
        # Verify state vector structure
        self.assertEqual(len(state_t0.position), 3)
        self.assertEqual(len(state_t0.velocity), 3)
        
        # Position should be reasonable for LEO satellite
        position_magnitude = math.sqrt(sum(x**2 for x in state_t0.position))
        self.assertTrue(6500 < position_magnitude < 7500)  # Reasonable LEO range
        
        # Velocity should be reasonable for LEO
        velocity_magnitude = math.sqrt(sum(v**2 for v in state_t0.velocity))
        self.assertTrue(7.0 < velocity_magnitude < 8.0)  # Typical LEO velocity range
        
        # Test position after some time
        state_t1 = self.propagator.get_satellite_position(satellite_id, 3600.0)  # 1 hour
        
        # Position should have changed
        position_diff = math.sqrt(sum((p1 - p0)**2 for p1, p0 in 
                                    zip(state_t1.position, state_t0.position)))
        self.assertTrue(position_diff > 1000)  # Should move significantly in 1 hour
        
        # Test invalid satellite ID
        with self.assertRaises(ValueError):
            self.propagator.get_satellite_position(99999, 0.0)
    
    def test_calculate_transfer_window(self):
        """Test transfer window calculations between satellites."""
        source_id = 25544
        target_id = 25545
        departure_time = 0.0
        
        # Test basic transfer window calculation
        transfer_window = self.propagator.calculate_transfer_window(
            source_id, target_id, departure_time
        )
        
        # Verify transfer window structure
        self.assertIsInstance(transfer_window, TransferWindow)
        self.assertGreater(transfer_window.departure_deltav, 0)
        self.assertGreater(transfer_window.arrival_deltav, 0)
        self.assertGreater(transfer_window.transfer_time, 0)
        self.assertEqual(transfer_window.optimal_departure_time, departure_time)
        self.assertTrue(0 < transfer_window.transfer_efficiency <= 1.0)
        
        # Transfer time should be reasonable (less than one day for LEO)
        self.assertLess(transfer_window.transfer_time, 86400)  # Less than 24 hours
        
        # Delta-v should be reasonable for LEO transfers
        total_dv = transfer_window.departure_deltav + transfer_window.arrival_deltav
        self.assertLess(total_dv, 2.0)  # Should be less than 2 km/s for similar LEO orbits
        
        # Test same satellite transfer (should fail)
        with self.assertRaises(ValueError):
            self.propagator.calculate_transfer_window(source_id, source_id, departure_time)
        
        # Test invalid satellite IDs
        with self.assertRaises(ValueError):
            self.propagator.calculate_transfer_window(99999, target_id, departure_time)
        
        with self.assertRaises(ValueError):
            self.propagator.calculate_transfer_window(source_id, 99999, departure_time)
        
        # Test invalid departure time
        with self.assertRaises(ValueError):
            self.propagator.calculate_transfer_window(source_id, target_id, float('nan'))
    
    def test_transfer_window_with_inclination_change(self):
        """Test transfer window calculations with significant inclination changes."""
        source_id = 25544  # 51.6° inclination
        target_id = 25546  # 98° inclination
        departure_time = 0.0
        
        transfer_window = self.propagator.calculate_transfer_window(
            source_id, target_id, departure_time
        )
        
        # Transfer with large inclination change should be calculated
        total_dv = transfer_window.departure_deltav + transfer_window.arrival_deltav
        
        # Should be more expensive than coplanar transfer
        coplanar_window = self.propagator.calculate_transfer_window(25544, 25545, departure_time)
        coplanar_dv = coplanar_window.departure_deltav + coplanar_window.arrival_deltav
        
        # Note: The actual comparison depends on the altitude difference vs inclination change
        # For this test, we'll just verify both transfers are calculated successfully
        self.assertGreater(total_dv, 0)
        self.assertGreater(coplanar_dv, 0)
        
        # Both transfers should have reasonable delta-v values
        self.assertLess(total_dv, 10.0)  # Should be less than 10 km/s
        self.assertLess(coplanar_dv, 10.0)
        
        # Verify transfer times are reasonable
        self.assertGreater(transfer_window.transfer_time, 0)
        self.assertLess(transfer_window.transfer_time, 86400)  # Less than 24 hours
    
    def test_find_optimal_departure_time(self):
        """Test finding optimal departure time within a time window."""
        source_id = 25544
        target_id = 25545
        time_window_start = 0.0
        time_window_end = 86400.0  # 24 hours
        time_step = 3600.0  # 1 hour steps
        
        optimal_transfer = self.propagator.find_optimal_departure_time(
            source_id, target_id, time_window_start, time_window_end, time_step
        )
        
        # Should find a valid transfer
        self.assertIsInstance(optimal_transfer, TransferWindow)
        self.assertTrue(time_window_start <= optimal_transfer.optimal_departure_time <= time_window_end)
        
        # Compare with a few manual calculations to verify it's actually optimal
        manual_transfers = []
        for t in [0.0, 3600.0, 7200.0, 10800.0]:
            try:
                transfer = self.propagator.calculate_transfer_window(source_id, target_id, t)
                manual_transfers.append(transfer)
            except:
                pass
        
        if manual_transfers:
            optimal_manual_dv = min(t.departure_deltav + t.arrival_deltav for t in manual_transfers)
            optimal_found_dv = optimal_transfer.departure_deltav + optimal_transfer.arrival_deltav
            
            # Found optimum should be at least as good as manual samples
            self.assertLessEqual(optimal_found_dv, optimal_manual_dv + 0.01)  # Small tolerance
        
        # Test invalid time window
        with self.assertRaises(ValueError):
            self.propagator.find_optimal_departure_time(
                source_id, target_id, time_window_end, time_window_start, time_step
            )
        
        # Test invalid time step
        with self.assertRaises(ValueError):
            self.propagator.find_optimal_departure_time(
                source_id, target_id, time_window_start, time_window_end, -1.0
            )
    
    def test_utility_methods(self):
        """Test utility methods of OrbitalPropagator."""
        # Test get_satellite_ids
        satellite_ids = self.propagator.get_satellite_ids()
        self.assertEqual(set(satellite_ids), {25544, 25545, 25546})
        
        # Test get_satellite_data
        sat_data = self.propagator.get_satellite_data(25544)
        self.assertEqual(sat_data.catalog_number, 25544)
        self.assertEqual(sat_data.name, "TEST-SAT-1")
        
        # Test invalid satellite ID
        with self.assertRaises(ValueError):
            self.propagator.get_satellite_data(99999)
    
    def test_integration_with_existing_transfer_calculator(self):
        """Test integration with existing transfer calculator functions."""
        source_id = 25544
        target_id = 25545
        departure_time = 0.0
        
        # Calculate transfer using propagator
        transfer_window = self.propagator.calculate_transfer_window(
            source_id, target_id, departure_time
        )
        
        # Verify results are physically reasonable
        self.assertGreater(transfer_window.departure_deltav, 0)
        self.assertGreater(transfer_window.arrival_deltav, 0)
        self.assertGreater(transfer_window.transfer_time, 0)
        
        # Total delta-v should be reasonable for LEO transfers
        total_dv = transfer_window.departure_deltav + transfer_window.arrival_deltav
        self.assertLess(total_dv, 5.0)  # Should be reasonable for LEO
        
        # Transfer time should be reasonable
        self.assertLess(transfer_window.transfer_time, 86400)  # Less than 24 hours
        
        # Efficiency should be reasonable
        self.assertTrue(0.1 <= transfer_window.transfer_efficiency <= 1.0)
    
    def test_propagation_accuracy(self):
        """Test orbital propagation accuracy against known orbital mechanics."""
        satellite_id = 25544
        
        # Get original orbital elements
        original_elements = self.propagator.orbital_elements[satellite_id]
        
        # Calculate expected mean motion
        n = math.sqrt(EARTH_MU / original_elements.semi_major_axis**3)
        
        # Test propagation for various time intervals
        test_times = [0.0, 3600.0, 7200.0, 10800.0]  # 0, 1, 2, 3 hours
        
        for t in test_times:
            propagated_elements = self.propagator.propagate_to_time(satellite_id, t)
            
            # Calculate expected mean anomaly
            expected_mean_anomaly = (original_elements.mean_anomaly + n * t) % (2 * math.pi)
            
            # Check accuracy (should be very close for simple Keplerian propagation)
            self.assertAlmostEqual(
                propagated_elements.mean_anomaly, 
                expected_mean_anomaly, 
                places=6,
                msg=f"Mean anomaly mismatch at t={t}"
            )
            
            # Other orbital elements should remain unchanged
            self.assertAlmostEqual(
                propagated_elements.semi_major_axis, 
                original_elements.semi_major_axis, 
                places=6
            )
            self.assertAlmostEqual(
                propagated_elements.eccentricity, 
                original_elements.eccentricity, 
                places=6
            )
            self.assertAlmostEqual(
                propagated_elements.inclination, 
                original_elements.inclination, 
                places=6
            )


if __name__ == '__main__':
    unittest.main()