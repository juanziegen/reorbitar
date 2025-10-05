"""
Unit tests for orbital mechanics calculations.

Tests Hohmann transfer calculations, plane change delta-v calculations,
and coordinate transformations against analytical solutions.
"""

import unittest
import math
from src.orbital_mechanics import (
    OrbitalElements, StateVector, TransferOrbit, CombinedManeuverResult,
    tle_to_orbital_elements, orbital_elements_to_state_vector,
    calculate_hohmann_transfer, calculate_plane_change_dv,
    calculate_combined_maneuver_dv, optimize_transfer_timing,
    EARTH_MU, EARTH_RADIUS
)
from src.tle_parser import SatelliteData
from datetime import datetime


class TestOrbitalMechanics(unittest.TestCase):
    """Test cases for orbital mechanics calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test satellite data for circular LEO orbit (400 km altitude)
        self.leo_altitude = 400  # km
        self.leo_radius = EARTH_RADIUS + self.leo_altitude
        self.leo_velocity = math.sqrt(EARTH_MU / self.leo_radius)
        
        # Create test satellite data
        self.test_satellite_leo = SatelliteData(
            catalog_number=12345,
            name="TEST-LEO",
            epoch=datetime.now(),
            mean_motion=15.5,  # Approximately correct for 400km
            eccentricity=0.0,  # Circular orbit
            inclination=51.6,  # ISS-like inclination
            raan=0.0,
            arg_perigee=0.0,
            mean_anomaly=0.0,
            semi_major_axis=self.leo_radius,
            orbital_period=90.0
        )
        
        # Create test satellite data for higher orbit (800 km altitude)
        self.geo_altitude = 800  # km
        self.geo_radius = EARTH_RADIUS + self.geo_altitude
        
        self.test_satellite_higher = SatelliteData(
            catalog_number=67890,
            name="TEST-HIGHER",
            epoch=datetime.now(),
            mean_motion=12.0,  # Lower for higher orbit
            eccentricity=0.0,
            inclination=51.6,  # Same inclination
            raan=0.0,
            arg_perigee=0.0,
            mean_anomaly=0.0,
            semi_major_axis=self.geo_radius,
            orbital_period=120.0
        )
    
    def test_tle_to_orbital_elements_conversion(self):
        """Test conversion from TLE data to orbital elements."""
        elements = tle_to_orbital_elements(self.test_satellite_leo)
        
        self.assertAlmostEqual(elements.semi_major_axis, self.leo_radius, places=1)
        self.assertAlmostEqual(elements.eccentricity, 0.0, places=6)
        self.assertAlmostEqual(elements.inclination, math.radians(51.6), places=6)
        self.assertAlmostEqual(elements.raan, 0.0, places=6)
        self.assertAlmostEqual(elements.arg_perigee, 0.0, places=6)
        self.assertAlmostEqual(elements.mean_anomaly, 0.0, places=6)
    
    def test_orbital_elements_to_state_vector(self):
        """Test conversion from orbital elements to state vector."""
        elements = tle_to_orbital_elements(self.test_satellite_leo)
        state_vector = orbital_elements_to_state_vector(elements, 0.0)
        
        # For a circular orbit at mean anomaly = 0, satellite should be at perigee
        # Position should be approximately (r, 0, 0) in ECI coordinates
        position_magnitude = math.sqrt(sum(x**2 for x in state_vector.position))
        velocity_magnitude = math.sqrt(sum(v**2 for v in state_vector.velocity))
        
        self.assertAlmostEqual(position_magnitude, self.leo_radius, places=1)
        self.assertAlmostEqual(velocity_magnitude, self.leo_velocity, places=3)
    
    def test_hohmann_transfer_calculation(self):
        """Test Hohmann transfer calculations against analytical solutions."""
        source_elements = tle_to_orbital_elements(self.test_satellite_leo)
        target_elements = tle_to_orbital_elements(self.test_satellite_higher)
        
        transfer = calculate_hohmann_transfer(source_elements, target_elements)
        
        # Verify transfer orbit parameters
        expected_apogee = self.geo_radius
        expected_perigee = self.leo_radius
        expected_transfer_sma = (self.leo_radius + self.geo_radius) / 2
        
        self.assertAlmostEqual(transfer.apogee, expected_apogee, places=1)
        self.assertAlmostEqual(transfer.perigee, expected_perigee, places=1)
        
        # Calculate expected delta-v analytically
        v1 = math.sqrt(EARTH_MU / self.leo_radius)
        v2 = math.sqrt(EARTH_MU / self.geo_radius)
        v_transfer_perigee = math.sqrt(EARTH_MU * (2/self.leo_radius - 1/expected_transfer_sma))
        v_transfer_apogee = math.sqrt(EARTH_MU * (2/self.geo_radius - 1/expected_transfer_sma))
        
        expected_departure_dv = v_transfer_perigee - v1
        expected_arrival_dv = v2 - v_transfer_apogee
        
        self.assertAlmostEqual(transfer.departure_dv, expected_departure_dv, places=3)
        self.assertAlmostEqual(transfer.arrival_dv, expected_arrival_dv, places=3)
        
        # Verify transfer time (half orbital period of transfer orbit)
        expected_transfer_period = 2 * math.pi * math.sqrt(expected_transfer_sma**3 / EARTH_MU)
        expected_transfer_time = expected_transfer_period / 2
        
        self.assertAlmostEqual(transfer.transfer_time, expected_transfer_time, places=1)
    
    def test_plane_change_delta_v(self):
        """Test plane change delta-v calculations."""
        # Test zero inclination change
        dv_zero = calculate_plane_change_dv(0.0, self.leo_velocity)
        self.assertAlmostEqual(dv_zero, 0.0, places=6)
        
        # Test 90-degree plane change
        inclination_change = math.pi / 2  # 90 degrees in radians
        dv_90deg = calculate_plane_change_dv(inclination_change, self.leo_velocity)
        
        # For 90-degree change: dv = 2 * v * sin(45°) = v * sqrt(2)
        expected_dv = self.leo_velocity * math.sqrt(2)
        self.assertAlmostEqual(dv_90deg, expected_dv, places=3)
        
        # Test small inclination change (linear approximation should hold)
        small_change = math.radians(5)  # 5 degrees
        dv_small = calculate_plane_change_dv(small_change, self.leo_velocity)
        expected_small_dv = self.leo_velocity * small_change  # Linear approximation
        
        # Should be close for small angles
        self.assertAlmostEqual(dv_small, expected_small_dv, delta=0.1)
    
    def test_combined_maneuver_calculation(self):
        """Test combined altitude and plane change maneuvers."""
        # Create target with different inclination
        target_satellite = SatelliteData(
            catalog_number=99999,
            name="TEST-INCLINED",
            epoch=datetime.now(),
            mean_motion=12.0,
            eccentricity=0.0,
            inclination=60.0,  # Different inclination
            raan=0.0,
            arg_perigee=0.0,
            mean_anomaly=0.0,
            semi_major_axis=self.geo_radius,
            orbital_period=120.0
        )
        
        source_elements = tle_to_orbital_elements(self.test_satellite_leo)
        target_elements = tle_to_orbital_elements(target_satellite)
        
        result = calculate_combined_maneuver_dv(source_elements, target_elements)
        
        # All delta-v components should be positive
        self.assertGreater(result.hohmann_departure_dv, 0)
        self.assertGreater(result.hohmann_arrival_dv, 0)
        self.assertGreater(result.plane_change_dv, 0)
        self.assertGreater(result.total_dv, 0)
        
        # Total should be sum of components
        expected_total = result.hohmann_departure_dv + result.hohmann_arrival_dv + result.plane_change_dv
        self.assertAlmostEqual(result.total_dv, expected_total, places=3)
        
        # Should have a valid strategy
        self.assertIn(result.maneuver_strategy, [
            "plane_change_at_departure", 
            "plane_change_at_arrival", 
            "plane_change_at_apogee"
        ])
        
        # Plane change should be significant for 8.4-degree difference
        inclination_diff = math.radians(60.0 - 51.6)
        self.assertGreater(result.plane_change_dv, 0.1)  # Should be substantial
    
    def test_transfer_timing_optimization(self):
        """Test optimal transfer timing calculations."""
        source_elements = tle_to_orbital_elements(self.test_satellite_leo)
        target_elements = tle_to_orbital_elements(self.test_satellite_higher)
        
        phase_angle = optimize_transfer_timing(source_elements, target_elements)
        
        # Phase angle should be between 0 and 2π
        self.assertGreaterEqual(phase_angle, 0)
        self.assertLessEqual(phase_angle, 2 * math.pi)
        
        # For transfer from lower to higher orbit, phase angle should be less than π
        # (target needs to be "behind" the source at transfer initiation)
        self.assertLess(phase_angle, math.pi)
    
    def test_circular_orbit_edge_cases(self):
        """Test edge cases for circular orbits."""
        # Test transfer between identical orbits
        source_elements = tle_to_orbital_elements(self.test_satellite_leo)
        target_elements = tle_to_orbital_elements(self.test_satellite_leo)
        
        transfer = calculate_hohmann_transfer(source_elements, target_elements)
        
        # Delta-v should be zero for identical orbits
        self.assertAlmostEqual(transfer.departure_dv, 0.0, places=6)
        self.assertAlmostEqual(transfer.arrival_dv, 0.0, places=6)
        
        # For identical orbits, transfer time is still half the orbital period
        # This is expected behavior for the Hohmann transfer calculation
        self.assertGreater(transfer.transfer_time, 0)
    
    def test_eccentric_orbit_handling(self):
        """Test handling of eccentric orbits."""
        # Create eccentric orbit satellite
        eccentric_satellite = SatelliteData(
            catalog_number=11111,
            name="TEST-ECCENTRIC",
            epoch=datetime.now(),
            mean_motion=14.0,
            eccentricity=0.2,  # Moderate eccentricity
            inclination=0.0,
            raan=0.0,
            arg_perigee=0.0,
            mean_anomaly=0.0,
            semi_major_axis=7000.0,  # km
            orbital_period=100.0
        )
        
        eccentric_elements = tle_to_orbital_elements(eccentric_satellite)
        state_vector = orbital_elements_to_state_vector(eccentric_elements, 0.0)
        
        # Should produce valid state vector
        position_magnitude = math.sqrt(sum(x**2 for x in state_vector.position))
        velocity_magnitude = math.sqrt(sum(v**2 for v in state_vector.velocity))
        
        self.assertGreater(position_magnitude, 0)
        self.assertGreater(velocity_magnitude, 0)
        
        # Position should be reasonable for the given orbit
        # For eccentric orbit at mean anomaly = 0 (perigee), position should be at perigee distance
        expected_perigee = eccentric_satellite.semi_major_axis * (1 - eccentric_satellite.eccentricity)
        self.assertAlmostEqual(position_magnitude, expected_perigee, places=1)
        self.assertLess(position_magnitude, 20000)  # Reasonable upper bound
    
    def test_hohmann_transfer_analytical_solutions(self):
        """Test Hohmann transfer calculations against known analytical solutions."""
        # Test case 1: Earth to Mars transfer (scaled down to LEO altitudes)
        # Using well-known orbital mechanics example
        r1 = 7000.0  # km (source orbit radius)
        r2 = 10000.0  # km (target orbit radius)
        
        # Create test satellites for this scenario
        source_sat = SatelliteData(
            catalog_number=1001, name="SOURCE", epoch=datetime.now(),
            mean_motion=14.0, eccentricity=0.0, inclination=0.0,
            raan=0.0, arg_perigee=0.0, mean_anomaly=0.0,
            semi_major_axis=r1, orbital_period=90.0
        )
        
        target_sat = SatelliteData(
            catalog_number=1002, name="TARGET", epoch=datetime.now(),
            mean_motion=11.0, eccentricity=0.0, inclination=0.0,
            raan=0.0, arg_perigee=0.0, mean_anomaly=0.0,
            semi_major_axis=r2, orbital_period=120.0
        )
        
        source_elements = tle_to_orbital_elements(source_sat)
        target_elements = tle_to_orbital_elements(target_sat)
        
        transfer = calculate_hohmann_transfer(source_elements, target_elements)
        
        # Calculate analytical solution
        a_transfer = (r1 + r2) / 2
        v1 = math.sqrt(EARTH_MU / r1)
        v2 = math.sqrt(EARTH_MU / r2)
        v_transfer_perigee = math.sqrt(EARTH_MU * (2/r1 - 1/a_transfer))
        v_transfer_apogee = math.sqrt(EARTH_MU * (2/r2 - 1/a_transfer))
        
        expected_departure_dv = v_transfer_perigee - v1
        expected_arrival_dv = v2 - v_transfer_apogee
        expected_total_dv = expected_departure_dv + expected_arrival_dv
        
        # Verify against analytical solution
        self.assertAlmostEqual(transfer.departure_dv, expected_departure_dv, places=4)
        self.assertAlmostEqual(transfer.arrival_dv, expected_arrival_dv, places=4)
        
        # Test case 2: Descending transfer (higher to lower orbit)
        transfer_desc = calculate_hohmann_transfer(target_elements, source_elements)
        
        # For descending transfer, delta-v magnitudes should be the same but applied differently
        expected_desc_departure_dv = v2 - v_transfer_apogee  # Decelerate at apogee
        expected_desc_arrival_dv = v_transfer_perigee - v1   # Decelerate at perigee
        
        self.assertAlmostEqual(transfer_desc.departure_dv, expected_desc_departure_dv, places=4)
        self.assertAlmostEqual(transfer_desc.arrival_dv, expected_desc_arrival_dv, places=4)
    
    def test_plane_change_analytical_verification(self):
        """Verify plane change delta-v calculations against analytical solutions."""
        # Test various inclination changes with known analytical results
        test_velocity = 7.5  # km/s (typical LEO velocity)
        
        # Test case 1: 30-degree plane change
        inclination_change_30 = math.radians(30)
        dv_30 = calculate_plane_change_dv(inclination_change_30, test_velocity)
        expected_dv_30 = 2 * test_velocity * math.sin(inclination_change_30 / 2)
        self.assertAlmostEqual(dv_30, expected_dv_30, places=6)
        
        # Test case 2: 60-degree plane change
        inclination_change_60 = math.radians(60)
        dv_60 = calculate_plane_change_dv(inclination_change_60, test_velocity)
        expected_dv_60 = 2 * test_velocity * math.sin(inclination_change_60 / 2)
        self.assertAlmostEqual(dv_60, expected_dv_60, places=6)
        
        # Test case 3: 180-degree plane change (orbit reversal)
        inclination_change_180 = math.radians(180)
        dv_180 = calculate_plane_change_dv(inclination_change_180, test_velocity)
        expected_dv_180 = 2 * test_velocity  # sin(90°) = 1
        self.assertAlmostEqual(dv_180, expected_dv_180, places=6)
        
        # Test case 4: Small angle approximation (should be nearly linear)
        small_angles = [1, 2, 3, 4, 5]  # degrees
        for angle_deg in small_angles:
            angle_rad = math.radians(angle_deg)
            dv_calculated = calculate_plane_change_dv(angle_rad, test_velocity)
            dv_linear_approx = test_velocity * angle_rad  # Small angle approximation
            
            # For small angles, the difference should be minimal
            relative_error = abs(dv_calculated - dv_linear_approx) / dv_linear_approx
            self.assertLess(relative_error, 0.01, f"Small angle approximation failed for {angle_deg}°")
    
    def test_edge_cases_circular_orbits(self):
        """Test edge cases specifically for circular orbits."""
        # Test case 1: Very low Earth orbit (200 km altitude)
        leo_200_sat = SatelliteData(
            catalog_number=2001, name="LEO-200", epoch=datetime.now(),
            mean_motion=15.8, eccentricity=0.0, inclination=0.0,
            raan=0.0, arg_perigee=0.0, mean_anomaly=0.0,
            semi_major_axis=EARTH_RADIUS + 200, orbital_period=88.0
        )
        
        # Test case 2: High Earth orbit (2000 km altitude)
        heo_2000_sat = SatelliteData(
            catalog_number=2002, name="HEO-2000", epoch=datetime.now(),
            mean_motion=10.0, eccentricity=0.0, inclination=0.0,
            raan=0.0, arg_perigee=0.0, mean_anomaly=0.0,
            semi_major_axis=EARTH_RADIUS + 2000, orbital_period=150.0
        )
        
        leo_elements = tle_to_orbital_elements(leo_200_sat)
        heo_elements = tle_to_orbital_elements(heo_2000_sat)
        
        # Test extreme altitude difference transfer
        transfer_extreme = calculate_hohmann_transfer(leo_elements, heo_elements)
        
        # Should produce reasonable results
        self.assertGreater(transfer_extreme.departure_dv, 0)
        self.assertGreater(transfer_extreme.arrival_dv, 0)
        self.assertGreater(transfer_extreme.transfer_time, 0)
        
        # Total delta-v should be substantial for large altitude difference
        total_dv = transfer_extreme.departure_dv + transfer_extreme.arrival_dv
        self.assertGreater(total_dv, 0.5)  # Should be substantial for 1800 km altitude change
        
        # Test case 3: Minimal altitude difference (should produce small delta-v)
        leo_400_sat = SatelliteData(
            catalog_number=2003, name="LEO-400", epoch=datetime.now(),
            mean_motion=15.5, eccentricity=0.0, inclination=0.0,
            raan=0.0, arg_perigee=0.0, mean_anomaly=0.0,
            semi_major_axis=EARTH_RADIUS + 400, orbital_period=92.0
        )
        
        leo_410_sat = SatelliteData(
            catalog_number=2004, name="LEO-410", epoch=datetime.now(),
            mean_motion=15.4, eccentricity=0.0, inclination=0.0,
            raan=0.0, arg_perigee=0.0, mean_anomaly=0.0,
            semi_major_axis=EARTH_RADIUS + 410, orbital_period=93.0
        )
        
        leo_400_elements = tle_to_orbital_elements(leo_400_sat)
        leo_410_elements = tle_to_orbital_elements(leo_410_sat)
        
        transfer_small = calculate_hohmann_transfer(leo_400_elements, leo_410_elements)
        
        # Small altitude difference should produce small delta-v
        total_dv_small = transfer_small.departure_dv + transfer_small.arrival_dv
        self.assertLess(total_dv_small, 0.01)  # Should be very small for 10 km difference
    
    def test_edge_cases_eccentric_orbits(self):
        """Test edge cases for eccentric orbits."""
        # Test case 1: Highly eccentric orbit (e = 0.8)
        highly_eccentric_sat = SatelliteData(
            catalog_number=3001, name="HIGHLY-ECC", epoch=datetime.now(),
            mean_motion=12.0, eccentricity=0.8, inclination=0.0,
            raan=0.0, arg_perigee=0.0, mean_anomaly=0.0,
            semi_major_axis=10000.0, orbital_period=140.0
        )
        
        eccentric_elements = tle_to_orbital_elements(highly_eccentric_sat)
        
        # Test state vector calculation at different mean anomalies
        state_at_perigee = orbital_elements_to_state_vector(eccentric_elements, 0.0)
        
        # Modify elements for apogee (mean anomaly = π)
        eccentric_elements_apogee = OrbitalElements(
            semi_major_axis=eccentric_elements.semi_major_axis,
            eccentricity=eccentric_elements.eccentricity,
            inclination=eccentric_elements.inclination,
            raan=eccentric_elements.raan,
            arg_perigee=eccentric_elements.arg_perigee,
            mean_anomaly=math.pi
        )
        
        state_at_apogee = orbital_elements_to_state_vector(eccentric_elements_apogee, 0.0)
        
        # Calculate expected distances
        a = highly_eccentric_sat.semi_major_axis
        e = highly_eccentric_sat.eccentricity
        expected_perigee_distance = a * (1 - e)
        expected_apogee_distance = a * (1 + e)
        
        perigee_distance = math.sqrt(sum(x**2 for x in state_at_perigee.position))
        apogee_distance = math.sqrt(sum(x**2 for x in state_at_apogee.position))
        
        # Verify distances are correct
        self.assertAlmostEqual(perigee_distance, expected_perigee_distance, places=1)
        self.assertAlmostEqual(apogee_distance, expected_apogee_distance, places=1)
        
        # Test case 2: Nearly parabolic orbit (e = 0.99)
        nearly_parabolic_sat = SatelliteData(
            catalog_number=3002, name="NEARLY-PARABOLIC", epoch=datetime.now(),
            mean_motion=8.0, eccentricity=0.99, inclination=0.0,
            raan=0.0, arg_perigee=0.0, mean_anomaly=0.0,
            semi_major_axis=15000.0, orbital_period=200.0
        )
        
        parabolic_elements = tle_to_orbital_elements(nearly_parabolic_sat)
        state_parabolic = orbital_elements_to_state_vector(parabolic_elements, 0.0)
        
        # Should still produce valid state vector
        position_magnitude = math.sqrt(sum(x**2 for x in state_parabolic.position))
        velocity_magnitude = math.sqrt(sum(v**2 for v in state_parabolic.velocity))
        
        self.assertGreater(position_magnitude, 0)
        self.assertGreater(velocity_magnitude, 0)
        
        # Position should be at perigee for nearly parabolic orbit
        expected_perigee = nearly_parabolic_sat.semi_major_axis * (1 - nearly_parabolic_sat.eccentricity)
        self.assertAlmostEqual(position_magnitude, expected_perigee, places=1)
    
    def test_comprehensive_plane_change_scenarios(self):
        """Test comprehensive plane change scenarios with different orbital configurations."""
        # Create satellites with various inclination differences
        base_sat = SatelliteData(
            catalog_number=4001, name="BASE", epoch=datetime.now(),
            mean_motion=15.0, eccentricity=0.0, inclination=0.0,  # Equatorial
            raan=0.0, arg_perigee=0.0, mean_anomaly=0.0,
            semi_major_axis=7000.0, orbital_period=95.0
        )
        
        # Test different inclination targets
        inclination_scenarios = [
            ("POLAR", 90.0),      # Polar orbit
            ("SUN-SYNC", 98.0),   # Sun-synchronous
            ("MOLNIYA", 63.4),    # Molniya orbit inclination
            ("RETROGRADE", 120.0) # Retrograde orbit
        ]
        
        base_elements = tle_to_orbital_elements(base_sat)
        
        for name, target_inclination in inclination_scenarios:
            target_sat = SatelliteData(
                catalog_number=4000 + len(inclination_scenarios), name=name, epoch=datetime.now(),
                mean_motion=15.0, eccentricity=0.0, inclination=target_inclination,
                raan=0.0, arg_perigee=0.0, mean_anomaly=0.0,
                semi_major_axis=7000.0, orbital_period=95.0
            )
            
            target_elements = tle_to_orbital_elements(target_sat)
            combined_result = calculate_combined_maneuver_dv(base_elements, target_elements)
            
            # All scenarios should produce valid results
            self.assertGreater(combined_result.total_dv, 0)
            self.assertGreater(combined_result.plane_change_dv, 0)
            
            # Larger inclination changes should require more delta-v
            inclination_change_deg = abs(target_inclination - 0.0)
            if inclination_change_deg > 45:
                self.assertGreater(combined_result.plane_change_dv, 5.0)  # Should be substantial
            
            # Verify strategy selection makes sense
            self.assertIsNotNone(combined_result.maneuver_strategy)
            self.assertGreater(combined_result.optimal_maneuver_radius, 0)


if __name__ == '__main__':
    unittest.main()