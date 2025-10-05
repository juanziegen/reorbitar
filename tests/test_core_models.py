"""
Test the core data models for the satellite debris removal service.
"""

import unittest
from datetime import datetime, timedelta
from debris_removal_service.models import (
    Satellite, OrbitalElements, Route, Hop, ManeuverDetails,
    PropellantMass, MissionCost, DetailedCost,
    ServiceRequest, TimelineConstraints, BudgetConstraints, 
    ProcessingPreferences, RequestStatus, ProcessingType
)
from debris_removal_service.utils import SatelliteDataValidator, TLEParser


class TestCoreModels(unittest.TestCase):
    """Test core data models functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Sample TLE data for testing (ISS) - valid checksums
        self.sample_tle_line1 = "1 25544U 98067A   21001.00000000  .00002182  00000-0  40768-4 0  9992"
        self.sample_tle_line2 = "2 25544  51.6461 339.2911 0002829 242.9350 117.0717 15.48919103123456"
        
        self.sample_satellite = Satellite(
            id="25544",
            name="ISS (ZARYA)",
            tle_line1=self.sample_tle_line1,
            tle_line2=self.sample_tle_line2,
            mass=420000.0,
            material_composition={
                'aluminum': 0.4,
                'steel': 0.3,
                'electronics': 0.2,
                'other': 0.1
            },
            decommission_date=datetime(2030, 1, 1)
        )
    
    def test_satellite_creation(self):
        """Test satellite object creation and validation."""
        satellite = self.sample_satellite
        
        # Check basic properties
        self.assertEqual(satellite.id, "25544")
        self.assertEqual(satellite.name, "ISS (ZARYA)")
        self.assertEqual(satellite.mass, 420000.0)
        
        # Check orbital elements were parsed
        self.assertIsNotNone(satellite.orbital_elements)
        self.assertGreater(satellite.orbital_elements.semi_major_axis, 6371)  # Above Earth's surface
        
        # Check validation
        self.assertTrue(satellite.is_valid())
    
    def test_propellant_mass(self):
        """Test propellant mass calculations."""
        propellant = PropellantMass(
            fuel_kg=10.0,
            oxidizer_kg=19.0,
            total_kg=29.0
        )
        
        self.assertEqual(propellant.total_kg, 29.0)
        self.assertAlmostEqual(propellant.get_mass_ratio(), 1.9, places=1)
        
        fuel_vol, ox_vol = propellant.get_volume_estimate()
        self.assertGreater(fuel_vol, 0)
        self.assertGreater(ox_vol, 0)
    
    def test_mission_cost(self):
        """Test mission cost calculations."""
        cost = MissionCost(
            collection_cost=1000.0,
            processing_cost=500.0,
            storage_cost=200.0,
            operational_overhead=300.0,
            total_cost=2000.0,
            cost_per_satellite=500.0
        )
        
        self.assertEqual(cost.total_cost, 2000.0)
        
        summary = cost.get_cost_summary()
        self.assertEqual(summary['total_cost_usd'], 2000.0)
        
        metrics = cost.get_cost_efficiency_metrics(4, 1000.0)
        self.assertEqual(metrics['cost_per_satellite'], 500.0)
        self.assertEqual(metrics['cost_per_delta_v_ms'], 2.0)
    
    def test_service_request(self):
        """Test service request creation and validation."""
        timeline = TimelineConstraints(
            earliest_start=datetime(2024, 1, 1),
            latest_completion=datetime(2024, 12, 31)
        )
        
        budget = BudgetConstraints(
            max_total_cost=100000.0,
            preferred_cost=80000.0
        )
        
        processing = ProcessingPreferences(
            preferred_processing_types=[ProcessingType.ISS_RECYCLING, ProcessingType.HEO_STORAGE]
        )
        
        request = ServiceRequest(
            client_id="client_001",
            satellites=["25544", "12345"],
            timeline_requirements=timeline,
            budget_constraints=budget,
            processing_preferences=processing
        )
        
        self.assertEqual(request.client_id, "client_001")
        self.assertEqual(request.get_satellite_count(), 2)
        self.assertEqual(request.status, RequestStatus.PENDING)
        
        # Test status update
        request.update_status(RequestStatus.PROCESSING, "Started processing")
        self.assertEqual(request.status, RequestStatus.PROCESSING)
        self.assertIsNotNone(request.notes)
    
    def test_satellite_validator(self):
        """Test satellite data validation."""
        is_valid, errors = SatelliteDataValidator.validate_satellite(self.sample_satellite)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Test invalid satellite
        invalid_satellite = Satellite(
            id="",  # Invalid empty ID
            name="Test",
            tle_line1="invalid_tle",
            tle_line2="invalid_tle",
            mass=-100.0,  # Invalid negative mass
            material_composition={'aluminum': 2.0},  # Invalid composition > 1.0
            decommission_date=datetime(2024, 1, 1)
        )
        
        is_valid, errors = SatelliteDataValidator.validate_satellite(invalid_satellite)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_tle_parser(self):
        """Test TLE parsing functionality."""
        # Test orbital info extraction
        orbital_info = TLEParser.extract_orbital_info(self.sample_tle_line1, self.sample_tle_line2)
        
        self.assertIn('inclination', orbital_info)
        self.assertIn('semi_major_axis', orbital_info)
        self.assertGreater(orbital_info['semi_major_axis'], 6371)
        
        # Test TLE validation
        is_valid, errors = TLEParser.validate_tle_format(self.sample_tle_line1, self.sample_tle_line2)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Test satellite creation from TLE
        satellite = TLEParser.parse_tle_lines("TEST SAT", self.sample_tle_line1, self.sample_tle_line2)
        self.assertIsNotNone(satellite)
        self.assertEqual(satellite.name, "TEST SAT")


if __name__ == '__main__':
    unittest.main()