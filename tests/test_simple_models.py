"""
Simple test for basic models without TLE parsing.
"""

import unittest
from datetime import datetime, timedelta
from debris_removal_service.models.cost import PropellantMass, MissionCost, DetailedCost
from debris_removal_service.models.service_request import (
    ServiceRequest, TimelineConstraints, BudgetConstraints, 
    ProcessingPreferences, RequestStatus, ProcessingType
)


class TestSimpleModels(unittest.TestCase):
    """Test basic models without complex TLE parsing."""
    
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
    
    def test_detailed_cost(self):
        """Test detailed cost calculations."""
        cost = DetailedCost(
            propellant_cost=500.0,
            operational_cost=300.0,
            processing_cost=200.0,
            storage_cost=100.0,
            total_cost=1100.0
        )
        
        self.assertEqual(cost.total_cost, 1100.0)
        
        percentages = cost.get_cost_percentages()
        self.assertAlmostEqual(percentages['propellant'], 45.45, places=1)
        
        largest_component, value = cost.get_largest_cost_component()
        self.assertEqual(largest_component, 'propellant')
        self.assertEqual(value, 500.0)
    
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
    
    def test_timeline_constraints(self):
        """Test timeline constraints."""
        timeline = TimelineConstraints(
            earliest_start=datetime(2024, 1, 1),
            latest_completion=datetime(2024, 12, 31)
        )
        
        duration = timeline.get_available_duration()
        self.assertGreater(duration.days, 300)  # About a year
        
        # Test time availability
        test_start = datetime(2024, 6, 1)
        test_duration = timedelta(days=30)
        self.assertTrue(timeline.is_time_available(test_start, test_duration))
    
    def test_budget_constraints(self):
        """Test budget constraints."""
        budget = BudgetConstraints(
            max_total_cost=100000.0,
            preferred_cost=80000.0
        )
        
        self.assertTrue(budget.is_within_budget(75000.0))
        self.assertFalse(budget.is_within_budget(150000.0))
    
    def test_processing_preferences(self):
        """Test processing preferences."""
        processing = ProcessingPreferences(
            preferred_processing_types=[ProcessingType.ISS_RECYCLING, ProcessingType.HEO_STORAGE]
        )
        
        priority = processing.get_processing_priority(ProcessingType.ISS_RECYCLING)
        self.assertEqual(priority, 2)  # First in list gets highest priority
        
        priority = processing.get_processing_priority(ProcessingType.SOLAR_FORGE)
        self.assertEqual(priority, 0)  # Not in list gets 0 priority
    
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
        
        # Test urgency check
        self.assertFalse(request.is_urgent())  # Year-long timeline is not urgent
        
        # Test summary
        summary = request.get_request_summary()
        self.assertEqual(summary['satellite_count'], 2)
        self.assertEqual(summary['max_budget_usd'], 100000.0)


if __name__ == '__main__':
    unittest.main()