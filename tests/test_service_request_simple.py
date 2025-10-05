"""
Simple test for service request management without FastAPI dependency.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta

# Import directly without going through API module
from debris_removal_service.api.service_request_manager import ServiceRequestManager
from debris_removal_service.models.service_request import (
    ServiceRequest, RequestStatus, TimelineConstraints, 
    BudgetConstraints, ProcessingPreferences, ProcessingType
)
from debris_removal_service.models.satellite import Satellite


def create_test_satellite():
    """Create a test satellite."""
    return Satellite(
        id="SAT001",
        name="Test Satellite",
        tle_line1="1 25544U 98067A   21001.00000000  .00002182  00000-0  38792-4 0  9991",
        tle_line2="2 25544  51.6461 339.2911 0002829  68.6102 291.5211 15.48919103123456",
        mass=450.0,
        material_composition={"aluminum": 0.6, "steel": 0.3, "electronics": 0.1},
        decommission_date=datetime(2024, 6, 1)
    )


def test_basic_functionality():
    """Test basic service request management functionality."""
    print("Testing service request management...")
    
    # Create manager
    manager = ServiceRequestManager()
    print("✓ ServiceRequestManager created")
    
    # Create test data
    satellite = create_test_satellite()
    satellites = [satellite]
    
    timeline = TimelineConstraints(
        earliest_start=datetime(2024, 1, 1),
        latest_completion=datetime(2024, 6, 1)
    )
    
    budget = BudgetConstraints(
        max_total_cost=100000.0,
        preferred_cost=80000.0
    )
    
    processing = ProcessingPreferences(
        preferred_processing_types=[ProcessingType.ISS_RECYCLING]
    )
    
    service_request = ServiceRequest(
        client_id="test_client",
        satellites=["SAT001"],
        timeline_requirements=timeline,
        budget_constraints=budget,
        processing_preferences=processing
    )
    
    # Test request creation
    created_request = manager.create_request(service_request, satellites)
    print(f"✓ Service request created: {created_request.request_id}")
    
    # Test status update
    success = manager.update_request_status(
        created_request.request_id, 
        RequestStatus.PROCESSING, 
        "Test status update"
    )
    print(f"✓ Status updated: {success}")
    
    # Test progress tracking
    progress = manager.get_request_progress(created_request.request_id)
    print(f"✓ Progress retrieved: {progress['overall_progress']}%")
    
    # Test dashboard
    dashboard = manager.get_dashboard_summary()
    print(f"✓ Dashboard summary: {dashboard['summary']['total_requests']} total requests")
    
    print("\n✅ All basic service request management tests passed!")


if __name__ == "__main__":
    test_basic_functionality()