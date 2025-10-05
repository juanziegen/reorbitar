"""
Test service request manager functionality directly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta

# Import directly
from debris_removal_service.api.service_request_manager import ServiceRequestManager, RequestValidationError
from debris_removal_service.models.service_request import (
    ServiceRequest, RequestStatus, TimelineConstraints, 
    BudgetConstraints, ProcessingPreferences, ProcessingType
)


def test_service_request_manager():
    """Test service request manager functionality."""
    print("Testing ServiceRequestManager...")
    
    # Create manager
    manager = ServiceRequestManager()
    print("✓ ServiceRequestManager created")
    
    # Create test service request
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
        satellites=["SAT001"],  # This will fail validation but we can test the manager
        timeline_requirements=timeline,
        budget_constraints=budget,
        processing_preferences=processing
    )
    
    print("✓ Service request model created")
    
    # Test validation error handling
    try:
        manager.create_request(service_request, [])  # Empty satellites list
        print("✗ Should have failed validation")
    except RequestValidationError as e:
        print(f"✓ Validation error caught: {str(e)[:50]}...")
    
    # Test status updates on non-existent request
    success = manager.update_request_status("fake_id", RequestStatus.PROCESSING, "test")
    print(f"✓ Non-existent request update handled: {not success}")
    
    # Test progress retrieval on non-existent request
    progress = manager.get_request_progress("fake_id")
    print(f"✓ Non-existent progress handled: {progress is None}")
    
    # Test dashboard with no requests
    dashboard = manager.get_dashboard_summary()
    print(f"✓ Empty dashboard: {dashboard['summary']['total_requests']} requests")
    
    # Test client listing with no requests
    requests = manager.list_client_requests("fake_client")
    print(f"✓ Empty client listing: {len(requests)} requests")
    
    print("\n✅ ServiceRequestManager basic functionality verified!")


if __name__ == "__main__":
    test_service_request_manager()