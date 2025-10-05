"""
Test service request management functionality.

This test verifies client request processing and validation,
mission status tracking and progress updates, and request
history and quote management functionality.
"""

from datetime import datetime, timedelta
from debris_removal_service.api.service_request_manager import ServiceRequestManager
from debris_removal_service.models.service_request import (
    ServiceRequest, RequestStatus, TimelineConstraints, 
    BudgetConstraints, ProcessingPreferences, ProcessingType
)
from debris_removal_service.models.satellite import Satellite, OrbitalElements


def create_test_satellites():
    """Create test satellites for validation."""
    satellites = []
    
    # Create sample satellites
    for i in range(3):
        sat_id = f"SAT{i+1:03d}"
        satellite = Satellite(
            id=sat_id,
            name=f"Test Satellite {i+1}",
            tle_line1=f"1 2554{i}U 98067A   21001.00000000  .00002182  00000-0  38792-4 0  999{i}",
            tle_line2=f"2 2554{i}  51.6461 339.2911 0002829  68.6102 291.5211 15.48919103123456",
            mass=450.0 + i * 50,
            material_composition={"aluminum": 0.6, "steel": 0.3, "electronics": 0.1},
            decommission_date=datetime(2024, 6, 1)
        )
        satellites.append(satellite)
    
    return satellites


def create_test_service_request():
    """Create a test service request."""
    timeline = TimelineConstraints(
        earliest_start=datetime(2024, 1, 1),
        latest_completion=datetime(2024, 6, 1)
    )
    
    budget = BudgetConstraints(
        max_total_cost=100000.0,
        preferred_cost=80000.0
    )
    
    processing = ProcessingPreferences(
        preferred_processing_types=[ProcessingType.ISS_RECYCLING, ProcessingType.HEO_STORAGE]
    )
    
    return ServiceRequest(
        client_id="test_client_001",
        satellites=["SAT001", "SAT002", "SAT003"],
        timeline_requirements=timeline,
        budget_constraints=budget,
        processing_preferences=processing
    )


def test_service_request_creation():
    """Test service request creation and validation."""
    print("Testing service request creation...")
    
    manager = ServiceRequestManager()
    satellites = create_test_satellites()
    service_request = create_test_service_request()
    
    # Test successful creation
    created_request = manager.create_request(service_request, satellites)
    assert created_request.request_id is not None
    assert created_request.status == RequestStatus.PENDING
    print("✓ Service request created successfully")
    
    # Test retrieval
    retrieved_request = manager.get_request(created_request.request_id)
    assert retrieved_request is not None
    assert retrieved_request.client_id == "test_client_001"
    print("✓ Service request retrieved successfully")
    
    return created_request.request_id


def test_status_tracking():
    """Test mission status tracking and progress updates."""
    print("Testing status tracking...")
    
    manager = ServiceRequestManager()
    satellites = create_test_satellites()
    service_request = create_test_service_request()
    
    created_request = manager.create_request(service_request, satellites)
    request_id = created_request.request_id
    
    # Test status updates
    success = manager.update_request_status(
        request_id, 
        RequestStatus.PROCESSING, 
        "Starting route optimization"
    )
    assert success
    print("✓ Status updated to PROCESSING")
    
    # Test progress tracking
    progress = manager.get_request_progress(request_id)
    assert progress is not None
    assert progress["overall_progress"] == 25.0
    assert progress["current_phase"] == "mission_planning"
    print("✓ Progress tracking working")
    
    # Test multiple status updates
    manager.update_request_status(request_id, RequestStatus.QUOTED, "Quote generated")
    manager.update_request_status(request_id, RequestStatus.APPROVED, "Client approved")
    
    final_progress = manager.get_request_progress(request_id)
    assert final_progress["overall_progress"] == 60.0
    print("✓ Multiple status updates working")
    
    return request_id


def test_request_history():
    """Test request history and management functionality."""
    print("Testing request history...")
    
    manager = ServiceRequestManager()
    satellites = create_test_satellites()
    service_request = create_test_service_request()
    
    created_request = manager.create_request(service_request, satellites)
    request_id = created_request.request_id
    
    # Add some history
    manager.update_request_status(request_id, RequestStatus.PROCESSING, "Processing started")
    manager.update_request_status(request_id, RequestStatus.QUOTED, "Quote ready")
    
    # Test history retrieval
    history = manager.get_request_history(request_id)
    assert len(history) >= 3  # Created + 2 updates
    assert history[0]["action"] == "created"
    print("✓ Request history tracking working")
    
    return request_id


def test_client_request_listing():
    """Test listing requests for a client."""
    print("Testing client request listing...")
    
    manager = ServiceRequestManager()
    satellites = create_test_satellites()
    
    # Create multiple requests for the same client
    for i in range(3):
        service_request = create_test_service_request()
        service_request.client_id = "test_client_multi"
        manager.create_request(service_request, satellites)
    
    # Test listing all requests
    requests = manager.list_client_requests("test_client_multi")
    assert len(requests) == 3
    print("✓ Client request listing working")
    
    # Test status filtering
    # Update one request status
    first_request = requests[0]
    manager.update_request_status(first_request.request_id, RequestStatus.PROCESSING)
    
    processing_requests = manager.list_client_requests(
        "test_client_multi", 
        RequestStatus.PROCESSING
    )
    assert len(processing_requests) == 1
    print("✓ Status filtering working")


def test_request_approval_cancellation():
    """Test request approval and cancellation."""
    print("Testing request approval and cancellation...")
    
    manager = ServiceRequestManager()
    satellites = create_test_satellites()
    
    # Test approval
    service_request1 = create_test_service_request()
    created_request1 = manager.create_request(service_request1, satellites)
    manager.update_request_status(created_request1.request_id, RequestStatus.QUOTED)
    
    success = manager.approve_request(created_request1.request_id, "Client approved quote")
    assert success
    
    approved_request = manager.get_request(created_request1.request_id)
    assert approved_request.status == RequestStatus.APPROVED
    print("✓ Request approval working")
    
    # Test cancellation
    service_request2 = create_test_service_request()
    created_request2 = manager.create_request(service_request2, satellites)
    
    success = manager.cancel_request(created_request2.request_id, "Client cancelled")
    assert success
    
    cancelled_request = manager.get_request(created_request2.request_id)
    assert cancelled_request.status == RequestStatus.CANCELLED
    print("✓ Request cancellation working")


def test_dashboard_summary():
    """Test dashboard summary functionality."""
    print("Testing dashboard summary...")
    
    manager = ServiceRequestManager()
    satellites = create_test_satellites()
    
    # Create some requests with different statuses
    for i in range(5):
        service_request = create_test_service_request()
        service_request.client_id = f"dashboard_client_{i}"
        created_request = manager.create_request(service_request, satellites)
        
        # Set different statuses
        if i == 0:
            manager.update_request_status(created_request.request_id, RequestStatus.COMPLETED)
        elif i == 1:
            manager.update_request_status(created_request.request_id, RequestStatus.IN_PROGRESS)
        elif i == 2:
            manager.update_request_status(created_request.request_id, RequestStatus.QUOTED)
    
    # Test dashboard
    dashboard = manager.get_dashboard_summary()
    assert dashboard["summary"]["total_requests"] >= 5
    assert "status_breakdown" in dashboard
    assert "recent_activity" in dashboard
    assert "performance_metrics" in dashboard
    print("✓ Dashboard summary working")


def test_validation_errors():
    """Test request validation error handling."""
    print("Testing validation errors...")
    
    manager = ServiceRequestManager()
    satellites = create_test_satellites()
    
    # Test invalid satellite ID
    invalid_request = create_test_service_request()
    invalid_request.satellites = ["INVALID_SAT"]
    
    try:
        manager.create_request(invalid_request, satellites)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "not found" in str(e)
        print("✓ Invalid satellite ID validation working")
    
    # Test invalid timeline
    invalid_timeline_request = create_test_service_request()
    invalid_timeline_request.timeline_requirements.latest_completion = datetime(2023, 1, 1)  # Before start
    
    try:
        manager.create_request(invalid_timeline_request, satellites)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "before" in str(e)
        print("✓ Invalid timeline validation working")
    
    # Test insufficient budget
    low_budget_request = create_test_service_request()
    low_budget_request.budget_constraints.max_total_cost = 100.0  # Too low
    
    try:
        manager.create_request(low_budget_request, satellites)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "Budget too low" in str(e)
        print("✓ Budget validation working")


def main():
    """Run all service request management tests."""
    print("Running Service Request Management Tests...")
    print("=" * 50)
    
    try:
        test_service_request_creation()
        print()
        
        test_status_tracking()
        print()
        
        test_request_history()
        print()
        
        test_client_request_listing()
        print()
        
        test_request_approval_cancellation()
        print()
        
        test_dashboard_summary()
        print()
        
        test_validation_errors()
        print()
        
        print("=" * 50)
        print("✅ All service request management tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()