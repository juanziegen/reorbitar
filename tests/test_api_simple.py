"""
Simple test for core API endpoints without pytest dependency.
"""

from fastapi.testclient import TestClient
from debris_removal_service.api.main import app

def test_basic_endpoints():
    """Test basic API endpoints."""
    client = TestClient(app)
    
    print("Testing root endpoint...")
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Root endpoint: {data['service']}")
    
    print("Testing health check...")
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Health check: {data['status']}")
    
    print("Testing satellite data retrieval...")
    response = client.get("/api/satellite/SAT001")
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Satellite data: {data['satellite']['name']}")
    
    print("Testing satellite listing...")
    response = client.get("/api/satellites")
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Satellite listing: {len(data)} satellites found")
    
    print("\nAll basic API tests passed!")

if __name__ == "__main__":
    test_basic_endpoints()