"""
Test processing services integration functionality.
"""

from datetime import datetime, timedelta
from debris_removal_service.models.satellite import Satellite, OrbitalElements
from debris_removal_service.services.iss_recycling_service import ISSRecyclingService, MaterialType
from debris_removal_service.services.solar_forge_service import SolarForgeService, OutputMaterialGrade
from debris_removal_service.services.heo_storage_service import HEOStorageService, StorageOrbitType


def create_test_satellite():
    """Create a test satellite for processing services tests."""
    from debris_removal_service.models.satellite import OrbitalElements
    
    # Create orbital elements manually to avoid TLE parsing issues
    orbital_elements = OrbitalElements(
        semi_major_axis=7000.0,  # km
        eccentricity=0.001,
        inclination=51.6,  # degrees
        raan=339.3,  # degrees
        argument_of_perigee=68.6,  # degrees
        mean_anomaly=291.5,  # degrees
        mean_motion=15.5,  # revolutions per day
        epoch=datetime(2025, 10, 3)
    )
    
    satellite = Satellite(
        id="TEST-SAT-001",
        name="Test Satellite",
        tle_line1="1 00005U 58002B   25276.57478752  .00000086  00000-0  13412-3 0  9995",
        tle_line2="2 00005  34.2488  26.3129 1841753  61.4980 315.8676 10.85924841415385",
        mass=500.0,
        material_composition={
            "aluminum": 0.4,
            "titanium": 0.2,
            "electronics": 0.15,
            "solar_panels": 0.1,
            "carbon_fiber": 0.1,
            "batteries": 0.05
        },
        decommission_date=datetime(2025, 12, 31),
        orbital_elements=orbital_elements  # Provide pre-computed orbital elements
    )
    
    return satellite


def test_iss_recycling_service_basic():
    """Test basic ISS recycling service functionality."""
    service = ISSRecyclingService()
    
    # Test cost calculation for aluminum
    cost = service.calculate_processing_cost("aluminum", 100.0)
    assert cost > 0
    assert isinstance(cost, float)
    
    # Test processing time estimation
    processing_time = service.estimate_processing_time("aluminum", 100.0)
    assert isinstance(processing_time, timedelta)
    assert processing_time.total_seconds() > 0
    
    # Test capacity availability check
    start_time = datetime.utcnow() + timedelta(days=30)
    duration = timedelta(days=5)
    availability = service.check_capacity_availability(start_time, duration)
    assert isinstance(availability, bool)


def test_iss_recycling_satellite_processing():
    """Test ISS recycling service with complete satellite."""
    service = ISSRecyclingService()
    satellite = create_test_satellite()
    
    # Test satellite processing cost calculation
    detailed_cost = service.calculate_satellite_processing_cost(satellite)
    assert detailed_cost.total_cost > 0
    assert detailed_cost.processing_cost >= 0
    assert detailed_cost.operational_cost >= 0
    assert detailed_cost.storage_cost >= 0  # Transport cost
    
    # Test satellite processing time estimation
    processing_time = service.estimate_satellite_processing_time(satellite)
    assert isinstance(processing_time, timedelta)
    assert processing_time.total_seconds() > 0
    
    # Test comprehensive quote generation
    quote = service.get_processing_quote([satellite])
    assert quote['satellite_count'] == 1
    assert quote['total_cost_usd'] > 0
    assert quote['total_mass_kg'] == satellite.mass
    assert 'estimated_start_time' in quote
    assert 'detailed_satellite_costs' in quote


def test_solar_forge_service_basic():
    """Test basic solar forge service functionality."""
    service = SolarForgeService()
    
    # Test refinement cost calculation
    cost = service.calculate_refinement_cost("aluminum", 100.0, OutputMaterialGrade.AEROSPACE)
    assert cost > 0
    assert isinstance(cost, float)
    
    # Test refinement time estimation
    refinement_time = service.estimate_refinement_time("aluminum", 100.0, OutputMaterialGrade.AEROSPACE)
    assert isinstance(refinement_time, timedelta)
    assert refinement_time.total_seconds() > 0
    
    # Test output material prediction
    input_materials = [("aluminum", 100.0), ("titanium", 50.0)]
    output_materials = service.get_output_materials(input_materials, OutputMaterialGrade.AEROSPACE)
    assert len(output_materials) > 0
    for material in output_materials:
        assert material.mass_kg > 0
        assert material.market_value_per_kg > 0


def test_solar_forge_satellite_processing():
    """Test solar forge service with complete satellite."""
    service = SolarForgeService()
    satellite = create_test_satellite()
    
    # Test satellite refinement cost calculation
    detailed_cost = service.calculate_satellite_refinement_cost(satellite, OutputMaterialGrade.AEROSPACE)
    assert detailed_cost.total_cost > 0
    assert detailed_cost.processing_cost >= 0
    assert detailed_cost.operational_cost >= 0
    
    # Test comprehensive quote generation
    quote = service.get_refinement_quote([satellite], OutputMaterialGrade.AEROSPACE)
    assert quote['satellite_count'] == 1
    assert quote['total_refinement_cost_usd'] > 0
    assert quote['total_input_mass_kg'] == satellite.mass
    assert quote['total_output_value_usd'] > 0
    assert 'detailed_satellite_costs' in quote
    assert 'selected_station' in quote


def test_heo_storage_service_basic():
    """Test basic HEO storage service functionality."""
    service = HEOStorageService()
    
    # Test storage orbit calculation
    orbit = service.calculate_storage_orbit(1000.0, timedelta(days=365))
    assert orbit.storage_capacity_kg > 0
    assert orbit.delta_v_from_leo_ms > 0
    
    # Test storage cost calculation
    cost = service.calculate_storage_cost(1000.0, timedelta(days=365))
    assert cost > 0
    assert isinstance(cost, float)
    
    # Test retrieval cost calculation
    retrieval_cost = service.calculate_retrieval_cost(orbit, 1000.0)
    assert retrieval_cost > 0
    assert isinstance(retrieval_cost, float)


def test_heo_storage_satellite_processing():
    """Test HEO storage service with complete satellite."""
    service = HEOStorageService()
    satellite = create_test_satellite()
    storage_duration = timedelta(days=730)  # 2 years
    
    # Test storage allocation optimization
    allocations = service.optimize_storage_allocation([satellite], storage_duration)
    assert len(allocations) == 1
    allocation = allocations[0]
    assert allocation.satellite_id == satellite.id
    assert allocation.total_mass_kg == satellite.mass
    assert allocation.total_storage_cost > 0
    assert len(allocation.storage_containers) > 0
    
    # Test comprehensive quote generation
    quote = service.get_storage_quote([satellite], storage_duration)
    assert quote['satellite_count'] == 1
    assert quote['total_storage_cost_usd'] > 0
    assert quote['total_mass_kg'] == satellite.mass
    assert quote['storage_duration_days'] == 730
    assert 'storage_allocations' in quote
    assert 'total_containers_required' in quote


def test_processing_services_integration():
    """Test integration between all processing services."""
    iss_service = ISSRecyclingService()
    forge_service = SolarForgeService()
    heo_service = HEOStorageService()
    
    satellite = create_test_satellite()
    
    # Get quotes from all services
    iss_quote = iss_service.get_processing_quote([satellite])
    forge_quote = forge_service.get_refinement_quote([satellite], OutputMaterialGrade.AEROSPACE)
    heo_quote = heo_service.get_storage_quote([satellite], timedelta(days=365))
    
    # Verify all services can process the same satellite
    assert iss_quote['satellite_count'] == 1
    assert forge_quote['satellite_count'] == 1
    assert heo_quote['satellite_count'] == 1
    
    # Verify cost structures are different but reasonable
    assert iss_quote['total_cost_usd'] != forge_quote['total_refinement_cost_usd']
    assert forge_quote['total_refinement_cost_usd'] != heo_quote['total_storage_cost_usd']
    
    # Verify all services provide comprehensive information
    assert 'service_capabilities' in iss_quote
    assert 'service_capabilities' in forge_quote
    assert 'service_capabilities' in heo_quote


if __name__ == "__main__":
    # Run basic tests
    test_iss_recycling_service_basic()
    test_iss_recycling_satellite_processing()
    test_solar_forge_service_basic()
    test_solar_forge_satellite_processing()
    test_heo_storage_service_basic()
    test_heo_storage_satellite_processing()
    test_processing_services_integration()
    
    print("All processing services integration tests passed!")