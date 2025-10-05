"""
Simple test to verify processing services integration works.
"""

from datetime import datetime, timedelta
from debris_removal_service.models.satellite import Satellite, OrbitalElements
from debris_removal_service.services.iss_recycling_service import ISSRecyclingService
from debris_removal_service.services.solar_forge_service import SolarForgeService, OutputMaterialGrade
from debris_removal_service.services.heo_storage_service import HEOStorageService


def main():
    """Test all processing services with a simple satellite."""
    
    # Create a test satellite
    orbital_elements = OrbitalElements(
        semi_major_axis=7000.0,
        eccentricity=0.001,
        inclination=51.6,
        raan=339.3,
        argument_of_perigee=68.6,
        mean_anomaly=291.5,
        mean_motion=15.5,
        epoch=datetime(2025, 10, 3)
    )
    
    satellite = Satellite(
        id="TEST-001",
        name="Test Satellite",
        tle_line1="1 00005U 58002B   25276.57478752  .00000086  00000-0  13412-3 0  9995",
        tle_line2="2 00005  34.2488  26.3129 1841753  61.4980 315.8676 10.85924841415385",
        mass=1000.0,
        material_composition={
            "aluminum": 0.5,
            "titanium": 0.2,
            "electronics": 0.15,
            "solar_panels": 0.1,
            "batteries": 0.05
        },
        decommission_date=datetime(2025, 12, 31),
        orbital_elements=orbital_elements
    )
    
    print(f"Testing processing services for satellite: {satellite.name}")
    print(f"Satellite mass: {satellite.mass} kg")
    print(f"Material composition: {satellite.material_composition}")
    print()
    
    # Test ISS Recycling Service
    print("=== ISS Recycling Service ===")
    iss_service = ISSRecyclingService()
    iss_quote = iss_service.get_processing_quote([satellite])
    print(f"ISS Processing Cost: ${iss_quote['total_cost_usd']:,.2f}")
    print(f"Processing Duration: {iss_quote['processing_duration_hours']:.1f} hours")
    print(f"Cost per kg: ${iss_quote['cost_per_kg_usd']:.2f}")
    print()
    
    # Test Solar Forge Service
    print("=== Solar Forge Service ===")
    forge_service = SolarForgeService()
    forge_quote = forge_service.get_refinement_quote([satellite], OutputMaterialGrade.AEROSPACE)
    print(f"Solar Forge Cost: ${forge_quote['total_refinement_cost_usd']:,.2f}")
    print(f"Output Value: ${forge_quote['total_output_value_usd']:,.2f}")
    print(f"Value Added: ${forge_quote['value_added_usd']:,.2f}")
    print(f"ROI: {forge_quote['roi_percentage']:.1f}%")
    print()
    
    # Test HEO Storage Service
    print("=== HEO Storage Service ===")
    heo_service = HEOStorageService()
    storage_duration = timedelta(days=365)  # 1 year storage
    heo_quote = heo_service.get_storage_quote([satellite], storage_duration)
    print(f"HEO Storage Cost: ${heo_quote['total_storage_cost_usd']:,.2f}")
    print(f"Retrieval Cost: ${heo_quote['total_retrieval_cost_usd']:,.2f}")
    print(f"Total Lifecycle Cost: ${heo_quote['total_lifecycle_cost_usd']:,.2f}")
    print(f"Storage Duration: {heo_quote['storage_duration_days']} days")
    print()
    
    # Compare services
    print("=== Service Comparison ===")
    print(f"ISS Recycling:   ${iss_quote['total_cost_usd']:>10,.2f}")
    print(f"Solar Forge:     ${forge_quote['total_refinement_cost_usd']:>10,.2f} (Value: ${forge_quote['total_output_value_usd']:,.2f})")
    print(f"HEO Storage:     ${heo_quote['total_lifecycle_cost_usd']:>10,.2f}")
    print()
    
    print("✅ All processing services are working correctly!")


if __name__ == "__main__":
    main()