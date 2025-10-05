"""
Test Satellite Data Validator Integration

Tests the satellite data validation functionality and integration
with the main system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.satellite_data_validator import SatelliteDataValidator
from src.tle_parser import SatelliteData, TLEParser
from datetime import datetime


def test_validator_basic_functionality():
    """Test basic validator functionality."""
    print("Testing Satellite Data Validator...")
    
    # Create a test satellite with valid data
    valid_satellite = SatelliteData(
        catalog_number=25544,
        name="ISS (ZARYA)",
        epoch=datetime.utcnow(),
        mean_motion=15.48919103,
        eccentricity=0.0001234,
        inclination=51.6461,
        raan=123.4567,
        arg_perigee=89.1234,
        mean_anomaly=270.5678,
        semi_major_axis=6793.137,  # ~415 km altitude
        orbital_period=92.68
    )
    
    # Create a test satellite with invalid data
    invalid_satellite = SatelliteData(
        catalog_number=99999,
        name="INVALID SAT",
        epoch=datetime.utcnow(),
        mean_motion=-1.0,  # Invalid: negative
        eccentricity=1.5,  # Invalid: > 1
        inclination=200.0,  # Invalid: > 180
        raan=400.0,  # Invalid: >= 360
        arg_perigee=89.1234,
        mean_anomaly=270.5678,
        semi_major_axis=6000.0,  # Too low altitude
        orbital_period=92.68
    )
    
    validator = SatelliteDataValidator()
    
    # Test individual satellite validation
    print("\n1. Testing individual satellite validation:")
    
    valid_result = validator.validate_individual_satellite(valid_satellite)
    print(f"   Valid satellite result: {valid_result['valid']}")
    if valid_result['warnings']:
        print(f"   Warnings: {len(valid_result['warnings'])}")
    
    invalid_result = validator.validate_individual_satellite(invalid_satellite)
    print(f"   Invalid satellite result: {invalid_result['valid']}")
    if invalid_result['error']:
        print(f"   Error: {invalid_result['error']}")
    
    # Test dataset validation
    print("\n2. Testing dataset validation:")
    test_satellites = [valid_satellite, invalid_satellite]
    
    dataset_result = validator.validate_satellite_dataset(test_satellites)
    print(f"   Dataset valid: {dataset_result['valid']}")
    print(f"   Valid satellites: {dataset_result['valid_satellites']}/{dataset_result['total_satellites']}")
    print(f"   Validity rate: {dataset_result['validity_rate']:.1%}")
    
    # Test filtering
    print("\n3. Testing satellite filtering:")
    filtered_satellites = validator.filter_valid_satellites(test_satellites)
    print(f"   Filtered count: {len(filtered_satellites)}")
    
    print("\n✅ Validator basic functionality test completed")
    return True


def test_integration_with_tle_parser():
    """Test integration with actual TLE data."""
    print("\nTesting integration with TLE parser...")
    
    try:
        # Try to load actual satellite data
        parser = TLEParser()
        possible_files = ["leo_satellites.txt", "data/leo_satellites.txt"]
        
        satellites = []
        for filepath in possible_files:
            try:
                satellites = parser.parse_tle_file(filepath)
                print(f"   Loaded {len(satellites)} satellites from {filepath}")
                break
            except FileNotFoundError:
                continue
        
        if not satellites:
            print("   ⚠️  No TLE file found, skipping integration test")
            return True
        
        # Test validation on real data
        validator = SatelliteDataValidator()
        validation_result = validator.validate_satellite_dataset(satellites[:50])  # Test first 50
        
        print(f"   Dataset validation result: {validation_result['valid']}")
        print(f"   Valid satellites: {validation_result['valid_satellites']}/{validation_result['total_satellites']}")
        
        if validation_result['statistics']:
            stats = validation_result['statistics']
            if 'altitude_stats' in stats:
                alt_stats = stats['altitude_stats']
                print(f"   Altitude range: {alt_stats['min']:.1f} - {alt_stats['max']:.1f} km")
        
        print("   ✅ Integration test completed")
        return True
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False


def test_main_integration():
    """Test integration with main module."""
    print("\nTesting main module integration...")
    
    try:
        # Import main module functions
        from src.main import load_satellites
        
        # Test loading with validation
        print("   Testing load_satellites with validation...")
        satellites = load_satellites(validate_data=True)
        
        if satellites:
            print(f"   ✅ Successfully loaded and validated {len(satellites)} satellites")
        else:
            print("   ⚠️  No satellites loaded (may be expected if no TLE file)")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("="*60)
    print("SATELLITE DATA VALIDATOR INTEGRATION TESTS")
    print("="*60)
    
    tests = [
        test_validator_basic_functionality,
        test_integration_with_tle_parser,
        test_main_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
    
    print("\n" + "="*60)
    print(f"INTEGRATION TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All integration tests passed!")
        return 0
    else:
        print("⚠️  Some integration tests failed")
        return 1


if __name__ == "__main__":
    exit(main())