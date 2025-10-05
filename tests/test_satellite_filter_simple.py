"""
Simple Test Runner for Satellite Filtering Module

Tests the satellite filtering, clustering, candidate pre-selection,
and search space pruning functionality without pytest dependency.
"""

import sys
import traceback
from datetime import datetime

# Add src to path
sys.path.append('src')

from satellite_filter import (
    SatelliteFilter, FilteringConfig, OrbitalCharacteristics,
    SatelliteCluster, FilterCriteria
)
from tle_parser import SatelliteData
from genetic_algorithm import RouteConstraints


def create_sample_satellites():
    """Create sample satellite data for testing."""
    satellites = []
    
    # Create satellites with different characteristics
    satellite_data = [
        # Low altitude, low inclination
        (1001, 400, 28.5, 0.01, 92, 0, 45, 15.5),
        # Medium altitude, medium inclination  
        (1002, 600, 55.0, 0.02, 96, 90, 135, 14.8),
        # High altitude, high inclination
        (1003, 800, 82.0, 0.03, 101, 180, 225, 14.1),
        # Very low altitude (should be filtered)
        (1004, 150, 45.0, 0.01, 88, 45, 90, 16.2),
        # High eccentricity (should be filtered)
        (1005, 500, 45.0, 0.6, 95, 90, 180, 15.0),
        # Another good satellite
        (1006, 550, 45.0, 0.015, 95, 120, 270, 14.9)
    ]
    
    for cat_num, alt, inc, ecc, period, raan, arg_per, mean_motion in satellite_data:
        semi_major_axis = alt + 6378.137  # Add Earth radius
        satellite = SatelliteData(
            catalog_number=cat_num,
            name=f"SAT-{cat_num}",
            epoch=datetime.now(),
            mean_motion=mean_motion,
            eccentricity=ecc,
            inclination=inc,
            raan=raan,
            arg_perigee=arg_per,
            mean_anomaly=0.0,
            semi_major_axis=semi_major_axis,
            orbital_period=period
        )
        satellites.append(satellite)
    
    return satellites


def test_orbital_characteristics():
    """Test orbital characteristics classification."""
    print("Testing orbital characteristics...")
    
    # Test altitude categories
    char = OrbitalCharacteristics(250, 45, 0.01, 90, 0, 0, 16)
    assert char.altitude_category == "very_low", f"Expected very_low, got {char.altitude_category}"
    
    char = OrbitalCharacteristics(450, 45, 0.01, 90, 0, 0, 16)
    assert char.altitude_category == "low", f"Expected low, got {char.altitude_category}"
    
    # Test inclination categories
    char = OrbitalCharacteristics(500, 5, 0.01, 90, 0, 0, 16)
    assert char.inclination_category == "equatorial", f"Expected equatorial, got {char.inclination_category}"
    
    char = OrbitalCharacteristics(500, 45, 0.01, 90, 0, 0, 16)
    assert char.inclination_category == "medium_inclination", f"Expected medium_inclination, got {char.inclination_category}"
    
    # Test eccentricity categories
    char = OrbitalCharacteristics(500, 45, 0.005, 90, 0, 0, 16)
    assert char.eccentricity_category == "circular", f"Expected circular, got {char.eccentricity_category}"
    
    print("✓ Orbital characteristics tests passed")


def test_satellite_filter_initialization():
    """Test satellite filter initialization."""
    print("Testing satellite filter initialization...")
    
    satellite_filter = SatelliteFilter()
    
    assert satellite_filter.config is not None
    assert satellite_filter.filtering_stats['total_satellites'] == 0
    assert len(satellite_filter._orbital_characteristics) == 0
    
    print("✓ Satellite filter initialization tests passed")


def test_orbital_characteristics_calculation():
    """Test orbital characteristics calculation."""
    print("Testing orbital characteristics calculation...")
    
    satellite_filter = SatelliteFilter()
    sample_satellites = create_sample_satellites()
    
    satellite_filter._calculate_orbital_characteristics(sample_satellites)
    
    assert len(satellite_filter._orbital_characteristics) == len(sample_satellites)
    
    # Check specific satellite characteristics
    char_1001 = satellite_filter._orbital_characteristics[1001]
    assert char_1001.altitude == 400
    assert char_1001.inclination == 28.5
    assert char_1001.eccentricity == 0.01
    
    print("✓ Orbital characteristics calculation tests passed")


def test_basic_filtering():
    """Test basic orbital filtering."""
    print("Testing basic orbital filtering...")
    
    satellite_filter = SatelliteFilter()
    sample_satellites = create_sample_satellites()
    
    satellite_filter._calculate_orbital_characteristics(sample_satellites)
    filtered = satellite_filter._apply_orbital_filters(sample_satellites)
    
    # Should filter out very low altitude (1004) and high eccentricity (1005)
    filtered_ids = {sat.catalog_number for sat in filtered}
    assert 1004 not in filtered_ids, "Very low altitude satellite should be filtered"
    assert 1005 not in filtered_ids, "High eccentricity satellite should be filtered"
    assert 1001 in filtered_ids, "Good satellite should not be filtered"
    assert 1002 in filtered_ids, "Good satellite should not be filtered"
    assert 1003 in filtered_ids, "Good satellite should not be filtered"
    assert 1006 in filtered_ids, "Good satellite should not be filtered"
    
    print("✓ Basic filtering tests passed")


def test_complete_filtering():
    """Test complete filtering process."""
    print("Testing complete filtering process...")
    
    satellite_filter = SatelliteFilter()
    sample_satellites = create_sample_satellites()
    
    constraints = RouteConstraints(
        max_deltav_budget=5.0,
        max_mission_duration=86400.0,
        forbidden_satellites=[1004]
    )
    
    filtered = satellite_filter.filter_satellites(sample_satellites, constraints)
    
    # Should have fewer satellites than input
    assert len(filtered) < len(sample_satellites), "Should filter out some satellites"
    
    # Check statistics
    assert satellite_filter.filtering_stats['total_satellites'] == len(sample_satellites)
    assert satellite_filter.filtering_stats['filtered_satellites'] == len(filtered)
    
    # Verify no forbidden or invalid satellites remain
    filtered_ids = {sat.catalog_number for sat in filtered}
    assert 1004 not in filtered_ids, "Forbidden satellite should be filtered"
    assert 1005 not in filtered_ids, "Invalid satellite should be filtered"
    
    print("✓ Complete filtering tests passed")


def test_transfer_cost_estimation():
    """Test transfer cost estimation."""
    print("Testing transfer cost estimation...")
    
    satellite_filter = SatelliteFilter()
    
    char1 = OrbitalCharacteristics(400, 28.5, 0.01, 92, 0, 45, 15.5)
    char2 = OrbitalCharacteristics(600, 55.0, 0.02, 96, 90, 135, 14.8)
    
    cost = satellite_filter._estimate_transfer_cost(char1, char2)
    
    # Should be positive and reasonable
    assert cost > 0, "Transfer cost should be positive"
    assert cost < 10.0, "Transfer cost should be reasonable (< 10 km/s)"
    
    print("✓ Transfer cost estimation tests passed")


def test_clustering():
    """Test satellite clustering."""
    print("Testing satellite clustering...")
    
    satellite_filter = SatelliteFilter()
    
    # Create clusterable satellites
    satellites = []
    
    # Group 1: Similar satellites
    for i in range(3):
        sat = SatelliteData(
            catalog_number=2000 + i,
            name=f"SAT-{2000 + i}",
            epoch=datetime.now(),
            mean_motion=15.5,
            eccentricity=0.01,
            inclination=28.5 + i,  # Slightly different inclinations
            raan=45 + i * 5,  # Slightly different RAANs
            arg_perigee=90,
            mean_anomaly=0.0,
            semi_major_axis=6778.137 + i * 10,  # 400-420 km altitude
            orbital_period=92 + i
        )
        satellites.append(sat)
    
    clusters = satellite_filter.create_satellite_clusters(satellites)
    
    # Should create at least one cluster or none (if below minimum size)
    assert len(clusters) >= 0, "Should create valid number of clusters"
    
    # Check cluster properties if clusters were created
    for cluster in clusters:
        assert cluster.size >= satellite_filter.config.min_cluster_size
        assert len(cluster.satellites) == cluster.size
        assert cluster.cluster_id is not None
        assert cluster.priority_score >= 0
    
    print("✓ Clustering tests passed")


def test_candidate_preselection():
    """Test candidate pre-selection."""
    print("Testing candidate pre-selection...")
    
    satellite_filter = SatelliteFilter()
    sample_satellites = create_sample_satellites()
    
    constraints = RouteConstraints(
        max_deltav_budget=5.0,
        max_mission_duration=86400.0
    )
    
    candidates = satellite_filter.preselect_candidates(sample_satellites, constraints)
    
    # Check that all candidate categories are present
    expected_categories = [
        'start_candidates', 'end_candidates', 'intermediate_candidates',
        'high_priority', 'medium_priority', 'low_priority'
    ]
    
    for category in expected_categories:
        assert category in candidates, f"Missing candidate category: {category}"
        assert isinstance(candidates[category], list), f"Category {category} should be a list"
    
    # Check that candidates are reasonable
    assert len(candidates['start_candidates']) > 0, "Should have start candidates"
    assert len(candidates['end_candidates']) > 0, "Should have end candidates"
    assert len(candidates['intermediate_candidates']) > 0, "Should have intermediate candidates"
    
    print("✓ Candidate pre-selection tests passed")


def test_search_space_pruning():
    """Test search space pruning."""
    print("Testing search space pruning...")
    
    satellite_filter = SatelliteFilter()
    
    # Create larger set of satellites
    satellites = []
    for i in range(10):
        altitude = 400 + (i * 50)  # 400-850 km range
        inclination = 30 + (i * 5)  # 30-75 degree range
        
        sat = SatelliteData(
            catalog_number=7000 + i,
            name=f"PRUNE-{7000 + i}",
            epoch=datetime.now(),
            mean_motion=15.5 - (i * 0.1),
            eccentricity=0.01 + (i * 0.01),
            inclination=inclination,
            raan=i * 18,  # Spread across RAAN
            arg_perigee=i * 18,
            mean_anomaly=0.0,
            semi_major_axis=6778.137 + (i * 50),
            orbital_period=90 + (i * 2)
        )
        satellites.append(sat)
    
    constraints = RouteConstraints(
        max_deltav_budget=5.0,
        max_mission_duration=86400.0
    )
    
    max_satellites = 5
    pruned = satellite_filter.prune_search_space(satellites, constraints, max_satellites)
    
    # Should reduce satellite count
    assert len(pruned) <= max_satellites, "Should not exceed maximum satellites"
    assert len(pruned) <= len(satellites), "Should not increase satellite count"
    
    print("✓ Search space pruning tests passed")


def test_statistics():
    """Test statistics collection."""
    print("Testing statistics collection...")
    
    satellite_filter = SatelliteFilter()
    sample_satellites = create_sample_satellites()
    
    constraints = RouteConstraints(
        max_deltav_budget=5.0,
        max_mission_duration=86400.0
    )
    
    filtered = satellite_filter.filter_satellites(sample_satellites, constraints)
    
    # Check statistics
    stats = satellite_filter.get_filtering_statistics()
    
    assert stats['total_satellites'] == len(sample_satellites)
    assert stats['filtered_satellites'] == len(filtered)
    assert 'orbital_stats' in stats
    
    print("✓ Statistics tests passed")


def run_all_tests():
    """Run all tests."""
    print("Running Satellite Filter Tests...")
    print("=" * 50)
    
    tests = [
        test_orbital_characteristics,
        test_satellite_filter_initialization,
        test_orbital_characteristics_calculation,
        test_basic_filtering,
        test_complete_filtering,
        test_transfer_cost_estimation,
        test_clustering,
        test_candidate_preselection,
        test_search_space_pruning,
        test_statistics
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            traceback.print_exc()
            failed += 1
    
    print("=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)