"""
Unit Tests for Satellite Filtering and Preprocessing Module

Tests the satellite filtering, clustering, candidate pre-selection,
and search space pruning functionality.
"""

import pytest
import math
from unittest.mock import Mock, patch
from typing import List

from src.satellite_filter import (
    SatelliteFilter, FilteringConfig, OrbitalCharacteristics,
    SatelliteCluster, FilterCriteria
)
from src.tle_parser import SatelliteData
from src.genetic_algorithm import RouteConstraints
from datetime import datetime


class TestOrbitalCharacteristics:
    """Test orbital characteristics classification."""
    
    def test_altitude_categories(self):
        """Test altitude categorization."""
        # Very low altitude
        char = OrbitalCharacteristics(250, 45, 0.01, 90, 0, 0, 16)
        assert char.altitude_category == "very_low"
        
        # Low altitude
        char = OrbitalCharacteristics(450, 45, 0.01, 90, 0, 0, 16)
        assert char.altitude_category == "low"
        
        # Medium low altitude
        char = OrbitalCharacteristics(750, 45, 0.01, 90, 0, 0, 16)
        assert char.altitude_category == "medium_low"
        
        # Medium altitude
        char = OrbitalCharacteristics(1200, 45, 0.01, 90, 0, 0, 16)
        assert char.altitude_category == "medium"
        
        # High altitude
        char = OrbitalCharacteristics(1800, 45, 0.01, 90, 0, 0, 16)
        assert char.altitude_category == "high"
    
    def test_inclination_categories(self):
        """Test inclination categorization."""
        # Equatorial
        char = OrbitalCharacteristics(500, 5, 0.01, 90, 0, 0, 16)
        assert char.inclination_category == "equatorial"
        
        # Low inclination
        char = OrbitalCharacteristics(500, 25, 0.01, 90, 0, 0, 16)
        assert char.inclination_category == "low_inclination"
        
        # Medium inclination
        char = OrbitalCharacteristics(500, 45, 0.01, 90, 0, 0, 16)
        assert char.inclination_category == "medium_inclination"
        
        # High inclination
        char = OrbitalCharacteristics(500, 75, 0.01, 90, 0, 0, 16)
        assert char.inclination_category == "high_inclination"
        
        # Polar
        char = OrbitalCharacteristics(500, 135, 0.01, 90, 0, 0, 16)
        assert char.inclination_category == "polar"
    
    def test_eccentricity_categories(self):
        """Test eccentricity categorization."""
        # Circular
        char = OrbitalCharacteristics(500, 45, 0.005, 90, 0, 0, 16)
        assert char.eccentricity_category == "circular"
        
        # Low eccentric
        char = OrbitalCharacteristics(500, 45, 0.05, 90, 0, 0, 16)
        assert char.eccentricity_category == "low_eccentric"
        
        # Moderate eccentric
        char = OrbitalCharacteristics(500, 45, 0.15, 90, 0, 0, 16)
        assert char.eccentricity_category == "moderate_eccentric"
        
        # High eccentric
        char = OrbitalCharacteristics(500, 45, 0.4, 90, 0, 0, 16)
        assert char.eccentricity_category == "high_eccentric"


class TestSatelliteCluster:
    """Test satellite cluster functionality."""
    
    def test_cluster_creation(self):
        """Test basic cluster creation."""
        satellites = [1001, 1002, 1003]
        characteristics = OrbitalCharacteristics(500, 45, 0.01, 90, 0, 0, 16)
        
        cluster = SatelliteCluster(
            cluster_id="test_cluster",
            satellites=satellites,
            characteristics=characteristics,
            cluster_center=(500, 45, 0),
            cluster_radius=50.0
        )
        
        assert cluster.cluster_id == "test_cluster"
        assert cluster.size == 3
        assert cluster.satellites == satellites
        assert cluster.cluster_radius == 50.0
    
    def test_empty_cluster(self):
        """Test empty cluster handling."""
        cluster = SatelliteCluster(
            cluster_id="empty",
            satellites=[],
            characteristics=OrbitalCharacteristics(500, 45, 0.01, 90, 0, 0, 16),
            cluster_center=(500, 45, 0),
            cluster_radius=0.0
        )
        
        assert cluster.size == 0


class TestFilteringConfig:
    """Test filtering configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FilteringConfig()
        
        assert config.min_altitude == 200.0
        assert config.max_altitude == 2000.0
        assert config.min_inclination == 0.0
        assert config.max_inclination == 180.0
        assert config.max_eccentricity == 0.5
        assert config.enable_hierarchical_clustering is True
        assert config.enable_smart_pruning is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = FilteringConfig(
            min_altitude=300.0,
            max_altitude=1500.0,
            max_eccentricity=0.3,
            enable_smart_pruning=False
        )
        
        assert config.min_altitude == 300.0
        assert config.max_altitude == 1500.0
        assert config.max_eccentricity == 0.3
        assert config.enable_smart_pruning is False


class TestSatelliteFilter:
    """Test main satellite filter functionality."""
    
    @pytest.fixture
    def sample_satellites(self):
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
    
    @pytest.fixture
    def satellite_filter(self):
        """Create satellite filter instance."""
        return SatelliteFilter()
    
    @pytest.fixture
    def route_constraints(self):
        """Create sample route constraints."""
        return RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400.0,  # 1 day
            forbidden_satellites=[1004]  # Forbid the very low altitude satellite
        )
    
    def test_filter_initialization(self, satellite_filter):
        """Test filter initialization."""
        assert satellite_filter.config is not None
        assert satellite_filter.filtering_stats['total_satellites'] == 0
        assert len(satellite_filter._orbital_characteristics) == 0
    
    def test_orbital_characteristics_calculation(self, satellite_filter, sample_satellites):
        """Test orbital characteristics calculation."""
        satellite_filter._calculate_orbital_characteristics(sample_satellites)
        
        assert len(satellite_filter._orbital_characteristics) == len(sample_satellites)
        
        # Check specific satellite characteristics
        char_1001 = satellite_filter._orbital_characteristics[1001]
        assert char_1001.altitude == 400
        assert char_1001.inclination == 28.5
        assert char_1001.eccentricity == 0.01
    
    def test_basic_orbital_filtering(self, satellite_filter, sample_satellites):
        """Test basic orbital characteristic filtering."""
        satellite_filter._calculate_orbital_characteristics(sample_satellites)
        filtered = satellite_filter._apply_orbital_filters(sample_satellites)
        
        # Should filter out very low altitude (1004) and high eccentricity (1005)
        filtered_ids = {sat.catalog_number for sat in filtered}
        assert 1004 not in filtered_ids  # Too low altitude
        assert 1005 not in filtered_ids  # Too high eccentricity
        assert 1001 in filtered_ids  # Good satellite
        assert 1002 in filtered_ids  # Good satellite
        assert 1003 in filtered_ids  # Good satellite
        assert 1006 in filtered_ids  # Good satellite
    
    def test_constraint_based_filtering(self, satellite_filter, sample_satellites, route_constraints):
        """Test constraint-based filtering."""
        filtered = satellite_filter._apply_constraint_filters(sample_satellites, route_constraints)
        
        # Should filter out forbidden satellite
        filtered_ids = {sat.catalog_number for sat in filtered}
        assert 1004 not in filtered_ids  # Forbidden satellite
        assert len(filtered_ids) == len(sample_satellites) - 1
    
    def test_complete_filtering_process(self, satellite_filter, sample_satellites, route_constraints):
        """Test complete filtering process."""
        filtered = satellite_filter.filter_satellites(sample_satellites, route_constraints)
        
        # Should have fewer satellites than input
        assert len(filtered) < len(sample_satellites)
        
        # Check statistics
        assert satellite_filter.filtering_stats['total_satellites'] == len(sample_satellites)
        assert satellite_filter.filtering_stats['filtered_satellites'] == len(filtered)
        
        # Verify no forbidden or invalid satellites remain
        filtered_ids = {sat.catalog_number for sat in filtered}
        assert 1004 not in filtered_ids  # Forbidden and too low altitude
        assert 1005 not in filtered_ids  # Too high eccentricity
    
    def test_empty_satellite_list(self, satellite_filter):
        """Test filtering with empty satellite list."""
        filtered = satellite_filter.filter_satellites([])
        assert filtered == []
        assert satellite_filter.filtering_stats['total_satellites'] == 0
    
    def test_transfer_cost_estimation(self, satellite_filter):
        """Test transfer cost estimation between satellites."""
        char1 = OrbitalCharacteristics(400, 28.5, 0.01, 92, 0, 45, 15.5)
        char2 = OrbitalCharacteristics(600, 55.0, 0.02, 96, 90, 135, 14.8)
        
        cost = satellite_filter._estimate_transfer_cost(char1, char2)
        
        # Should be positive and reasonable
        assert cost > 0
        assert cost < 10.0  # Should be less than 10 km/s for reasonable transfers
    
    def test_orbital_similarity_check(self, satellite_filter):
        """Test orbital similarity checking."""
        # Similar satellites
        char1 = OrbitalCharacteristics(500, 45, 0.01, 95, 90, 180, 15.0)
        char2 = OrbitalCharacteristics(520, 47, 0.015, 96, 95, 185, 14.9)
        
        assert satellite_filter._are_orbitally_similar(char1, char2)
        
        # Dissimilar satellites
        char3 = OrbitalCharacteristics(800, 82, 0.03, 101, 180, 225, 14.1)
        
        assert not satellite_filter._are_orbitally_similar(char1, char3)


class TestSatelliteClustering:
    """Test satellite clustering functionality."""
    
    @pytest.fixture
    def satellite_filter(self):
        """Create satellite filter with clustering enabled."""
        config = FilteringConfig(
            enable_hierarchical_clustering=True,
            min_cluster_size=2,
            max_clusters=10
        )
        return SatelliteFilter(config)
    
    @pytest.fixture
    def clusterable_satellites(self):
        """Create satellites suitable for clustering."""
        satellites = []
        
        # Create two groups of similar satellites
        # Group 1: Low altitude, low inclination
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
        
        # Group 2: Medium altitude, medium inclination
        for i in range(3):
            sat = SatelliteData(
                catalog_number=3000 + i,
                name=f"SAT-{3000 + i}",
                epoch=datetime.now(),
                mean_motion=14.8,
                eccentricity=0.02,
                inclination=55.0 + i,  # Slightly different inclinations
                raan=135 + i * 5,  # Slightly different RAANs
                arg_perigee=180,
                mean_anomaly=0.0,
                semi_major_axis=6978.137 + i * 10,  # 600-620 km altitude
                orbital_period=96 + i
            )
            satellites.append(sat)
        
        return satellites
    
    def test_simple_clustering(self, satellite_filter, clusterable_satellites):
        """Test simple clustering algorithm."""
        satellite_filter.config.enable_hierarchical_clustering = False
        clusters = satellite_filter.create_satellite_clusters(clusterable_satellites)
        
        # Should create at least one cluster
        assert len(clusters) > 0
        
        # Check cluster properties
        for cluster in clusters:
            assert cluster.size >= satellite_filter.config.min_cluster_size
            assert len(cluster.satellites) == cluster.size
            assert cluster.cluster_id is not None
    
    def test_hierarchical_clustering(self, satellite_filter, clusterable_satellites):
        """Test hierarchical clustering algorithm."""
        clusters = satellite_filter.create_satellite_clusters(clusterable_satellites)
        
        # Should create clusters
        assert len(clusters) > 0
        
        # Should have reasonable cluster sizes
        total_satellites_in_clusters = sum(cluster.size for cluster in clusters)
        assert total_satellites_in_clusters <= len(clusterable_satellites)
        
        # Check cluster priorities are calculated
        for cluster in clusters:
            assert cluster.priority_score >= 0
    
    def test_cluster_characteristics_calculation(self, satellite_filter, clusterable_satellites):
        """Test cluster characteristics calculation."""
        clusters = satellite_filter.create_satellite_clusters(clusterable_satellites)
        
        for cluster in clusters:
            # Check that characteristics are reasonable averages
            assert cluster.characteristics.altitude > 0
            assert 0 <= cluster.characteristics.inclination <= 180
            assert 0 <= cluster.characteristics.eccentricity < 1
            assert cluster.cluster_radius >= 0
    
    def test_empty_clustering(self, satellite_filter):
        """Test clustering with empty satellite list."""
        clusters = satellite_filter.create_satellite_clusters([])
        assert clusters == []
    
    def test_single_satellite_clustering(self, satellite_filter, clusterable_satellites):
        """Test clustering with single satellite."""
        single_satellite = [clusterable_satellites[0]]
        clusters = satellite_filter.create_satellite_clusters(single_satellite)
        
        # Should not create clusters (below minimum size)
        assert len(clusters) == 0


class TestCandidatePreselection:
    """Test candidate pre-selection functionality."""
    
    @pytest.fixture
    def satellite_filter(self):
        """Create satellite filter instance."""
        return SatelliteFilter()
    
    @pytest.fixture
    def diverse_satellites(self):
        """Create satellites with diverse characteristics for candidate selection."""
        satellites = []
        
        # Low altitude satellites (good for starting)
        for i in range(2):
            sat = SatelliteData(
                catalog_number=4000 + i,
                name=f"LOW-{4000 + i}",
                epoch=datetime.now(),
                mean_motion=15.8,
                eccentricity=0.01,
                inclination=45,
                raan=90,
                arg_perigee=180,
                mean_anomaly=0.0,
                semi_major_axis=6678.137,  # 300 km altitude
                orbital_period=90
            )
            satellites.append(sat)
        
        # High altitude satellites (good for ending)
        for i in range(2):
            sat = SatelliteData(
                catalog_number=5000 + i,
                name=f"HIGH-{5000 + i}",
                epoch=datetime.now(),
                mean_motion=14.0,
                eccentricity=0.02,
                inclination=45,
                raan=90,
                arg_perigee=180,
                mean_anomaly=0.0,
                semi_major_axis=7378.137,  # 1000 km altitude
                orbital_period=105
            )
            satellites.append(sat)
        
        # Medium altitude satellites (good for intermediate)
        for i in range(2):
            sat = SatelliteData(
                catalog_number=6000 + i,
                name=f"MED-{6000 + i}",
                epoch=datetime.now(),
                mean_motion=14.9,
                eccentricity=0.015,
                inclination=55,
                raan=135,
                arg_perigee=270,
                mean_anomaly=0.0,
                semi_major_axis=6978.137,  # 600 km altitude
                orbital_period=95
            )
            satellites.append(sat)
        
        return satellites
    
    def test_candidate_preselection(self, satellite_filter, diverse_satellites):
        """Test candidate pre-selection process."""
        constraints = RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400.0
        )
        
        candidates = satellite_filter.preselect_candidates(diverse_satellites, constraints)
        
        # Check that all candidate categories are present
        expected_categories = [
            'start_candidates', 'end_candidates', 'intermediate_candidates',
            'high_priority', 'medium_priority', 'low_priority'
        ]
        
        for category in expected_categories:
            assert category in candidates
            assert isinstance(candidates[category], list)
        
        # Check that candidates are reasonable
        assert len(candidates['start_candidates']) > 0
        assert len(candidates['end_candidates']) > 0
        assert len(candidates['intermediate_candidates']) > 0
    
    def test_fixed_start_end_satellites(self, satellite_filter, diverse_satellites):
        """Test candidate selection with fixed start/end satellites."""
        constraints = RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400.0,
            start_satellite_id=4000,
            end_satellite_id=5000
        )
        
        candidates = satellite_filter.preselect_candidates(diverse_satellites, constraints)
        
        # Should use fixed satellites
        assert candidates['start_candidates'] == [4000]
        assert candidates['end_candidates'] == [5000]
    
    def test_start_candidate_selection(self, satellite_filter, diverse_satellites):
        """Test start candidate selection logic."""
        start_candidates = satellite_filter._select_start_candidates(diverse_satellites)
        
        # Should prefer lower altitude satellites
        assert len(start_candidates) > 0
        
        # Check that lower altitude satellites are preferred
        satellite_filter._calculate_orbital_characteristics(diverse_satellites)
        for sat_id in start_candidates[:2]:  # Check first two candidates
            char = satellite_filter._orbital_characteristics[sat_id]
            assert char.altitude <= 400  # Should be low altitude satellites
    
    def test_end_candidate_selection(self, satellite_filter, diverse_satellites):
        """Test end candidate selection logic."""
        end_candidates = satellite_filter._select_end_candidates(diverse_satellites)
        
        # Should prefer higher altitude satellites
        assert len(end_candidates) > 0
        
        # Check that higher altitude satellites are preferred
        satellite_filter._calculate_orbital_characteristics(diverse_satellites)
        for sat_id in end_candidates[:2]:  # Check first two candidates
            char = satellite_filter._orbital_characteristics[sat_id]
            assert char.altitude >= 600  # Should be higher altitude satellites


class TestSearchSpacePruning:
    """Test search space pruning functionality."""
    
    @pytest.fixture
    def satellite_filter(self):
        """Create satellite filter with pruning enabled."""
        config = FilteringConfig(enable_smart_pruning=True)
        return SatelliteFilter(config)
    
    @pytest.fixture
    def large_satellite_set(self):
        """Create large set of satellites for pruning tests."""
        satellites = []
        
        # Create 20 satellites with varying characteristics
        for i in range(20):
            altitude = 400 + (i * 50)  # 400-1350 km range
            inclination = 30 + (i * 5)  # 30-125 degree range
            
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
        
        return satellites
    
    def test_search_space_pruning(self, satellite_filter, large_satellite_set):
        """Test search space pruning process."""
        constraints = RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400.0
        )
        
        max_satellites = 10
        pruned = satellite_filter.prune_search_space(
            large_satellite_set, constraints, max_satellites
        )
        
        # Should reduce satellite count
        assert len(pruned) <= max_satellites
        assert len(pruned) < len(large_satellite_set)
        
        # Should maintain some diversity
        satellite_filter._calculate_orbital_characteristics(pruned)
        altitudes = [satellite_filter._orbital_characteristics[sat.catalog_number].altitude 
                    for sat in pruned]
        
        # Should have some altitude diversity
        altitude_range = max(altitudes) - min(altitudes)
        assert altitude_range > 100  # At least 100 km range
    
    def test_satellite_scoring(self, satellite_filter, large_satellite_set):
        """Test satellite scoring for pruning."""
        constraints = RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400.0
        )
        
        satellite_filter._calculate_orbital_characteristics(large_satellite_set)
        scores = satellite_filter._calculate_satellite_scores(large_satellite_set, constraints)
        
        # Should have scores for all satellites
        assert len(scores) == len(large_satellite_set)
        
        # Scores should be reasonable
        for sat in large_satellite_set:
            score = scores[sat.catalog_number]
            assert isinstance(score, (int, float))
            assert score >= 0  # Scores should be non-negative
    
    def test_diversity_aware_pruning(self, satellite_filter, large_satellite_set):
        """Test diversity-aware pruning."""
        # Create uniform scores to test diversity mechanism
        scores = {sat.catalog_number: 100.0 for sat in large_satellite_set}
        
        max_satellites = 8
        pruned = satellite_filter._diversity_aware_pruning(
            large_satellite_set, scores, max_satellites
        )
        
        assert len(pruned) == max_satellites
        
        # Check that pruned set has some diversity
        satellite_filter._calculate_orbital_characteristics(pruned)
        categories = set()
        for sat in pruned:
            category = satellite_filter._get_satellite_category(sat)
            categories.add(category)
        
        # Should have multiple categories represented
        assert len(categories) > 1
    
    def test_connectivity_preservation(self, satellite_filter, large_satellite_set):
        """Test connectivity preservation in pruning."""
        constraints = RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400.0
        )
        
        # Take a subset that should be connected
        connected_subset = large_satellite_set[:8]  # Similar satellites should be connected
        
        result = satellite_filter._ensure_connectivity(connected_subset, constraints)
        
        # Should return satellites (may be same or reduced set)
        assert len(result) > 0
        assert len(result) <= len(connected_subset)
    
    def test_empty_pruning(self, satellite_filter):
        """Test pruning with empty satellite list."""
        constraints = RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400.0
        )
        
        pruned = satellite_filter.prune_search_space([], constraints, 10)
        assert pruned == []
    
    def test_pruning_below_limit(self, satellite_filter, large_satellite_set):
        """Test pruning when satellite count is already below limit."""
        constraints = RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400.0
        )
        
        # Set limit higher than available satellites
        max_satellites = 50
        pruned = satellite_filter.prune_search_space(
            large_satellite_set, constraints, max_satellites
        )
        
        # Should return all satellites
        assert len(pruned) == len(large_satellite_set)


class TestFilteringStatistics:
    """Test filtering statistics and reporting."""
    
    def test_statistics_collection(self):
        """Test statistics collection during filtering."""
        satellite_filter = SatelliteFilter()
        
        # Create test satellites
        satellites = []
        for i in range(5):
            sat = SatelliteData(
                catalog_number=8000 + i,
                name=f"STAT-{8000 + i}",
                epoch=datetime.now(),
                mean_motion=15.0,
                eccentricity=0.01,
                inclination=45,
                raan=90,
                arg_perigee=180,
                mean_anomaly=0.0,
                semi_major_axis=6778.137,
                orbital_period=95
            )
            satellites.append(sat)
        
        # Apply filtering
        constraints = RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400.0
        )
        
        filtered = satellite_filter.filter_satellites(satellites, constraints)
        
        # Check statistics
        stats = satellite_filter.get_filtering_statistics()
        
        assert stats['total_satellites'] == 5
        assert stats['filtered_satellites'] == len(filtered)
        assert 'orbital_stats' in stats
    
    def test_clustering_statistics(self):
        """Test clustering statistics collection."""
        satellite_filter = SatelliteFilter()
        
        # Create clusterable satellites
        satellites = []
        for i in range(6):
            sat = SatelliteData(
                catalog_number=9000 + i,
                name=f"CLUST-{9000 + i}",
                epoch=datetime.now(),
                mean_motion=15.0,
                eccentricity=0.01,
                inclination=45 + (i % 2) * 5,  # Two groups
                raan=90 + (i % 2) * 30,
                arg_perigee=180,
                mean_anomaly=0.0,
                semi_major_axis=6778.137 + (i % 2) * 100,
                orbital_period=95
            )
            satellites.append(sat)
        
        # Create clusters
        clusters = satellite_filter.create_satellite_clusters(satellites)
        
        # Check statistics
        stats = satellite_filter.get_filtering_statistics()
        
        assert 'cluster_stats' in stats
        assert stats['cluster_stats']['total_clusters'] == len(clusters)
        if clusters:
            assert stats['cluster_stats']['avg_cluster_size'] > 0
            assert stats['cluster_stats']['min_cluster_size'] > 0
            assert stats['cluster_stats']['max_cluster_size'] > 0


if __name__ == "__main__":
    pytest.main([__file__])