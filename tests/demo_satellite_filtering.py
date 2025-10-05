"""
Demonstration of Satellite Filtering and Preprocessing System

This script demonstrates the key features of the satellite filtering module:
- Orbital characteristic-based filtering
- Hierarchical clustering for efficient satellite grouping
- Candidate satellite pre-selection based on constraints
- Smart satellite pruning to reduce search space
"""

import sys
from datetime import datetime

# Add src to path
sys.path.append('src')

from satellite_filter import SatelliteFilter, FilteringConfig, OrbitalCharacteristics
from tle_parser import SatelliteData
from genetic_algorithm import RouteConstraints


def create_diverse_satellite_constellation():
    """Create a diverse constellation of satellites for demonstration."""
    satellites = []
    
    # Define different orbital regimes
    orbital_regimes = [
        # (altitude_km, inclination_deg, eccentricity, name_prefix)
        (300, 28.5, 0.01, "ISS"),      # ISS-like orbit
        (400, 51.6, 0.02, "LEO1"),     # Low Earth orbit
        (550, 53.0, 0.01, "STAR"),     # Starlink-like
        (600, 97.8, 0.02, "SUN"),      # Sun-synchronous
        (800, 82.0, 0.03, "POLAR"),    # Polar orbit
        (1000, 45.0, 0.01, "MED"),     # Medium altitude
        (1200, 63.4, 0.02, "MOL"),     # Molniya-type
        (1400, 99.1, 0.01, "SSUN"),    # Sun-synchronous high
    ]
    
    satellite_id = 10000
    
    # Create multiple satellites for each regime
    for altitude, inclination, eccentricity, prefix in orbital_regimes:
        for i in range(8):  # 8 satellites per regime
            # Add some variation within each regime
            alt_variation = altitude + (i * 25) - 100  # ±100 km variation
            inc_variation = inclination + (i * 2) - 7   # ±7 degree variation
            ecc_variation = eccentricity + (i * 0.005) - 0.02  # Small eccentricity variation
            
            # Ensure reasonable bounds
            alt_variation = max(200, min(2000, alt_variation))
            inc_variation = max(0, min(180, inc_variation))
            ecc_variation = max(0.001, min(0.5, ecc_variation))
            
            # Calculate orbital parameters
            semi_major_axis = alt_variation + 6378.137
            mean_motion = 15.5 - (alt_variation - 400) * 0.001  # Approximate
            orbital_period = 90 + (alt_variation - 400) * 0.05  # Approximate
            
            satellite = SatelliteData(
                catalog_number=satellite_id,
                name=f"{prefix}-{i+1:02d}",
                epoch=datetime.now(),
                mean_motion=mean_motion,
                eccentricity=ecc_variation,
                inclination=inc_variation,
                raan=i * 45,  # Spread across different orbital planes
                arg_perigee=i * 30,
                mean_anomaly=i * 20,
                semi_major_axis=semi_major_axis,
                orbital_period=orbital_period
            )
            
            satellites.append(satellite)
            satellite_id += 1
    
    return satellites


def demonstrate_basic_filtering():
    """Demonstrate basic orbital characteristic filtering."""
    print("\n" + "="*60)
    print("BASIC ORBITAL FILTERING DEMONSTRATION")
    print("="*60)
    
    # Create satellite constellation
    satellites = create_diverse_satellite_constellation()
    print(f"Created constellation with {len(satellites)} satellites")
    
    # Create filter with default configuration
    satellite_filter = SatelliteFilter()
    
    # Create route constraints
    constraints = RouteConstraints(
        max_deltav_budget=8.0,
        max_mission_duration=172800.0,  # 2 days
        forbidden_satellites=[10005, 10015, 10025]  # Forbid some satellites
    )
    
    # Apply filtering
    print("\nApplying orbital characteristic filters...")
    filtered_satellites = satellite_filter.filter_satellites(satellites, constraints)
    
    # Show results
    print(f"Filtered satellites: {len(filtered_satellites)}/{len(satellites)} retained")
    
    # Show filtering statistics
    stats = satellite_filter.get_filtering_statistics()
    print(f"Filtering efficiency: {(stats['filtered_satellites']/stats['total_satellites']*100):.1f}%")
    
    if 'orbital_stats' in stats:
        orbital_stats = stats['orbital_stats']
        print(f"Altitude range: {orbital_stats['altitude_range'][0]:.0f} - {orbital_stats['altitude_range'][1]:.0f} km")
        print(f"Average altitude: {orbital_stats['avg_altitude']:.0f} km")
        print(f"Inclination range: {orbital_stats['inclination_range'][0]:.1f}° - {orbital_stats['inclination_range'][1]:.1f}°")
    
    return filtered_satellites


def demonstrate_hierarchical_clustering():
    """Demonstrate hierarchical clustering of satellites."""
    print("\n" + "="*60)
    print("HIERARCHICAL CLUSTERING DEMONSTRATION")
    print("="*60)
    
    # Create satellite constellation
    satellites = create_diverse_satellite_constellation()
    
    # Create filter with clustering enabled
    config = FilteringConfig(
        enable_hierarchical_clustering=True,
        min_cluster_size=3,
        max_clusters=15,
        altitude_tolerance=100.0,
        inclination_tolerance=10.0,
        raan_tolerance=45.0
    )
    
    satellite_filter = SatelliteFilter(config)
    
    # Create clusters
    print("Creating hierarchical clusters...")
    clusters = satellite_filter.create_satellite_clusters(satellites)
    
    print(f"Created {len(clusters)} clusters from {len(satellites)} satellites")
    
    # Show cluster details
    print("\nCluster Details:")
    print("-" * 80)
    print(f"{'Cluster ID':<20} {'Size':<6} {'Altitude':<10} {'Inclination':<12} {'Priority':<10}")
    print("-" * 80)
    
    for cluster in clusters[:10]:  # Show top 10 clusters
        char = cluster.characteristics
        print(f"{cluster.cluster_id:<20} {cluster.size:<6} {char.altitude:<10.0f} "
              f"{char.inclination:<12.1f} {cluster.priority_score:<10.1f}")
    
    # Show cluster statistics
    if clusters:
        cluster_sizes = [c.size for c in clusters]
        total_clustered = sum(cluster_sizes)
        print(f"\nCluster Statistics:")
        print(f"Total satellites clustered: {total_clustered}/{len(satellites)}")
        print(f"Average cluster size: {sum(cluster_sizes)/len(cluster_sizes):.1f}")
        print(f"Largest cluster: {max(cluster_sizes)} satellites")
        print(f"Smallest cluster: {min(cluster_sizes)} satellites")
    
    return clusters


def demonstrate_candidate_preselection():
    """Demonstrate candidate pre-selection functionality."""
    print("\n" + "="*60)
    print("CANDIDATE PRE-SELECTION DEMONSTRATION")
    print("="*60)
    
    # Create satellite constellation
    satellites = create_diverse_satellite_constellation()
    
    # Create filter
    satellite_filter = SatelliteFilter()
    
    # Create route constraints with specific requirements
    constraints = RouteConstraints(
        max_deltav_budget=6.0,
        max_mission_duration=86400.0,  # 1 day
        min_hops=3,
        max_hops=8,
        forbidden_satellites=[10010, 10020, 10030]
    )
    
    # Pre-select candidates
    print("Pre-selecting candidate satellites...")
    candidates = satellite_filter.preselect_candidates(satellites, constraints)
    
    # Show candidate categories
    print("\nCandidate Categories:")
    print("-" * 50)
    for category, satellite_list in candidates.items():
        print(f"{category:<25}: {len(satellite_list):>3} satellites")
    
    # Show some specific candidates
    print(f"\nStart Candidates (first 5): {candidates['start_candidates'][:5]}")
    print(f"End Candidates (first 5): {candidates['end_candidates'][:5]}")
    print(f"High Priority (first 10): {candidates['high_priority'][:10]}")
    
    return candidates


def demonstrate_search_space_pruning():
    """Demonstrate smart search space pruning."""
    print("\n" + "="*60)
    print("SEARCH SPACE PRUNING DEMONSTRATION")
    print("="*60)
    
    # Create large satellite constellation
    satellites = create_diverse_satellite_constellation()
    print(f"Starting with {len(satellites)} satellites")
    
    # Create filter with pruning enabled
    config = FilteringConfig(
        enable_smart_pruning=True,
        max_plane_change_cost=3.0
    )
    
    satellite_filter = SatelliteFilter(config)
    
    # Create route constraints
    constraints = RouteConstraints(
        max_deltav_budget=5.0,
        max_mission_duration=86400.0
    )
    
    # Test different pruning levels
    pruning_levels = [50, 30, 20, 10]
    
    print("\nPruning Results:")
    print("-" * 60)
    print(f"{'Target Size':<12} {'Actual Size':<12} {'Pruning %':<12} {'Diversity':<12}")
    print("-" * 60)
    
    for target_size in pruning_levels:
        pruned = satellite_filter.prune_search_space(satellites, constraints, target_size)
        pruning_percentage = (1 - len(pruned)/len(satellites)) * 100
        
        # Calculate diversity metric (altitude range)
        satellite_filter._calculate_orbital_characteristics(pruned)
        altitudes = [satellite_filter._orbital_characteristics[sat.catalog_number].altitude 
                    for sat in pruned]
        diversity = max(altitudes) - min(altitudes) if altitudes else 0
        
        print(f"{target_size:<12} {len(pruned):<12} {pruning_percentage:<12.1f} {diversity:<12.0f}")
    
    return pruned


def demonstrate_filtering_configuration():
    """Demonstrate different filtering configurations."""
    print("\n" + "="*60)
    print("FILTERING CONFIGURATION DEMONSTRATION")
    print("="*60)
    
    # Create satellite constellation
    satellites = create_diverse_satellite_constellation()
    
    # Test different configurations
    configurations = [
        ("Conservative", FilteringConfig(
            min_altitude=400.0,
            max_altitude=1000.0,
            max_eccentricity=0.1,
            enable_smart_pruning=False
        )),
        ("Aggressive", FilteringConfig(
            min_altitude=300.0,
            max_altitude=1500.0,
            max_eccentricity=0.3,
            enable_smart_pruning=True,
            max_plane_change_cost=4.0
        )),
        ("LEO Only", FilteringConfig(
            min_altitude=200.0,
            max_altitude=600.0,
            max_inclination=60.0,
            max_eccentricity=0.05
        )),
        ("High Altitude", FilteringConfig(
            min_altitude=800.0,
            max_altitude=2000.0,
            min_inclination=45.0,
            max_eccentricity=0.2
        ))
    ]
    
    constraints = RouteConstraints(
        max_deltav_budget=6.0,
        max_mission_duration=86400.0
    )
    
    print(f"Testing configurations with {len(satellites)} input satellites:")
    print("-" * 70)
    print(f"{'Configuration':<15} {'Filtered':<10} {'Retention %':<12} {'Clusters':<10}")
    print("-" * 70)
    
    for config_name, config in configurations:
        satellite_filter = SatelliteFilter(config)
        
        # Apply filtering
        filtered = satellite_filter.filter_satellites(satellites, constraints)
        retention_rate = (len(filtered) / len(satellites)) * 100
        
        # Create clusters
        clusters = satellite_filter.create_satellite_clusters(filtered)
        
        print(f"{config_name:<15} {len(filtered):<10} {retention_rate:<12.1f} {len(clusters):<10}")


def main():
    """Run all demonstrations."""
    print("SATELLITE FILTERING AND PREPROCESSING SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("This demonstration shows the key capabilities of the satellite filtering system:")
    print("• Orbital characteristic-based filtering")
    print("• Hierarchical clustering for efficient satellite grouping")
    print("• Candidate satellite pre-selection based on constraints")
    print("• Smart satellite pruning to reduce search space")
    
    try:
        # Run demonstrations
        filtered_satellites = demonstrate_basic_filtering()
        clusters = demonstrate_hierarchical_clustering()
        candidates = demonstrate_candidate_preselection()
        pruned_satellites = demonstrate_search_space_pruning()
        demonstrate_filtering_configuration()
        
        print("\n" + "="*60)
        print("DEMONSTRATION SUMMARY")
        print("="*60)
        print("✓ Basic filtering: Removes satellites outside operational parameters")
        print("✓ Hierarchical clustering: Groups similar satellites for efficient processing")
        print("✓ Candidate pre-selection: Identifies optimal start/end/intermediate satellites")
        print("✓ Search space pruning: Reduces satellite count while preserving diversity")
        print("✓ Configuration flexibility: Supports different mission requirements")
        
        print(f"\nThe satellite filtering system successfully processed satellite")
        print(f"constellations and reduced search space complexity while maintaining")
        print(f"solution quality for genetic algorithm optimization.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)