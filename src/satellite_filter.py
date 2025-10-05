"""
Satellite Filtering and Preprocessing Module

This module implements satellite filtering and preprocessing capabilities to optimize
genetic algorithm performance by reducing search space and grouping satellites
by orbital characteristics.
"""

import math
import logging
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

from src.tle_parser import SatelliteData
from src.genetic_algorithm import RouteConstraints
from src.orbital_mechanics import EARTH_RADIUS


class FilterCriteria(Enum):
    """Criteria for satellite filtering."""
    ALTITUDE_RANGE = "altitude_range"
    INCLINATION_RANGE = "inclination_range"
    ECCENTRICITY_LIMIT = "eccentricity_limit"
    ORBITAL_PERIOD_RANGE = "orbital_period_range"
    PLANE_SIMILARITY = "plane_similarity"
    ACCESSIBILITY = "accessibility"


@dataclass
class OrbitalCharacteristics:
    """Orbital characteristics for satellite classification."""
    altitude: float  # km above Earth surface
    inclination: float  # degrees
    eccentricity: float
    orbital_period: float  # minutes
    raan: float  # degrees (Right Ascension of Ascending Node)
    arg_perigee: float  # degrees
    mean_motion: float  # revolutions per day
    
    @property
    def altitude_category(self) -> str:
        """Categorize satellite by altitude."""
        if self.altitude < 300:
            return "very_low"
        elif self.altitude < 600:
            return "low"
        elif self.altitude < 1000:
            return "medium_low"
        elif self.altitude < 1500:
            return "medium"
        else:
            return "high"
    
    @property
    def inclination_category(self) -> str:
        """Categorize satellite by inclination."""
        if self.inclination < 10:
            return "equatorial"
        elif self.inclination < 30:
            return "low_inclination"
        elif self.inclination < 60:
            return "medium_inclination"
        elif self.inclination < 120:
            return "high_inclination"
        else:
            return "polar"
    
    @property
    def eccentricity_category(self) -> str:
        """Categorize satellite by eccentricity."""
        if self.eccentricity < 0.01:
            return "circular"
        elif self.eccentricity < 0.1:
            return "low_eccentric"
        elif self.eccentricity < 0.3:
            return "moderate_eccentric"
        else:
            return "high_eccentric"


@dataclass
class SatelliteCluster:
    """Cluster of satellites with similar orbital characteristics."""
    cluster_id: str
    satellites: List[int]  # Catalog numbers
    characteristics: OrbitalCharacteristics
    cluster_center: Tuple[float, float, float]  # (altitude, inclination, raan)
    cluster_radius: float
    priority_score: float = 0.0
    
    @property
    def size(self) -> int:
        """Number of satellites in cluster."""
        return len(self.satellites)


@dataclass
class FilteringConfig:
    """Configuration for satellite filtering and preprocessing."""
    # Altitude filtering
    min_altitude: float = 200.0  # km
    max_altitude: float = 2000.0  # km
    
    # Inclination filtering
    min_inclination: float = 0.0  # degrees
    max_inclination: float = 180.0  # degrees
    
    # Eccentricity filtering
    max_eccentricity: float = 0.5
    
    # Orbital period filtering
    min_orbital_period: float = 80.0  # minutes
    max_orbital_period: float = 200.0  # minutes
    
    # Clustering parameters
    altitude_tolerance: float = 50.0  # km
    inclination_tolerance: float = 5.0  # degrees
    raan_tolerance: float = 30.0  # degrees
    min_cluster_size: int = 2
    max_clusters: int = 50
    
    # Performance optimization
    max_satellites_per_cluster: int = 100
    enable_hierarchical_clustering: bool = True
    enable_smart_pruning: bool = True
    
    # Accessibility filtering
    min_transfer_efficiency: float = 0.1  # Minimum transfer efficiency threshold
    max_plane_change_cost: float = 2.0  # km/s maximum plane change cost


class SatelliteFilter:
    """
    Main satellite filtering and preprocessing system.
    
    Provides orbital characteristic-based filtering, hierarchical clustering,
    candidate pre-selection, and smart satellite pruning capabilities.
    """
    
    def __init__(self, config: FilteringConfig = None):
        """
        Initialize satellite filter.
        
        Args:
            config: Filtering configuration parameters
        """
        self.config = config or FilteringConfig()
        self.logger = logging.getLogger(__name__)
        
        # Cached data
        self._orbital_characteristics: Dict[int, OrbitalCharacteristics] = {}
        self._satellite_clusters: List[SatelliteCluster] = []
        self._accessibility_matrix: Dict[Tuple[int, int], float] = {}
        self._filtered_satellites: Set[int] = set()
        
        # Statistics
        self.filtering_stats = {
            'total_satellites': 0,
            'filtered_satellites': 0,
            'clusters_created': 0,
            'accessibility_calculations': 0
        }
    
    def filter_satellites(self, satellites: List[SatelliteData], 
                         constraints: RouteConstraints = None) -> List[SatelliteData]:
        """
        Apply comprehensive satellite filtering based on orbital characteristics.
        
        Args:
            satellites: List of satellite data to filter
            constraints: Optional route constraints for additional filtering
            
        Returns:
            Filtered list of satellites suitable for route optimization
        """
        if not satellites:
            return []
        
        self.filtering_stats['total_satellites'] = len(satellites)
        self.logger.info(f"Starting satellite filtering for {len(satellites)} satellites")
        
        # Step 1: Calculate orbital characteristics
        self._calculate_orbital_characteristics(satellites)
        
        # Step 2: Apply basic orbital filters
        filtered_satellites = self._apply_orbital_filters(satellites)
        
        # Step 3: Apply constraint-based filtering if provided
        if constraints:
            filtered_satellites = self._apply_constraint_filters(filtered_satellites, constraints)
        
        # Step 4: Apply accessibility filtering
        if self.config.enable_smart_pruning:
            filtered_satellites = self._apply_accessibility_filters(filtered_satellites)
        
        # Update statistics
        self.filtering_stats['filtered_satellites'] = len(filtered_satellites)
        self._filtered_satellites = {sat.catalog_number for sat in filtered_satellites}
        
        filter_ratio = (len(filtered_satellites) / len(satellites)) * 100
        self.logger.info(f"Satellite filtering complete: {len(filtered_satellites)}/{len(satellites)} "
                        f"satellites retained ({filter_ratio:.1f}%)")
        
        return filtered_satellites
    
    def create_satellite_clusters(self, satellites: List[SatelliteData]) -> List[SatelliteCluster]:
        """
        Create hierarchical clusters of satellites based on orbital characteristics.
        
        Args:
            satellites: List of satellites to cluster
            
        Returns:
            List of satellite clusters organized by orbital similarity
        """
        if not satellites:
            return []
        
        self.logger.info(f"Creating satellite clusters for {len(satellites)} satellites")
        
        # Ensure orbital characteristics are calculated
        if not self._orbital_characteristics:
            self._calculate_orbital_characteristics(satellites)
        
        # Create clusters using hierarchical clustering
        if self.config.enable_hierarchical_clustering:
            clusters = self._hierarchical_clustering(satellites)
        else:
            clusters = self._simple_clustering(satellites)
        
        # Calculate cluster priorities
        self._calculate_cluster_priorities(clusters)
        
        # Sort clusters by priority
        clusters.sort(key=lambda c: c.priority_score, reverse=True)
        
        # Limit number of clusters
        if len(clusters) > self.config.max_clusters:
            clusters = clusters[:self.config.max_clusters]
        
        self._satellite_clusters = clusters
        self.filtering_stats['clusters_created'] = len(clusters)
        
        self.logger.info(f"Created {len(clusters)} satellite clusters")
        return clusters
    
    def preselect_candidates(self, satellites: List[SatelliteData], 
                           constraints: RouteConstraints) -> Dict[str, List[int]]:
        """
        Pre-select candidate satellites based on constraints and optimization goals.
        
        Args:
            satellites: List of available satellites
            constraints: Route optimization constraints
            
        Returns:
            Dictionary mapping selection criteria to lists of satellite catalog numbers
        """
        # Ensure orbital characteristics are calculated
        if not self._orbital_characteristics:
            self._calculate_orbital_characteristics(satellites)
        
        candidates = {
            'start_candidates': [],
            'end_candidates': [],
            'intermediate_candidates': [],
            'high_priority': [],
            'medium_priority': [],
            'low_priority': []
        }
        
        # Handle fixed start/end satellites
        if constraints.start_satellite_id:
            candidates['start_candidates'] = [constraints.start_satellite_id]
        else:
            # Select good starting candidates (lower altitude, accessible)
            candidates['start_candidates'] = self._select_start_candidates(satellites)
        
        if constraints.end_satellite_id:
            candidates['end_candidates'] = [constraints.end_satellite_id]
        else:
            # Select good ending candidates (higher altitude, accessible)
            candidates['end_candidates'] = self._select_end_candidates(satellites)
        
        # Select intermediate candidates
        candidates['intermediate_candidates'] = self._select_intermediate_candidates(
            satellites, constraints
        )
        
        # Prioritize satellites based on accessibility and characteristics
        priority_groups = self._prioritize_satellites(satellites, constraints)
        candidates.update(priority_groups)
        
        # Log candidate statistics
        total_candidates = sum(len(group) for group in candidates.values())
        self.logger.info(f"Pre-selected {total_candidates} candidate satellites across "
                        f"{len(candidates)} categories")
        
        return candidates
    
    def prune_search_space(self, satellites: List[SatelliteData], 
                          constraints: RouteConstraints,
                          max_satellites: int = None) -> List[SatelliteData]:
        """
        Smart satellite pruning to reduce search space while preserving solution quality.
        
        Args:
            satellites: List of satellites to prune
            constraints: Route constraints for pruning decisions
            max_satellites: Maximum number of satellites to retain
            
        Returns:
            Pruned list of satellites optimized for genetic algorithm performance
        """
        if not satellites:
            return []
        
        if max_satellites is None:
            max_satellites = min(500, len(satellites))  # Default limit
        
        self.logger.info(f"Pruning search space from {len(satellites)} to {max_satellites} satellites")
        
        # Ensure orbital characteristics are calculated
        if not self._orbital_characteristics:
            self._calculate_orbital_characteristics(satellites)
        
        # Step 1: Score satellites based on multiple criteria
        satellite_scores = self._calculate_satellite_scores(satellites, constraints)
        
        # Step 2: Ensure diversity in pruned set
        pruned_satellites = self._diversity_aware_pruning(
            satellites, satellite_scores, max_satellites
        )
        
        # Step 3: Ensure connectivity (satellites can reach each other)
        pruned_satellites = self._ensure_connectivity(pruned_satellites, constraints)
        
        pruning_ratio = (len(pruned_satellites) / len(satellites)) * 100
        self.logger.info(f"Search space pruning complete: {len(pruned_satellites)}/{len(satellites)} "
                        f"satellites retained ({pruning_ratio:.1f}%)")
        
        return pruned_satellites
    
    def get_filtering_statistics(self) -> Dict[str, Any]:
        """Get comprehensive filtering and preprocessing statistics."""
        stats = self.filtering_stats.copy()
        
        if self._satellite_clusters:
            cluster_sizes = [cluster.size for cluster in self._satellite_clusters]
            stats.update({
                'cluster_stats': {
                    'total_clusters': len(self._satellite_clusters),
                    'avg_cluster_size': sum(cluster_sizes) / len(cluster_sizes),
                    'min_cluster_size': min(cluster_sizes),
                    'max_cluster_size': max(cluster_sizes),
                    'largest_cluster_id': max(self._satellite_clusters, key=lambda c: c.size).cluster_id
                }
            })
        
        if self._orbital_characteristics:
            altitudes = [char.altitude for char in self._orbital_characteristics.values()]
            inclinations = [char.inclination for char in self._orbital_characteristics.values()]
            
            stats.update({
                'orbital_stats': {
                    'altitude_range': (min(altitudes), max(altitudes)),
                    'inclination_range': (min(inclinations), max(inclinations)),
                    'avg_altitude': sum(altitudes) / len(altitudes),
                    'avg_inclination': sum(inclinations) / len(inclinations)
                }
            })
        
        return stats
    
    def _calculate_orbital_characteristics(self, satellites: List[SatelliteData]):
        """Calculate and cache orbital characteristics for all satellites."""
        for satellite in satellites:
            altitude = satellite.semi_major_axis - EARTH_RADIUS
            
            characteristics = OrbitalCharacteristics(
                altitude=altitude,
                inclination=satellite.inclination,
                eccentricity=satellite.eccentricity,
                orbital_period=satellite.orbital_period,
                raan=satellite.raan,
                arg_perigee=satellite.arg_perigee,
                mean_motion=satellite.mean_motion
            )
            
            self._orbital_characteristics[satellite.catalog_number] = characteristics
    
    def _apply_orbital_filters(self, satellites: List[SatelliteData]) -> List[SatelliteData]:
        """Apply basic orbital characteristic filters."""
        filtered = []
        
        for satellite in satellites:
            char = self._orbital_characteristics[satellite.catalog_number]
            
            # Altitude filter
            if not (self.config.min_altitude <= char.altitude <= self.config.max_altitude):
                continue
            
            # Inclination filter
            if not (self.config.min_inclination <= char.inclination <= self.config.max_inclination):
                continue
            
            # Eccentricity filter
            if char.eccentricity > self.config.max_eccentricity:
                continue
            
            # Orbital period filter
            if not (self.config.min_orbital_period <= char.orbital_period <= self.config.max_orbital_period):
                continue
            
            filtered.append(satellite)
        
        return filtered
    
    def _apply_constraint_filters(self, satellites: List[SatelliteData], 
                                constraints: RouteConstraints) -> List[SatelliteData]:
        """Apply constraint-based filtering."""
        filtered = []
        
        for satellite in satellites:
            # Skip forbidden satellites
            if satellite.catalog_number in constraints.forbidden_satellites:
                continue
            
            # Include fixed start/end satellites regardless of other filters
            if (satellite.catalog_number == constraints.start_satellite_id or
                satellite.catalog_number == constraints.end_satellite_id):
                filtered.append(satellite)
                continue
            
            # Apply additional constraint-based filters here
            # (e.g., based on mission duration, delta-v budget)
            
            filtered.append(satellite)
        
        return filtered
    
    def _apply_accessibility_filters(self, satellites: List[SatelliteData]) -> List[SatelliteData]:
        """Apply accessibility-based filtering to remove isolated satellites."""
        if len(satellites) < 2:
            return satellites
        
        # Calculate accessibility matrix for sample of satellites
        sample_size = min(100, len(satellites))
        sample_satellites = satellites[:sample_size]
        
        accessible_satellites = set()
        
        for i, sat1 in enumerate(sample_satellites):
            char1 = self._orbital_characteristics[sat1.catalog_number]
            connection_count = 0
            
            for j, sat2 in enumerate(sample_satellites):
                if i == j:
                    continue
                
                char2 = self._orbital_characteristics[sat2.catalog_number]
                
                # Estimate transfer cost based on orbital differences
                transfer_cost = self._estimate_transfer_cost(char1, char2)
                
                if transfer_cost < self.config.max_plane_change_cost:
                    connection_count += 1
            
            # Keep satellites that can connect to at least 2 others
            if connection_count >= 2:
                accessible_satellites.add(sat1.catalog_number)
        
        # Filter satellites based on accessibility
        filtered = [sat for sat in satellites 
                   if sat.catalog_number in accessible_satellites]
        
        return filtered
    
    def _estimate_transfer_cost(self, char1: OrbitalCharacteristics, 
                              char2: OrbitalCharacteristics) -> float:
        """Estimate transfer cost between two satellites based on orbital characteristics."""
        # Altitude difference cost (Hohmann transfer approximation)
        altitude_diff = abs(char1.altitude - char2.altitude)
        altitude_cost = altitude_diff * 0.001  # Rough approximation: 1 m/s per km altitude difference
        
        # Inclination change cost
        inclination_diff = abs(char1.inclination - char2.inclination)
        inclination_cost = inclination_diff * 0.1  # Rough approximation: 100 m/s per degree
        
        # RAAN difference cost (simplified)
        raan_diff = abs(char1.raan - char2.raan)
        if raan_diff > 180:
            raan_diff = 360 - raan_diff
        raan_cost = raan_diff * 0.05  # Rough approximation
        
        total_cost = altitude_cost + inclination_cost + raan_cost
        return total_cost
    
    def _hierarchical_clustering(self, satellites: List[SatelliteData]) -> List[SatelliteCluster]:
        """Create hierarchical clusters based on orbital similarity."""
        clusters = []
        
        # Group satellites by major orbital characteristics
        altitude_groups = self._group_by_altitude(satellites)
        
        for altitude_category, alt_satellites in altitude_groups.items():
            # Further subdivide by inclination
            inclination_groups = self._group_by_inclination(alt_satellites)
            
            for inc_category, inc_satellites in inclination_groups.items():
                # Final subdivision by RAAN (orbital plane)
                raan_groups = self._group_by_raan(inc_satellites)
                
                for raan_range, raan_satellites in raan_groups.items():
                    if len(raan_satellites) >= self.config.min_cluster_size:
                        # Convert satellites to catalog numbers for cluster creation
                        satellite_ids = [sat.catalog_number for sat in raan_satellites]
                        cluster = self._create_cluster(
                            f"{altitude_category}_{inc_category}_{raan_range}",
                            satellite_ids
                        )
                        clusters.append(cluster)
        
        return clusters
    
    def _simple_clustering(self, satellites: List[SatelliteData]) -> List[SatelliteCluster]:
        """Create simple clusters based on orbital characteristics."""
        clusters = []
        used_satellites = set()
        
        for i, satellite in enumerate(satellites):
            if satellite.catalog_number in used_satellites:
                continue
            
            char1 = self._orbital_characteristics[satellite.catalog_number]
            cluster_satellites = [satellite.catalog_number]
            used_satellites.add(satellite.catalog_number)
            
            # Find similar satellites
            for j, other_satellite in enumerate(satellites[i+1:], i+1):
                if other_satellite.catalog_number in used_satellites:
                    continue
                
                char2 = self._orbital_characteristics[other_satellite.catalog_number]
                
                if self._are_orbitally_similar(char1, char2):
                    cluster_satellites.append(other_satellite.catalog_number)
                    used_satellites.add(other_satellite.catalog_number)
                    
                    if len(cluster_satellites) >= self.config.max_satellites_per_cluster:
                        break
            
            if len(cluster_satellites) >= self.config.min_cluster_size:
                cluster = self._create_cluster(f"cluster_{len(clusters)}", cluster_satellites)
                clusters.append(cluster)
        
        return clusters
    
    def _are_orbitally_similar(self, char1: OrbitalCharacteristics, 
                             char2: OrbitalCharacteristics) -> bool:
        """Check if two satellites have similar orbital characteristics."""
        altitude_similar = abs(char1.altitude - char2.altitude) <= self.config.altitude_tolerance
        inclination_similar = abs(char1.inclination - char2.inclination) <= self.config.inclination_tolerance
        raan_similar = abs(char1.raan - char2.raan) <= self.config.raan_tolerance
        
        return altitude_similar and inclination_similar and raan_similar
    
    def _group_by_altitude(self, satellites: List[SatelliteData]) -> Dict[str, List[SatelliteData]]:
        """Group satellites by altitude category."""
        groups = defaultdict(list)
        
        for satellite in satellites:
            char = self._orbital_characteristics[satellite.catalog_number]
            category = char.altitude_category
            groups[category].append(satellite)
        
        return dict(groups)
    
    def _group_by_inclination(self, satellites: List[SatelliteData]) -> Dict[str, List[SatelliteData]]:
        """Group satellites by inclination category."""
        groups = defaultdict(list)
        
        for satellite in satellites:
            char = self._orbital_characteristics[satellite.catalog_number]
            category = char.inclination_category
            groups[category].append(satellite)
        
        return dict(groups)
    
    def _group_by_raan(self, satellites: List[SatelliteData]) -> Dict[str, List[SatelliteData]]:
        """Group satellites by RAAN ranges."""
        groups = defaultdict(list)
        
        for satellite in satellites:
            char = self._orbital_characteristics[satellite.catalog_number]
            # Group by 60-degree RAAN ranges
            raan_range = int(char.raan // 60) * 60
            range_key = f"raan_{raan_range}_{raan_range + 60}"
            groups[range_key].append(satellite)
        
        return dict(groups)
    
    def _create_cluster(self, cluster_id: str, satellite_ids: List[int]) -> SatelliteCluster:
        """Create a satellite cluster with calculated characteristics."""
        # Calculate cluster center and characteristics
        characteristics_list = [self._orbital_characteristics[sat_id] for sat_id in satellite_ids]
        
        avg_altitude = sum(char.altitude for char in characteristics_list) / len(characteristics_list)
        avg_inclination = sum(char.inclination for char in characteristics_list) / len(characteristics_list)
        avg_raan = sum(char.raan for char in characteristics_list) / len(characteristics_list)
        
        # Calculate cluster radius (maximum distance from center)
        cluster_radius = 0.0
        for char in characteristics_list:
            distance = math.sqrt(
                (char.altitude - avg_altitude) ** 2 +
                (char.inclination - avg_inclination) ** 2 +
                (char.raan - avg_raan) ** 2
            )
            cluster_radius = max(cluster_radius, distance)
        
        # Create representative characteristics
        avg_characteristics = OrbitalCharacteristics(
            altitude=avg_altitude,
            inclination=avg_inclination,
            eccentricity=sum(char.eccentricity for char in characteristics_list) / len(characteristics_list),
            orbital_period=sum(char.orbital_period for char in characteristics_list) / len(characteristics_list),
            raan=avg_raan,
            arg_perigee=sum(char.arg_perigee for char in characteristics_list) / len(characteristics_list),
            mean_motion=sum(char.mean_motion for char in characteristics_list) / len(characteristics_list)
        )
        
        return SatelliteCluster(
            cluster_id=cluster_id,
            satellites=satellite_ids,
            characteristics=avg_characteristics,
            cluster_center=(avg_altitude, avg_inclination, avg_raan),
            cluster_radius=cluster_radius
        )
    
    def _calculate_cluster_priorities(self, clusters: List[SatelliteCluster]):
        """Calculate priority scores for clusters based on various factors."""
        for cluster in clusters:
            score = 0.0
            
            # Size factor (larger clusters are more important)
            score += cluster.size * 10
            
            # Altitude factor (medium altitudes preferred for transfers)
            ideal_altitude = 600  # km
            altitude_penalty = abs(cluster.characteristics.altitude - ideal_altitude) / 100
            score -= altitude_penalty
            
            # Inclination factor (moderate inclinations preferred)
            if 30 <= cluster.characteristics.inclination <= 90:
                score += 50  # Bonus for useful inclinations
            
            # Eccentricity factor (prefer circular orbits)
            if cluster.characteristics.eccentricity < 0.1:
                score += 20
            
            # Compactness factor (tighter clusters are better)
            if cluster.cluster_radius < 100:
                score += 30
            
            cluster.priority_score = max(0.0, score)
    
    def _select_start_candidates(self, satellites: List[SatelliteData]) -> List[int]:
        """Select good starting satellite candidates."""
        candidates = []
        
        # Prefer lower altitude satellites for starting (easier to reach)
        sorted_satellites = sorted(
            satellites,
            key=lambda s: self._orbital_characteristics[s.catalog_number].altitude
        )
        
        # Take bottom 20% as start candidates
        num_candidates = max(1, len(sorted_satellites) // 5)
        candidates = [sat.catalog_number for sat in sorted_satellites[:num_candidates]]
        
        return candidates
    
    def _select_end_candidates(self, satellites: List[SatelliteData]) -> List[int]:
        """Select good ending satellite candidates."""
        candidates = []
        
        # Prefer higher altitude satellites for ending (natural progression)
        sorted_satellites = sorted(
            satellites,
            key=lambda s: self._orbital_characteristics[s.catalog_number].altitude,
            reverse=True
        )
        
        # Take top 20% as end candidates
        num_candidates = max(1, len(sorted_satellites) // 5)
        candidates = [sat.catalog_number for sat in sorted_satellites[:num_candidates]]
        
        return candidates
    
    def _select_intermediate_candidates(self, satellites: List[SatelliteData], 
                                      constraints: RouteConstraints) -> List[int]:
        """Select good intermediate satellite candidates."""
        # All satellites except forbidden ones are potential intermediate candidates
        candidates = [
            sat.catalog_number for sat in satellites
            if sat.catalog_number not in constraints.forbidden_satellites
        ]
        
        return candidates
    
    def _prioritize_satellites(self, satellites: List[SatelliteData], 
                             constraints: RouteConstraints) -> Dict[str, List[int]]:
        """Prioritize satellites based on accessibility and characteristics."""
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for satellite in satellites:
            char = self._orbital_characteristics[satellite.catalog_number]
            priority_score = 0
            
            # Altitude scoring (medium altitudes preferred)
            if 400 <= char.altitude <= 800:
                priority_score += 3
            elif 300 <= char.altitude <= 1000:
                priority_score += 2
            else:
                priority_score += 1
            
            # Inclination scoring
            if 30 <= char.inclination <= 90:
                priority_score += 2
            elif char.inclination <= 120:
                priority_score += 1
            
            # Eccentricity scoring (prefer circular)
            if char.eccentricity < 0.05:
                priority_score += 2
            elif char.eccentricity < 0.2:
                priority_score += 1
            
            # Categorize by priority
            if priority_score >= 6:
                high_priority.append(satellite.catalog_number)
            elif priority_score >= 4:
                medium_priority.append(satellite.catalog_number)
            else:
                low_priority.append(satellite.catalog_number)
        
        return {
            'high_priority': high_priority,
            'medium_priority': medium_priority,
            'low_priority': low_priority
        }
    
    def _calculate_satellite_scores(self, satellites: List[SatelliteData], 
                                  constraints: RouteConstraints) -> Dict[int, float]:
        """Calculate comprehensive scores for satellites."""
        scores = {}
        
        for satellite in satellites:
            char = self._orbital_characteristics[satellite.catalog_number]
            score = 0.0
            
            # Base score from orbital characteristics
            score += self._score_orbital_characteristics(char)
            
            # Accessibility score (estimated connectivity)
            score += self._score_accessibility(satellite, satellites)
            
            # Constraint compatibility score
            score += self._score_constraint_compatibility(satellite, constraints)
            
            scores[satellite.catalog_number] = score
        
        return scores
    
    def _score_orbital_characteristics(self, char: OrbitalCharacteristics) -> float:
        """Score satellite based on orbital characteristics."""
        score = 0.0
        
        # Altitude scoring (bell curve around 600 km)
        ideal_altitude = 600
        altitude_factor = 1.0 - (abs(char.altitude - ideal_altitude) / 1000)
        score += max(0, altitude_factor * 50)
        
        # Inclination scoring (prefer useful inclinations)
        if 30 <= char.inclination <= 90:
            score += 30
        elif char.inclination <= 120:
            score += 20
        else:
            score += 10
        
        # Eccentricity scoring (prefer circular)
        eccentricity_factor = 1.0 - char.eccentricity
        score += eccentricity_factor * 20
        
        return score
    
    def _score_accessibility(self, satellite: SatelliteData, 
                           all_satellites: List[SatelliteData]) -> float:
        """Score satellite based on estimated accessibility to others."""
        char1 = self._orbital_characteristics[satellite.catalog_number]
        accessible_count = 0
        
        # Sample a subset for performance
        sample_size = min(50, len(all_satellites))
        sample_satellites = all_satellites[:sample_size]
        
        for other_satellite in sample_satellites:
            if other_satellite.catalog_number == satellite.catalog_number:
                continue
            
            char2 = self._orbital_characteristics[other_satellite.catalog_number]
            transfer_cost = self._estimate_transfer_cost(char1, char2)
            
            if transfer_cost < self.config.max_plane_change_cost:
                accessible_count += 1
        
        # Score based on connectivity
        connectivity_ratio = accessible_count / sample_size
        return connectivity_ratio * 50
    
    def _score_constraint_compatibility(self, satellite: SatelliteData, 
                                      constraints: RouteConstraints) -> float:
        """Score satellite based on constraint compatibility."""
        score = 0.0
        
        # Bonus for fixed start/end satellites
        if (satellite.catalog_number == constraints.start_satellite_id or
            satellite.catalog_number == constraints.end_satellite_id):
            score += 100
        
        # Penalty for forbidden satellites
        if satellite.catalog_number in constraints.forbidden_satellites:
            score -= 1000
        
        return score
    
    def _diversity_aware_pruning(self, satellites: List[SatelliteData], 
                               scores: Dict[int, float], 
                               max_satellites: int) -> List[SatelliteData]:
        """Prune satellites while maintaining diversity."""
        if len(satellites) <= max_satellites:
            return satellites
        
        # Sort by score
        sorted_satellites = sorted(
            satellites,
            key=lambda s: scores[s.catalog_number],
            reverse=True
        )
        
        # Take top candidates
        selected = sorted_satellites[:max_satellites]
        
        # Ensure diversity by replacing some high-scoring satellites
        # with representatives from different orbital categories
        diversity_replacements = max_satellites // 10  # Replace 10%
        
        if diversity_replacements > 0:
            # Group remaining satellites by characteristics
            remaining = sorted_satellites[max_satellites:]
            category_groups = self._group_satellites_by_categories(remaining)
            
            # Replace lowest-scoring selected satellites with diverse candidates
            replacements = 0
            for category, candidates in category_groups.items():
                if replacements >= diversity_replacements:
                    break
                
                if candidates:
                    # Check if this category is underrepresented in selected
                    selected_in_category = sum(
                        1 for sat in selected
                        if self._get_satellite_category(sat) == category
                    )
                    
                    if selected_in_category == 0:  # Category not represented
                        # Replace lowest scoring selected satellite
                        best_candidate = max(candidates, key=lambda s: scores[s.catalog_number])
                        selected[-1-replacements] = best_candidate
                        replacements += 1
        
        return selected
    
    def _group_satellites_by_categories(self, satellites: List[SatelliteData]) -> Dict[str, List[SatelliteData]]:
        """Group satellites by combined orbital categories."""
        groups = defaultdict(list)
        
        for satellite in satellites:
            char = self._orbital_characteristics[satellite.catalog_number]
            category = f"{char.altitude_category}_{char.inclination_category}"
            groups[category].append(satellite)
        
        return dict(groups)
    
    def _get_satellite_category(self, satellite: SatelliteData) -> str:
        """Get combined category for a satellite."""
        char = self._orbital_characteristics[satellite.catalog_number]
        return f"{char.altitude_category}_{char.inclination_category}"
    
    def _ensure_connectivity(self, satellites: List[SatelliteData], 
                           constraints: RouteConstraints) -> List[SatelliteData]:
        """Ensure pruned satellite set maintains connectivity."""
        if len(satellites) < 2:
            return satellites
        
        # Build connectivity graph
        connectivity_graph = defaultdict(set)
        
        for i, sat1 in enumerate(satellites):
            char1 = self._orbital_characteristics[sat1.catalog_number]
            
            for j, sat2 in enumerate(satellites[i+1:], i+1):
                char2 = self._orbital_characteristics[sat2.catalog_number]
                
                transfer_cost = self._estimate_transfer_cost(char1, char2)
                
                if transfer_cost < self.config.max_plane_change_cost:
                    connectivity_graph[sat1.catalog_number].add(sat2.catalog_number)
                    connectivity_graph[sat2.catalog_number].add(sat1.catalog_number)
        
        # Find connected components
        connected_components = self._find_connected_components(
            [sat.catalog_number for sat in satellites],
            connectivity_graph
        )
        
        # Keep largest connected component
        if connected_components:
            largest_component = max(connected_components, key=len)
            connected_satellites = [
                sat for sat in satellites
                if sat.catalog_number in largest_component
            ]
            
            return connected_satellites
        
        return satellites
    
    def _find_connected_components(self, satellite_ids: List[int], 
                                 connectivity_graph: Dict[int, Set[int]]) -> List[Set[int]]:
        """Find connected components in satellite connectivity graph."""
        visited = set()
        components = []
        
        for sat_id in satellite_ids:
            if sat_id not in visited:
                component = set()
                self._dfs_component(sat_id, connectivity_graph, visited, component)
                if component:
                    components.append(component)
        
        return components
    
    def _dfs_component(self, sat_id: int, graph: Dict[int, Set[int]], 
                      visited: Set[int], component: Set[int]):
        """Depth-first search to find connected component."""
        visited.add(sat_id)
        component.add(sat_id)
        
        for neighbor in graph.get(sat_id, set()):
            if neighbor not in visited:
                self._dfs_component(neighbor, graph, visited, component)