"""
Chromosome Initialization Module

This module contains methods for initializing RouteChromosome populations
using various strategies including random, greedy, altitude-based, and
orbital-plane-based initialization.
"""

import random
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from src.genetic_algorithm import RouteChromosome, RouteConstraints
from src.tle_parser import SatelliteData
from src.orbital_mechanics import OrbitalElements, tle_to_orbital_elements


@dataclass
class InitializationConfig:
    """Configuration for chromosome initialization strategies."""
    random_ratio: float = 0.4          # Fraction of population using random initialization
    greedy_ratio: float = 0.3          # Fraction using greedy nearest-neighbor
    altitude_ratio: float = 0.2        # Fraction using altitude-based patterns
    orbital_plane_ratio: float = 0.1   # Fraction using orbital plane clustering
    min_route_length: int = 2          # Minimum satellites in route
    max_route_length: int = 10         # Maximum satellites in route
    time_window_hours: float = 24.0    # Time window for departure scheduling (hours)


class ChromosomeInitializer:
    """
    Handles initialization of RouteChromosome populations using various strategies.
    
    This class implements multiple initialization strategies to create diverse
    initial populations for the genetic algorithm, improving convergence and
    solution quality.
    """
    
    def __init__(self, satellites: List[SatelliteData], config: InitializationConfig = None):
        """
        Initialize the chromosome initializer.
        
        Args:
            satellites: List of available satellites
            config: Configuration for initialization strategies
        """
        self.satellites = satellites
        self.config = config or InitializationConfig()
        self.satellite_dict = {sat.catalog_number: sat for sat in satellites}
        self.orbital_elements_cache = {}
        
        # Pre-compute orbital elements for all satellites
        self._precompute_orbital_elements()
        
        # Pre-compute satellite groupings for efficient initialization
        self._precompute_satellite_groups()
    
    def _precompute_orbital_elements(self):
        """Pre-compute orbital elements for all satellites."""
        for satellite in self.satellites:
            try:
                elements = tle_to_orbital_elements(satellite)
                self.orbital_elements_cache[satellite.catalog_number] = elements
            except Exception as e:
                # Skip satellites with invalid orbital data
                print(f"Warning: Skipping satellite {satellite.catalog_number} due to orbital data error: {e}")
    
    def _precompute_satellite_groups(self):
        """Pre-compute satellite groupings for efficient initialization."""
        # Group by altitude ranges
        self.altitude_groups = self._group_satellites_by_altitude()
        
        # Group by orbital planes (similar inclination and RAAN)
        self.orbital_plane_groups = self._group_satellites_by_orbital_plane()
        
        # Create nearest neighbor lookup for greedy initialization
        self.nearest_neighbors = self._compute_nearest_neighbors()
    
    def _group_satellites_by_altitude(self) -> Dict[str, List[int]]:
        """Group satellites by altitude ranges."""
        groups = {
            'low': [],      # < 600 km
            'medium': [],   # 600-1200 km
            'high': []      # > 1200 km
        }
        
        for sat_id, elements in self.orbital_elements_cache.items():
            altitude = elements.semi_major_axis - 6371.0  # Earth radius approximation
            if altitude < 600:
                groups['low'].append(sat_id)
            elif altitude < 1200:
                groups['medium'].append(sat_id)
            else:
                groups['high'].append(sat_id)
        
        return groups
    
    def _group_satellites_by_orbital_plane(self) -> Dict[int, List[int]]:
        """Group satellites by similar orbital planes (inclination and RAAN)."""
        groups = {}
        plane_tolerance = 5.0  # degrees
        
        for sat_id, elements in self.orbital_elements_cache.items():
            # Create a plane identifier based on inclination and RAAN
            inc_group = round(elements.inclination / plane_tolerance)
            raan_group = round(elements.raan / plane_tolerance)
            plane_id = (inc_group, raan_group)
            
            if plane_id not in groups:
                groups[plane_id] = []
            groups[plane_id].append(sat_id)
        
        # Convert to simple integer keys and filter small groups
        filtered_groups = {}
        group_id = 0
        for plane_id, satellites in groups.items():
            if len(satellites) >= 2:  # Only keep groups with multiple satellites
                filtered_groups[group_id] = satellites
                group_id += 1
        
        return filtered_groups
    
    def _compute_nearest_neighbors(self) -> Dict[int, List[int]]:
        """Compute nearest neighbors for each satellite based on orbital similarity."""
        neighbors = {}
        
        for sat_id in self.orbital_elements_cache.keys():
            neighbors[sat_id] = self._find_nearest_satellites(sat_id, count=10)
        
        return neighbors
    
    def _find_nearest_satellites(self, satellite_id: int, count: int = 10) -> List[int]:
        """Find nearest satellites to given satellite based on orbital elements."""
        if satellite_id not in self.orbital_elements_cache:
            return []
        
        target_elements = self.orbital_elements_cache[satellite_id]
        distances = []
        
        for other_id, other_elements in self.orbital_elements_cache.items():
            if other_id == satellite_id:
                continue
            
            # Calculate orbital similarity distance
            distance = self._calculate_orbital_distance(target_elements, other_elements)
            distances.append((distance, other_id))
        
        # Sort by distance and return closest satellites
        distances.sort(key=lambda x: x[0])
        return [sat_id for _, sat_id in distances[:count]]
    
    def _calculate_orbital_distance(self, elem1: OrbitalElements, elem2: OrbitalElements) -> float:
        """Calculate orbital similarity distance between two satellites."""
        # Normalize differences by typical ranges
        sma_diff = abs(elem1.semi_major_axis - elem2.semi_major_axis) / 1000.0  # km
        ecc_diff = abs(elem1.eccentricity - elem2.eccentricity) * 1000.0
        inc_diff = abs(elem1.inclination - elem2.inclination) / 10.0  # degrees
        raan_diff = abs(elem1.raan - elem2.raan) / 10.0  # degrees
        
        # Handle RAAN wraparound
        if raan_diff > 18.0:  # 180 degrees / 10
            raan_diff = 36.0 - raan_diff
        
        return math.sqrt(sma_diff**2 + ecc_diff**2 + inc_diff**2 + raan_diff**2)
    
    def initialize_population(self, population_size: int, constraints: RouteConstraints) -> List[RouteChromosome]:
        """
        Initialize a population of chromosomes using mixed strategies.
        
        Args:
            population_size: Number of chromosomes to create
            constraints: Route constraints to satisfy
            
        Returns:
            List of initialized chromosomes
        """
        population = []
        
        # Calculate counts for each initialization strategy
        random_count = int(population_size * self.config.random_ratio)
        greedy_count = int(population_size * self.config.greedy_ratio)
        altitude_count = int(population_size * self.config.altitude_ratio)
        orbital_count = population_size - random_count - greedy_count - altitude_count
        
        # Generate chromosomes using different strategies
        population.extend(self._create_random_chromosomes(random_count, constraints))
        population.extend(self._create_greedy_chromosomes(greedy_count, constraints))
        population.extend(self._create_altitude_based_chromosomes(altitude_count, constraints))
        population.extend(self._create_orbital_plane_chromosomes(orbital_count, constraints))
        
        # Shuffle to mix strategies
        random.shuffle(population)
        
        return population
    
    def _create_random_chromosomes(self, count: int, constraints: RouteConstraints) -> List[RouteChromosome]:
        """Create chromosomes with random satellite sequences."""
        chromosomes = []
        available_satellites = self._get_available_satellites(constraints)
        
        for _ in range(count):
            route_length = random.randint(
                max(constraints.min_hops + 1, self.config.min_route_length),
                min(constraints.max_hops + 1, self.config.max_route_length, len(available_satellites))
            )
            
            # Select random satellites
            satellite_sequence = random.sample(available_satellites, route_length)
            
            # Handle fixed start/end constraints
            satellite_sequence = self._apply_endpoint_constraints(satellite_sequence, constraints)
            
            # Generate random departure times
            departure_times = self._generate_departure_times(len(satellite_sequence))
            
            chromosome = RouteChromosome(
                satellite_sequence=satellite_sequence,
                departure_times=departure_times
            )
            chromosomes.append(chromosome)
        
        return chromosomes
    
    def _create_greedy_chromosomes(self, count: int, constraints: RouteConstraints) -> List[RouteChromosome]:
        """Create chromosomes using greedy nearest-neighbor heuristic."""
        chromosomes = []
        available_satellites = self._get_available_satellites(constraints)
        
        for _ in range(count):
            route_length = random.randint(
                max(constraints.min_hops + 1, self.config.min_route_length),
                min(constraints.max_hops + 1, self.config.max_route_length, len(available_satellites))
            )
            
            # Start with random or constrained satellite
            if constraints.start_satellite_id and constraints.start_satellite_id in available_satellites:
                current_satellite = constraints.start_satellite_id
            else:
                current_satellite = random.choice(available_satellites)
            
            satellite_sequence = [current_satellite]
            used_satellites = {current_satellite}
            
            # Build route using nearest neighbors
            while len(satellite_sequence) < route_length:
                neighbors = [s for s in self.nearest_neighbors.get(current_satellite, []) 
                           if s in available_satellites and s not in used_satellites]
                
                if not neighbors:
                    # Fallback to random selection
                    remaining = [s for s in available_satellites if s not in used_satellites]
                    if not remaining:
                        break
                    next_satellite = random.choice(remaining)
                else:
                    # Select from nearest neighbors with some randomness
                    next_satellite = random.choice(neighbors[:min(3, len(neighbors))])
                
                satellite_sequence.append(next_satellite)
                used_satellites.add(next_satellite)
                current_satellite = next_satellite
            
            # Apply endpoint constraints
            satellite_sequence = self._apply_endpoint_constraints(satellite_sequence, constraints)
            
            # Generate departure times
            departure_times = self._generate_departure_times(len(satellite_sequence))
            
            chromosome = RouteChromosome(
                satellite_sequence=satellite_sequence,
                departure_times=departure_times
            )
            chromosomes.append(chromosome)
        
        return chromosomes
    
    def _create_altitude_based_chromosomes(self, count: int, constraints: RouteConstraints) -> List[RouteChromosome]:
        """Create chromosomes following altitude patterns (ascending/descending)."""
        chromosomes = []
        
        for _ in range(count):
            # Choose ascending or descending pattern
            ascending = random.choice([True, False])
            
            # Select satellites from different altitude groups
            satellite_sequence = []
            
            if ascending:
                # Low -> Medium -> High
                groups_order = ['low', 'medium', 'high']
            else:
                # High -> Medium -> Low
                groups_order = ['high', 'medium', 'low']
            
            route_length = random.randint(
                max(constraints.min_hops + 1, self.config.min_route_length),
                min(constraints.max_hops + 1, self.config.max_route_length)
            )
            
            # Distribute satellites across altitude groups
            satellites_per_group = max(1, route_length // len(groups_order))
            
            for group_name in groups_order:
                available_in_group = [s for s in self.altitude_groups.get(group_name, []) 
                                    if self._is_satellite_available(s, constraints)]
                
                if available_in_group:
                    count_from_group = min(satellites_per_group, len(available_in_group), 
                                         route_length - len(satellite_sequence))
                    selected = random.sample(available_in_group, count_from_group)
                    satellite_sequence.extend(selected)
                
                if len(satellite_sequence) >= route_length:
                    break
            
            # Fill remaining slots if needed
            if len(satellite_sequence) < route_length:
                available_satellites = self._get_available_satellites(constraints)
                remaining_needed = route_length - len(satellite_sequence)
                unused_satellites = [s for s in available_satellites if s not in satellite_sequence]
                
                if unused_satellites:
                    additional = random.sample(unused_satellites, 
                                             min(remaining_needed, len(unused_satellites)))
                    satellite_sequence.extend(additional)
            
            # Apply endpoint constraints
            satellite_sequence = self._apply_endpoint_constraints(satellite_sequence, constraints)
            
            # Generate departure times
            departure_times = self._generate_departure_times(len(satellite_sequence))
            
            chromosome = RouteChromosome(
                satellite_sequence=satellite_sequence,
                departure_times=departure_times
            )
            chromosomes.append(chromosome)
        
        return chromosomes
    
    def _create_orbital_plane_chromosomes(self, count: int, constraints: RouteConstraints) -> List[RouteChromosome]:
        """Create chromosomes using satellites from similar orbital planes."""
        chromosomes = []
        
        # Get available orbital plane groups
        available_groups = []
        for group_id, satellites in self.orbital_plane_groups.items():
            available_sats = [s for s in satellites if self._is_satellite_available(s, constraints)]
            if len(available_sats) >= 2:
                available_groups.append(available_sats)
        
        if not available_groups:
            # Fallback to random initialization
            return self._create_random_chromosomes(count, constraints)
        
        for _ in range(count):
            route_length = random.randint(
                max(constraints.min_hops + 1, self.config.min_route_length),
                min(constraints.max_hops + 1, self.config.max_route_length)
            )
            
            satellite_sequence = []
            
            # Select primary orbital plane group
            primary_group = random.choice(available_groups)
            
            # Take most satellites from primary group
            primary_count = min(route_length // 2 + 1, len(primary_group))
            satellite_sequence.extend(random.sample(primary_group, primary_count))
            
            # Fill remaining slots from other groups or general population
            remaining_needed = route_length - len(satellite_sequence)
            if remaining_needed > 0:
                available_satellites = self._get_available_satellites(constraints)
                unused_satellites = [s for s in available_satellites if s not in satellite_sequence]
                
                if unused_satellites:
                    additional = random.sample(unused_satellites, 
                                             min(remaining_needed, len(unused_satellites)))
                    satellite_sequence.extend(additional)
            
            # Apply endpoint constraints
            satellite_sequence = self._apply_endpoint_constraints(satellite_sequence, constraints)
            
            # Generate departure times
            departure_times = self._generate_departure_times(len(satellite_sequence))
            
            chromosome = RouteChromosome(
                satellite_sequence=satellite_sequence,
                departure_times=departure_times
            )
            chromosomes.append(chromosome)
        
        return chromosomes
    
    def _get_available_satellites(self, constraints: RouteConstraints) -> List[int]:
        """Get list of satellites available for route construction."""
        available = []
        
        for sat_id in self.orbital_elements_cache.keys():
            if self._is_satellite_available(sat_id, constraints):
                available.append(sat_id)
        
        return available
    
    def _is_satellite_available(self, satellite_id: int, constraints: RouteConstraints) -> bool:
        """Check if satellite is available for use in routes."""
        return satellite_id not in constraints.forbidden_satellites
    
    def _apply_endpoint_constraints(self, satellite_sequence: List[int], 
                                  constraints: RouteConstraints) -> List[int]:
        """Apply start and end satellite constraints to sequence."""
        if not satellite_sequence:
            return satellite_sequence
        
        sequence = satellite_sequence.copy()
        
        # Apply start constraint
        if constraints.start_satellite_id is not None:
            if constraints.start_satellite_id in sequence:
                # Move to front
                sequence.remove(constraints.start_satellite_id)
                sequence.insert(0, constraints.start_satellite_id)
            else:
                # Add to front if not forbidden
                if self._is_satellite_available(constraints.start_satellite_id, constraints):
                    sequence.insert(0, constraints.start_satellite_id)
        
        # Apply end constraint
        if constraints.end_satellite_id is not None:
            if constraints.end_satellite_id in sequence:
                # Move to end
                sequence.remove(constraints.end_satellite_id)
                sequence.append(constraints.end_satellite_id)
            else:
                # Add to end if not forbidden
                if self._is_satellite_available(constraints.end_satellite_id, constraints):
                    sequence.append(constraints.end_satellite_id)
        
        return sequence
    
    def _generate_departure_times(self, sequence_length: int) -> List[float]:
        """Generate departure times for satellite sequence."""
        if sequence_length == 0:
            return []
        
        # Start time (random within first few hours)
        start_time = random.uniform(0, 3600 * 4)  # 0-4 hours
        
        if sequence_length == 1:
            return [start_time]
        
        # Generate increasing departure times
        departure_times = [start_time]
        current_time = start_time
        
        # Time window in seconds
        time_window = self.config.time_window_hours * 3600
        
        for i in range(1, sequence_length):
            # Random interval between departures (1-6 hours)
            interval = random.uniform(3600, 6 * 3600)
            current_time += interval
            departure_times.append(current_time)
        
        return departure_times