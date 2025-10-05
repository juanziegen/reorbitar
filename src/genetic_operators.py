"""
Genetic Algorithm Operators

This module implements crossover, mutation, and selection operations
for the genetic algorithm route optimization system.
"""

import random
import math
from typing import List, Tuple, Optional, Set
from src.genetic_algorithm import RouteChromosome, RouteConstraints, GAConfig


class CrossoverOperator:
    """
    Implements various crossover operations for RouteChromosome.
    
    Crossover operations combine genetic material from two parent chromosomes
    to create offspring that inherit characteristics from both parents.
    """
    
    def __init__(self, config: GAConfig):
        """Initialize crossover operator with configuration."""
        self.config = config
    
    def order_crossover(self, parent1: RouteChromosome, parent2: RouteChromosome, 
                       constraints: RouteConstraints) -> Tuple[RouteChromosome, RouteChromosome]:
        """
        Order Crossover (OX) - preserves relative order of satellites.
        
        This crossover maintains the relative ordering of satellites from one parent
        while filling in missing satellites from the other parent.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            constraints: Route constraints to validate offspring
            
        Returns:
            Tuple of two offspring chromosomes
        """
        seq1, seq2 = parent1.satellite_sequence, parent2.satellite_sequence
        
        if len(seq1) < 2 or len(seq2) < 2:
            # Cannot perform crossover on sequences too short
            return self._create_valid_offspring(parent1, parent2, constraints)
        
        # Create offspring using order crossover
        offspring1_seq = self._perform_order_crossover(seq1, seq2)
        offspring2_seq = self._perform_order_crossover(seq2, seq1)
        
        # Generate timing for offspring
        offspring1_times = self._crossover_timing(parent1.departure_times, parent2.departure_times, 
                                                len(offspring1_seq))
        offspring2_times = self._crossover_timing(parent2.departure_times, parent1.departure_times, 
                                                len(offspring2_seq))
        
        # Create offspring chromosomes
        offspring1 = RouteChromosome(
            satellite_sequence=offspring1_seq,
            departure_times=offspring1_times
        )
        offspring2 = RouteChromosome(
            satellite_sequence=offspring2_seq,
            departure_times=offspring2_times
        )
        
        # Validate and repair if necessary
        offspring1 = self._validate_and_repair(offspring1, constraints)
        offspring2 = self._validate_and_repair(offspring2, constraints)
        
        return offspring1, offspring2
    
    def _perform_order_crossover(self, seq1: List[int], seq2: List[int]) -> List[int]:
        """Perform the core order crossover operation."""
        length = min(len(seq1), len(seq2))
        if length < 2:
            return seq1.copy()
        
        # Select crossover points
        start = random.randint(0, length - 2)
        end = random.randint(start + 1, length - 1)
        
        # Create offspring sequence
        offspring = [None] * length
        
        # Copy segment from first parent
        for i in range(start, end + 1):
            offspring[i] = seq1[i]
        
        # Fill remaining positions with satellites from second parent
        used_satellites = set(offspring[start:end + 1])
        fill_index = 0
        
        for satellite in seq2:
            if satellite not in used_satellites:
                # Find next empty position
                while fill_index < length and offspring[fill_index] is not None:
                    fill_index += 1
                
                if fill_index < length:
                    offspring[fill_index] = satellite
                    fill_index += 1
        
        # Handle any remaining None values (shouldn't happen in normal cases)
        for i in range(length):
            if offspring[i] is None:
                # Find unused satellite from seq1
                for satellite in seq1:
                    if satellite not in offspring:
                        offspring[i] = satellite
                        break
        
        return offspring
    
    def partially_mapped_crossover(self, parent1: RouteChromosome, parent2: RouteChromosome,
                                 constraints: RouteConstraints) -> Tuple[RouteChromosome, RouteChromosome]:
        """
        Partially Mapped Crossover (PMX) - maintains satellite relationships.
        
        PMX creates a mapping between satellites in the crossover region
        and uses this mapping to resolve conflicts in the rest of the sequence.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            constraints: Route constraints to validate offspring
            
        Returns:
            Tuple of two offspring chromosomes
        """
        seq1, seq2 = parent1.satellite_sequence, parent2.satellite_sequence
        
        if len(seq1) < 2 or len(seq2) < 2:
            return self._create_valid_offspring(parent1, parent2, constraints)
        
        # Create offspring using PMX
        offspring1_seq = self._perform_pmx(seq1, seq2)
        offspring2_seq = self._perform_pmx(seq2, seq1)
        
        # Generate timing for offspring
        offspring1_times = self._crossover_timing(parent1.departure_times, parent2.departure_times,
                                                len(offspring1_seq))
        offspring2_times = self._crossover_timing(parent2.departure_times, parent1.departure_times,
                                                len(offspring2_seq))
        
        # Create offspring chromosomes
        offspring1 = RouteChromosome(
            satellite_sequence=offspring1_seq,
            departure_times=offspring1_times
        )
        offspring2 = RouteChromosome(
            satellite_sequence=offspring2_seq,
            departure_times=offspring2_times
        )
        
        # Validate and repair if necessary
        offspring1 = self._validate_and_repair(offspring1, constraints)
        offspring2 = self._validate_and_repair(offspring2, constraints)
        
        return offspring1, offspring2
    
    def _perform_pmx(self, seq1: List[int], seq2: List[int]) -> List[int]:
        """Perform the core PMX operation."""
        length = min(len(seq1), len(seq2))
        if length < 2:
            return seq1.copy()
        
        # Select crossover points
        start = random.randint(0, length - 2)
        end = random.randint(start + 1, length - 1)
        
        # Initialize offspring with first parent
        offspring = seq1.copy()
        
        # Create mapping from crossover region
        mapping = {}
        for i in range(start, end + 1):
            if seq1[i] != seq2[i]:
                mapping[seq1[i]] = seq2[i]
                offspring[i] = seq2[i]
        
        # Resolve conflicts outside crossover region
        for i in range(length):
            if i < start or i > end:
                current = offspring[i]
                while current in mapping:
                    current = mapping[current]
                offspring[i] = current
        
        return offspring
    
    def timing_crossover(self, parent1: RouteChromosome, parent2: RouteChromosome,
                        constraints: RouteConstraints) -> Tuple[RouteChromosome, RouteChromosome]:
        """
        Timing-focused crossover that optimizes departure times.
        
        This crossover focuses on combining timing information while
        keeping satellite sequences relatively intact.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            constraints: Route constraints to validate offspring
            
        Returns:
            Tuple of two offspring chromosomes
        """
        # Use sequences from parents but cross timing information
        offspring1 = RouteChromosome(
            satellite_sequence=parent1.satellite_sequence.copy(),
            departure_times=self._blend_timing(parent1.departure_times, parent2.departure_times)
        )
        
        offspring2 = RouteChromosome(
            satellite_sequence=parent2.satellite_sequence.copy(),
            departure_times=self._blend_timing(parent2.departure_times, parent1.departure_times)
        )
        
        # Validate and repair if necessary
        offspring1 = self._validate_and_repair(offspring1, constraints)
        offspring2 = self._validate_and_repair(offspring2, constraints)
        
        return offspring1, offspring2
    
    def _crossover_timing(self, times1: List[float], times2: List[float], 
                         target_length: int) -> List[float]:
        """Create timing for offspring by combining parent timings."""
        if target_length == 0:
            return []
        
        if target_length == 1:
            # Single departure time - blend from parents
            if times1 and times2:
                return [(times1[0] + times2[0]) / 2]
            elif times1:
                return [times1[0]]
            elif times2:
                return [times2[0]]
            else:
                return [0.0]
        
        # For multiple departures, interpolate and blend
        result_times = []
        
        # Start time - blend from parents
        start_time = 0.0
        if times1 and times2:
            start_time = (times1[0] + times2[0]) / 2
        elif times1:
            start_time = times1[0]
        elif times2:
            start_time = times2[0]
        
        result_times.append(start_time)
        
        # Generate subsequent times with blended intervals
        current_time = start_time
        for i in range(1, target_length):
            # Calculate intervals from parents
            interval1 = 3600.0  # Default 1 hour
            interval2 = 3600.0
            
            if i < len(times1) and i > 0:
                interval1 = times1[i] - times1[i-1]
            if i < len(times2) and i > 0:
                interval2 = times2[i] - times2[i-1]
            
            # Blend intervals with some randomness
            blended_interval = (interval1 + interval2) / 2
            blended_interval *= random.uniform(0.8, 1.2)  # Add variation
            
            current_time += blended_interval
            result_times.append(current_time)
        
        return result_times
    
    def _blend_timing(self, times1: List[float], times2: List[float]) -> List[float]:
        """Blend timing information from two parents."""
        if not times1:
            return times2.copy() if times2 else []
        if not times2:
            return times1.copy()
        
        # Blend corresponding time points
        min_length = min(len(times1), len(times2))
        blended = []
        
        for i in range(min_length):
            # Weighted blend with some randomness
            weight = random.uniform(0.3, 0.7)
            blended_time = weight * times1[i] + (1 - weight) * times2[i]
            blended.append(blended_time)
        
        # If one parent has more times, use them with some modification
        if len(times1) > min_length:
            for i in range(min_length, len(times1)):
                modified_time = times1[i] * random.uniform(0.9, 1.1)
                blended.append(modified_time)
        elif len(times2) > min_length:
            for i in range(min_length, len(times2)):
                modified_time = times2[i] * random.uniform(0.9, 1.1)
                blended.append(modified_time)
        
        return blended
    
    def _create_valid_offspring(self, parent1: RouteChromosome, parent2: RouteChromosome,
                              constraints: RouteConstraints) -> Tuple[RouteChromosome, RouteChromosome]:
        """Create valid offspring when normal crossover cannot be performed."""
        # Simple fallback - return copies of parents with minor modifications
        offspring1 = RouteChromosome(
            satellite_sequence=parent1.satellite_sequence.copy(),
            departure_times=[t * random.uniform(0.95, 1.05) for t in parent1.departure_times]
        )
        
        offspring2 = RouteChromosome(
            satellite_sequence=parent2.satellite_sequence.copy(),
            departure_times=[t * random.uniform(0.95, 1.05) for t in parent2.departure_times]
        )
        
        return offspring1, offspring2
    
    def _validate_and_repair(self, chromosome: RouteChromosome, 
                           constraints: RouteConstraints) -> RouteChromosome:
        """Validate chromosome and repair basic constraint violations."""
        # Check for duplicate satellites
        sequence = chromosome.satellite_sequence
        if len(set(sequence)) != len(sequence):
            # Remove duplicates while preserving order
            seen = set()
            unique_sequence = []
            for sat_id in sequence:
                if sat_id not in seen:
                    unique_sequence.append(sat_id)
                    seen.add(sat_id)
            
            # Adjust timing to match new sequence length
            if len(unique_sequence) != len(sequence):
                new_times = chromosome.departure_times[:len(unique_sequence)]
                chromosome = RouteChromosome(
                    satellite_sequence=unique_sequence,
                    departure_times=new_times
                )
        
        # Ensure timing list matches sequence length
        if len(chromosome.departure_times) != len(chromosome.satellite_sequence):
            # Adjust timing to match sequence
            seq_len = len(chromosome.satellite_sequence)
            if seq_len == 0:
                times = []
            elif seq_len == 1:
                times = [chromosome.departure_times[0] if chromosome.departure_times else 0.0]
            else:
                # Interpolate or extend timing
                if len(chromosome.departure_times) >= 2:
                    # Interpolate
                    start_time = chromosome.departure_times[0]
                    if len(chromosome.departure_times) > 1:
                        avg_interval = (chromosome.departure_times[-1] - chromosome.departure_times[0]) / (len(chromosome.departure_times) - 1)
                    else:
                        avg_interval = 3600.0  # 1 hour default
                    
                    times = [start_time + i * avg_interval for i in range(seq_len)]
                else:
                    # Generate new timing
                    start_time = chromosome.departure_times[0] if chromosome.departure_times else 0.0
                    times = [start_time + i * 3600.0 for i in range(seq_len)]
            
            chromosome = RouteChromosome(
                satellite_sequence=chromosome.satellite_sequence,
                departure_times=times
            )
        
        # Apply endpoint constraints if specified
        if constraints.start_satellite_id is not None or constraints.end_satellite_id is not None:
            sequence = chromosome.satellite_sequence.copy()
            times = chromosome.departure_times.copy()
            
            # Handle start constraint
            if (constraints.start_satellite_id is not None and 
                constraints.start_satellite_id not in constraints.forbidden_satellites):
                if constraints.start_satellite_id in sequence:
                    # Move to front
                    idx = sequence.index(constraints.start_satellite_id)
                    sequence.pop(idx)
                    time_val = times.pop(idx)
                    sequence.insert(0, constraints.start_satellite_id)
                    times.insert(0, time_val)
                else:
                    # Add to front
                    sequence.insert(0, constraints.start_satellite_id)
                    times.insert(0, times[0] - 3600.0 if times else 0.0)
            
            # Handle end constraint
            if (constraints.end_satellite_id is not None and 
                constraints.end_satellite_id not in constraints.forbidden_satellites):
                if constraints.end_satellite_id in sequence:
                    # Move to end
                    idx = sequence.index(constraints.end_satellite_id)
                    sequence.pop(idx)
                    time_val = times.pop(idx)
                    sequence.append(constraints.end_satellite_id)
                    times.append(time_val)
                else:
                    # Add to end
                    sequence.append(constraints.end_satellite_id)
                    times.append(times[-1] + 3600.0 if times else 3600.0)
            
            chromosome = RouteChromosome(
                satellite_sequence=sequence,
                departure_times=times
            )
        
        return chromosome


class MutationOperator:
    """
    Implements various mutation operations for RouteChromosome.
    
    Mutation operations introduce small random changes to chromosomes
    to maintain genetic diversity and explore new solution spaces.
    """
    
    def __init__(self, config: GAConfig):
        """Initialize mutation operator with configuration."""
        self.config = config
        self.base_mutation_rate = config.mutation_rate
    
    def swap_mutation(self, chromosome: RouteChromosome, constraints: RouteConstraints,
                     mutation_rate: Optional[float] = None) -> RouteChromosome:
        """
        Swap mutation - exchanges positions of two satellites in route.
        
        This mutation swaps two randomly selected satellites in the sequence,
        which can help explore different route orderings.
        
        Args:
            chromosome: Chromosome to mutate
            constraints: Route constraints to respect
            mutation_rate: Override default mutation rate
            
        Returns:
            Mutated chromosome
        """
        rate = mutation_rate if mutation_rate is not None else self.base_mutation_rate
        
        if random.random() > rate or len(chromosome.satellite_sequence) < 2:
            return chromosome
        
        # Create copy for mutation
        new_sequence = chromosome.satellite_sequence.copy()
        new_times = chromosome.departure_times.copy()
        
        # Select two different positions to swap
        pos1 = random.randint(0, len(new_sequence) - 1)
        pos2 = random.randint(0, len(new_sequence) - 1)
        
        # Ensure different positions
        while pos2 == pos1 and len(new_sequence) > 1:
            pos2 = random.randint(0, len(new_sequence) - 1)
        
        # Check endpoint constraints before swapping
        if self._can_swap_positions(pos1, pos2, constraints):
            # Swap satellites and corresponding times
            new_sequence[pos1], new_sequence[pos2] = new_sequence[pos2], new_sequence[pos1]
            if pos1 < len(new_times) and pos2 < len(new_times):
                new_times[pos1], new_times[pos2] = new_times[pos2], new_times[pos1]
        
        mutated = RouteChromosome(
            satellite_sequence=new_sequence,
            departure_times=new_times
        )
        
        return self._validate_mutation(mutated, constraints)
    
    def insert_mutation(self, chromosome: RouteChromosome, constraints: RouteConstraints,
                       mutation_rate: Optional[float] = None) -> RouteChromosome:
        """
        Insert mutation - removes satellite and inserts at different position.
        
        This mutation removes a satellite from one position and inserts it
        at another position, changing the route structure.
        
        Args:
            chromosome: Chromosome to mutate
            constraints: Route constraints to respect
            mutation_rate: Override default mutation rate
            
        Returns:
            Mutated chromosome
        """
        rate = mutation_rate if mutation_rate is not None else self.base_mutation_rate
        
        if random.random() > rate or len(chromosome.satellite_sequence) < 3:
            return chromosome
        
        # Create copy for mutation
        new_sequence = chromosome.satellite_sequence.copy()
        new_times = chromosome.departure_times.copy()
        
        # Select position to remove
        remove_pos = random.randint(0, len(new_sequence) - 1)
        
        # Check if this position can be moved (respect endpoint constraints)
        if not self._can_move_position(remove_pos, constraints):
            return chromosome
        
        # Remove satellite and time
        satellite = new_sequence.pop(remove_pos)
        time_val = new_times.pop(remove_pos) if remove_pos < len(new_times) else 0.0
        
        # Select new insertion position
        insert_pos = random.randint(0, len(new_sequence))
        
        # Insert at new position
        new_sequence.insert(insert_pos, satellite)
        new_times.insert(insert_pos, time_val)
        
        mutated = RouteChromosome(
            satellite_sequence=new_sequence,
            departure_times=new_times
        )
        
        return self._validate_mutation(mutated, constraints)
    
    def time_shift_mutation(self, chromosome: RouteChromosome, constraints: RouteConstraints,
                           mutation_rate: Optional[float] = None) -> RouteChromosome:
        """
        Time shift mutation - adjusts departure times while keeping sequence.
        
        This mutation modifies departure times to explore different timing
        strategies without changing the satellite sequence.
        
        Args:
            chromosome: Chromosome to mutate
            constraints: Route constraints to respect
            mutation_rate: Override default mutation rate
            
        Returns:
            Mutated chromosome
        """
        rate = mutation_rate if mutation_rate is not None else self.base_mutation_rate
        
        if random.random() > rate or not chromosome.departure_times:
            return chromosome
        
        # Create copy for mutation
        new_times = chromosome.departure_times.copy()
        
        # Select mutation type
        mutation_type = random.choice(['shift_all', 'shift_individual', 'scale_intervals'])
        
        if mutation_type == 'shift_all':
            # Shift all times by same amount
            shift = random.uniform(-3600, 3600)  # ±1 hour
            new_times = [max(0, t + shift) for t in new_times]
        
        elif mutation_type == 'shift_individual':
            # Shift individual departure time
            pos = random.randint(0, len(new_times) - 1)
            shift = random.uniform(-1800, 1800)  # ±30 minutes
            new_times[pos] = max(0, new_times[pos] + shift)
        
        elif mutation_type == 'scale_intervals':
            # Scale intervals between departures
            if len(new_times) > 1:
                scale_factor = random.uniform(0.8, 1.2)
                start_time = new_times[0]
                
                for i in range(1, len(new_times)):
                    interval = new_times[i] - new_times[i-1]
                    scaled_interval = interval * scale_factor
                    new_times[i] = new_times[i-1] + scaled_interval
        
        # Ensure times are in ascending order
        new_times = self._ensure_ascending_times(new_times)
        
        mutated = RouteChromosome(
            satellite_sequence=chromosome.satellite_sequence.copy(),
            departure_times=new_times
        )
        
        return self._validate_mutation(mutated, constraints)
    
    def inversion_mutation(self, chromosome: RouteChromosome, constraints: RouteConstraints,
                          mutation_rate: Optional[float] = None) -> RouteChromosome:
        """
        Inversion mutation - reverses order of route segment.
        
        This mutation reverses the order of satellites in a randomly
        selected segment of the route.
        
        Args:
            chromosome: Chromosome to mutate
            constraints: Route constraints to respect
            mutation_rate: Override default mutation rate
            
        Returns:
            Mutated chromosome
        """
        rate = mutation_rate if mutation_rate is not None else self.base_mutation_rate
        
        if random.random() > rate or len(chromosome.satellite_sequence) < 3:
            return chromosome
        
        # Create copy for mutation
        new_sequence = chromosome.satellite_sequence.copy()
        new_times = chromosome.departure_times.copy()
        
        # Select segment to invert
        start = random.randint(0, len(new_sequence) - 2)
        end = random.randint(start + 1, len(new_sequence) - 1)
        
        # Check if inversion respects endpoint constraints
        if not self._can_invert_segment(start, end, constraints):
            return chromosome
        
        # Invert the segment
        new_sequence[start:end+1] = new_sequence[start:end+1][::-1]
        if start < len(new_times) and end < len(new_times):
            new_times[start:end+1] = new_times[start:end+1][::-1]
        
        mutated = RouteChromosome(
            satellite_sequence=new_sequence,
            departure_times=new_times
        )
        
        return self._validate_mutation(mutated, constraints)
    
    def adaptive_mutation(self, chromosome: RouteChromosome, constraints: RouteConstraints,
                         population_diversity: float) -> RouteChromosome:
        """
        Adaptive mutation - adjusts mutation rate based on population diversity.
        
        This mutation adapts the mutation rate based on the current population
        diversity to balance exploration and exploitation.
        
        Args:
            chromosome: Chromosome to mutate
            constraints: Route constraints to respect
            population_diversity: Current population diversity metric (0-1)
            
        Returns:
            Mutated chromosome
        """
        # Adapt mutation rate based on diversity
        # Low diversity -> higher mutation rate
        # High diversity -> lower mutation rate
        adapted_rate = self.base_mutation_rate * (2.0 - population_diversity)
        adapted_rate = max(0.01, min(0.5, adapted_rate))  # Clamp between 1% and 50%
        
        # Apply multiple mutation types with adapted rate
        mutated = chromosome
        
        # Apply mutations in sequence with decreasing probability
        if random.random() < adapted_rate:
            mutated = self.swap_mutation(mutated, constraints, adapted_rate * 0.8)
        
        if random.random() < adapted_rate * 0.7:
            mutated = self.time_shift_mutation(mutated, constraints, adapted_rate * 0.6)
        
        if random.random() < adapted_rate * 0.5:
            mutated = self.insert_mutation(mutated, constraints, adapted_rate * 0.4)
        
        if random.random() < adapted_rate * 0.3:
            mutated = self.inversion_mutation(mutated, constraints, adapted_rate * 0.2)
        
        return mutated
    
    def _can_swap_positions(self, pos1: int, pos2: int, constraints: RouteConstraints) -> bool:
        """Check if two positions can be swapped given constraints."""
        # Cannot swap if one position is constrained start/end
        if constraints.start_satellite_id is not None and (pos1 == 0 or pos2 == 0):
            return False
        
        # For end constraint, we need to know sequence length, so we'll be permissive here
        # and let validation handle it
        return True
    
    def _can_move_position(self, pos: int, constraints: RouteConstraints) -> bool:
        """Check if a position can be moved given constraints."""
        # Cannot move constrained start position
        if constraints.start_satellite_id is not None and pos == 0:
            return False
        
        # For end constraint, we'll be permissive and let validation handle it
        return True
    
    def _can_invert_segment(self, start: int, end: int, constraints: RouteConstraints) -> bool:
        """Check if a segment can be inverted given constraints."""
        # Cannot invert if it includes constrained start position
        if constraints.start_satellite_id is not None and start == 0:
            return False
        
        # For end constraint, we'll be permissive and let validation handle it
        return True
    
    def _ensure_ascending_times(self, times: List[float]) -> List[float]:
        """Ensure departure times are in ascending order."""
        if len(times) <= 1:
            return times
        
        # Sort times while maintaining relative spacing
        sorted_times = times.copy()
        
        # Ensure each time is at least as large as the previous
        for i in range(1, len(sorted_times)):
            if sorted_times[i] <= sorted_times[i-1]:
                # Add small increment to maintain order
                sorted_times[i] = sorted_times[i-1] + random.uniform(60, 600)  # 1-10 minutes
        
        return sorted_times
    
    def _validate_mutation(self, chromosome: RouteChromosome, 
                          constraints: RouteConstraints) -> RouteChromosome:
        """Validate mutated chromosome and apply constraint repairs."""
        # Remove forbidden satellites
        if constraints.forbidden_satellites:
            new_sequence = []
            new_times = []
            
            for i, sat_id in enumerate(chromosome.satellite_sequence):
                if sat_id not in constraints.forbidden_satellites:
                    new_sequence.append(sat_id)
                    if i < len(chromosome.departure_times):
                        new_times.append(chromosome.departure_times[i])
            
            if len(new_sequence) != len(chromosome.satellite_sequence):
                chromosome = RouteChromosome(
                    satellite_sequence=new_sequence,
                    departure_times=new_times
                )
        
        # Ensure endpoint constraints are satisfied
        sequence = chromosome.satellite_sequence.copy()
        times = chromosome.departure_times.copy()
        
        # Handle start constraint
        if (constraints.start_satellite_id is not None and 
            constraints.start_satellite_id not in constraints.forbidden_satellites):
            if sequence and sequence[0] != constraints.start_satellite_id:
                if constraints.start_satellite_id in sequence:
                    # Move to front
                    idx = sequence.index(constraints.start_satellite_id)
                    sequence.pop(idx)
                    if idx < len(times):
                        time_val = times.pop(idx)
                    else:
                        time_val = times[0] - 3600.0 if times else 0.0
                    sequence.insert(0, constraints.start_satellite_id)
                    times.insert(0, time_val)
        
        # Handle end constraint
        if (constraints.end_satellite_id is not None and 
            constraints.end_satellite_id not in constraints.forbidden_satellites):
            if sequence and sequence[-1] != constraints.end_satellite_id:
                if constraints.end_satellite_id in sequence:
                    # Move to end
                    idx = sequence.index(constraints.end_satellite_id)
                    sequence.pop(idx)
                    if idx < len(times):
                        time_val = times.pop(idx)
                    else:
                        time_val = times[-1] + 3600.0 if times else 3600.0
                    sequence.append(constraints.end_satellite_id)
                    times.append(time_val)
        
        # Ensure times are ascending
        times = self._ensure_ascending_times(times)
        
        return RouteChromosome(
            satellite_sequence=sequence,
            departure_times=times
        )


class SelectionOperator:
    """
    Implements various selection mechanisms for genetic algorithm.
    
    Selection operations choose which chromosomes will be parents
    for the next generation based on their fitness values.
    """
    
    def __init__(self, config: GAConfig):
        """Initialize selection operator with configuration."""
        self.config = config
    
    def tournament_selection(self, population: List[RouteChromosome], 
                           fitness_scores: List[float],
                           num_parents: int) -> List[RouteChromosome]:
        """
        Tournament selection with configurable tournament size.
        
        Selects parents by running tournaments between randomly chosen
        individuals and selecting the best from each tournament.
        
        Args:
            population: Population of chromosomes
            fitness_scores: Fitness scores for each chromosome
            num_parents: Number of parents to select
            
        Returns:
            List of selected parent chromosomes
        """
        if not population or not fitness_scores:
            return []
        
        parents = []
        tournament_size = min(self.config.tournament_size, len(population))
        
        for _ in range(num_parents):
            # Select random individuals for tournament
            tournament_indices = random.sample(range(len(population)), tournament_size)
            
            # Find best individual in tournament (highest fitness)
            best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
            parents.append(population[best_idx])
        
        return parents
    
    def elitism_selection(self, population: List[RouteChromosome],
                         fitness_scores: List[float]) -> List[RouteChromosome]:
        """
        Elitism selection - preserves best solutions across generations.
        
        Selects the top-performing chromosomes to carry forward
        to the next generation unchanged.
        
        Args:
            population: Population of chromosomes
            fitness_scores: Fitness scores for each chromosome
            
        Returns:
            List of elite chromosomes
        """
        if not population or not fitness_scores:
            return []
        
        elite_count = min(self.config.elitism_count, len(population))
        
        # Sort by fitness (descending - higher is better)
        sorted_indices = sorted(range(len(population)), 
                              key=lambda i: fitness_scores[i], reverse=True)
        
        # Select top performers
        elites = [population[i] for i in sorted_indices[:elite_count]]
        
        return elites
    
    def fitness_proportionate_selection(self, population: List[RouteChromosome],
                                      fitness_scores: List[float],
                                      num_parents: int) -> List[RouteChromosome]:
        """
        Fitness-proportionate selection (roulette wheel).
        
        Selects parents with probability proportional to their fitness.
        Higher fitness chromosomes have higher probability of selection.
        
        Args:
            population: Population of chromosomes
            fitness_scores: Fitness scores for each chromosome
            num_parents: Number of parents to select
            
        Returns:
            List of selected parent chromosomes
        """
        if not population or not fitness_scores:
            return []
        
        # Handle negative fitness scores by shifting
        min_fitness = min(fitness_scores)
        if min_fitness < 0:
            adjusted_scores = [score - min_fitness + 1 for score in fitness_scores]
        else:
            adjusted_scores = fitness_scores.copy()
        
        # Calculate total fitness
        total_fitness = sum(adjusted_scores)
        if total_fitness == 0:
            # Fallback to uniform selection
            return random.choices(population, k=num_parents)
        
        # Calculate selection probabilities
        probabilities = [score / total_fitness for score in adjusted_scores]
        
        # Select parents based on probabilities
        parents = random.choices(population, weights=probabilities, k=num_parents)
        
        return parents
    
    def rank_selection(self, population: List[RouteChromosome],
                      fitness_scores: List[float],
                      num_parents: int) -> List[RouteChromosome]:
        """
        Rank-based selection.
        
        Selects parents based on fitness ranking rather than absolute
        fitness values, which helps when fitness values have large variance.
        
        Args:
            population: Population of chromosomes
            fitness_scores: Fitness scores for each chromosome
            num_parents: Number of parents to select
            
        Returns:
            List of selected parent chromosomes
        """
        if not population or not fitness_scores:
            return []
        
        # Create rank-based weights (linear ranking)
        n = len(population)
        sorted_indices = sorted(range(n), key=lambda i: fitness_scores[i])
        
        # Assign ranks (1 to n, where n is best)
        ranks = [0] * n
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        
        # Use ranks as weights for selection
        parents = random.choices(population, weights=ranks, k=num_parents)
        
        return parents
    
    def diversity_preserving_selection(self, population: List[RouteChromosome],
                                     fitness_scores: List[float],
                                     num_parents: int) -> List[RouteChromosome]:
        """
        Diversity-preserving selection to prevent premature convergence.
        
        Combines fitness-based selection with diversity considerations
        to maintain genetic diversity in the population.
        
        Args:
            population: Population of chromosomes
            fitness_scores: Fitness scores for each chromosome
            num_parents: Number of parents to select
            
        Returns:
            List of selected parent chromosomes
        """
        if not population or not fitness_scores:
            return []
        
        parents = []
        remaining_population = list(zip(population, fitness_scores))
        
        for _ in range(num_parents):
            if not remaining_population:
                break
            
            # Calculate diversity scores
            diversity_scores = []
            for i, (candidate, fitness) in enumerate(remaining_population):
                diversity = self._calculate_diversity_score(candidate, parents)
                # Combine fitness and diversity (weighted)
                combined_score = 0.7 * fitness + 0.3 * diversity
                diversity_scores.append(combined_score)
            
            # Select based on combined score
            if diversity_scores:
                best_idx = max(range(len(diversity_scores)), 
                             key=lambda i: diversity_scores[i])
                selected_parent, _ = remaining_population.pop(best_idx)
                parents.append(selected_parent)
        
        return parents
    
    def _calculate_diversity_score(self, candidate: RouteChromosome, 
                                 existing_parents: List[RouteChromosome]) -> float:
        """Calculate diversity score for a candidate relative to existing parents."""
        if not existing_parents:
            return 1.0  # Maximum diversity if no parents selected yet
        
        # Calculate minimum distance to existing parents
        min_distance = float('inf')
        
        for parent in existing_parents:
            distance = self._calculate_chromosome_distance(candidate, parent)
            min_distance = min(min_distance, distance)
        
        # Normalize distance to [0, 1] range
        max_possible_distance = len(candidate.satellite_sequence) + 1
        normalized_distance = min_distance / max_possible_distance
        
        return normalized_distance
    
    def _calculate_chromosome_distance(self, chrom1: RouteChromosome, 
                                     chrom2: RouteChromosome) -> float:
        """Calculate distance between two chromosomes."""
        seq1, seq2 = chrom1.satellite_sequence, chrom2.satellite_sequence
        
        # Calculate sequence similarity (Jaccard distance)
        set1, set2 = set(seq1), set(seq2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            jaccard_similarity = 1.0
        else:
            jaccard_similarity = intersection / union
        
        jaccard_distance = 1.0 - jaccard_similarity
        
        # Calculate timing difference
        times1, times2 = chrom1.departure_times, chrom2.departure_times
        time_distance = 0.0
        
        if times1 and times2:
            # Compare normalized timing patterns
            min_len = min(len(times1), len(times2))
            if min_len > 1:
                # Calculate average intervals
                avg_interval1 = (times1[-1] - times1[0]) / (len(times1) - 1) if len(times1) > 1 else 0
                avg_interval2 = (times2[-1] - times2[0]) / (len(times2) - 1) if len(times2) > 1 else 0
                
                if avg_interval1 + avg_interval2 > 0:
                    time_distance = abs(avg_interval1 - avg_interval2) / (avg_interval1 + avg_interval2)
        
        # Combine distances
        combined_distance = 0.8 * jaccard_distance + 0.2 * time_distance
        
        return combined_distance
    
    def select_next_generation(self, population: List[RouteChromosome],
                             fitness_scores: List[float],
                             selection_method: str = 'tournament') -> List[RouteChromosome]:
        """
        Select chromosomes for the next generation using specified method.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for each chromosome
            selection_method: Selection method to use
            
        Returns:
            Selected chromosomes for next generation
        """
        if not population:
            return []
        
        next_generation = []
        
        # Always preserve elites
        elites = self.elitism_selection(population, fitness_scores)
        next_generation.extend(elites)
        
        # Fill remaining slots with selected parents
        remaining_slots = len(population) - len(elites)
        
        if remaining_slots > 0:
            if selection_method == 'tournament':
                parents = self.tournament_selection(population, fitness_scores, remaining_slots)
            elif selection_method == 'fitness_proportionate':
                parents = self.fitness_proportionate_selection(population, fitness_scores, remaining_slots)
            elif selection_method == 'rank':
                parents = self.rank_selection(population, fitness_scores, remaining_slots)
            elif selection_method == 'diversity':
                parents = self.diversity_preserving_selection(population, fitness_scores, remaining_slots)
            else:
                # Default to tournament selection
                parents = self.tournament_selection(population, fitness_scores, remaining_slots)
            
            next_generation.extend(parents)
        
        return next_generation
    
    def calculate_population_diversity(self, population: List[RouteChromosome]) -> float:
        """
        Calculate diversity metric for the population.
        
        Args:
            population: Population to analyze
            
        Returns:
            Diversity metric (0.0 = no diversity, 1.0 = maximum diversity)
        """
        if len(population) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        # Calculate pairwise distances
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = self._calculate_chromosome_distance(population[i], population[j])
                total_distance += distance
                comparisons += 1
        
        if comparisons == 0:
            return 0.0
        
        # Return average pairwise distance
        return total_distance / comparisons