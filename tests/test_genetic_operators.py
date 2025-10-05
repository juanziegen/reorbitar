"""
Unit tests for genetic algorithm operators.

Tests crossover, mutation, and selection operations to ensure they
maintain chromosome validity and respect constraints.
"""

import unittest
import random
from typing import List
from src.genetic_algorithm import RouteChromosome, RouteConstraints, GAConfig
from src.genetic_operators import CrossoverOperator, MutationOperator, SelectionOperator


class TestCrossoverOperator(unittest.TestCase):
    """Test crossover operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GAConfig(
            population_size=50,
            mutation_rate=0.1,
            crossover_rate=0.8,
            tournament_size=3
        )
        self.crossover_op = CrossoverOperator(self.config)
        
        # Create test constraints
        self.constraints = RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400 * 7,  # 7 days
            min_hops=2,
            max_hops=10
        )
        
        # Create test chromosomes
        self.parent1 = RouteChromosome(
            satellite_sequence=[1001, 1002, 1003, 1004, 1005],
            departure_times=[0.0, 3600.0, 7200.0, 10800.0, 14400.0]
        )
        
        self.parent2 = RouteChromosome(
            satellite_sequence=[1003, 1006, 1001, 1007, 1008],
            departure_times=[1800.0, 5400.0, 9000.0, 12600.0, 16200.0]
        )
    
    def test_order_crossover_preserves_satellites(self):
        """Test that order crossover preserves valid satellite sequences."""
        offspring1, offspring2 = self.crossover_op.order_crossover(
            self.parent1, self.parent2, self.constraints
        )
        
        # Check that offspring contain valid satellite sequences
        self.assertIsInstance(offspring1.satellite_sequence, list)
        self.assertIsInstance(offspring2.satellite_sequence, list)
        
        # Check no duplicate satellites in offspring
        self.assertEqual(len(offspring1.satellite_sequence), 
                        len(set(offspring1.satellite_sequence)))
        self.assertEqual(len(offspring2.satellite_sequence), 
                        len(set(offspring2.satellite_sequence)))
        
        # Check timing matches sequence length
        self.assertEqual(len(offspring1.satellite_sequence), 
                        len(offspring1.departure_times))
        self.assertEqual(len(offspring2.satellite_sequence), 
                        len(offspring2.departure_times))
    
    def test_pmx_crossover_maintains_relationships(self):
        """Test that PMX crossover maintains satellite relationships."""
        offspring1, offspring2 = self.crossover_op.partially_mapped_crossover(
            self.parent1, self.parent2, self.constraints
        )
        
        # Check basic validity
        self.assertIsInstance(offspring1, RouteChromosome)
        self.assertIsInstance(offspring2, RouteChromosome)
        
        # Check no duplicates
        self.assertEqual(len(offspring1.satellite_sequence), 
                        len(set(offspring1.satellite_sequence)))
        self.assertEqual(len(offspring2.satellite_sequence), 
                        len(set(offspring2.satellite_sequence)))
    
    def test_timing_crossover_preserves_sequences(self):
        """Test that timing crossover preserves satellite sequences."""
        offspring1, offspring2 = self.crossover_op.timing_crossover(
            self.parent1, self.parent2, self.constraints
        )
        
        # Sequences should be preserved from parents
        self.assertEqual(offspring1.satellite_sequence, self.parent1.satellite_sequence)
        self.assertEqual(offspring2.satellite_sequence, self.parent2.satellite_sequence)
        
        # Times should be modified
        self.assertNotEqual(offspring1.departure_times, self.parent1.departure_times)
        self.assertNotEqual(offspring2.departure_times, self.parent2.departure_times)
    
    def test_crossover_with_constraints(self):
        """Test crossover operations respect endpoint constraints."""
        constrained = RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400 * 7,
            start_satellite_id=1001,
            end_satellite_id=1008,
            min_hops=2,
            max_hops=10
        )
        
        offspring1, offspring2 = self.crossover_op.order_crossover(
            self.parent1, self.parent2, constrained
        )
        
        # Check start constraint if satellite is available
        if 1001 in offspring1.satellite_sequence:
            self.assertEqual(offspring1.satellite_sequence[0], 1001)
        if 1001 in offspring2.satellite_sequence:
            self.assertEqual(offspring2.satellite_sequence[0], 1001)
        
        # Check end constraint if satellite is available
        if 1008 in offspring1.satellite_sequence:
            self.assertEqual(offspring1.satellite_sequence[-1], 1008)
        if 1008 in offspring2.satellite_sequence:
            self.assertEqual(offspring2.satellite_sequence[-1], 1008)
    
    def test_crossover_with_short_sequences(self):
        """Test crossover handles short sequences gracefully."""
        short_parent1 = RouteChromosome(
            satellite_sequence=[1001],
            departure_times=[0.0]
        )
        
        short_parent2 = RouteChromosome(
            satellite_sequence=[1002],
            departure_times=[1800.0]
        )
        
        offspring1, offspring2 = self.crossover_op.order_crossover(
            short_parent1, short_parent2, self.constraints
        )
        
        # Should handle gracefully without errors
        self.assertIsInstance(offspring1, RouteChromosome)
        self.assertIsInstance(offspring2, RouteChromosome)


class TestMutationOperator(unittest.TestCase):
    """Test mutation operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GAConfig(mutation_rate=0.5)  # High rate for testing
        self.mutation_op = MutationOperator(self.config)
        
        self.constraints = RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400 * 7,
            min_hops=2,
            max_hops=10
        )
        
        self.chromosome = RouteChromosome(
            satellite_sequence=[1001, 1002, 1003, 1004, 1005],
            departure_times=[0.0, 3600.0, 7200.0, 10800.0, 14400.0]
        )
    
    def test_swap_mutation_maintains_validity(self):
        """Test that swap mutation maintains chromosome validity."""
        mutated = self.mutation_op.swap_mutation(self.chromosome, self.constraints)
        
        # Check basic structure
        self.assertIsInstance(mutated, RouteChromosome)
        self.assertEqual(len(mutated.satellite_sequence), len(self.chromosome.satellite_sequence))
        self.assertEqual(len(mutated.departure_times), len(self.chromosome.departure_times))
        
        # Check no duplicates
        self.assertEqual(len(mutated.satellite_sequence), len(set(mutated.satellite_sequence)))
        
        # Check same satellites (just reordered)
        self.assertEqual(set(mutated.satellite_sequence), set(self.chromosome.satellite_sequence))
    
    def test_insert_mutation_preserves_satellites(self):
        """Test that insert mutation preserves satellite set."""
        mutated = self.mutation_op.insert_mutation(self.chromosome, self.constraints)
        
        # Check basic validity
        self.assertIsInstance(mutated, RouteChromosome)
        self.assertEqual(len(mutated.satellite_sequence), len(set(mutated.satellite_sequence)))
        
        # Should have same satellites
        self.assertEqual(set(mutated.satellite_sequence), set(self.chromosome.satellite_sequence))
    
    def test_time_shift_mutation_preserves_sequence(self):
        """Test that time shift mutation preserves satellite sequence."""
        mutated = self.mutation_op.time_shift_mutation(self.chromosome, self.constraints)
        
        # Sequence should be unchanged
        self.assertEqual(mutated.satellite_sequence, self.chromosome.satellite_sequence)
        
        # Times should be modified (with high mutation rate)
        # Note: Due to randomness, times might occasionally be the same
        self.assertEqual(len(mutated.departure_times), len(self.chromosome.departure_times))
    
    def test_inversion_mutation_maintains_satellites(self):
        """Test that inversion mutation maintains satellite set."""
        mutated = self.mutation_op.inversion_mutation(self.chromosome, self.constraints)
        
        # Check basic validity
        self.assertIsInstance(mutated, RouteChromosome)
        self.assertEqual(len(mutated.satellite_sequence), len(set(mutated.satellite_sequence)))
        
        # Should have same satellites
        self.assertEqual(set(mutated.satellite_sequence), set(self.chromosome.satellite_sequence))
    
    def test_adaptive_mutation_responds_to_diversity(self):
        """Test that adaptive mutation responds to population diversity."""
        # Test with low diversity (should increase mutation)
        low_diversity_mutated = self.mutation_op.adaptive_mutation(
            self.chromosome, self.constraints, population_diversity=0.1
        )
        
        # Test with high diversity (should decrease mutation)
        high_diversity_mutated = self.mutation_op.adaptive_mutation(
            self.chromosome, self.constraints, population_diversity=0.9
        )
        
        # Both should be valid
        self.assertIsInstance(low_diversity_mutated, RouteChromosome)
        self.assertIsInstance(high_diversity_mutated, RouteChromosome)
    
    def test_mutation_respects_constraints(self):
        """Test that mutations respect endpoint constraints."""
        constrained = RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400 * 7,
            start_satellite_id=1001,
            end_satellite_id=1005,
            min_hops=2,
            max_hops=10
        )
        
        mutated = self.mutation_op.swap_mutation(self.chromosome, constrained)
        
        # Should respect start constraint
        if mutated.satellite_sequence:
            self.assertEqual(mutated.satellite_sequence[0], 1001)
        
        # Should respect end constraint
        if mutated.satellite_sequence:
            self.assertEqual(mutated.satellite_sequence[-1], 1005)
    
    def test_mutation_handles_forbidden_satellites(self):
        """Test that mutations handle forbidden satellites."""
        # Create chromosome without forbidden satellites
        clean_chromosome = RouteChromosome(
            satellite_sequence=[1001, 1002, 1005, 1006, 1007],
            departure_times=[0.0, 3600.0, 7200.0, 10800.0, 14400.0]
        )
        
        constrained = RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400 * 7,
            forbidden_satellites=[1003, 1004],
            min_hops=2,
            max_hops=10
        )
        
        mutated = self.mutation_op.swap_mutation(clean_chromosome, constrained)
        
        # Should not contain forbidden satellites
        for sat_id in mutated.satellite_sequence:
            self.assertNotIn(sat_id, constrained.forbidden_satellites)
        
        # Test that mutation removes forbidden satellites if they exist
        contaminated_chromosome = RouteChromosome(
            satellite_sequence=[1001, 1003, 1005, 1004, 1007],  # Contains forbidden satellites
            departure_times=[0.0, 3600.0, 7200.0, 10800.0, 14400.0]
        )
        
        mutated_contaminated = self.mutation_op.swap_mutation(contaminated_chromosome, constrained, mutation_rate=1.0)
        
        # Should have removed forbidden satellites
        for sat_id in mutated_contaminated.satellite_sequence:
            self.assertNotIn(sat_id, constrained.forbidden_satellites)


class TestSelectionOperator(unittest.TestCase):
    """Test selection operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GAConfig(
            elitism_count=2,
            tournament_size=3
        )
        self.selection_op = SelectionOperator(self.config)
        
        # Create test population
        self.population = [
            RouteChromosome([1001, 1002], [0.0, 3600.0]),
            RouteChromosome([1003, 1004], [1800.0, 5400.0]),
            RouteChromosome([1005, 1006], [3600.0, 7200.0]),
            RouteChromosome([1007, 1008], [5400.0, 9000.0]),
            RouteChromosome([1009, 1010], [7200.0, 10800.0])
        ]
        
        # Fitness scores (higher is better)
        self.fitness_scores = [10.0, 8.0, 6.0, 4.0, 2.0]
    
    def test_tournament_selection_returns_valid_parents(self):
        """Test that tournament selection returns valid parents."""
        parents = self.selection_op.tournament_selection(
            self.population, self.fitness_scores, num_parents=3
        )
        
        self.assertEqual(len(parents), 3)
        for parent in parents:
            self.assertIn(parent, self.population)
    
    def test_elitism_selection_preserves_best(self):
        """Test that elitism selection preserves best chromosomes."""
        elites = self.selection_op.elitism_selection(self.population, self.fitness_scores)
        
        self.assertEqual(len(elites), self.config.elitism_count)
        
        # Should select chromosomes with highest fitness
        self.assertEqual(elites[0], self.population[0])  # Fitness 10.0
        self.assertEqual(elites[1], self.population[1])  # Fitness 8.0
    
    def test_fitness_proportionate_selection(self):
        """Test fitness-proportionate selection."""
        parents = self.selection_op.fitness_proportionate_selection(
            self.population, self.fitness_scores, num_parents=3
        )
        
        self.assertEqual(len(parents), 3)
        for parent in parents:
            self.assertIn(parent, self.population)
    
    def test_rank_selection(self):
        """Test rank-based selection."""
        parents = self.selection_op.rank_selection(
            self.population, self.fitness_scores, num_parents=3
        )
        
        self.assertEqual(len(parents), 3)
        for parent in parents:
            self.assertIn(parent, self.population)
    
    def test_diversity_preserving_selection(self):
        """Test diversity-preserving selection."""
        parents = self.selection_op.diversity_preserving_selection(
            self.population, self.fitness_scores, num_parents=3
        )
        
        self.assertEqual(len(parents), 3)
        for parent in parents:
            self.assertIn(parent, self.population)
    
    def test_population_diversity_calculation(self):
        """Test population diversity calculation."""
        diversity = self.selection_op.calculate_population_diversity(self.population)
        
        self.assertIsInstance(diversity, float)
        self.assertGreaterEqual(diversity, 0.0)
        self.assertLessEqual(diversity, 1.0)
    
    def test_select_next_generation(self):
        """Test complete next generation selection."""
        next_gen = self.selection_op.select_next_generation(
            self.population, self.fitness_scores, selection_method='tournament'
        )
        
        self.assertEqual(len(next_gen), len(self.population))
        
        # Should include elites
        self.assertIn(self.population[0], next_gen)  # Best fitness
        self.assertIn(self.population[1], next_gen)  # Second best
    
    def test_selection_with_empty_population(self):
        """Test selection operations handle empty populations."""
        empty_pop = []
        empty_fitness = []
        
        parents = self.selection_op.tournament_selection(empty_pop, empty_fitness, 3)
        self.assertEqual(len(parents), 0)
        
        elites = self.selection_op.elitism_selection(empty_pop, empty_fitness)
        self.assertEqual(len(elites), 0)
    
    def test_selection_with_negative_fitness(self):
        """Test selection handles negative fitness scores."""
        negative_fitness = [-5.0, -3.0, -1.0, 0.0, 2.0]
        
        parents = self.selection_op.fitness_proportionate_selection(
            self.population, negative_fitness, num_parents=3
        )
        
        self.assertEqual(len(parents), 3)
        for parent in parents:
            self.assertIn(parent, self.population)


class TestGeneticOperatorsIntegration(unittest.TestCase):
    """Integration tests for genetic operators."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GAConfig(
            population_size=10,
            mutation_rate=0.2,
            crossover_rate=0.8,
            elitism_count=2,
            tournament_size=3
        )
        
        self.crossover_op = CrossoverOperator(self.config)
        self.mutation_op = MutationOperator(self.config)
        self.selection_op = SelectionOperator(self.config)
        
        self.constraints = RouteConstraints(
            max_deltav_budget=5.0,
            max_mission_duration=86400 * 7,
            min_hops=2,
            max_hops=8
        )
    
    def test_complete_genetic_cycle(self):
        """Test a complete genetic algorithm cycle."""
        # Create initial population
        population = []
        for i in range(self.config.population_size):
            chromosome = RouteChromosome(
                satellite_sequence=[1000 + j for j in range(i % 5 + 2)],
                departure_times=[j * 3600.0 for j in range(i % 5 + 2)]
            )
            population.append(chromosome)
        
        # Mock fitness scores
        fitness_scores = [random.uniform(1.0, 10.0) for _ in population]
        
        # Selection
        parents = self.selection_op.tournament_selection(
            population, fitness_scores, num_parents=6
        )
        
        # Crossover
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            child1, child2 = self.crossover_op.order_crossover(
                parents[i], parents[i + 1], self.constraints
            )
            offspring.extend([child1, child2])
        
        # Mutation
        mutated_offspring = []
        for child in offspring:
            mutated = self.mutation_op.swap_mutation(child, self.constraints)
            mutated_offspring.append(mutated)
        
        # Verify all operations completed successfully
        self.assertGreater(len(parents), 0)
        self.assertGreater(len(offspring), 0)
        self.assertEqual(len(mutated_offspring), len(offspring))
        
        # Verify chromosome validity
        for chromosome in mutated_offspring:
            self.assertIsInstance(chromosome, RouteChromosome)
            self.assertEqual(len(chromosome.satellite_sequence), 
                           len(set(chromosome.satellite_sequence)))
            self.assertEqual(len(chromosome.satellite_sequence), 
                           len(chromosome.departure_times))


if __name__ == '__main__':
    # Set random seed for reproducible tests
    random.seed(42)
    unittest.main()