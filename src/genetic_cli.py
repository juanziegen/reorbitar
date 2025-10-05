"""
Genetic Algorithm CLI Module

Command-line interface for genetic algorithm-based satellite route optimization.
Provides interactive and batch modes for route optimization with comprehensive
parameter configuration and result visualization.
"""

import argparse
import sys
import time
import json
import os
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

from src.tle_parser import SatelliteData, TLEParser
from src.genetic_route_optimizer import GeneticRouteOptimizer
from src.genetic_algorithm import (
    GAConfig, RouteConstraints, OptimizationResult, 
    RouteChromosome, OptimizationStatus
)


class GeneticCLI:
    """Command-line interface for genetic algorithm route optimization."""
    
    def __init__(self):
        """Initialize CLI with default settings."""
        self.satellites: List[SatelliteData] = []
        self.optimizer: Optional[GeneticRouteOptimizer] = None
        self.verbose = False
        self.interactive_mode = False
        
    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Main entry point for CLI execution.
        
        Args:
            args: Command line arguments (None to use sys.argv)
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            # Parse command line arguments
            parsed_args = self._parse_arguments(args)
            
            # Set verbosity
            self.verbose = parsed_args.verbose
            self.interactive_mode = parsed_args.interactive
            
            # Load satellite data
            if not self._load_satellites(parsed_args.tle_file):
                return 1
            
            # Initialize optimizer
            self._initialize_optimizer(parsed_args)
            
            # Run optimization based on mode
            if self.interactive_mode:
                return self._run_interactive_mode()
            else:
                return self._run_batch_mode(parsed_args)
                
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user.")
            return 130
        except Exception as e:
            print(f"Error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def _parse_arguments(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="Genetic Algorithm Satellite Route Optimizer",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic optimization with 5 km/s budget over 2 years
  python -m src.genetic_cli --deltav-budget 5.0 --timeframe 63072000
  
  # Interactive mode for parameter tuning
  python -m src.genetic_cli --interactive
  
  # Custom GA parameters with specific start/end satellites
  python -m src.genetic_cli --deltav-budget 3.0 --timeframe 31536000 \\
    --population-size 200 --generations 1000 --start-satellite 25544 --end-satellite 39084
  
  # Save results to custom file
  python -m src.genetic_cli --deltav-budget 4.0 --timeframe 94608000 \\
    --output results/optimization_run_1.json
            """
        )
        
        # Required parameters
        parser.add_argument(
            '--deltav-budget', '-d',
            type=float,
            help='Maximum delta-v budget in km/s (required unless --interactive)'
        )
        
        parser.add_argument(
            '--timeframe', '-t',
            type=float,
            help='Mission timeframe in seconds (required unless --interactive)'
        )
        
        # Optional route constraints
        parser.add_argument(
            '--start-satellite', '-s',
            type=int,
            help='Starting satellite catalog number (optional)'
        )
        
        parser.add_argument(
            '--end-satellite', '-e',
            type=int,
            help='Ending satellite catalog number (optional)'
        )
        
        parser.add_argument(
            '--min-hops',
            type=int,
            default=1,
            help='Minimum number of hops (default: 1)'
        )
        
        parser.add_argument(
            '--max-hops',
            type=int,
            default=50,
            help='Maximum number of hops (default: 50)'
        )
        
        parser.add_argument(
            '--forbidden-satellites',
            type=str,
            help='Comma-separated list of forbidden satellite catalog numbers'
        )
        
        # Genetic algorithm parameters
        parser.add_argument(
            '--population-size', '-p',
            type=int,
            default=100,
            help='GA population size (default: 100)'
        )
        
        parser.add_argument(
            '--generations', '-g',
            type=int,
            default=500,
            help='Maximum generations (default: 500)'
        )
        
        parser.add_argument(
            '--mutation-rate', '-m',
            type=float,
            default=0.1,
            help='Mutation rate (0.0-1.0, default: 0.1)'
        )
        
        parser.add_argument(
            '--crossover-rate', '-c',
            type=float,
            default=0.8,
            help='Crossover rate (0.0-1.0, default: 0.8)'
        )
        
        parser.add_argument(
            '--elitism-count',
            type=int,
            default=5,
            help='Number of elite chromosomes to preserve (default: 5)'
        )
        
        parser.add_argument(
            '--tournament-size',
            type=int,
            default=3,
            help='Tournament selection size (default: 3)'
        )
        
        # Input/output options
        parser.add_argument(
            '--tle-file',
            type=str,
            default='leo_satellites.txt',
            help='TLE data file path (default: leo_satellites.txt)'
        )
        
        parser.add_argument(
            '--output', '-o',
            type=str,
            help='Output file for results (default: auto-generated)'
        )
        
        parser.add_argument(
            '--format',
            choices=['json', 'text', 'csv'],
            default='text',
            help='Output format (default: text)'
        )
        
        # Execution options
        parser.add_argument(
            '--interactive', '-i',
            action='store_true',
            help='Run in interactive mode for parameter tuning'
        )
        
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )
        
        parser.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress progress output'
        )
        
        # Parse arguments
        parsed = parser.parse_args(args)
        
        # Validate required arguments for non-interactive mode
        if not parsed.interactive:
            if parsed.deltav_budget is None:
                parser.error("--deltav-budget is required unless using --interactive mode")
            if parsed.timeframe is None:
                parser.error("--timeframe is required unless using --interactive mode")
        
        # Validate parameter ranges
        if parsed.deltav_budget is not None and parsed.deltav_budget <= 0:
            parser.error("Delta-v budget must be positive")
        
        if parsed.timeframe is not None and parsed.timeframe <= 0:
            parser.error("Timeframe must be positive")
        
        if not 0.0 <= parsed.mutation_rate <= 1.0:
            parser.error("Mutation rate must be between 0.0 and 1.0")
        
        if not 0.0 <= parsed.crossover_rate <= 1.0:
            parser.error("Crossover rate must be between 0.0 and 1.0")
        
        if parsed.population_size < 2:
            parser.error("Population size must be at least 2")
        
        if parsed.generations < 1:
            parser.error("Generations must be at least 1")
        
        return parsed
    
    def _load_satellites(self, tle_file: str) -> bool:
        """Load satellite data from TLE file."""
        try:
            if self.verbose:
                print(f"Loading satellite data from {tle_file}...")
            
            parser = TLEParser()
            self.satellites = parser.parse_tle_file(tle_file)
            
            if not self.satellites:
                print(f"Error: No satellites loaded from {tle_file}")
                return False
            
            print(f"Successfully loaded {len(self.satellites)} satellites")
            
            if self.verbose:
                self._print_satellite_summary()
            
            return True
            
        except FileNotFoundError:
            print(f"Error: TLE file '{tle_file}' not found")
            print("Please ensure the file exists in the current directory or provide the correct path")
            return False
        except Exception as e:
            print(f"Error loading satellites: {e}")
            return False
    
    def _print_satellite_summary(self):
        """Print summary of loaded satellites."""
        if not self.satellites:
            return
        
        altitudes = [sat.semi_major_axis - 6378.137 for sat in self.satellites]
        inclinations = [sat.inclination for sat in self.satellites]
        
        print("\nSatellite Summary:")
        print(f"  Count: {len(self.satellites)}")
        print(f"  Altitude range: {min(altitudes):.1f} - {max(altitudes):.1f} km")
        print(f"  Inclination range: {min(inclinations):.1f} - {max(inclinations):.1f}°")
        print(f"  Catalog numbers: {min(sat.catalog_number for sat in self.satellites)} - "
              f"{max(sat.catalog_number for sat in self.satellites)}")
    
    def _initialize_optimizer(self, args: argparse.Namespace):
        """Initialize genetic algorithm optimizer."""
        # Create GA configuration
        ga_config = GAConfig(
            population_size=args.population_size,
            max_generations=args.generations,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            elitism_count=args.elitism_count,
            tournament_size=args.tournament_size
        )
        
        # Initialize optimizer
        self.optimizer = GeneticRouteOptimizer(self.satellites, ga_config)
        
        # Set up progress callback if not quiet
        if not args.quiet:
            self.optimizer.progress_callback = self._progress_callback
        
        if self.verbose:
            print(f"\nGenetic Algorithm Configuration:")
            print(f"  Population size: {ga_config.population_size}")
            print(f"  Max generations: {ga_config.max_generations}")
            print(f"  Mutation rate: {ga_config.mutation_rate}")
            print(f"  Crossover rate: {ga_config.crossover_rate}")
            print(f"  Elitism count: {ga_config.elitism_count}")
            print(f"  Tournament size: {ga_config.tournament_size}")
    
    def _run_batch_mode(self, args: argparse.Namespace) -> int:
        """Run optimization in batch mode."""
        try:
            # Create route constraints
            constraints = self._create_constraints(args)
            
            if self.verbose:
                self._print_constraints(constraints)
            
            # Run optimization
            print("\nStarting genetic algorithm optimization...")
            start_time = time.time()
            
            result = self.optimizer.optimize_route(constraints)
            
            execution_time = time.time() - start_time
            
            # Display results
            self._display_results(result, execution_time)
            
            # Save results if requested
            if args.output or args.format != 'text':
                self._save_results(result, args.output, args.format)
            
            return 0 if result.success else 1
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return 1
    
    def _run_interactive_mode(self) -> int:
        """Run optimization in interactive mode."""
        print("\n" + "="*60)
        print("GENETIC ALGORITHM INTERACTIVE MODE")
        print("="*60)
        print("Configure optimization parameters interactively")
        
        try:
            while True:
                print("\nOptions:")
                print("1. Configure route constraints")
                print("2. Configure genetic algorithm parameters")
                print("3. Run optimization")
                print("4. View satellite information")
                print("5. Exit")
                
                choice = input("\nSelect option (1-5): ").strip()
                
                if choice == '1':
                    constraints = self._interactive_constraints()
                    if constraints:
                        self._print_constraints(constraints)
                elif choice == '2':
                    self._interactive_ga_config()
                elif choice == '3':
                    if hasattr(self, '_current_constraints'):
                        result = self._interactive_optimization()
                        if result:
                            self._display_results(result, result.execution_time)
                    else:
                        print("Please configure route constraints first (option 1)")
                elif choice == '4':
                    self._interactive_satellite_info()
                elif choice == '5':
                    print("Goodbye!")
                    return 0
                else:
                    print("Invalid choice. Please select 1-5.")
                    
        except KeyboardInterrupt:
            print("\nInteractive mode interrupted by user.")
            return 130
    
    def _create_constraints(self, args: argparse.Namespace) -> RouteConstraints:
        """Create route constraints from command line arguments."""
        forbidden_satellites = []
        if args.forbidden_satellites:
            try:
                forbidden_satellites = [
                    int(x.strip()) for x in args.forbidden_satellites.split(',')
                ]
            except ValueError:
                raise ValueError("Invalid forbidden satellites format. Use comma-separated integers.")
        
        return RouteConstraints(
            max_deltav_budget=args.deltav_budget,
            max_mission_duration=args.timeframe,
            start_satellite_id=args.start_satellite,
            end_satellite_id=args.end_satellite,
            min_hops=args.min_hops,
            max_hops=args.max_hops,
            forbidden_satellites=forbidden_satellites
        )
    
    def _interactive_constraints(self) -> Optional[RouteConstraints]:
        """Interactive configuration of route constraints."""
        print("\n" + "-"*40)
        print("ROUTE CONSTRAINTS CONFIGURATION")
        print("-"*40)
        
        try:
            # Delta-v budget
            deltav_budget = self._get_float_input(
                "Delta-v budget (km/s)", 
                min_val=0.1, max_val=20.0, default=5.0
            )
            
            # Mission timeframe
            print("\nMission timeframe options:")
            print("1. Enter in seconds")
            print("2. Enter in days")
            print("3. Enter in years")
            
            time_choice = input("Choose option (1-3, default: 3): ").strip() or "3"
            
            if time_choice == "1":
                timeframe = self._get_float_input(
                    "Mission timeframe (seconds)",
                    min_val=3600, max_val=315360000, default=63072000
                )
            elif time_choice == "2":
                days = self._get_float_input(
                    "Mission timeframe (days)",
                    min_val=1, max_val=3650, default=730
                )
                timeframe = days * 86400
            else:  # years
                years = self._get_float_input(
                    "Mission timeframe (years)",
                    min_val=0.1, max_val=10.0, default=2.0
                )
                timeframe = years * 365.25 * 86400
            
            # Optional constraints
            start_sat = self._get_optional_int_input("Starting satellite catalog number")
            end_sat = self._get_optional_int_input("Ending satellite catalog number")
            
            min_hops = self._get_int_input("Minimum hops", min_val=1, max_val=100, default=1)
            max_hops = self._get_int_input("Maximum hops", min_val=min_hops, max_val=100, default=50)
            
            # Forbidden satellites
            forbidden_input = input("Forbidden satellites (comma-separated catalog numbers, or Enter to skip): ").strip()
            forbidden_satellites = []
            if forbidden_input:
                try:
                    forbidden_satellites = [int(x.strip()) for x in forbidden_input.split(',')]
                except ValueError:
                    print("Warning: Invalid forbidden satellites format. Ignoring.")
            
            constraints = RouteConstraints(
                max_deltav_budget=deltav_budget,
                max_mission_duration=timeframe,
                start_satellite_id=start_sat,
                end_satellite_id=end_sat,
                min_hops=min_hops,
                max_hops=max_hops,
                forbidden_satellites=forbidden_satellites
            )
            
            self._current_constraints = constraints
            return constraints
            
        except KeyboardInterrupt:
            print("\nConstraint configuration cancelled.")
            return None
    
    def _interactive_ga_config(self):
        """Interactive configuration of genetic algorithm parameters."""
        print("\n" + "-"*40)
        print("GENETIC ALGORITHM CONFIGURATION")
        print("-"*40)
        
        try:
            current_config = self.optimizer.config
            
            population_size = self._get_int_input(
                "Population size", 
                min_val=10, max_val=1000, default=current_config.population_size
            )
            
            generations = self._get_int_input(
                "Maximum generations",
                min_val=10, max_val=10000, default=current_config.max_generations
            )
            
            mutation_rate = self._get_float_input(
                "Mutation rate",
                min_val=0.01, max_val=0.5, default=current_config.mutation_rate
            )
            
            crossover_rate = self._get_float_input(
                "Crossover rate",
                min_val=0.1, max_val=1.0, default=current_config.crossover_rate
            )
            
            elitism_count = self._get_int_input(
                "Elitism count",
                min_val=1, max_val=population_size//2, default=current_config.elitism_count
            )
            
            tournament_size = self._get_int_input(
                "Tournament size",
                min_val=2, max_val=population_size//4, default=current_config.tournament_size
            )
            
            # Update optimizer configuration
            new_config = GAConfig(
                population_size=population_size,
                max_generations=generations,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                elitism_count=elitism_count,
                tournament_size=tournament_size
            )
            
            self.optimizer = GeneticRouteOptimizer(self.satellites, new_config)
            self.optimizer.progress_callback = self._progress_callback
            
            print("Genetic algorithm configuration updated successfully!")
            
        except KeyboardInterrupt:
            print("\nGA configuration cancelled.")
    
    def _interactive_optimization(self) -> Optional[OptimizationResult]:
        """Run optimization interactively."""
        print("\n" + "-"*40)
        print("RUNNING OPTIMIZATION")
        print("-"*40)
        
        try:
            print("Starting genetic algorithm optimization...")
            print("Press Ctrl+C to interrupt if needed")
            
            start_time = time.time()
            result = self.optimizer.optimize_route(self._current_constraints)
            execution_time = time.time() - start_time
            
            return result
            
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user.")
            return None
    
    def _interactive_satellite_info(self):
        """Display satellite information interactively."""
        print("\n" + "-"*40)
        print("SATELLITE INFORMATION")
        print("-"*40)
        
        print("Options:")
        print("1. Show satellite summary")
        print("2. Search satellites by catalog number")
        print("3. Filter satellites by altitude")
        print("4. Filter satellites by inclination")
        
        choice = input("Select option (1-4): ").strip()
        
        if choice == '1':
            self._print_satellite_summary()
            self._print_satellite_list(self.satellites[:20])  # Show first 20
            if len(self.satellites) > 20:
                print(f"... and {len(self.satellites) - 20} more satellites")
        elif choice == '2':
            self._search_satellites_interactive()
        elif choice == '3':
            self._filter_satellites_by_altitude()
        elif choice == '4':
            self._filter_satellites_by_inclination()
        else:
            print("Invalid choice.")
    
    def _search_satellites_interactive(self):
        """Interactive satellite search."""
        try:
            catalog_num = int(input("Enter catalog number: ").strip())
            found = [sat for sat in self.satellites if sat.catalog_number == catalog_num]
            
            if found:
                print(f"\nFound satellite:")
                self._print_satellite_details(found[0])
            else:
                print(f"No satellite found with catalog number {catalog_num}")
                
        except ValueError:
            print("Invalid catalog number format.")
    
    def _filter_satellites_by_altitude(self):
        """Filter satellites by altitude range."""
        try:
            min_alt = self._get_optional_float_input("Minimum altitude (km)")
            max_alt = self._get_optional_float_input("Maximum altitude (km)")
            
            filtered = []
            for sat in self.satellites:
                altitude = sat.semi_major_axis - 6378.137
                if (min_alt is None or altitude >= min_alt) and \
                   (max_alt is None or altitude <= max_alt):
                    filtered.append(sat)
            
            print(f"\nFound {len(filtered)} satellites matching altitude criteria:")
            self._print_satellite_list(filtered[:10])  # Show first 10
            
        except ValueError:
            print("Invalid altitude values.")
    
    def _filter_satellites_by_inclination(self):
        """Filter satellites by inclination range."""
        try:
            min_inc = self._get_optional_float_input("Minimum inclination (degrees)")
            max_inc = self._get_optional_float_input("Maximum inclination (degrees)")
            
            filtered = []
            for sat in self.satellites:
                if (min_inc is None or sat.inclination >= min_inc) and \
                   (max_inc is None or sat.inclination <= max_inc):
                    filtered.append(sat)
            
            print(f"\nFound {len(filtered)} satellites matching inclination criteria:")
            self._print_satellite_list(filtered[:10])  # Show first 10
            
        except ValueError:
            print("Invalid inclination values.")
    
    def _print_satellite_list(self, satellites: List[SatelliteData]):
        """Print formatted list of satellites."""
        if not satellites:
            print("No satellites to display.")
            return
        
        print(f"\n{'Catalog':<8} {'Name':<20} {'Altitude (km)':<12} {'Inclination':<12}")
        print("-" * 60)
        
        for sat in satellites:
            altitude = sat.semi_major_axis - 6378.137
            print(f"{sat.catalog_number:<8} {sat.name:<20} {altitude:<12.1f} {sat.inclination:<12.1f}")
    
    def _print_satellite_details(self, satellite: SatelliteData):
        """Print detailed satellite information."""
        altitude = satellite.semi_major_axis - 6378.137
        
        print(f"\nSatellite Details:")
        print(f"  Catalog Number: {satellite.catalog_number}")
        print(f"  Name: {satellite.name}")
        print(f"  Epoch: {satellite.epoch.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Altitude: {altitude:.1f} km")
        print(f"  Semi-major Axis: {satellite.semi_major_axis:.1f} km")
        print(f"  Eccentricity: {satellite.eccentricity:.6f}")
        print(f"  Inclination: {satellite.inclination:.2f}°")
        print(f"  RAAN: {satellite.raan:.2f}°")
        print(f"  Arg. of Perigee: {satellite.arg_perigee:.2f}°")
        print(f"  Mean Anomaly: {satellite.mean_anomaly:.2f}°")
        print(f"  Mean Motion: {satellite.mean_motion:.8f} rev/day")
        print(f"  Orbital Period: {satellite.orbital_period:.1f} minutes")
    
    def _print_constraints(self, constraints: RouteConstraints):
        """Print route constraints summary."""
        print(f"\nRoute Constraints:")
        print(f"  Delta-v budget: {constraints.max_deltav_budget:.2f} km/s")
        print(f"  Mission duration: {constraints.max_mission_duration/86400:.1f} days "
              f"({constraints.max_mission_duration/31536000:.2f} years)")
        
        if constraints.start_satellite_id:
            print(f"  Start satellite: {constraints.start_satellite_id}")
        if constraints.end_satellite_id:
            print(f"  End satellite: {constraints.end_satellite_id}")
        
        print(f"  Hop range: {constraints.min_hops} - {constraints.max_hops}")
        
        if constraints.forbidden_satellites:
            print(f"  Forbidden satellites: {constraints.forbidden_satellites}")
    
    def _display_results(self, result: OptimizationResult, execution_time: float):
        """Display optimization results."""
        print("\n" + "="*80)
        print("GENETIC ALGORITHM OPTIMIZATION RESULTS")
        print("="*80)
        
        # Status and timing
        print(f"\nOptimization Status: {result.status.value.upper()}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Generations Completed: {result.optimization_stats.generations_completed}")
        
        if result.error_message:
            print(f"Error: {result.error_message}")
        
        # Best route results
        if result.best_route:
            route = result.best_route
            print(f"\nBest Route Found:")
            print(f"  Total Hops: {route.hop_count}")
            print(f"  Total Delta-v: {route.total_deltav:.3f} km/s")
            print(f"  Mission Duration: {route.mission_duration/86400:.1f} days")
            print(f"  Valid Route: {'Yes' if route.is_valid else 'No'}")
            
            if route.constraint_violations:
                print(f"  Constraint Violations:")
                for violation in route.constraint_violations:
                    print(f"    - {violation}")
            
            # Route details
            print(f"\nRoute Sequence:")
            print(f"{'Hop':<4} {'Satellite':<10} {'Name':<20} {'Departure Time':<15} {'Delta-v (m/s)':<12}")
            print("-" * 70)
            
            for i, sat_id in enumerate(route.satellite_sequence):
                # Find satellite info
                sat_info = next((s for s in self.satellites if s.catalog_number == sat_id), None)
                sat_name = sat_info.name if sat_info else f"SAT-{sat_id}"
                
                departure_time = route.departure_times[i] if i < len(route.departure_times) else 0
                departure_str = f"{departure_time/3600:.1f}h"
                
                # Calculate segment delta-v (simplified)
                segment_deltav = 0
                if i > 0 and hasattr(route, 'segment_deltavs'):
                    segment_deltav = route.segment_deltavs[i-1] * 1000  # Convert to m/s
                
                print(f"{i+1:<4} {sat_id:<10} {sat_name:<20} {departure_str:<15} {segment_deltav:<12.1f}")
        
        else:
            print("\nNo valid route found.")
        
        # Optimization statistics
        stats = result.optimization_stats
        print(f"\nOptimization Statistics:")
        print(f"  Best Fitness: {stats.best_fitness:.6f}")
        print(f"  Average Fitness: {stats.average_fitness:.6f}")
        print(f"  Population Diversity: {stats.population_diversity:.3f}")
        print(f"  Constraint Satisfaction Rate: {stats.constraint_satisfaction_rate:.1%}")
        
        if stats.convergence_generation:
            print(f"  Converged at Generation: {stats.convergence_generation}")
        
        # Convergence history summary
        if result.convergence_history:
            print(f"\nConvergence Summary:")
            history = result.convergence_history
            
            print(f"  Initial Best Fitness: {history[0].best_fitness:.6f}")
            print(f"  Final Best Fitness: {history[-1].best_fitness:.6f}")
            
            if len(history) > 1:
                improvement = history[-1].best_fitness - history[0].best_fitness
                print(f"  Total Improvement: {improvement:.6f}")
                print(f"  Average Improvement per Generation: {improvement/len(history):.6f}")
    
    def _save_results(self, result: OptimizationResult, output_file: Optional[str], format_type: str):
        """Save optimization results to file."""
        try:
            # Generate filename if not provided
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if format_type == 'json':
                    output_file = f"genetic_optimization_{timestamp}.json"
                elif format_type == 'csv':
                    output_file = f"genetic_optimization_{timestamp}.csv"
                else:
                    output_file = f"genetic_optimization_{timestamp}.txt"
            
            # Save based on format
            if format_type == 'json':
                self._save_json_results(result, output_file)
            elif format_type == 'csv':
                self._save_csv_results(result, output_file)
            else:
                self._save_text_results(result, output_file)
            
            print(f"\nResults saved to: {output_file}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def _save_json_results(self, result: OptimizationResult, filename: str):
        """Save results in JSON format."""
        data = {
            'optimization_status': result.status.value,
            'execution_time': result.execution_time,
            'success': result.success,
            'error_message': result.error_message,
            'best_route': {
                'satellite_sequence': result.best_route.satellite_sequence if result.best_route else [],
                'departure_times': result.best_route.departure_times if result.best_route else [],
                'total_deltav': result.best_route.total_deltav if result.best_route else 0,
                'hop_count': result.best_route.hop_count if result.best_route else 0,
                'mission_duration': result.best_route.mission_duration if result.best_route else 0,
                'is_valid': result.best_route.is_valid if result.best_route else False,
                'constraint_violations': result.best_route.constraint_violations if result.best_route else []
            },
            'optimization_stats': {
                'generations_completed': result.optimization_stats.generations_completed,
                'best_fitness': result.optimization_stats.best_fitness,
                'average_fitness': result.optimization_stats.average_fitness,
                'population_diversity': result.optimization_stats.population_diversity,
                'constraint_satisfaction_rate': result.optimization_stats.constraint_satisfaction_rate,
                'convergence_generation': result.optimization_stats.convergence_generation,
                'stagnant_generations': result.optimization_stats.stagnant_generations
            },
            'convergence_history': [
                {
                    'generation': gen.generation,
                    'best_fitness': gen.best_fitness,
                    'average_fitness': gen.average_fitness,
                    'worst_fitness': gen.worst_fitness,
                    'diversity_metric': gen.diversity_metric,
                    'valid_solutions_count': gen.valid_solutions_count,
                    'constraint_satisfaction_rate': gen.constraint_satisfaction_rate
                }
                for gen in result.convergence_history
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_text_results(self, result: OptimizationResult, filename: str):
        """Save results in text format."""
        with open(filename, 'w') as f:
            f.write("GENETIC ALGORITHM OPTIMIZATION RESULTS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Optimization Status: {result.status.value.upper()}\n")
            f.write(f"Execution Time: {result.execution_time:.2f} seconds\n")
            f.write(f"Generations Completed: {result.optimization_stats.generations_completed}\n")
            
            if result.error_message:
                f.write(f"Error: {result.error_message}\n")
            
            if result.best_route:
                route = result.best_route
                f.write(f"\nBest Route Found:\n")
                f.write(f"  Total Hops: {route.hop_count}\n")
                f.write(f"  Total Delta-v: {route.total_deltav:.3f} km/s\n")
                f.write(f"  Mission Duration: {route.mission_duration/86400:.1f} days\n")
                f.write(f"  Valid Route: {'Yes' if route.is_valid else 'No'}\n")
                
                if route.constraint_violations:
                    f.write(f"  Constraint Violations:\n")
                    for violation in route.constraint_violations:
                        f.write(f"    - {violation}\n")
                
                f.write(f"\nRoute Sequence:\n")
                for i, sat_id in enumerate(route.satellite_sequence):
                    departure_time = route.departure_times[i] if i < len(route.departure_times) else 0
                    f.write(f"  {i+1}. Satellite {sat_id} at {departure_time/3600:.1f}h\n")
            
            # Statistics
            stats = result.optimization_stats
            f.write(f"\nOptimization Statistics:\n")
            f.write(f"  Best Fitness: {stats.best_fitness:.6f}\n")
            f.write(f"  Average Fitness: {stats.average_fitness:.6f}\n")
            f.write(f"  Population Diversity: {stats.population_diversity:.3f}\n")
            f.write(f"  Constraint Satisfaction Rate: {stats.constraint_satisfaction_rate:.1%}\n")
    
    def _save_csv_results(self, result: OptimizationResult, filename: str):
        """Save convergence history in CSV format."""
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Generation', 'Best_Fitness', 'Average_Fitness', 'Worst_Fitness',
                'Diversity_Metric', 'Valid_Solutions_Count', 'Constraint_Satisfaction_Rate'
            ])
            
            # Data
            for gen in result.convergence_history:
                writer.writerow([
                    gen.generation, gen.best_fitness, gen.average_fitness, gen.worst_fitness,
                    gen.diversity_metric, gen.valid_solutions_count, gen.constraint_satisfaction_rate
                ])
    
    def _progress_callback(self, progress_data: Dict[str, Any]):
        """Progress callback for optimization updates."""
        generation = progress_data.get('generation', 0)
        best_fitness = progress_data.get('best_fitness', 0)
        
        if not self.verbose:
            # Simple progress indicator - show generation and best fitness
            print(f"Generation {generation:4d}: Best fitness = {best_fitness:.6f}")
        else:
            # Detailed progress
            avg_fitness = progress_data.get('average_fitness', 0)
            diversity = progress_data.get('diversity', 0)
            valid_count = progress_data.get('valid_solutions', 0)
            
            print(f"Generation {generation:4d}: Best={best_fitness:.6f}, Avg={avg_fitness:.6f}, "
                  f"Diversity={diversity:.3f}, Valid={valid_count}")
    
    def _get_float_input(self, prompt: str, min_val: float = None, max_val: float = None, 
                        default: float = None) -> float:
        """Get validated float input from user."""
        while True:
            try:
                default_str = f" (default: {default})" if default is not None else ""
                range_str = ""
                if min_val is not None and max_val is not None:
                    range_str = f" [{min_val}-{max_val}]"
                elif min_val is not None:
                    range_str = f" [>={min_val}]"
                elif max_val is not None:
                    range_str = f" [<={max_val}]"
                
                user_input = input(f"{prompt}{range_str}{default_str}: ").strip()
                
                if not user_input and default is not None:
                    return default
                
                value = float(user_input)
                
                if min_val is not None and value < min_val:
                    print(f"Value must be >= {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"Value must be <= {max_val}")
                    continue
                
                return value
                
            except ValueError:
                print("Please enter a valid number.")
    
    def _get_int_input(self, prompt: str, min_val: int = None, max_val: int = None, 
                      default: int = None) -> int:
        """Get validated integer input from user."""
        while True:
            try:
                default_str = f" (default: {default})" if default is not None else ""
                range_str = ""
                if min_val is not None and max_val is not None:
                    range_str = f" [{min_val}-{max_val}]"
                elif min_val is not None:
                    range_str = f" [>={min_val}]"
                elif max_val is not None:
                    range_str = f" [<={max_val}]"
                
                user_input = input(f"{prompt}{range_str}{default_str}: ").strip()
                
                if not user_input and default is not None:
                    return default
                
                value = int(user_input)
                
                if min_val is not None and value < min_val:
                    print(f"Value must be >= {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"Value must be <= {max_val}")
                    continue
                
                return value
                
            except ValueError:
                print("Please enter a valid integer.")
    
    def _get_optional_int_input(self, prompt: str) -> Optional[int]:
        """Get optional integer input from user."""
        try:
            user_input = input(f"{prompt} (or Enter to skip): ").strip()
            if not user_input:
                return None
            return int(user_input)
        except ValueError:
            print("Invalid integer format. Skipping.")
            return None
    
    def _get_optional_float_input(self, prompt: str) -> Optional[float]:
        """Get optional float input from user."""
        try:
            user_input = input(f"{prompt} (or Enter to skip): ").strip()
            if not user_input:
                return None
            return float(user_input)
        except ValueError:
            print("Invalid number format. Skipping.")
            return None


def main():
    """Main entry point for CLI."""
    cli = GeneticCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())