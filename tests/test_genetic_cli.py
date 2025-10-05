#!/usr/bin/env python3
"""
Test script for Genetic Algorithm CLI

Tests basic functionality of the genetic algorithm command-line interface.
"""

import sys
import os
import tempfile
import json
from unittest.mock import patch, MagicMock

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.genetic_cli import GeneticCLI
from src.genetic_algorithm import GAConfig, RouteConstraints, OptimizationResult, OptimizationStatus, OptimizationStats
from src.tle_parser import SatelliteData
from datetime import datetime


def create_test_satellites():
    """Create test satellite data."""
    satellites = []
    
    # Create a few test satellites with different characteristics
    for i in range(5):
        sat = SatelliteData(
            catalog_number=25544 + i,
            name=f"TEST-SAT-{i+1}",
            epoch=datetime.now(),
            mean_motion=15.5 + i * 0.1,
            eccentricity=0.001 + i * 0.0001,
            inclination=51.6 + i * 2.0,
            raan=100.0 + i * 10.0,
            arg_perigee=200.0 + i * 15.0,
            mean_anomaly=150.0 + i * 20.0,
            semi_major_axis=6800.0 + i * 50.0,
            orbital_period=90.0 + i * 2.0
        )
        satellites.append(sat)
    
    return satellites


def test_cli_argument_parsing():
    """Test CLI argument parsing."""
    print("Testing CLI argument parsing...")
    
    cli = GeneticCLI()
    
    # Test basic arguments
    args = cli._parse_arguments([
        '--deltav-budget', '5.0',
        '--timeframe', '63072000',
        '--population-size', '50',
        '--generations', '100',
        '--verbose'
    ])
    
    assert args.deltav_budget == 5.0
    assert args.timeframe == 63072000
    assert args.population_size == 50
    assert args.generations == 100
    assert args.verbose == True
    
    print("✓ Basic argument parsing works")
    
    # Test optional arguments
    args = cli._parse_arguments([
        '--deltav-budget', '3.0',
        '--timeframe', '31536000',
        '--start-satellite', '25544',
        '--end-satellite', '39084',
        '--forbidden-satellites', '12345,67890',
        '--output', 'test_results.json',
        '--format', 'json'
    ])
    
    assert args.start_satellite == 25544
    assert args.end_satellite == 39084
    assert args.forbidden_satellites == '12345,67890'
    assert args.output == 'test_results.json'
    assert args.format == 'json'
    
    print("✓ Optional argument parsing works")


def test_constraint_creation():
    """Test route constraint creation."""
    print("Testing constraint creation...")
    
    cli = GeneticCLI()
    
    # Mock arguments
    class MockArgs:
        deltav_budget = 5.0
        timeframe = 63072000
        start_satellite = 25544
        end_satellite = None
        min_hops = 1
        max_hops = 10
        forbidden_satellites = '12345,67890'
    
    args = MockArgs()
    constraints = cli._create_constraints(args)
    
    assert constraints.max_deltav_budget == 5.0
    assert constraints.max_mission_duration == 63072000
    assert constraints.start_satellite_id == 25544
    assert constraints.end_satellite_id is None
    assert constraints.min_hops == 1
    assert constraints.max_hops == 10
    assert constraints.forbidden_satellites == [12345, 67890]
    
    print("✓ Constraint creation works")


def test_satellite_loading():
    """Test satellite data loading."""
    print("Testing satellite loading...")
    
    cli = GeneticCLI()
    
    # Create temporary TLE file with valid data from actual file
    tle_content = """1 00005U 58002B   25276.57478752  .00000086  00000-0  13412-3 0  9995
2 00005  34.2488  26.3129 1841753  61.4980 315.8676 10.85924841415385
1 00011U 59001A   25276.55959630  .00001342  00000-0  71021-3 0  9995
2 00011  32.8685 336.1027 1447658  36.8902 332.2996 11.90057977504886"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(tle_content)
        temp_file = f.name
    
    try:
        success = cli._load_satellites(temp_file)
        assert success == True
        assert len(cli.satellites) == 2
        assert cli.satellites[0].catalog_number == 5
        assert cli.satellites[1].catalog_number == 11
        
        print("✓ Satellite loading works")
        
    finally:
        os.unlink(temp_file)


def test_result_formatting():
    """Test result display and saving."""
    print("Testing result formatting...")
    
    cli = GeneticCLI()
    cli.satellites = create_test_satellites()
    
    # Create mock optimization result
    from src.genetic_algorithm import RouteChromosome, GenerationStats
    
    best_route = RouteChromosome(
        satellite_sequence=[25544, 25545, 25546],
        departure_times=[0, 3600, 7200],
        total_deltav=2.5,
        is_valid=True,
        constraint_violations=[]
    )
    
    stats = OptimizationStats(
        generations_completed=100,
        best_fitness=0.8,
        average_fitness=0.6,
        population_diversity=0.4,
        constraint_satisfaction_rate=0.9
    )
    
    gen_stats = [
        GenerationStats(
            generation=0,
            best_fitness=0.5,
            average_fitness=0.3,
            worst_fitness=0.1,
            diversity_metric=0.8,
            valid_solutions_count=80,
            constraint_satisfaction_rate=0.8
        ),
        GenerationStats(
            generation=99,
            best_fitness=0.8,
            average_fitness=0.6,
            worst_fitness=0.4,
            diversity_metric=0.4,
            valid_solutions_count=90,
            constraint_satisfaction_rate=0.9
        )
    ]
    
    result = OptimizationResult(
        best_route=best_route,
        optimization_stats=stats,
        convergence_history=gen_stats,
        execution_time=45.5,
        status=OptimizationStatus.CONVERGED,
        error_message=None
    )
    
    # Test display (capture output)
    with patch('builtins.print') as mock_print:
        cli._display_results(result, 45.5)
        
        # Check that print was called (basic test)
        assert mock_print.called
        print("✓ Result display works")
    
    # Test JSON saving
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        cli._save_json_results(result, temp_file)
        
        # Verify JSON file was created and contains expected data
        with open(temp_file, 'r') as f:
            data = json.load(f)
        
        assert data['optimization_status'] == 'converged'
        assert data['execution_time'] == 45.5
        assert data['best_route']['total_deltav'] == 2.5
        assert len(data['best_route']['satellite_sequence']) == 3
        assert len(data['convergence_history']) == 2
        
        print("✓ JSON result saving works")
        
    finally:
        os.unlink(temp_file)


def test_input_validation():
    """Test input validation functions."""
    print("Testing input validation...")
    
    cli = GeneticCLI()
    
    # Test float input validation with mock input
    with patch('builtins.input', return_value='5.5'):
        result = cli._get_float_input("Test", min_val=1.0, max_val=10.0, default=3.0)
        assert result == 5.5
    
    # Test default value
    with patch('builtins.input', return_value=''):
        result = cli._get_float_input("Test", default=3.0)
        assert result == 3.0
    
    # Test integer input validation
    with patch('builtins.input', return_value='42'):
        result = cli._get_int_input("Test", min_val=1, max_val=100, default=10)
        assert result == 42
    
    # Test optional input
    with patch('builtins.input', return_value=''):
        result = cli._get_optional_int_input("Test")
        assert result is None
    
    with patch('builtins.input', return_value='123'):
        result = cli._get_optional_int_input("Test")
        assert result == 123
    
    print("✓ Input validation works")


def run_all_tests():
    """Run all CLI tests."""
    print("Running Genetic Algorithm CLI Tests")
    print("=" * 50)
    
    try:
        test_cli_argument_parsing()
        test_constraint_creation()
        test_satellite_loading()
        test_result_formatting()
        test_input_validation()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)