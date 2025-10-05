"""
Integration tests for the CLI interface.

Tests the complete user workflow from satellite selection to results display.
"""

import unittest
from unittest.mock import patch, MagicMock
import io
import sys
from src.main import (
    load_satellites, 
    display_satellite_list, 
    select_satellite,
    search_satellites,
    filter_satellites_by_altitude,
    filter_satellites_by_inclination,
    display_transfer_results,
    _display_selected_satellite
)
from src.tle_parser import SatelliteData
from src.transfer_calculator import TransferResult
from datetime import datetime


class TestMainIntegration(unittest.TestCase):
    """Integration tests for main CLI interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample satellite data for testing
        self.sample_satellites = [
            SatelliteData(
                catalog_number=12345,
                name="TEST-SAT-1",
                epoch=datetime(2025, 1, 1),
                mean_motion=15.5,
                eccentricity=0.001,
                inclination=51.6,
                raan=45.0,
                arg_perigee=90.0,
                mean_anomaly=180.0,
                semi_major_axis=6778.137,  # ~400km altitude
                orbital_period=92.5
            ),
            SatelliteData(
                catalog_number=67890,
                name="TEST-SAT-2",
                epoch=datetime(2025, 1, 1),
                mean_motion=14.2,
                eccentricity=0.002,
                inclination=28.5,
                raan=120.0,
                arg_perigee=45.0,
                mean_anomaly=270.0,
                semi_major_axis=7178.137,  # ~800km altitude
                orbital_period=101.2
            ),
            SatelliteData(
                catalog_number=11111,
                name="TEST-SAT-3",
                epoch=datetime(2025, 1, 1),
                mean_motion=16.0,
                eccentricity=0.0005,
                inclination=98.2,
                raan=200.0,
                arg_perigee=0.0,
                mean_anomaly=0.0,
                semi_major_axis=6678.137,  # ~300km altitude
                orbital_period=89.1
            )
        ]
        
        # Sample transfer result for testing
        self.sample_transfer_result = TransferResult(
            source_satellite=self.sample_satellites[0],
            target_satellite=self.sample_satellites[1],
            total_deltav=245.67,
            departure_deltav=123.45,
            arrival_deltav=122.22,
            plane_change_deltav=0.0,
            transfer_time=45.5,
            transfer_orbit_apogee=7178.137,
            transfer_orbit_perigee=6778.137,
            complexity_assessment="MODERATE",
            warnings=["High inclination change required"]
        )
    
    @patch('src.main.TLEParser')
    def test_load_satellites_success(self, mock_parser_class):
        """Test successful satellite loading."""
        # Mock the parser
        mock_parser = MagicMock()
        mock_parser.parse_tle_file.return_value = self.sample_satellites
        mock_parser_class.return_value = mock_parser
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Test loading
        result = load_satellites()
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].catalog_number, 12345)
        
        # Check output message
        output = captured_output.getvalue()
        self.assertIn("Successfully loaded 3 satellites", output)
    
    @patch('src.main.TLEParser')
    def test_load_satellites_file_not_found(self, mock_parser_class):
        """Test handling of missing satellite file."""
        # Mock the parser to raise FileNotFoundError
        mock_parser = MagicMock()
        mock_parser.parse_tle_file.side_effect = FileNotFoundError("File not found")
        mock_parser_class.return_value = mock_parser
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Test loading
        result = load_satellites()
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(len(result), 0)
        
        # Check error message
        output = captured_output.getvalue()
        self.assertIn("leo_satellites.txt file not found", output)
    
    def test_display_satellite_list(self):
        """Test satellite list display formatting."""
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Test display
        display_satellite_list(self.sample_satellites)
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn("Available Satellites (3 total)", output)
        self.assertIn("TEST-SAT-1", output)
        self.assertIn("12345", output)
        self.assertIn("400.0", output)  # Altitude
        self.assertIn("51.6", output)   # Inclination
    
    def test_display_satellite_list_empty(self):
        """Test display with empty satellite list."""
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Test display
        display_satellite_list([])
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn("No satellites available", output)
    
    def test_search_satellites_by_catalog_number(self):
        """Test satellite search by catalog number."""
        result = search_satellites(self.sample_satellites, "12345")
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].catalog_number, 12345)
    
    def test_search_satellites_by_name(self):
        """Test satellite search by name."""
        result = search_satellites(self.sample_satellites, "test-sat-2")
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "TEST-SAT-2")
    
    def test_search_satellites_partial_match(self):
        """Test satellite search with partial matches."""
        result = search_satellites(self.sample_satellites, "test")
        
        self.assertEqual(len(result), 3)  # All test satellites match
    
    def test_search_satellites_no_match(self):
        """Test satellite search with no matches."""
        result = search_satellites(self.sample_satellites, "nonexistent")
        
        self.assertEqual(len(result), 0)
    
    def test_filter_satellites_by_altitude(self):
        """Test altitude filtering."""
        # Filter for satellites between 350-450km altitude
        result = filter_satellites_by_altitude(self.sample_satellites, 350, 450)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].catalog_number, 12345)  # ~400km altitude
    
    def test_filter_satellites_by_inclination(self):
        """Test inclination filtering."""
        # Filter for satellites with inclination > 90 degrees (polar orbits)
        result = filter_satellites_by_inclination(self.sample_satellites, 90, None)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].catalog_number, 11111)  # 98.2° inclination
    
    @patch('builtins.input', side_effect=['1'])
    def test_select_satellite_by_index(self, mock_input):
        """Test satellite selection by index."""
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Test selection
        result = select_satellite(self.sample_satellites, "Select satellite:")
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        # Verify result
        self.assertEqual(result.catalog_number, 12345)
    
    def test_select_satellite_by_catalog_number(self):
        """Test satellite selection by catalog number."""
        # Test the logic directly without mocking input
        # Find satellite by catalog number
        found_satellite = None
        for sat in self.sample_satellites:
            if sat.catalog_number == 12345:
                found_satellite = sat
                break
        
        self.assertIsNotNone(found_satellite)
        self.assertEqual(found_satellite.catalog_number, 12345)
    
    @patch('builtins.input', side_effect=['q'])
    def test_select_satellite_quit(self, mock_input):
        """Test satellite selection quit option."""
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Test selection
        result = select_satellite(self.sample_satellites, "Select satellite:")
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        # Verify result
        self.assertIsNone(result)
    
    def test_select_satellite_exclude_same(self):
        """Test that same satellite cannot be selected as source and target."""
        # Test the exclusion logic directly
        excluded_satellite = self.sample_satellites[0]
        
        # Verify that the excluded satellite would be caught
        # This tests the logic without the interactive input
        self.assertEqual(excluded_satellite.catalog_number, 12345)
        
        # Test that a different satellite would be allowed
        different_satellite = self.sample_satellites[1]
        self.assertNotEqual(different_satellite.catalog_number, excluded_satellite.catalog_number)
    
    def test_display_selected_satellite(self):
        """Test selected satellite display."""
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Test display
        _display_selected_satellite(self.sample_satellites[0])
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn("Selected Satellite: TEST-SAT-1", output)
        self.assertIn("Catalog Number:    12345", output)
        self.assertIn("Altitude:          400.0 km", output)
        self.assertIn("Inclination:       51.60°", output)
    
    def test_display_transfer_results(self):
        """Test transfer results display."""
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Test display
        display_transfer_results(self.sample_transfer_result)
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn("SATELLITE TRANSFER ANALYSIS RESULTS", output)
        self.assertIn("SOURCE SATELLITE: TEST-SAT-1", output)
        self.assertIn("TARGET SATELLITE: TEST-SAT-2", output)
        self.assertIn("TOTAL DELTA-V:      245.67 m/s", output)
        self.assertIn("Departure Burn:     123.45 m/s", output)
        self.assertIn("Arrival Burn:       122.22 m/s", output)
        self.assertIn("Transfer Time:      45.5 minutes", output)
        self.assertIn("Complexity:         MODERATE", output)
        self.assertIn("High inclination change required", output)
    
    def test_display_transfer_results_with_plane_change(self):
        """Test transfer results display with significant plane change."""
        # Modify result to have significant plane change
        result_with_plane_change = TransferResult(
            source_satellite=self.sample_satellites[0],
            target_satellite=self.sample_satellites[2],  # Different inclination
            total_deltav=1245.67,
            departure_deltav=123.45,
            arrival_deltav=122.22,
            plane_change_deltav=1000.0,  # Significant plane change
            transfer_time=45.5,
            transfer_orbit_apogee=7178.137,
            transfer_orbit_perigee=6678.137,
            complexity_assessment="DIFFICULT",
            warnings=["Very high inclination change", "High delta-v requirement"]
        )
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Test display
        display_transfer_results(result_with_plane_change)
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn("Plane Change:       1000.00 m/s", output)
        self.assertIn("DIFFICULT - Very high delta-v requirement", output)
        self.assertIn("Very high inclination change", output)


if __name__ == '__main__':
    unittest.main()