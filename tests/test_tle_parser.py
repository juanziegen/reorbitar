"""
Unit tests for TLE Parser Module
"""

import unittest
import tempfile
import os
from datetime import datetime
from src.tle_parser import TLEParser, SatelliteData


class TestTLEParser(unittest.TestCase):
    """Test cases for TLE parsing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = TLEParser()
        
        # Known good TLE data for testing
        self.valid_tle_line1 = "1 00005U 58002B   25276.57478752  .00000086  00000-0  13412-3 0  9995"
        self.valid_tle_line2 = "2 00005  34.2488  26.3129 1841753  61.4980 315.8676 10.85924841415385"
        
        # Invalid TLE data for testing
        self.invalid_checksum_line1 = "1 00005U 58002B   25276.57478752  .00000086  00000-0  13412-3 0  9999"  # Wrong checksum
        # Create a line with valid checksum but invalid catalog number
        invalid_line_base = "1 ABCDE 58002B   25276.57478752  .00000086  00000-0  13412-3 0  "
        # Calculate checksum for the base line
        checksum = 0
        for char in invalid_line_base:
            if char.isdigit():
                checksum += int(char)
            elif char == '-':
                checksum += 1
        # Pad to make it 69 characters total
        padding_needed = 68 - len(invalid_line_base)
        padded_line = invalid_line_base + '0' * padding_needed
        self.invalid_format_line1 = padded_line + str(checksum % 10)
        self.short_line = "1 00005U 58002B   25276.57478752"  # Too short
    
    def test_validate_tle_checksum_valid(self):
        """Test checksum validation with valid TLE lines."""
        self.assertTrue(self.parser.validate_tle_checksum(self.valid_tle_line1))
        self.assertTrue(self.parser.validate_tle_checksum(self.valid_tle_line2))
    
    def test_validate_tle_checksum_invalid(self):
        """Test checksum validation with invalid TLE lines."""
        self.assertFalse(self.parser.validate_tle_checksum(self.invalid_checksum_line1))
        self.assertFalse(self.parser.validate_tle_checksum(self.short_line))
    
    def test_parse_tle_entry_valid(self):
        """Test parsing of valid TLE entry."""
        satellite = self.parser.parse_tle_entry(self.valid_tle_line1, self.valid_tle_line2)
        
        self.assertIsInstance(satellite, SatelliteData)
        self.assertEqual(satellite.catalog_number, 5)
        self.assertEqual(satellite.name, "SATELLITE-00005")
        self.assertAlmostEqual(satellite.inclination, 34.2488, places=4)
        self.assertAlmostEqual(satellite.eccentricity, 0.1841753, places=7)
        self.assertAlmostEqual(satellite.mean_motion, 10.85924841, places=8)
        self.assertAlmostEqual(satellite.raan, 26.3129, places=4)
        self.assertAlmostEqual(satellite.arg_perigee, 61.4980, places=4)
        self.assertAlmostEqual(satellite.mean_anomaly, 315.8676, places=4)
        
        # Check derived parameters
        self.assertGreater(satellite.semi_major_axis, 0)
        self.assertGreater(satellite.orbital_period, 0)
        
        # Check epoch parsing
        self.assertIsInstance(satellite.epoch, datetime)
        self.assertEqual(satellite.epoch.year, 2025)
    
    def test_parse_tle_entry_invalid_checksum(self):
        """Test parsing with invalid checksum."""
        with self.assertRaises(ValueError) as context:
            self.parser.parse_tle_entry(self.invalid_checksum_line1, self.valid_tle_line2)
        self.assertIn("checksum", str(context.exception))
    
    def test_parse_tle_entry_mismatched_catalog(self):
        """Test parsing with mismatched catalog numbers."""
        line1_cat5 = self.valid_tle_line1
        line2_cat6 = "2 00006  34.2488  26.3129 1841753  61.4980 315.8676 10.85924841415386"  # Fixed checksum
        
        with self.assertRaises(ValueError) as context:
            self.parser.parse_tle_entry(line1_cat5, line2_cat6)
        self.assertIn("Catalog numbers don't match", str(context.exception))
    
    def test_parse_tle_entry_short_lines(self):
        """Test parsing with lines that are too short."""
        with self.assertRaises(ValueError) as context:
            self.parser.parse_tle_entry(self.short_line, self.valid_tle_line2)
        self.assertIn("too short", str(context.exception))
    
    def test_parse_tle_entry_invalid_format(self):
        """Test parsing with invalid numeric format."""
        with self.assertRaises(ValueError) as context:
            self.parser.parse_tle_entry(self.invalid_format_line1, self.valid_tle_line2)
        self.assertIn("Invalid catalog number format", str(context.exception))
    
    def test_parse_tle_file_valid(self):
        """Test parsing a valid TLE file."""
        # Create temporary file with valid TLE data
        tle_content = f"{self.valid_tle_line1}\n{self.valid_tle_line2}\n"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(tle_content)
            temp_filename = f.name
        
        try:
            satellites = self.parser.parse_tle_file(temp_filename)
            self.assertEqual(len(satellites), 1)
            self.assertEqual(satellites[0].catalog_number, 5)
        finally:
            os.unlink(temp_filename)
    
    def test_parse_tle_file_mixed_valid_invalid(self):
        """Test parsing file with mix of valid and invalid entries."""
        # Create file with valid and invalid TLE data
        tle_content = f"""{self.valid_tle_line1}
{self.valid_tle_line2}
{self.invalid_checksum_line1}
{self.valid_tle_line2}
{self.valid_tle_line1}
{self.valid_tle_line2}
"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(tle_content)
            temp_filename = f.name
        
        try:
            # Should parse 2 valid entries and skip 1 invalid
            satellites = self.parser.parse_tle_file(temp_filename)
            self.assertEqual(len(satellites), 2)
        finally:
            os.unlink(temp_filename)
    
    def test_parse_tle_file_nonexistent(self):
        """Test parsing non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.parser.parse_tle_file("nonexistent_file.txt")
    
    def test_epoch_from_year_day(self):
        """Test epoch conversion from year and day."""
        # Test known epoch conversion
        epoch = self.parser._epoch_from_year_day(2025, 276.57478752)
        
        self.assertEqual(epoch.year, 2025)
        self.assertEqual(epoch.month, 10)  # Day 276 should be in October
        self.assertEqual(epoch.day, 3)     # Day 276 should be October 3rd
    
    def test_calculate_semi_major_axis(self):
        """Test semi-major axis calculation."""
        # Test with known mean motion
        mean_motion = 15.0  # revolutions per day (typical for LEO)
        semi_major_axis = self.parser._calculate_semi_major_axis(mean_motion)
        
        # Should be reasonable for LEO (around 6600-7000 km)
        self.assertGreater(semi_major_axis, 6000)
        self.assertLess(semi_major_axis, 8000)
    
    def test_calculate_orbital_period(self):
        """Test orbital period calculation."""
        # Test with typical LEO semi-major axis
        semi_major_axis = 6800  # km
        period = self.parser._calculate_orbital_period(semi_major_axis)
        
        # Should be around 90-100 minutes for LEO
        self.assertGreater(period, 80)
        self.assertLess(period, 120)


if __name__ == '__main__':
    unittest.main()