"""
Main Interface Module

Command-line interface for the satellite delta-v calculator.
Provides user interaction and result display functionality.
"""

import math
from typing import List, Optional, Dict
from .tle_parser import SatelliteData, TLEParser
from .transfer_calculator import TransferResult, calculate_transfer_deltav
from .satellite_data_validator import SatelliteDataValidator


def check_genetic_algorithm_system_requirements() -> Dict[str, any]:
    """Check system requirements for genetic algorithm operations."""
    import sys
    import os
    
    requirements = {
        'compatible': True,
        'warnings': [],
        'errors': [],
        'recommendations': []
    }
    
    try:
        # Check Python version
        if sys.version_info < (3, 7):
            requirements['errors'].append(f"Python {sys.version_info.major}.{sys.version_info.minor} detected. Python 3.7+ required.")
            requirements['compatible'] = False
        
        # Try to check system resources with psutil if available
        try:
            import psutil
            
            # Check available memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < 1.0:
                requirements['errors'].append(f"Low memory: {available_gb:.1f} GB available. At least 1 GB recommended.")
                requirements['compatible'] = False
            elif available_gb < 2.0:
                requirements['warnings'].append(f"Limited memory: {available_gb:.1f} GB available. 2+ GB recommended for large constellations.")
            
            # Check CPU cores
            cpu_count = psutil.cpu_count()
            if cpu_count < 2:
                requirements['warnings'].append(f"Single CPU core detected. Multi-core systems recommended for better performance.")
            
            # Check disk space for results
            disk_usage = psutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)
            
            if free_gb < 0.1:
                requirements['warnings'].append(f"Low disk space: {free_gb:.1f} GB free. May not be able to save results.")
            
            # Add recommendations based on system specs
            if requirements['compatible']:
                if available_gb >= 4.0 and cpu_count >= 4:
                    requirements['recommendations'].append("System well-suited for genetic algorithm optimization")
                else:
                    requirements['recommendations'].append("System meets minimum requirements")
            
            if cpu_count >= 4:
                requirements['recommendations'].append("Multi-core system detected - good for parallel processing")
            
            if available_gb >= 8.0:
                requirements['recommendations'].append("Ample memory available for large constellation optimization")
                
        except ImportError:
            requirements['warnings'].append("psutil not available - cannot check system resources")
            requirements['recommendations'].append("Install psutil for detailed system monitoring")
        
        # Check for required modules (basic ones that should be available)
        required_modules = ['datetime', 'json', 'time', 'random', 'math']
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            requirements['errors'].append(f"Missing required modules: {', '.join(missing_modules)}")
            requirements['compatible'] = False
        
        # Check for genetic algorithm specific modules
        try:
            from src.genetic_cli import GeneticCLI
            requirements['recommendations'].append("Genetic algorithm modules available")
        except ImportError as e:
            requirements['errors'].append(f"Genetic algorithm modules not available: {e}")
            requirements['compatible'] = False
        
        # Basic compatibility check passed
        if requirements['compatible'] and not requirements['recommendations']:
            requirements['recommendations'].append("System compatible with genetic algorithm optimization")
        
    except Exception as e:
        requirements['errors'].append(f"System check failed: {e}")
        requirements['compatible'] = False
    
    return requirements


def load_satellites(validate_data: bool = True) -> List[SatelliteData]:
    """Load and parse all satellites from the data file with optional validation."""
    parser = TLEParser()
    
    # Try multiple possible file locations
    possible_files = ["leo_satellites.txt", "data/leo_satellites.txt", "./leo_satellites.txt"]
    
    for filepath in possible_files:
        try:
            satellites = parser.parse_tle_file(filepath)
            print(f"Successfully loaded {len(satellites)} satellites from {filepath}")
            
            if len(satellites) == 0:
                print("Warning: No valid satellites found in file")
                return satellites
            
            # Validate satellite data if requested
            if validate_data:
                print("Validating satellite data quality...")
                validator = SatelliteDataValidator()
                validation_result = validator.validate_satellite_dataset(satellites)
                
                # Print validation summary
                if validation_result['valid']:
                    print(f"✅ Satellite data validation passed")
                    print(f"   Valid satellites: {validation_result['valid_satellites']}/{validation_result['total_satellites']}")
                    
                    # Show genetic algorithm readiness
                    if validation_result.get('genetic_algorithm_ready'):
                        print(f"   ✅ Ready for genetic algorithm optimization")
                    else:
                        print(f"   ⚠️  Limited genetic algorithm compatibility")
                    
                    # Show warnings if any
                    if validation_result['warnings']:
                        warning_count = len(validation_result['warnings'])
                        print(f"   ⚠️  {warning_count} warnings found (use verbose mode for details)")
                else:
                    print(f"⚠️  Satellite data validation issues detected")
                    print(f"   Valid satellites: {validation_result['valid_satellites']}/{validation_result['total_satellites']}")
                    
                    # Show critical errors
                    if validation_result['errors']:
                        error_count = len(validation_result['errors'])
                        print(f"   ❌ {error_count} errors found")
                        
                        # Show first few errors
                        for error in validation_result['errors'][:3]:
                            print(f"      - {error}")
                        if len(validation_result['errors']) > 3:
                            print(f"      ... and {len(validation_result['errors']) - 3} more errors")
                
                # Return filtered valid satellites if validation failed
                if not validation_result['valid'] and validation_result['valid_satellites'] > 0:
                    print(f"Using {validation_result['valid_satellites']} valid satellites for calculations")
                    return validation_result['valid_satellite_list']
                elif not validation_result['valid']:
                    print("Error: No valid satellites found after validation")
                    return []
            
            return satellites
            
        except FileNotFoundError:
            continue
        except PermissionError as e:
            print(f"Error: Permission denied accessing {filepath}: {e}")
            return []
        except ValueError as e:
            print(f"Error: Invalid data in {filepath}: {e}")
            return []
        except Exception as e:
            print(f"Error loading satellites from {filepath}: {e}")
            return []
    
    # If we get here, no file was found
    print("Error: leo_satellites.txt file not found in any of the expected locations:")
    for filepath in possible_files:
        print(f"  - {filepath}")
    print("Please ensure the TLE data file exists and is accessible.")
    return []


def display_satellite_list(satellites: List[SatelliteData]) -> None:
    """Display list of available satellites."""
    if not satellites:
        print("No satellites available.")
        return
    
    print(f"\nAvailable Satellites ({len(satellites)} total):")
    print("-" * 80)
    print(f"{'Index':<6} {'Catalog':<8} {'Name':<18} {'Altitude (km)':<12} {'Inclination':<12} {'Period (min)':<12}")
    print("-" * 80)
    
    for i, sat in enumerate(satellites, 1):
        # Calculate approximate altitude (semi-major axis - Earth radius)
        altitude = sat.semi_major_axis - 6378.137  # Earth radius in km
        print(f"{i:<6} {sat.catalog_number:<8} {sat.name:<18} {altitude:<12.1f} {sat.inclination:<12.1f} {sat.orbital_period:<12.1f}")
    
    print("-" * 80)


def search_satellites(satellites: List[SatelliteData], search_term: str) -> List[SatelliteData]:
    """Search satellites by catalog number or name."""
    if not search_term:
        return satellites
    
    search_term = search_term.lower().strip()
    filtered = []
    
    for sat in satellites:
        # Search by catalog number
        if search_term in str(sat.catalog_number):
            filtered.append(sat)
        # Search by name
        elif search_term in sat.name.lower():
            filtered.append(sat)
    
    return filtered


def filter_satellites_by_altitude(satellites: List[SatelliteData], min_alt: Optional[float] = None, max_alt: Optional[float] = None) -> List[SatelliteData]:
    """Filter satellites by altitude range."""
    filtered = []
    
    for sat in satellites:
        altitude = sat.semi_major_axis - 6378.137  # Earth radius in km
        
        if min_alt is not None and altitude < min_alt:
            continue
        if max_alt is not None and altitude > max_alt:
            continue
            
        filtered.append(sat)
    
    return filtered


def filter_satellites_by_inclination(satellites: List[SatelliteData], min_inc: Optional[float] = None, max_inc: Optional[float] = None) -> List[SatelliteData]:
    """Filter satellites by inclination range."""
    filtered = []
    
    for sat in satellites:
        if min_inc is not None and sat.inclination < min_inc:
            continue
        if max_inc is not None and sat.inclination > max_inc:
            continue
            
        filtered.append(sat)
    
    return filtered


def select_satellite(satellites: List[SatelliteData], prompt: str, exclude_satellite: Optional[SatelliteData] = None) -> Optional[SatelliteData]:
    """Interactive satellite selection with validation."""
    if not satellites:
        print("No satellites available for selection.")
        return None
    
    max_attempts = 5
    attempt_count = 0
    
    while attempt_count < max_attempts:
        try:
            print(f"\n{prompt}")
            print("Options:")
            print("  - Enter satellite index (1-{})".format(len(satellites)))
            print("  - Enter catalog number")
            print("  - Enter 's' to search satellites")
            print("  - Enter 'f' to filter satellites")
            print("  - Enter 'l' to list all satellites")
            print("  - Enter 'q' to quit")
            
            choice = input("\nYour choice: ").strip()
            
            # Reset attempt counter on valid input
            if choice:
                attempt_count = 0
            else:
                attempt_count += 1
                if attempt_count >= max_attempts:
                    print("Too many empty inputs. Returning to main menu.")
                    return None
                print("Please enter a valid choice.")
                continue
            
            if choice.lower() == 'q':
                return None
            elif choice.lower() == 'l':
                display_satellite_list(satellites)
                continue
            elif choice.lower() == 's':
                search_term = input("Enter search term (catalog number or name): ").strip()
                filtered = search_satellites(satellites, search_term)
                if filtered:
                    print(f"\nFound {len(filtered)} matching satellites:")
                    display_satellite_list(filtered)
                    # Allow selection from filtered results
                    selected = _select_from_filtered(filtered, exclude_satellite)
                    if selected:
                        return selected
                else:
                    print("No satellites found matching your search.")
                continue
            elif choice.lower() == 'f':
                filtered = _interactive_filter(satellites)
                if filtered:
                    print(f"\nFiltered to {len(filtered)} satellites:")
                    display_satellite_list(filtered)
                    # Allow selection from filtered results
                    selected = _select_from_filtered(filtered, exclude_satellite)
                    if selected:
                        return selected
                else:
                    print("No satellites match the filter criteria.")
                continue
            
            # Try to parse as index
            try:
                index = int(choice)
                if 1 <= index <= len(satellites):
                    selected = satellites[index - 1]
                    if exclude_satellite and selected.catalog_number == exclude_satellite.catalog_number:
                        print("Error: Cannot select the same satellite as both source and target.")
                        attempt_count += 1
                        continue
                    _display_selected_satellite(selected)
                    return selected
                else:
                    print(f"Error: Index must be between 1 and {len(satellites)}")
                    attempt_count += 1
                    continue
            except ValueError:
                pass
            
            # Try to parse as catalog number
            try:
                catalog_num = int(choice)
                if catalog_num <= 0:
                    print("Error: Catalog number must be positive.")
                    attempt_count += 1
                    continue
                    
                for sat in satellites:
                    if sat.catalog_number == catalog_num:
                        if exclude_satellite and sat.catalog_number == exclude_satellite.catalog_number:
                            print("Error: Cannot select the same satellite as both source and target.")
                            attempt_count += 1
                            break
                        _display_selected_satellite(sat)
                        return sat
                print(f"Error: No satellite found with catalog number {catalog_num}")
                attempt_count += 1
                continue
            except ValueError:
                print("Error: Invalid input. Please enter a valid option.")
                attempt_count += 1
                continue
                
        except KeyboardInterrupt:
            print("\nSelection cancelled by user.")
            return None
        except Exception as e:
            print(f"Unexpected error during selection: {e}")
            attempt_count += 1
            continue
    
    print("Too many invalid attempts. Returning to main menu.")
    return None


def _select_from_filtered(filtered_satellites: List[SatelliteData], exclude_satellite: Optional[SatelliteData] = None) -> Optional[SatelliteData]:
    """Helper function to select from filtered satellite list."""
    while True:
        choice = input(f"\nEnter index (1-{len(filtered_satellites)}) or 'b' to go back: ").strip()
        
        if choice.lower() == 'b':
            return None
        
        try:
            index = int(choice)
            if 1 <= index <= len(filtered_satellites):
                selected = filtered_satellites[index - 1]
                if exclude_satellite and selected.catalog_number == exclude_satellite.catalog_number:
                    print("Error: Cannot select the same satellite as both source and target.")
                    continue
                return selected
            else:
                print(f"Error: Index must be between 1 and {len(filtered_satellites)}")
        except ValueError:
            print("Error: Invalid input. Please enter a valid index.")


def _interactive_filter(satellites: List[SatelliteData]) -> List[SatelliteData]:
    """Interactive filtering of satellites."""
    print("\nFilter Options:")
    print("1. Filter by altitude range")
    print("2. Filter by inclination range")
    print("3. No filter (return all)")
    
    choice = input("Choose filter type (1-3): ").strip()
    
    if choice == '1':
        try:
            min_alt_input = input("Minimum altitude (km, press Enter to skip): ").strip()
            max_alt_input = input("Maximum altitude (km, press Enter to skip): ").strip()
            
            min_alt = float(min_alt_input) if min_alt_input else None
            max_alt = float(max_alt_input) if max_alt_input else None
            
            return filter_satellites_by_altitude(satellites, min_alt, max_alt)
        except ValueError:
            print("Error: Invalid altitude values.")
            return satellites
    elif choice == '2':
        try:
            min_inc_input = input("Minimum inclination (degrees, press Enter to skip): ").strip()
            max_inc_input = input("Maximum inclination (degrees, press Enter to skip): ").strip()
            
            min_inc = float(min_inc_input) if min_inc_input else None
            max_inc = float(max_inc_input) if max_inc_input else None
            
            return filter_satellites_by_inclination(satellites, min_inc, max_inc)
        except ValueError:
            print("Error: Invalid inclination values.")
            return satellites
    else:
        return satellites


def _display_selected_satellite(satellite: SatelliteData) -> None:
    """Display detailed information about selected satellite."""
    altitude = satellite.semi_major_axis - 6378.137  # Earth radius in km
    
    print(f"\nSelected Satellite: {satellite.name}")
    print("-" * 40)
    print(f"Catalog Number:    {satellite.catalog_number}")
    print(f"Epoch:             {satellite.epoch.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Altitude:          {altitude:.1f} km")
    print(f"Semi-major Axis:   {satellite.semi_major_axis:.1f} km")
    print(f"Eccentricity:      {satellite.eccentricity:.6f}")
    print(f"Inclination:       {satellite.inclination:.2f}°")
    print(f"RAAN:              {satellite.raan:.2f}°")
    print(f"Arg. of Perigee:   {satellite.arg_perigee:.2f}°")
    print(f"Mean Anomaly:      {satellite.mean_anomaly:.2f}°")
    print(f"Mean Motion:       {satellite.mean_motion:.8f} rev/day")
    print(f"Orbital Period:    {satellite.orbital_period:.1f} minutes")
    print("-" * 40)


def display_transfer_results(result: TransferResult) -> None:
    """Display comprehensive transfer results."""
    print("\n" + "=" * 80)
    print("SATELLITE TRANSFER ANALYSIS RESULTS")
    print("=" * 80)
    
    # Source and target satellite info
    source_alt = result.source_satellite.semi_major_axis - 6378.137
    target_alt = result.target_satellite.semi_major_axis - 6378.137
    
    print(f"\nSOURCE SATELLITE: {result.source_satellite.name}")
    print(f"  Catalog Number: {result.source_satellite.catalog_number}")
    print(f"  Altitude:       {source_alt:.1f} km")
    print(f"  Inclination:    {result.source_satellite.inclination:.2f}°")
    
    print(f"\nTARGET SATELLITE: {result.target_satellite.name}")
    print(f"  Catalog Number: {result.target_satellite.catalog_number}")
    print(f"  Altitude:       {target_alt:.1f} km")
    print(f"  Inclination:    {result.target_satellite.inclination:.2f}°")
    
    # Transfer summary
    print(f"\nTRANSFER SUMMARY:")
    print(f"  Altitude Change:    {target_alt - source_alt:+.1f} km")
    print(f"  Inclination Change: {abs(result.target_satellite.inclination - result.source_satellite.inclination):.2f}°")
    print(f"  Transfer Time:      {result.transfer_time:.1f} minutes ({result.transfer_time/60:.2f} hours)")
    print(f"  Complexity:         {result.complexity_assessment}")
    
    # Delta-V breakdown
    print(f"\nDELTA-V REQUIREMENTS:")
    print(f"  Departure Burn:     {result.departure_deltav:.2f} m/s")
    print(f"  Arrival Burn:       {result.arrival_deltav:.2f} m/s")
    if result.plane_change_deltav > 0.01:  # Only show if significant
        print(f"  Plane Change:       {result.plane_change_deltav:.2f} m/s")
    print(f"  TOTAL DELTA-V:      {result.total_deltav:.2f} m/s")
    
    # Transfer orbit details
    print(f"\nTRANSFER ORBIT DETAILS:")
    print(f"  Apogee Altitude:    {result.transfer_orbit_apogee - 6378.137:.1f} km")
    print(f"  Perigee Altitude:   {result.transfer_orbit_perigee - 6378.137:.1f} km")
    print(f"  Semi-major Axis:    {(result.transfer_orbit_apogee + result.transfer_orbit_perigee) / 2:.1f} km")
    
    # Mission planning information
    print(f"\nMISSION PLANNING NOTES:")
    
    # Fuel estimation (rough approximation)
    # Using Tsiolkovsky rocket equation with typical Isp values
    isp_chemical = 300  # seconds (typical chemical propulsion)
    isp_electric = 3000  # seconds (typical electric propulsion)
    
    mass_ratio_chemical = math.exp(result.total_deltav / (isp_chemical * 9.81))
    mass_ratio_electric = math.exp(result.total_deltav / (isp_electric * 9.81))
    
    fuel_fraction_chemical = (mass_ratio_chemical - 1) / mass_ratio_chemical
    fuel_fraction_electric = (mass_ratio_electric - 1) / mass_ratio_electric
    
    print(f"  Fuel Requirements (Chemical Propulsion, Isp={isp_chemical}s):")
    print(f"    Mass Ratio:       {mass_ratio_chemical:.3f}")
    print(f"    Fuel Fraction:    {fuel_fraction_chemical:.1%}")
    
    print(f"  Fuel Requirements (Electric Propulsion, Isp={isp_electric}s):")
    print(f"    Mass Ratio:       {mass_ratio_electric:.3f}")
    print(f"    Fuel Fraction:    {fuel_fraction_electric:.1%}")
    
    # Display warnings if any
    if result.warnings:
        print(f"\nWARNINGS:")
        for warning in result.warnings:
            print(f"  ⚠️  {warning}")
    
    # Performance classification
    print(f"\nPERFORMANCE CLASSIFICATION:")
    if result.total_deltav < 100:
        classification = "EASY - Low delta-v requirement"
    elif result.total_deltav < 500:
        classification = "MODERATE - Reasonable delta-v requirement"
    elif result.total_deltav < 1000:
        classification = "CHALLENGING - High delta-v requirement"
    else:
        classification = "DIFFICULT - Very high delta-v requirement"
    
    print(f"  {classification}")
    
    print("=" * 80)


def main() -> None:
    """Main application entry point."""
    print("Satellite Delta-V Calculator")
    print("=" * 40)
    print("Calculate delta-v requirements for transfers between LEO satellites")
    print()
    
    try:
        # Load satellite data
        print("Loading satellite data...")
        satellites = load_satellites()
        
        if not satellites:
            print("No satellites loaded. Exiting.")
            return
        
        print(f"Loaded {len(satellites)} satellites successfully.")
        
        while True:
            print("\n" + "=" * 60)
            print("MAIN MENU")
            print("=" * 60)
            print("1. List all satellites")
            print("2. Search satellites")
            print("3. Calculate transfer delta-v")
            print("4. Quick transfer calculation (by catalog numbers)")
            print("5. Genetic algorithm route optimization")
            print("6. Validate satellite data quality")
            print("7. Exit")
            
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == '1':
                display_satellite_list(satellites)
                
            elif choice == '2':
                search_term = input("Enter search term (catalog number or name): ").strip()
                filtered = search_satellites(satellites, search_term)
                if filtered:
                    print(f"\nFound {len(filtered)} matching satellites:")
                    display_satellite_list(filtered)
                else:
                    print("No satellites found matching your search.")
                    
            elif choice == '3':
                # Interactive transfer calculation
                print("\n" + "-" * 50)
                print("TRANSFER CALCULATION")
                print("-" * 50)
                
                # Select source satellite
                print("\nStep 1: Select source satellite")
                source_satellite = select_satellite(satellites, "Select SOURCE satellite:")
                if not source_satellite:
                    print("Source satellite selection cancelled.")
                    continue
                
                # Select target satellite
                print("\nStep 2: Select target satellite")
                target_satellite = select_satellite(satellites, "Select TARGET satellite:", exclude_satellite=source_satellite)
                if not target_satellite:
                    print("Target satellite selection cancelled.")
                    continue
                
                # Calculate transfer
                print("\nCalculating transfer requirements...")
                try:
                    result = calculate_transfer_deltav(source_satellite, target_satellite)
                    display_transfer_results(result)
                    
                    # Ask if user wants to save results with retry mechanism
                    max_save_attempts = 3
                    for save_attempt in range(max_save_attempts):
                        try:
                            save_choice = input("\nSave results to file? (y/n): ").strip().lower()
                            if save_choice in ['y', 'yes']:
                                _save_transfer_results(result)
                                break
                            elif save_choice in ['n', 'no']:
                                break
                            else:
                                if save_attempt < max_save_attempts - 1:
                                    print("Please enter 'y' for yes or 'n' for no.")
                                else:
                                    print("Invalid input. Skipping save.")
                        except KeyboardInterrupt:
                            print("\nSave cancelled by user.")
                            break
                        except Exception as e:
                            print(f"Error during save prompt: {e}")
                            break
                        
                except ValueError as e:
                    print(f"Calculation error: {e}")
                    print("This may be due to invalid orbital parameters or incompatible satellite data.")
                except RuntimeError as e:
                    print(f"Runtime error during calculation: {e}")
                    print("Please try again or select different satellites.")
                except Exception as e:
                    print(f"Unexpected error calculating transfer: {e}")
                    print("Please check that both satellites have valid orbital data and try again.")
                    
            elif choice == '4':
                # Quick calculation by catalog numbers
                print("\n" + "-" * 50)
                print("QUICK TRANSFER CALCULATION")
                print("-" * 50)
                
                try:
                    # Get source catalog number with validation
                    source_cat = None
                    for attempt in range(3):
                        try:
                            source_input = input("Enter source satellite catalog number: ").strip()
                            if not source_input:
                                print("Please enter a catalog number.")
                                continue
                            source_cat = int(source_input)
                            if source_cat <= 0:
                                print("Catalog number must be positive.")
                                continue
                            break
                        except ValueError:
                            print("Please enter a valid integer catalog number.")
                            if attempt == 2:
                                print("Too many invalid attempts.")
                                continue
                    
                    if source_cat is None:
                        continue
                    
                    # Get target catalog number with validation
                    target_cat = None
                    for attempt in range(3):
                        try:
                            target_input = input("Enter target satellite catalog number: ").strip()
                            if not target_input:
                                print("Please enter a catalog number.")
                                continue
                            target_cat = int(target_input)
                            if target_cat <= 0:
                                print("Catalog number must be positive.")
                                continue
                            if target_cat == source_cat:
                                print("Error: Source and target cannot be the same satellite.")
                                continue
                            break
                        except ValueError:
                            print("Please enter a valid integer catalog number.")
                            if attempt == 2:
                                print("Too many invalid attempts.")
                                continue
                    
                    if target_cat is None:
                        continue
                    
                    # Find satellites by catalog number
                    source_sat = None
                    target_sat = None
                    
                    for sat in satellites:
                        if sat.catalog_number == source_cat:
                            source_sat = sat
                        if sat.catalog_number == target_cat:
                            target_sat = sat
                    
                    if not source_sat:
                        print(f"Error: Source satellite {source_cat} not found in dataset.")
                        print("Use option 1 to list available satellites.")
                        continue
                    if not target_sat:
                        print(f"Error: Target satellite {target_cat} not found in dataset.")
                        print("Use option 1 to list available satellites.")
                        continue
                    
                    print(f"\nCalculating transfer from {source_sat.name} to {target_sat.name}...")
                    result = calculate_transfer_deltav(source_sat, target_sat)
                    display_transfer_results(result)
                    
                    # Ask if user wants to save results with validation
                    for save_attempt in range(3):
                        try:
                            save_choice = input("\nSave results to file? (y/n): ").strip().lower()
                            if save_choice in ['y', 'yes']:
                                _save_transfer_results(result)
                                break
                            elif save_choice in ['n', 'no']:
                                break
                            else:
                                if save_attempt < 2:
                                    print("Please enter 'y' for yes or 'n' for no.")
                                else:
                                    print("Invalid input. Skipping save.")
                        except KeyboardInterrupt:
                            print("\nSave cancelled by user.")
                            break
                        
                except KeyboardInterrupt:
                    print("\nOperation cancelled by user.")
                    continue
                except ValueError as e:
                    print(f"Input error: {e}")
                except Exception as e:
                    print(f"Error calculating transfer: {e}")
                    print("Please verify the satellite data and try again.")
                    
            elif choice == '5':
                # Genetic algorithm route optimization
                try:
                    from src.genetic_cli import GeneticCLI
                    print("\n" + "-" * 50)
                    print("GENETIC ALGORITHM ROUTE OPTIMIZATION")
                    print("-" * 50)
                    
                    # Check system requirements
                    print("Checking system requirements...")
                    try:
                        system_check = check_genetic_algorithm_system_requirements()
                        
                        if not system_check['compatible']:
                            print("❌ System requirements not met:")
                            for error in system_check['errors']:
                                print(f"   - {error}")
                            print("Please address these issues before running genetic algorithm optimization.")
                            continue
                        
                        if system_check['warnings']:
                            print("⚠️  System warnings:")
                            for warning in system_check['warnings']:
                                print(f"   - {warning}")
                        
                        if system_check['recommendations']:
                            for rec in system_check['recommendations']:
                                print(f"✅ {rec}")
                    
                    except Exception as e:
                        print(f"⚠️  Could not check system requirements: {e}")
                        print("Proceeding with genetic algorithm...")
                    
                    # Validate satellite data for genetic algorithm compatibility
                    print("\nValidating satellite data for genetic algorithm...")
                    validator = SatelliteDataValidator()
                    validation_result = validator.validate_satellite_dataset(satellites)
                    
                    if not validation_result['valid']:
                        print("⚠️  Satellite data validation issues detected:")
                        print(f"   Valid satellites: {validation_result['valid_satellites']}/{validation_result['total_satellites']}")
                        print(f"   Validity rate: {validation_result.get('validity_rate', 0):.1%}")
                        
                        # Show summary of issues
                        if validation_result.get('errors'):
                            error_count = len(validation_result['errors'])
                            print(f"   ❌ {error_count} critical errors found")
                        
                        if validation_result.get('warnings'):
                            warning_count = len(validation_result['warnings'])
                            print(f"   ⚠️  {warning_count} warnings found")
                        
                        if validation_result['valid_satellites'] == 0:
                            print("❌ No valid satellites found. Cannot run genetic algorithm.")
                            print("Please check your satellite data file or use option 6 for detailed validation.")
                            continue
                        
                        # Ask user if they want to proceed with valid satellites only
                        proceed_choice = input(f"\nProceed with {validation_result['valid_satellites']} valid satellites? (y/n): ").strip().lower()
                        if proceed_choice not in ['y', 'yes']:
                            print("Genetic algorithm cancelled.")
                            continue
                        
                        # Use only valid satellites
                        ga_satellites = validation_result['valid_satellite_list']
                        print(f"Using {len(ga_satellites)} valid satellites for optimization.")
                    else:
                        ga_satellites = satellites
                        print(f"✅ All {len(ga_satellites)} satellites passed validation.")
                    
                    # Enhanced satellite data quality checks for genetic algorithm
                    print("Performing genetic algorithm compatibility checks...")
                    
                    # Check minimum satellite count for meaningful optimization
                    if len(ga_satellites) < 5:
                        print(f"❌ Error: Only {len(ga_satellites)} satellites available.")
                        print("Genetic algorithm requires at least 5 satellites for meaningful optimization.")
                        continue
                    elif len(ga_satellites) < 20:
                        print(f"⚠️  Warning: Only {len(ga_satellites)} satellites available.")
                        print("Genetic algorithm works best with larger satellite constellations (50+ satellites).")
                        print("Results may be limited with small constellations.")
                        
                        continue_choice = input("Continue anyway? (y/n): ").strip().lower()
                        if continue_choice not in ['y', 'yes']:
                            print("Genetic algorithm cancelled.")
                            continue
                    
                    # Check orbital diversity for meaningful route optimization
                    altitudes = [sat.semi_major_axis - 6378.137 for sat in ga_satellites]
                    altitude_range = max(altitudes) - min(altitudes)
                    
                    if altitude_range < 50:  # Less than 50km altitude range
                        print(f"⚠️  Warning: Limited altitude diversity ({altitude_range:.1f} km range).")
                        print("Route optimization works best with satellites at different altitudes.")
                    
                    inclinations = [sat.inclination for sat in ga_satellites]
                    inclination_range = max(inclinations) - min(inclinations)
                    
                    if inclination_range < 10:  # Less than 10 degree inclination range
                        print(f"⚠️  Warning: Limited inclination diversity ({inclination_range:.1f}° range).")
                        print("Route optimization benefits from satellites in different orbital planes.")
                    
                    # Check for recent epoch data
                    from datetime import datetime, timedelta
                    current_time = datetime.utcnow()
                    old_epochs = [sat for sat in ga_satellites 
                                if sat.epoch and (current_time - sat.epoch).days > 30]
                    
                    if old_epochs:
                        print(f"⚠️  Warning: {len(old_epochs)} satellites have epoch data older than 30 days.")
                        print("Old orbital data may affect route calculation accuracy.")
                    
                    print(f"✅ Satellite data ready for genetic algorithm optimization.")
                    print(f"   Constellation size: {len(ga_satellites)} satellites")
                    print(f"   Altitude range: {min(altitudes):.1f} - {max(altitudes):.1f} km")
                    print(f"   Inclination range: {min(inclinations):.1f} - {max(inclinations):.1f}°")
                    
                    print("\nLaunching genetic algorithm interface...")
                    
                    # Create genetic CLI instance and run in interactive mode
                    genetic_cli = GeneticCLI()
                    genetic_cli.satellites = ga_satellites  # Use validated satellites
                    genetic_cli.verbose = True
                    genetic_cli.interactive_mode = True
                    
                    # Run interactive genetic algorithm
                    exit_code = genetic_cli._run_interactive_mode()
                    
                    if exit_code == 0:
                        print("Genetic algorithm optimization completed successfully.")
                    elif exit_code == 130:
                        print("Genetic algorithm optimization interrupted by user.")
                    else:
                        print("Genetic algorithm optimization completed with issues.")
                    
                except ImportError as e:
                    print(f"Error: Genetic algorithm module not available: {e}")
                    print("Please ensure all genetic algorithm components are properly installed.")
                    print("Required modules: genetic_cli, genetic_route_optimizer, genetic_algorithm")
                    print("\nTroubleshooting:")
                    print("1. Check that all Python files are in the src/ directory")
                    print("2. Verify that __init__.py exists in the src/ directory")
                    print("3. Ensure no syntax errors in genetic algorithm modules")
                except ModuleNotFoundError as e:
                    print(f"Error: Required module not found: {e}")
                    print("The genetic algorithm system requires additional dependencies.")
                    print("Please ensure all required modules are available.")
                except Exception as e:
                    print(f"Error running genetic algorithm: {e}")
                    print("This may be due to:")
                    print("1. Incompatible satellite data format")
                    print("2. Insufficient system resources")
                    print("3. Configuration issues")
                    print("Use option 6 for detailed satellite data validation.")
                    print("Returning to main menu.")
                    
            elif choice == '6':
                # Satellite data validation
                print("\n" + "-" * 50)
                print("SATELLITE DATA VALIDATION")
                print("-" * 50)
                
                try:
                    validator = SatelliteDataValidator()
                    print("Running comprehensive satellite data validation...")
                    
                    validation_result = validator.validate_satellite_dataset(satellites)
                    validator.print_validation_report(validation_result)
                    
                    # Offer to filter to valid satellites only
                    if not validation_result['valid'] and validation_result['valid_satellites'] > 0:
                        filter_choice = input(f"\nFilter to {validation_result['valid_satellites']} valid satellites only? (y/n): ").strip().lower()
                        if filter_choice in ['y', 'yes']:
                            satellites = validation_result['valid_satellite_list']
                            print(f"Filtered to {len(satellites)} valid satellites.")
                    
                except Exception as e:
                    print(f"Error during validation: {e}")
                    
            elif choice == '7':
                print("\nThank you for using the Satellite Delta-V Calculator!")
                break
                
            else:
                print("Invalid choice. Please select 1-7.")
                
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except FileNotFoundError as e:
        print(f"\nFile not found error: {e}")
        print("Please ensure the leo_satellites.txt file is in the current directory.")
    except PermissionError as e:
        print(f"\nPermission error: {e}")
        print("Please check file permissions and try again.")
    except ValueError as e:
        print(f"\nData validation error: {e}")
        print("Please check your satellite data file for corrupted or invalid data.")
    except MemoryError:
        print("\nOut of memory error. The satellite data file may be too large.")
        print("Please try with a smaller dataset.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please check your satellite data file and system configuration.")
        print("If the problem persists, please report this issue.")


def _save_genetic_algorithm_results(result, filename: Optional[str] = None) -> None:
    """Save genetic algorithm optimization results to a file."""
    import datetime
    import json
    import os
    
    if result is None:
        print("Error: No results to save.")
        return
    
    try:
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"genetic_optimization_result_{timestamp}.json"
        
        # Ensure filename is valid for the filesystem
        filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        
        # Prepare results data for JSON serialization
        results_data = {
            'optimization_info': {
                'timestamp': datetime.datetime.now().isoformat(),
                'status': result.status.value if hasattr(result, 'status') else 'unknown',
                'execution_time': getattr(result, 'execution_time', 0),
                'success': getattr(result, 'success', False)
            },
            'best_route': {
                'satellite_sequence': getattr(result.best_route, 'satellite_sequence', []) if result.best_route else [],
                'total_deltav': getattr(result.best_route, 'total_deltav', 0) if result.best_route else 0,
                'hop_count': getattr(result.best_route, 'hop_count', 0) if result.best_route else 0,
                'mission_duration': getattr(result.best_route, 'mission_duration', 0) if result.best_route else 0,
                'is_valid': getattr(result.best_route, 'is_valid', False) if result.best_route else False
            },
            'optimization_stats': {
                'generations_completed': getattr(result.optimization_stats, 'generations_completed', 0) if hasattr(result, 'optimization_stats') else 0,
                'best_fitness': getattr(result.optimization_stats, 'best_fitness', 0) if hasattr(result, 'optimization_stats') else 0,
                'average_fitness': getattr(result.optimization_stats, 'average_fitness', 0) if hasattr(result, 'optimization_stats') else 0
            }
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Genetic algorithm results saved to: {filename}")
        
    except Exception as e:
        print(f"Error saving results: {e}")


def _save_transfer_results(result: TransferResult) -> None:
    """Save transfer results to a file."""
    import datetime
    import os
    
    if result is None:
        print("Error: No results to save.")
        return
    
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transfer_result_{result.source_satellite.catalog_number}_to_{result.target_satellite.catalog_number}_{timestamp}.txt"
        
        # Ensure filename is valid for the filesystem
        filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        
        # Check if file already exists and create unique name if needed
        counter = 1
        original_filename = filename
        while os.path.exists(filename):
            name, ext = os.path.splitext(original_filename)
            filename = f"{name}_{counter}{ext}"
            counter += 1
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("SATELLITE TRANSFER ANALYSIS RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            # Source and target info
            source_alt = result.source_satellite.semi_major_axis - 6378.137
            target_alt = result.target_satellite.semi_major_axis - 6378.137
            
            f.write(f"SOURCE SATELLITE: {result.source_satellite.name}\n")
            f.write(f"  Catalog Number: {result.source_satellite.catalog_number}\n")
            f.write(f"  Altitude:       {source_alt:.1f} km\n")
            f.write(f"  Inclination:    {result.source_satellite.inclination:.2f}°\n\n")
            
            f.write(f"TARGET SATELLITE: {result.target_satellite.name}\n")
            f.write(f"  Catalog Number: {result.target_satellite.catalog_number}\n")
            f.write(f"  Altitude:       {target_alt:.1f} km\n")
            f.write(f"  Inclination:    {result.target_satellite.inclination:.2f}°\n\n")
            
            # Transfer summary
            f.write("TRANSFER SUMMARY:\n")
            f.write(f"  Total Delta-V:      {result.total_deltav:.2f} m/s\n")
            f.write(f"  Departure Burn:     {result.departure_deltav:.2f} m/s\n")
            f.write(f"  Arrival Burn:       {result.arrival_deltav:.2f} m/s\n")
            if result.plane_change_deltav > 0.01:
                f.write(f"  Plane Change:       {result.plane_change_deltav:.2f} m/s\n")
            f.write(f"  Transfer Time:      {result.transfer_time:.1f} minutes\n")
            f.write(f"  Complexity:         {result.complexity_assessment}\n\n")
            
            # Warnings
            if result.warnings:
                f.write("WARNINGS:\n")
                for warning in result.warnings:
                    f.write(f"  - {warning}\n")
                f.write("\n")
            
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"Results saved to: {filename}")
        
    except PermissionError:
        print(f"Error: Permission denied writing to {filename}. Check file permissions.")
    except OSError as e:
        print(f"Error: Unable to write file {filename}: {e}")
    except Exception as e:
        print(f"Unexpected error saving results: {e}")
        print("Results could not be saved to file.")


if __name__ == "__main__":
    main()