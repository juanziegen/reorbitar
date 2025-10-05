"""
FastAPI dependencies for dependency injection.

This module provides dependency injection functions for the API endpoints,
including route simulator, satellite database, and other service dependencies.
"""

from typing import Dict, List, Optional
import logging
from datetime import datetime

from ..services.route_simulator import RouteSimulator
from ..models.satellite import Satellite, OrbitalElements
from ..utils.tle_parser import TLEParser
from .service_request_manager import ServiceRequestManager


logger = logging.getLogger(__name__)


class SatelliteDatabase:
    """
    Mock satellite database for demonstration purposes.
    In production, this would connect to a real database.
    """
    
    def __init__(self):
        """Initialize the satellite database with sample data."""
        self._satellites: Dict[str, Satellite] = {}
        self._load_sample_data()
    
    def _load_sample_data(self):
        """Load sample satellite data for testing."""
        # Sample TLE data for demonstration
        sample_satellites = [
            {
                "id": "SAT001",
                "name": "DEMO SAT 1",
                "tle_line1": "1 25544U 98067A   21001.00000000  .00002182  00000-0  38792-4 0  9991",
                "tle_line2": "2 25544  51.6461 339.2911 0002829  68.6102 291.5211 15.48919103123456",
                "mass": 450.0,
                "material_composition": {"aluminum": 0.6, "steel": 0.3, "electronics": 0.1},
                "decommission_date": datetime(2024, 6, 1)
            },
            {
                "id": "SAT002", 
                "name": "DEMO SAT 2",
                "tle_line1": "1 25545U 98067B   21001.00000000  .00001845  00000-0  32156-4 0  9992",
                "tle_line2": "2 25545  51.6455 340.1234 0003156  72.4567 287.6543 15.49012345123457",
                "mass": 320.0,
                "material_composition": {"aluminum": 0.65, "steel": 0.25, "electronics": 0.1},
                "decommission_date": datetime(2024, 7, 15)
            },
            {
                "id": "SAT003",
                "name": "DEMO SAT 3", 
                "tle_line1": "1 25546U 98067C   21001.00000000  .00001654  00000-0  28934-4 0  9993",
                "tle_line2": "2 25546  51.6449 341.5678 0002987  75.1234 284.8765 15.49156789123458",
                "mass": 280.0,
                "material_composition": {"aluminum": 0.7, "steel": 0.2, "electronics": 0.1},
                "decommission_date": datetime(2024, 8, 30)
            }
        ]
        
        for sat_data in sample_satellites:
            try:
                satellite = Satellite(**sat_data)
                self._satellites[satellite.id] = satellite
                logger.info(f"Loaded sample satellite: {satellite.id}")
            except Exception as e:
                logger.error(f"Failed to load sample satellite {sat_data['id']}: {str(e)}")
    
    def get_satellite(self, satellite_id: str) -> Optional[Satellite]:
        """Get satellite by ID."""
        return self._satellites.get(satellite_id)
    
    def list_satellites(self, limit: int = 100, offset: int = 0, 
                       filter_valid: bool = True) -> List[Satellite]:
        """List satellites with pagination and filtering."""
        satellites = list(self._satellites.values())
        
        if filter_valid:
            satellites = [sat for sat in satellites if sat.is_valid()]
        
        # Apply pagination
        start_idx = offset
        end_idx = offset + limit
        
        return satellites[start_idx:end_idx]
    
    def add_satellite(self, satellite: Satellite) -> bool:
        """Add a new satellite to the database."""
        try:
            if satellite.is_valid():
                self._satellites[satellite.id] = satellite
                logger.info(f"Added satellite: {satellite.id}")
                return True
            else:
                logger.warning(f"Attempted to add invalid satellite: {satellite.id}")
                return False
        except Exception as e:
            logger.error(f"Failed to add satellite {satellite.id}: {str(e)}")
            return False
    
    def update_satellite(self, satellite_id: str, satellite: Satellite) -> bool:
        """Update an existing satellite."""
        try:
            if satellite_id in self._satellites and satellite.is_valid():
                self._satellites[satellite_id] = satellite
                logger.info(f"Updated satellite: {satellite_id}")
                return True
            else:
                logger.warning(f"Failed to update satellite: {satellite_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to update satellite {satellite_id}: {str(e)}")
            return False
    
    def delete_satellite(self, satellite_id: str) -> bool:
        """Delete a satellite from the database."""
        try:
            if satellite_id in self._satellites:
                del self._satellites[satellite_id]
                logger.info(f"Deleted satellite: {satellite_id}")
                return True
            else:
                logger.warning(f"Satellite not found for deletion: {satellite_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete satellite {satellite_id}: {str(e)}")
            return False
    
    def search_satellites(self, query: str) -> List[Satellite]:
        """Search satellites by name or ID."""
        query_lower = query.lower()
        results = []
        
        for satellite in self._satellites.values():
            if (query_lower in satellite.id.lower() or 
                query_lower in satellite.name.lower()):
                results.append(satellite)
        
        return results
    
    def get_satellite_count(self) -> int:
        """Get total number of satellites in database."""
        return len(self._satellites)


# Global instances (in production, these would be properly configured)
_route_simulator = None
_satellite_database = None
_service_request_manager = None


def get_route_simulator() -> RouteSimulator:
    """
    Dependency injection for RouteSimulator.
    
    Returns:
        RouteSimulator instance
    """
    global _route_simulator
    
    if _route_simulator is None:
        try:
            _route_simulator = RouteSimulator(max_workers=4)
            logger.info("RouteSimulator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RouteSimulator: {str(e)}")
            raise
    
    return _route_simulator


def get_satellite_database() -> SatelliteDatabase:
    """
    Dependency injection for SatelliteDatabase.
    
    Returns:
        SatelliteDatabase instance
    """
    global _satellite_database
    
    if _satellite_database is None:
        try:
            _satellite_database = SatelliteDatabase()
            logger.info("SatelliteDatabase initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SatelliteDatabase: {str(e)}")
            raise
    
    return _satellite_database


def get_service_request_manager() -> ServiceRequestManager:
    """
    Dependency injection for ServiceRequestManager.
    
    Returns:
        ServiceRequestManager instance
    """
    global _service_request_manager
    
    if _service_request_manager is None:
        try:
            _service_request_manager = ServiceRequestManager()
            logger.info("ServiceRequestManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ServiceRequestManager: {str(e)}")
            raise
    
    return _service_request_manager


def get_tle_parser() -> TLEParser:
    """
    Dependency injection for TLEParser.
    
    Returns:
        TLEParser instance
    """
    try:
        return TLEParser()
    except Exception as e:
        logger.error(f"Failed to initialize TLEParser: {str(e)}")
        raise


# Configuration dependencies
def get_api_config() -> Dict[str, any]:
    """
    Get API configuration settings.
    
    Returns:
        Dictionary with API configuration
    """
    return {
        "max_satellites_per_request": 50,
        "max_optimization_time_seconds": 300,
        "default_quote_validity_days": 30,
        "enable_caching": True,
        "cache_ttl_seconds": 3600,
        "rate_limit_per_minute": 100
    }


def get_cost_model_config() -> Dict[str, float]:
    """
    Get cost model configuration parameters.
    
    Returns:
        Dictionary with cost model parameters
    """
    return {
        "cost_per_delta_v_ms": 1.27,  # USD per m/s
        "operational_overhead_factor": 0.15,  # 15%
        "processing_cost_factor": 0.1,  # 10% of collection cost
        "storage_cost_per_kg_per_day": 0.5,  # USD per kg per day
        "mission_complexity_multiplier": 1.0
    }


def get_optimization_config() -> Dict[str, any]:
    """
    Get genetic algorithm optimization configuration.
    
    Returns:
        Dictionary with optimization parameters
    """
    return {
        "population_size": 100,
        "max_generations": 500,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8,
        "convergence_threshold": 1e-6,
        "max_stagnant_generations": 50,
        "early_termination_enabled": True,
        "parallel_processing": True,
        "max_workers": 4
    }


# Health check dependencies
def check_service_health() -> Dict[str, str]:
    """
    Check the health of all service dependencies.
    
    Returns:
        Dictionary with health status of each service
    """
    health_status = {}
    
    # Check RouteSimulator
    try:
        simulator = get_route_simulator()
        health_status["route_simulator"] = "healthy"
    except Exception as e:
        health_status["route_simulator"] = f"unhealthy: {str(e)}"
    
    # Check SatelliteDatabase
    try:
        db = get_satellite_database()
        satellite_count = db.get_satellite_count()
        health_status["satellite_database"] = f"healthy ({satellite_count} satellites)"
    except Exception as e:
        health_status["satellite_database"] = f"unhealthy: {str(e)}"
    
    # Check ServiceRequestManager
    try:
        manager = get_service_request_manager()
        health_status["service_request_manager"] = "healthy"
    except Exception as e:
        health_status["service_request_manager"] = f"unhealthy: {str(e)}"
    
    # Check TLEParser
    try:
        parser = get_tle_parser()
        health_status["tle_parser"] = "healthy"
    except Exception as e:
        health_status["tle_parser"] = f"unhealthy: {str(e)}"
    
    return health_status


# Cleanup functions
def cleanup_dependencies():
    """Clean up global dependencies on shutdown."""
    global _route_simulator, _satellite_database, _service_request_manager
    
    try:
        if _route_simulator:
            # Cleanup any active simulations
            logger.info("Cleaning up RouteSimulator")
            _route_simulator = None
        
        if _satellite_database:
            logger.info("Cleaning up SatelliteDatabase")
            _satellite_database = None
        
        if _service_request_manager:
            logger.info("Cleaning up ServiceRequestManager")
            _service_request_manager = None
        
        logger.info("Dependencies cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during dependency cleanup: {str(e)}")