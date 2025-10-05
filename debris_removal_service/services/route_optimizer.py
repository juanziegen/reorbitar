"""
Route Optimization Service

This module integrates the genetic algorithm route optimizer with the debris removal service,
providing a high-level interface for route optimization with cost calculations.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging

from ..models.satellite import Satellite, OrbitalElements
from ..models.route import Route, Hop, ManeuverDetails
from ..models.cost import MissionCost, PropellantMass, DetailedCost
from ..models.service_request import ServiceRequest, TimelineConstraints, BudgetConstraints
from ..models.biprop_cost_model import BipropCostModel
from ..utils.tle_parser import TLEParser
from ..utils.validation import SatelliteDataValidator

# Import existing genetic algorithm components
import sys
import os
src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from genetic_route_optimizer import GeneticRouteOptimizer
    from genetic_algorithm import GAConfig, RouteConstraints, RouteChromosome
    from tle_parser import SatelliteData as GASatelliteData
    GENETIC_ALGORITHM_AVAILABLE = True
    SatelliteData = GASatelliteData
except ImportError as e:
    print(f"Warning: Genetic algorithm modules not available: {e}")
    GENETIC_ALGORITHM_AVAILABLE = False
    
    # Create dummy classes for graceful degradation
    class GAConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class RouteConstraints:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class RouteChromosome:
        def __init__(self, **kwargs):
            self.satellite_sequence = kwargs.get('satellite_sequence', [])
            self.departure_times = kwargs.get('departure_times', [])
            self.total_deltav = kwargs.get('total_deltav', 0.0)
            self.mission_duration = kwargs.get('mission_duration', 0.0)
    
    class SatelliteData:
        def __init__(self, **kwargs):
            # Accept any keyword arguments and set them as attributes
            for key, value in kwargs.items():
                setattr(self, key, value)


class RouteOptimizationService:
    """
    High-level service for satellite debris removal route optimization.
    
    This service integrates the genetic algorithm optimizer with the debris removal
    service models, providing cost-aware route optimization with comprehensive
    mission planning capabilities.
    """
    
    def __init__(self, cost_model: Optional[BipropCostModel] = None):
        """
        Initialize the route optimization service.
        
        Args:
            cost_model: Optional biprop cost model for cost calculations
        """
        self.cost_model = cost_model or BipropCostModel()
        self.logger = logging.getLogger(__name__)
        
    def optimize_route(self, service_request: ServiceRequest, 
                      satellites: List[Satellite]) -> Tuple[Route, MissionCost]:
        """
        Optimize a satellite collection route for a service request.
        
        Args:
            service_request: Client service request with constraints and preferences
            satellites: Available satellites for route optimization
            
        Returns:
            Tuple of optimized route and mission cost breakdown
            
        Raises:
            ValueError: If service request or satellites are invalid
            RuntimeError: If optimization fails
        """
        try:
            # Validate inputs
            self._validate_inputs(service_request, satellites)
            
            # Convert debris removal satellites to genetic algorithm format
            ga_satellites = self._convert_satellites_to_ga_format(satellites)
            
            # Create genetic algorithm configuration from service request
            ga_config = self._create_ga_config(service_request)
            
            # Create route constraints from service request
            constraints = self._create_route_constraints(service_request, satellites)
            
            if not GENETIC_ALGORITHM_AVAILABLE:
                # Fallback to simplified route optimization
                route = self._create_simple_route(satellites, service_request)
            else:
                # Initialize and run genetic algorithm optimizer
                optimizer = GeneticRouteOptimizer(ga_satellites, ga_config)
                optimization_result = optimizer.optimize_route(constraints)
                
                if not optimization_result.success:
                    raise RuntimeError(f"Route optimization failed: {optimization_result.error_message}")
                
                # Convert genetic algorithm result back to debris removal service format
                route = self._convert_ga_result_to_route(
                    optimization_result.best_chromosome, 
                    satellites, 
                    service_request
                )
            
            # Calculate comprehensive mission costs
            mission_cost = self._calculate_mission_cost(route, service_request)
            
            self.logger.info(f"Route optimization completed successfully. "
                           f"Route: {len(route.satellites)} satellites, "
                           f"Total cost: ${mission_cost.total_cost:.2f}")
            
            return route, mission_cost
            
        except Exception as e:
            self.logger.error(f"Route optimization failed: {str(e)}")
            raise
    
    def _validate_inputs(self, service_request: ServiceRequest, satellites: List[Satellite]):
        """Validate service request and satellite data."""
        if not service_request.satellites:
            raise ValueError("Service request must specify at least one satellite")
        
        if not satellites:
            raise ValueError("Satellites list cannot be empty")
        
        # Validate that requested satellites exist in the available satellites
        available_ids = {sat.id for sat in satellites}
        requested_ids = set(service_request.satellites)
        
        missing_satellites = requested_ids - available_ids
        if missing_satellites:
            raise ValueError(f"Requested satellites not found: {missing_satellites}")
        
        # Validate satellite data
        for satellite in satellites:
            is_valid, errors = SatelliteDataValidator.validate_satellite(satellite)
            if not is_valid:
                self.logger.warning(f"Satellite {satellite.id} has validation issues: {errors}")
    
    def _convert_satellites_to_ga_format(self, satellites: List[Satellite]) -> List[SatelliteData]:
        """Convert debris removal service satellites to genetic algorithm format."""
        ga_satellites = []
        
        for satellite in satellites:
            try:
                # Extract orbital elements
                elements = satellite.orbital_elements
                if not elements:
                    self.logger.warning(f"Satellite {satellite.id} missing orbital elements")
                    continue
                
                # Create SatelliteData for genetic algorithm
                ga_satellite = SatelliteData(
                    catalog_number=int(satellite.id),
                    name=satellite.name,
                    tle_line1=satellite.tle_line1,
                    tle_line2=satellite.tle_line2,
                    mass_kg=satellite.mass,
                    # Convert orbital elements
                    semi_major_axis=elements.semi_major_axis,
                    eccentricity=elements.eccentricity,
                    inclination=elements.inclination,
                    raan=elements.raan,
                    arg_perigee=elements.argument_of_perigee,
                    mean_anomaly=elements.mean_anomaly,
                    mean_motion=elements.mean_motion,
                    epoch=elements.epoch
                )
                
                ga_satellites.append(ga_satellite)
                
            except Exception as e:
                self.logger.warning(f"Failed to convert satellite {satellite.id}: {str(e)}")
                continue
        
        if not ga_satellites:
            # If no satellites could be converted, create simplified versions
            self.logger.warning("No satellites could be converted to GA format, creating simplified versions")
            for satellite in satellites:
                try:
                    ga_satellite = SatelliteData(
                        catalog_number=int(satellite.id),
                        name=satellite.name,
                        mass_kg=satellite.mass
                    )
                    ga_satellites.append(ga_satellite)
                except Exception as e:
                    self.logger.warning(f"Failed to create simplified satellite data for {satellite.id}: {str(e)}")
        
        if not ga_satellites:
            raise ValueError("No valid satellites available for optimization")
        
        return ga_satellites
    
    def _create_ga_config(self, service_request: ServiceRequest) -> GAConfig:
        """Create genetic algorithm configuration from service request."""
        # Determine population size based on number of satellites
        satellite_count = len(service_request.satellites)
        
        if satellite_count <= 10:
            population_size = 50
            max_generations = 200
        elif satellite_count <= 50:
            population_size = 100
            max_generations = 300
        else:
            population_size = 200
            max_generations = 500
        
        # Check if this is an urgent request
        is_urgent = service_request.is_urgent()
        if is_urgent:
            # Reduce generations for faster results
            max_generations = max_generations // 2
        
        return GAConfig(
            population_size=population_size,
            max_generations=max_generations,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elitism_count=max(2, population_size // 20),
            tournament_size=3,
            convergence_threshold=1e-6,
            max_stagnant_generations=50
        )
    
    def _create_route_constraints(self, service_request: ServiceRequest, 
                                satellites: List[Satellite]) -> RouteConstraints:
        """Create route constraints from service request."""
        # Convert budget constraint to delta-v budget
        max_budget = service_request.budget_constraints.max_total_cost
        
        # Estimate maximum delta-v from budget (using average cost per m/s)
        avg_cost_per_ms = 1.27  # USD per m/s from biprop cost model
        max_deltav_ms = max_budget / avg_cost_per_ms
        max_deltav_kms = max_deltav_ms / 1000.0  # Convert to km/s
        
        # Convert timeline constraint to mission duration
        timeline = service_request.timeline_requirements
        max_duration_seconds = timeline.get_available_duration().total_seconds()
        
        # Determine start/end satellites if specified
        start_satellite_id = None
        end_satellite_id = None
        
        # For now, we'll let the algorithm choose optimal start/end points
        # This could be enhanced to support user-specified start/end points
        
        return RouteConstraints(
            max_deltav_budget=max_deltav_kms,
            max_mission_duration=max_duration_seconds,
            start_satellite_id=start_satellite_id,
            end_satellite_id=end_satellite_id,
            min_hops=1,
            max_hops=min(len(service_request.satellites), 20),  # Reasonable upper limit
            forbidden_satellites=[]  # Could be enhanced to support forbidden satellites
        )
    
    def _convert_ga_result_to_route(self, chromosome: RouteChromosome, 
                                  satellites: List[Satellite],
                                  service_request: ServiceRequest) -> Route:
        """Convert genetic algorithm result back to debris removal service route format."""
        # Create satellite lookup
        satellite_lookup = {int(sat.id): sat for sat in satellites}
        
        # Get satellites in route order
        route_satellites = []
        for sat_id in chromosome.satellite_sequence:
            if sat_id in satellite_lookup:
                route_satellites.append(satellite_lookup[sat_id])
        
        if len(route_satellites) < 2:
            raise ValueError("Route must contain at least 2 satellites")
        
        # Create hops between consecutive satellites
        hops = []
        total_cost = 0.0
        
        for i in range(len(route_satellites) - 1):
            from_sat = route_satellites[i]
            to_sat = route_satellites[i + 1]
            
            # Calculate delta-v for this hop (simplified calculation)
            # In a real implementation, this would use orbital mechanics
            hop_deltav = self._estimate_hop_deltav(from_sat, to_sat)
            
            # Calculate cost for this hop
            hop_cost = self.cost_model.calculate_cost(hop_deltav * 1000)  # Convert km/s to m/s
            total_cost += hop_cost.cost_usd
            
            # Create maneuver details
            maneuver = ManeuverDetails(
                departure_burn_dv=hop_deltav * 500,  # Simplified
                arrival_burn_dv=hop_deltav * 500,
                plane_change_dv=0.0,  # Simplified
                total_dv=hop_deltav * 1000,
                transfer_type="hohmann",
                phase_angle=0.0,  # Simplified
                wait_time=timedelta(hours=6)  # Simplified
            )
            
            # Create hop
            hop = Hop(
                from_satellite=from_sat,
                to_satellite=to_sat,
                delta_v_required=hop_deltav * 1000,  # Convert to m/s
                transfer_time=timedelta(hours=12),  # Simplified
                cost=hop_cost.cost_usd,
                maneuver_details=maneuver,
                hop_number=i + 1
            )
            
            hops.append(hop)
        
        # Calculate mission duration
        mission_duration = timedelta(seconds=chromosome.mission_duration)
        
        # Create route
        route = Route(
            satellites=route_satellites,
            hops=hops,
            total_delta_v=chromosome.total_deltav * 1000,  # Convert km/s to m/s
            total_cost=total_cost,
            mission_duration=mission_duration,
            feasibility_score=0.8,  # Simplified
            route_id=f"route_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        return route
    
    def _estimate_hop_deltav(self, from_sat: Satellite, to_sat: Satellite) -> float:
        """
        Estimate delta-v required for hop between satellites.
        
        This is a simplified calculation. In a real implementation,
        this would use proper orbital mechanics calculations.
        """
        if not from_sat.orbital_elements or not to_sat.orbital_elements:
            return 1.0  # Default estimate in km/s
        
        # Simple estimate based on altitude difference
        from_alt = from_sat.orbital_elements.semi_major_axis
        to_alt = to_sat.orbital_elements.semi_major_axis
        
        altitude_diff = abs(to_alt - from_alt)
        
        # Rough estimate: 1 m/s per km altitude difference, minimum 100 m/s
        deltav_ms = max(100, altitude_diff)
        
        return deltav_ms / 1000.0  # Convert to km/s
    
    def _calculate_mission_cost(self, route: Route, service_request: ServiceRequest) -> MissionCost:
        """Calculate comprehensive mission cost breakdown."""
        # Collection cost is the route cost
        collection_cost = route.total_cost
        
        # Processing cost based on preferences
        processing_cost = self._calculate_processing_cost(route, service_request)
        
        # Storage cost (simplified)
        storage_cost = len(route.satellites) * 1000.0  # $1000 per satellite
        
        # Operational overhead (10% of total)
        subtotal = collection_cost + processing_cost + storage_cost
        operational_overhead = subtotal * 0.1
        
        total_cost = subtotal + operational_overhead
        cost_per_satellite = total_cost / len(route.satellites)
        
        # Calculate propellant mass
        propellant_mass = self._calculate_total_propellant_mass(route)
        
        return MissionCost(
            collection_cost=collection_cost,
            processing_cost=processing_cost,
            storage_cost=storage_cost,
            operational_overhead=operational_overhead,
            total_cost=total_cost,
            cost_per_satellite=cost_per_satellite,
            propellant_mass=propellant_mass
        )
    
    def _calculate_processing_cost(self, route: Route, service_request: ServiceRequest) -> float:
        """Calculate processing cost based on service preferences."""
        # Simplified processing cost calculation
        satellite_count = len(route.satellites)
        
        # Base processing cost per satellite
        base_cost_per_satellite = 5000.0
        
        # Adjust based on processing preferences
        processing_types = service_request.processing_preferences.preferred_processing_types
        
        if not processing_types:
            return satellite_count * base_cost_per_satellite
        
        # Different processing types have different costs
        primary_type = processing_types[0]
        
        if primary_type.value == "iss_recycling":
            multiplier = 1.0  # Base cost
        elif primary_type.value == "solar_forge":
            multiplier = 2.0  # More expensive
        elif primary_type.value == "heo_storage":
            multiplier = 0.5  # Cheaper storage
        else:
            multiplier = 1.0
        
        return satellite_count * base_cost_per_satellite * multiplier
    
    def _calculate_total_propellant_mass(self, route: Route) -> PropellantMass:
        """Calculate total propellant mass for the mission."""
        total_deltav_ms = route.total_delta_v
        
        # Use cost model to get propellant mass
        cost_result = self.cost_model.calculate_cost(total_deltav_ms)
        
        return PropellantMass(
            fuel_kg=cost_result.fuel_kg,
            oxidizer_kg=cost_result.ox_kg,
            total_kg=cost_result.total_prop_kg
        )
    
    def _create_simple_route(self, satellites: List[Satellite], 
                           service_request: ServiceRequest) -> Route:
        """
        Create a simple route when genetic algorithm is not available.
        
        This is a fallback method that creates a basic route by connecting
        satellites in order without optimization.
        """
        # Filter to requested satellites
        requested_satellites = [
            sat for sat in satellites 
            if sat.id in service_request.satellites
        ]
        
        if len(requested_satellites) < 2:
            raise ValueError("Need at least 2 satellites for route creation")
        
        # Create simple hops between consecutive satellites
        hops = []
        total_cost = 0.0
        total_deltav = 0.0
        
        for i in range(len(requested_satellites) - 1):
            from_sat = requested_satellites[i]
            to_sat = requested_satellites[i + 1]
            
            # Simple delta-v estimate
            hop_deltav = self._estimate_hop_deltav(from_sat, to_sat) * 1000  # Convert to m/s
            total_deltav += hop_deltav
            
            # Calculate cost for this hop
            hop_cost = self.cost_model.calculate_cost(hop_deltav)
            total_cost += hop_cost.cost_usd
            
            # Create simple maneuver details
            maneuver = ManeuverDetails(
                departure_burn_dv=hop_deltav * 0.5,
                arrival_burn_dv=hop_deltav * 0.5,
                plane_change_dv=0.0,
                total_dv=hop_deltav,
                transfer_type="simplified",
                phase_angle=0.0,
                wait_time=timedelta(hours=6)
            )
            
            # Create hop
            hop = Hop(
                from_satellite=from_sat,
                to_satellite=to_sat,
                delta_v_required=hop_deltav,
                transfer_time=timedelta(hours=12),
                cost=hop_cost.cost_usd,
                maneuver_details=maneuver,
                hop_number=i + 1
            )
            
            hops.append(hop)
        
        # Create route
        route = Route(
            satellites=requested_satellites,
            hops=hops,
            total_delta_v=total_deltav,
            total_cost=total_cost,
            mission_duration=timedelta(hours=len(hops) * 12),
            feasibility_score=0.7,  # Conservative estimate
            route_id=f"simple_route_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        return route