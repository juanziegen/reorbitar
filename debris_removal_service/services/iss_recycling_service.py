"""
ISS Recycling Service Module

Handles cost calculations, timeline estimation, and capacity management
for satellite debris processing at the International Space Station.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

from ..models.satellite import Satellite
from ..models.cost import DetailedCost, MissionCost


class MaterialType(Enum):
    """Types of materials that can be processed at ISS."""
    ALUMINUM = "aluminum"
    TITANIUM = "titanium"
    STEEL = "steel"
    CARBON_FIBER = "carbon_fiber"
    ELECTRONICS = "electronics"
    SOLAR_PANELS = "solar_panels"
    BATTERIES = "batteries"
    PROPELLANT_TANKS = "propellant_tanks"
    OTHER_METALS = "other_metals"
    COMPOSITES = "composites"


@dataclass
class ProcessingCapability:
    """ISS processing capability for a specific material type."""
    material_type: MaterialType
    max_daily_throughput_kg: float
    processing_cost_per_kg: float  # USD per kg
    processing_time_per_kg: timedelta  # Time per kg
    setup_time: timedelta  # One-time setup for this material
    efficiency_factor: float  # 0.0 to 1.0, processing efficiency
    
    def calculate_processing_time(self, mass_kg: float) -> timedelta:
        """Calculate total processing time for given mass."""
        base_time = self.processing_time_per_kg * mass_kg / self.efficiency_factor
        return self.setup_time + base_time
    
    def calculate_processing_cost(self, mass_kg: float) -> float:
        """Calculate processing cost for given mass."""
        return mass_kg * self.processing_cost_per_kg


@dataclass
class ISSCapacityStatus:
    """Current capacity status of ISS recycling operations."""
    current_utilization: float  # 0.0 to 1.0
    available_capacity_kg_per_day: float
    scheduled_maintenance: List[Tuple[datetime, datetime]]  # maintenance windows
    current_queue_size: int  # number of pending processing jobs
    estimated_queue_time: timedelta
    
    def is_available(self, start_time: datetime, duration: timedelta) -> bool:
        """Check if ISS capacity is available for the given time period."""
        end_time = start_time + duration
        
        # Check against maintenance windows
        for maint_start, maint_end in self.scheduled_maintenance:
            if not (end_time <= maint_start or start_time >= maint_end):
                return False
        
        return True
    
    def get_next_available_slot(self, required_duration: timedelta) -> datetime:
        """Get the next available time slot for processing."""
        current_time = datetime.utcnow()
        candidate_time = current_time + self.estimated_queue_time
        
        # Check against maintenance windows
        for maint_start, maint_end in self.scheduled_maintenance:
            if candidate_time < maint_end and candidate_time + required_duration > maint_start:
                candidate_time = maint_end
        
        return candidate_time


class ISSRecyclingService:
    """
    ISS Recycling Service for processing satellite debris materials.
    
    Provides cost calculations, timeline estimation, and capacity management
    for material processing at the International Space Station.
    """
    
    def __init__(self):
        """Initialize ISS recycling service with processing capabilities."""
        self.processing_capabilities = self._initialize_processing_capabilities()
        self.base_operational_cost = 5000.0  # USD per day base operational cost
        self.transport_cost_per_kg = 2200.0  # USD per kg to transport to ISS
        self.capacity_status = self._get_current_capacity_status()
    
    def _initialize_processing_capabilities(self) -> Dict[MaterialType, ProcessingCapability]:
        """Initialize processing capabilities for different material types."""
        return {
            MaterialType.ALUMINUM: ProcessingCapability(
                material_type=MaterialType.ALUMINUM,
                max_daily_throughput_kg=50.0,
                processing_cost_per_kg=15.0,
                processing_time_per_kg=timedelta(minutes=30),
                setup_time=timedelta(hours=2),
                efficiency_factor=0.85
            ),
            MaterialType.TITANIUM: ProcessingCapability(
                material_type=MaterialType.TITANIUM,
                max_daily_throughput_kg=20.0,
                processing_cost_per_kg=45.0,
                processing_time_per_kg=timedelta(hours=1.5),
                setup_time=timedelta(hours=4),
                efficiency_factor=0.75
            ),
            MaterialType.STEEL: ProcessingCapability(
                material_type=MaterialType.STEEL,
                max_daily_throughput_kg=40.0,
                processing_cost_per_kg=12.0,
                processing_time_per_kg=timedelta(minutes=45),
                setup_time=timedelta(hours=2),
                efficiency_factor=0.80
            ),
            MaterialType.CARBON_FIBER: ProcessingCapability(
                material_type=MaterialType.CARBON_FIBER,
                max_daily_throughput_kg=15.0,
                processing_cost_per_kg=85.0,
                processing_time_per_kg=timedelta(hours=2),
                setup_time=timedelta(hours=6),
                efficiency_factor=0.70
            ),
            MaterialType.ELECTRONICS: ProcessingCapability(
                material_type=MaterialType.ELECTRONICS,
                max_daily_throughput_kg=25.0,
                processing_cost_per_kg=120.0,
                processing_time_per_kg=timedelta(hours=3),
                setup_time=timedelta(hours=8),
                efficiency_factor=0.60
            ),
            MaterialType.SOLAR_PANELS: ProcessingCapability(
                material_type=MaterialType.SOLAR_PANELS,
                max_daily_throughput_kg=30.0,
                processing_cost_per_kg=95.0,
                processing_time_per_kg=timedelta(hours=2.5),
                setup_time=timedelta(hours=4),
                efficiency_factor=0.65
            ),
            MaterialType.BATTERIES: ProcessingCapability(
                material_type=MaterialType.BATTERIES,
                max_daily_throughput_kg=10.0,
                processing_cost_per_kg=200.0,
                processing_time_per_kg=timedelta(hours=4),
                setup_time=timedelta(hours=12),
                efficiency_factor=0.50
            ),
            MaterialType.PROPELLANT_TANKS: ProcessingCapability(
                material_type=MaterialType.PROPELLANT_TANKS,
                max_daily_throughput_kg=35.0,
                processing_cost_per_kg=25.0,
                processing_time_per_kg=timedelta(hours=1),
                setup_time=timedelta(hours=3),
                efficiency_factor=0.80
            ),
            MaterialType.OTHER_METALS: ProcessingCapability(
                material_type=MaterialType.OTHER_METALS,
                max_daily_throughput_kg=30.0,
                processing_cost_per_kg=20.0,
                processing_time_per_kg=timedelta(minutes=50),
                setup_time=timedelta(hours=2),
                efficiency_factor=0.75
            ),
            MaterialType.COMPOSITES: ProcessingCapability(
                material_type=MaterialType.COMPOSITES,
                max_daily_throughput_kg=20.0,
                processing_cost_per_kg=65.0,
                processing_time_per_kg=timedelta(hours=2.5),
                setup_time=timedelta(hours=5),
                efficiency_factor=0.70
            )
        }
    
    def _get_current_capacity_status(self) -> ISSCapacityStatus:
        """Get current ISS capacity status (would be from real-time data in production)."""
        # Simulated capacity status - in production this would come from ISS operations
        return ISSCapacityStatus(
            current_utilization=0.65,
            available_capacity_kg_per_day=150.0,
            scheduled_maintenance=[
                (datetime(2025, 11, 15), datetime(2025, 11, 18)),
                (datetime(2025, 12, 20), datetime(2025, 12, 25))
            ],
            current_queue_size=3,
            estimated_queue_time=timedelta(days=7)
        )
    
    def calculate_processing_cost(self, material_type: str, mass_kg: float) -> float:
        """
        Calculate processing cost for a specific material type and mass.
        
        Args:
            material_type: Type of material to process
            mass_kg: Mass of material in kilograms
            
        Returns:
            Processing cost in USD
        """
        try:
            mat_type = MaterialType(material_type.lower())
        except ValueError:
            # Default to other_metals for unknown materials
            mat_type = MaterialType.OTHER_METALS
        
        if mat_type not in self.processing_capabilities:
            raise ValueError(f"Processing capability not available for {material_type}")
        
        capability = self.processing_capabilities[mat_type]
        
        # Base processing cost
        processing_cost = capability.calculate_processing_cost(mass_kg)
        
        # Transport cost to ISS
        transport_cost = mass_kg * self.transport_cost_per_kg
        
        # Operational overhead based on processing time
        processing_time = capability.calculate_processing_time(mass_kg)
        operational_days = max(1, processing_time.total_seconds() / 86400)
        operational_cost = operational_days * self.base_operational_cost
        
        return processing_cost + transport_cost + operational_cost
    
    def estimate_processing_time(self, material_type: str, mass_kg: float) -> timedelta:
        """
        Estimate processing time for a specific material type and mass.
        
        Args:
            material_type: Type of material to process
            mass_kg: Mass of material in kilograms
            
        Returns:
            Estimated processing time
        """
        try:
            mat_type = MaterialType(material_type.lower())
        except ValueError:
            mat_type = MaterialType.OTHER_METALS
        
        if mat_type not in self.processing_capabilities:
            raise ValueError(f"Processing capability not available for {material_type}")
        
        capability = self.processing_capabilities[mat_type]
        return capability.calculate_processing_time(mass_kg)
    
    def check_capacity_availability(self, start_time: datetime, duration: timedelta) -> bool:
        """
        Check if ISS has available capacity for processing during the specified time.
        
        Args:
            start_time: Desired start time for processing
            duration: Required processing duration
            
        Returns:
            True if capacity is available, False otherwise
        """
        return self.capacity_status.is_available(start_time, duration)
    
    def get_next_available_slot(self, required_duration: timedelta) -> datetime:
        """
        Get the next available time slot for processing.
        
        Args:
            required_duration: Required processing duration
            
        Returns:
            Next available start time
        """
        return self.capacity_status.get_next_available_slot(required_duration)
    
    def calculate_satellite_processing_cost(self, satellite: Satellite) -> DetailedCost:
        """
        Calculate detailed processing cost for an entire satellite.
        
        Args:
            satellite: Satellite object with material composition
            
        Returns:
            Detailed cost breakdown
        """
        total_processing_cost = 0.0
        total_transport_cost = 0.0
        total_operational_cost = 0.0
        
        for material_type, percentage in satellite.material_composition.items():
            material_mass = satellite.mass * percentage
            
            # Calculate cost for this material
            material_cost = self.calculate_processing_cost(material_type, material_mass)
            
            # Break down the cost components
            try:
                mat_type = MaterialType(material_type.lower())
            except ValueError:
                mat_type = MaterialType.OTHER_METALS
            
            capability = self.processing_capabilities[mat_type]
            
            processing_cost = capability.calculate_processing_cost(material_mass)
            transport_cost = material_mass * self.transport_cost_per_kg
            
            processing_time = capability.calculate_processing_time(material_mass)
            operational_days = max(1, processing_time.total_seconds() / 86400)
            operational_cost = operational_days * self.base_operational_cost
            
            total_processing_cost += processing_cost
            total_transport_cost += transport_cost
            total_operational_cost += operational_cost
        
        total_cost = total_processing_cost + total_transport_cost + total_operational_cost
        
        return DetailedCost(
            propellant_cost=0.0,  # No propellant cost for ISS processing
            operational_cost=total_operational_cost,
            processing_cost=total_processing_cost,
            storage_cost=total_transport_cost,  # Using storage_cost for transport
            total_cost=total_cost,
            cost_breakdown={
                'material_processing': total_processing_cost,
                'transport_to_iss': total_transport_cost,
                'operational_overhead': total_operational_cost
            }
        )
    
    def estimate_satellite_processing_time(self, satellite: Satellite) -> timedelta:
        """
        Estimate total processing time for an entire satellite.
        
        Args:
            satellite: Satellite object with material composition
            
        Returns:
            Estimated total processing time
        """
        max_processing_time = timedelta()
        
        for material_type, percentage in satellite.material_composition.items():
            material_mass = satellite.mass * percentage
            processing_time = self.estimate_processing_time(material_type, material_mass)
            
            # Take the maximum time (assuming parallel processing where possible)
            if processing_time > max_processing_time:
                max_processing_time = processing_time
        
        return max_processing_time
    
    def get_processing_quote(self, satellites: List[Satellite], 
                           preferred_start_time: Optional[datetime] = None) -> Dict:
        """
        Generate a comprehensive processing quote for multiple satellites.
        
        Args:
            satellites: List of satellites to process
            preferred_start_time: Preferred start time (defaults to next available)
            
        Returns:
            Comprehensive quote with costs, timeline, and availability
        """
        total_cost = 0.0
        total_mass = 0.0
        max_processing_time = timedelta()
        detailed_costs = []
        
        for satellite in satellites:
            satellite_cost = self.calculate_satellite_processing_cost(satellite)
            satellite_time = self.estimate_satellite_processing_time(satellite)
            
            total_cost += satellite_cost.total_cost
            total_mass += satellite.mass
            
            if satellite_time > max_processing_time:
                max_processing_time = satellite_time
            
            detailed_costs.append({
                'satellite_id': satellite.id,
                'satellite_name': satellite.name,
                'mass_kg': satellite.mass,
                'processing_cost': satellite_cost.total_cost,
                'processing_time_hours': satellite_time.total_seconds() / 3600,
                'cost_breakdown': satellite_cost.cost_breakdown
            })
        
        # Determine start time
        if preferred_start_time is None:
            start_time = self.get_next_available_slot(max_processing_time)
        else:
            if self.check_capacity_availability(preferred_start_time, max_processing_time):
                start_time = preferred_start_time
            else:
                start_time = self.get_next_available_slot(max_processing_time)
        
        completion_time = start_time + max_processing_time
        
        return {
            'quote_id': f"ISS-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            'service_type': 'ISS Recycling',
            'satellite_count': len(satellites),
            'total_mass_kg': total_mass,
            'total_cost_usd': total_cost,
            'cost_per_kg_usd': total_cost / total_mass if total_mass > 0 else 0,
            'estimated_start_time': start_time.isoformat(),
            'estimated_completion_time': completion_time.isoformat(),
            'processing_duration_hours': max_processing_time.total_seconds() / 3600,
            'capacity_available': self.check_capacity_availability(start_time, max_processing_time),
            'detailed_satellite_costs': detailed_costs,
            'service_capabilities': {
                'max_daily_throughput_kg': self.capacity_status.available_capacity_kg_per_day,
                'current_queue_time_days': self.capacity_status.estimated_queue_time.total_seconds() / 86400,
                'supported_materials': [mat.value for mat in MaterialType]
            }
        }