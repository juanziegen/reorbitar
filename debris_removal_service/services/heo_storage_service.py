"""
HEO Storage Management Service Module

Handles orbital storage calculations, cost optimization, retrieval planning,
and inventory management for High Earth Orbit material storage operations.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math

from ..models.satellite import Satellite, OrbitalElements
from ..models.cost import DetailedCost, MissionCost


class StorageOrbitType(Enum):
    """Types of HEO storage orbits."""
    GEO_GRAVEYARD = "geo_graveyard"  # Above GEO
    SUPER_GEO = "super_geo"  # High circular orbit
    ELLIPTICAL_HEO = "elliptical_heo"  # Highly elliptical orbit
    LAGRANGE_L4 = "lagrange_l4"  # Earth-Moon L4
    LAGRANGE_L5 = "lagrange_l5"  # Earth-Moon L5


class StorageContainerType(Enum):
    """Types of storage containers for different materials."""
    STANDARD_MESH = "standard_mesh"  # For solid materials
    PRESSURIZED_TANK = "pressurized_tank"  # For gases/liquids
    SHIELDED_CONTAINER = "shielded_container"  # For electronics/sensitive materials
    BULK_CARGO_NET = "bulk_cargo_net"  # For large structural components
    TEMPERATURE_CONTROLLED = "temperature_controlled"  # For materials requiring thermal control


@dataclass
class StorageOrbit:
    """Orbital parameters for a storage location."""
    orbit_type: StorageOrbitType
    altitude_km: float
    eccentricity: float
    inclination_deg: float
    orbital_period_hours: float
    delta_v_from_leo_ms: float  # Delta-v required from LEO
    delta_v_to_leo_ms: float  # Delta-v required to return to LEO
    storage_capacity_kg: float  # Maximum storage capacity
    current_utilization: float  # 0.0 to 1.0
    
    def get_available_capacity(self) -> float:
        """Get available storage capacity in kg."""
        return self.storage_capacity_kg * (1.0 - self.current_utilization)
    
    def calculate_orbital_energy(self) -> float:
        """Calculate specific orbital energy (MJ/kg)."""
        # Simplified calculation: E = -GM/(2a) where a is semi-major axis
        GM = 398600.4418  # km^3/s^2
        earth_radius = 6371.0  # km
        semi_major_axis = earth_radius + self.altitude_km
        return -GM / (2 * semi_major_axis) / 1000  # Convert to MJ/kg
    
    def is_stable_orbit(self) -> bool:
        """Check if orbit is stable for long-term storage."""
        # Basic stability checks
        if self.altitude_km < 35786:  # Below GEO
            return False
        if self.eccentricity > 0.7:  # Too eccentric
            return False
        return True


@dataclass
class StorageContainer:
    """Storage container for materials in HEO."""
    container_id: str
    container_type: StorageContainerType
    capacity_kg: float
    current_mass_kg: float
    materials_stored: Dict[str, float]  # material_type -> mass_kg
    deployment_date: datetime
    retrieval_date: Optional[datetime]
    orbit_location: StorageOrbit
    maintenance_required: bool = False
    
    def get_utilization(self) -> float:
        """Get container utilization percentage."""
        return self.current_mass_kg / self.capacity_kg if self.capacity_kg > 0 else 0.0
    
    def get_available_capacity(self) -> float:
        """Get available capacity in kg."""
        return self.capacity_kg - self.current_mass_kg
    
    def can_store_material(self, material_type: str, mass_kg: float) -> bool:
        """Check if container can store additional material."""
        if mass_kg > self.get_available_capacity():
            return False
        
        # Check material compatibility with container type
        compatible_materials = {
            StorageContainerType.STANDARD_MESH: ["aluminum", "titanium", "steel", "other_metals"],
            StorageContainerType.PRESSURIZED_TANK: ["propellant_tanks", "batteries"],
            StorageContainerType.SHIELDED_CONTAINER: ["electronics", "solar_panels"],
            StorageContainerType.BULK_CARGO_NET: ["composites", "carbon_fiber"],
            StorageContainerType.TEMPERATURE_CONTROLLED: ["electronics", "batteries", "solar_panels"]
        }
        
        return material_type in compatible_materials.get(self.container_type, [])
    
    def add_material(self, material_type: str, mass_kg: float) -> bool:
        """Add material to container if possible."""
        if not self.can_store_material(material_type, mass_kg):
            return False
        
        if material_type in self.materials_stored:
            self.materials_stored[material_type] += mass_kg
        else:
            self.materials_stored[material_type] = mass_kg
        
        self.current_mass_kg += mass_kg
        return True


@dataclass
class StorageAllocation:
    """Allocation plan for storing materials."""
    satellite_id: str
    total_mass_kg: float
    storage_containers: List[StorageContainer]
    total_storage_cost: float
    deployment_delta_v: float
    retrieval_delta_v: float
    storage_duration: timedelta
    
    def get_total_containers(self) -> int:
        """Get total number of containers required."""
        return len(self.storage_containers)
    
    def get_storage_efficiency(self) -> float:
        """Get storage efficiency (utilized capacity / total capacity)."""
        total_capacity = sum(container.capacity_kg for container in self.storage_containers)
        return self.total_mass_kg / total_capacity if total_capacity > 0 else 0.0


class HEOStorageService:
    """
    High Earth Orbit Storage Management Service.
    
    Provides orbital storage calculations, cost optimization, retrieval planning,
    and inventory management for long-term material storage in HEO.
    """
    
    def __init__(self):
        """Initialize HEO storage service with orbital configurations."""
        self.storage_orbits = self._initialize_storage_orbits()
        self.container_types = self._initialize_container_types()
        self.active_containers = []  # List of deployed containers
        self.base_operational_cost = 2000.0  # USD per day per container
        self.deployment_cost_per_kg = 3500.0  # USD per kg to deploy to HEO
        self.retrieval_cost_per_kg = 4200.0  # USD per kg to retrieve from HEO
    
    def _initialize_storage_orbits(self) -> Dict[StorageOrbitType, StorageOrbit]:
        """Initialize available storage orbit configurations."""
        return {
            StorageOrbitType.GEO_GRAVEYARD: StorageOrbit(
                orbit_type=StorageOrbitType.GEO_GRAVEYARD,
                altitude_km=36086,  # 300 km above GEO
                eccentricity=0.01,
                inclination_deg=0.1,
                orbital_period_hours=24.2,
                delta_v_from_leo_ms=3900.0,
                delta_v_to_leo_ms=3900.0,
                storage_capacity_kg=50000.0,
                current_utilization=0.25
            ),
            StorageOrbitType.SUPER_GEO: StorageOrbit(
                orbit_type=StorageOrbitType.SUPER_GEO,
                altitude_km=50000,
                eccentricity=0.02,
                inclination_deg=0.0,
                orbital_period_hours=36.8,
                delta_v_from_leo_ms=4500.0,
                delta_v_to_leo_ms=4500.0,
                storage_capacity_kg=100000.0,
                current_utilization=0.15
            ),
            StorageOrbitType.ELLIPTICAL_HEO: StorageOrbit(
                orbit_type=StorageOrbitType.ELLIPTICAL_HEO,
                altitude_km=25000,  # Apogee altitude
                eccentricity=0.4,
                inclination_deg=28.5,
                orbital_period_hours=18.5,
                delta_v_from_leo_ms=3200.0,
                delta_v_to_leo_ms=3200.0,
                storage_capacity_kg=75000.0,
                current_utilization=0.35
            ),
            StorageOrbitType.LAGRANGE_L4: StorageOrbit(
                orbit_type=StorageOrbitType.LAGRANGE_L4,
                altitude_km=384400,  # Distance to L4 point
                eccentricity=0.0,
                inclination_deg=5.0,
                orbital_period_hours=672.0,  # 28 days
                delta_v_from_leo_ms=8500.0,
                delta_v_to_leo_ms=8500.0,
                storage_capacity_kg=500000.0,
                current_utilization=0.05
            ),
            StorageOrbitType.LAGRANGE_L5: StorageOrbit(
                orbit_type=StorageOrbitType.LAGRANGE_L5,
                altitude_km=384400,  # Distance to L5 point
                eccentricity=0.0,
                inclination_deg=5.0,
                orbital_period_hours=672.0,  # 28 days
                delta_v_from_leo_ms=8500.0,
                delta_v_to_leo_ms=8500.0,
                storage_capacity_kg=500000.0,
                current_utilization=0.03
            )
        }
    
    def _initialize_container_types(self) -> Dict[StorageContainerType, Dict]:
        """Initialize storage container specifications."""
        return {
            StorageContainerType.STANDARD_MESH: {
                'capacity_kg': 1000.0,
                'cost_per_container': 25000.0,
                'maintenance_interval_days': 365,
                'compatible_materials': ["aluminum", "titanium", "steel", "other_metals"]
            },
            StorageContainerType.PRESSURIZED_TANK: {
                'capacity_kg': 500.0,
                'cost_per_container': 45000.0,
                'maintenance_interval_days': 180,
                'compatible_materials': ["propellant_tanks", "batteries"]
            },
            StorageContainerType.SHIELDED_CONTAINER: {
                'capacity_kg': 200.0,
                'cost_per_container': 85000.0,
                'maintenance_interval_days': 90,
                'compatible_materials': ["electronics", "solar_panels"]
            },
            StorageContainerType.BULK_CARGO_NET: {
                'capacity_kg': 2000.0,
                'cost_per_container': 15000.0,
                'maintenance_interval_days': 730,
                'compatible_materials': ["composites", "carbon_fiber"]
            },
            StorageContainerType.TEMPERATURE_CONTROLLED: {
                'capacity_kg': 300.0,
                'cost_per_container': 120000.0,
                'maintenance_interval_days': 60,
                'compatible_materials': ["electronics", "batteries", "solar_panels"]
            }
        }
    
    def calculate_storage_orbit(self, mass_kg: float, duration: timedelta, 
                              preferred_orbit: Optional[StorageOrbitType] = None) -> StorageOrbit:
        """
        Calculate optimal storage orbit for given mass and duration.
        
        Args:
            mass_kg: Mass of materials to store
            duration: Storage duration
            preferred_orbit: Preferred orbit type (optional)
            
        Returns:
            Optimal storage orbit
        """
        if preferred_orbit and preferred_orbit in self.storage_orbits:
            orbit = self.storage_orbits[preferred_orbit]
            if orbit.get_available_capacity() >= mass_kg:
                return orbit
        
        # Find best orbit based on cost-efficiency and capacity
        suitable_orbits = []
        
        for orbit_type, orbit in self.storage_orbits.items():
            if orbit.get_available_capacity() >= mass_kg and orbit.is_stable_orbit():
                # Calculate cost-efficiency score
                storage_days = duration.total_seconds() / 86400
                deployment_cost = mass_kg * self.deployment_cost_per_kg
                daily_cost = mass_kg * self.base_operational_cost / 1000  # Scale by mass
                total_cost = deployment_cost + (daily_cost * storage_days)
                
                # Lower delta-v and cost is better
                efficiency_score = 1.0 / (orbit.delta_v_from_leo_ms + total_cost / 1000)
                
                suitable_orbits.append((orbit, efficiency_score))
        
        if not suitable_orbits:
            # Return largest capacity orbit as fallback
            return max(self.storage_orbits.values(), key=lambda o: o.storage_capacity_kg)
        
        # Return most efficient orbit
        return max(suitable_orbits, key=lambda x: x[1])[0]
    
    def calculate_storage_cost(self, mass_kg: float, duration: timedelta, 
                             orbit: Optional[StorageOrbit] = None) -> float:
        """
        Calculate storage cost for given mass and duration.
        
        Args:
            mass_kg: Mass of materials to store
            duration: Storage duration
            orbit: Storage orbit (calculated if not provided)
            
        Returns:
            Total storage cost in USD
        """
        if orbit is None:
            orbit = self.calculate_storage_orbit(mass_kg, duration)
        
        # Deployment cost
        deployment_cost = mass_kg * self.deployment_cost_per_kg
        
        # Operational cost
        storage_days = duration.total_seconds() / 86400
        daily_operational_cost = mass_kg * self.base_operational_cost / 1000  # Scale by mass
        operational_cost = daily_operational_cost * storage_days
        
        # Container cost (amortized over storage period)
        containers_needed = self._calculate_containers_needed(mass_kg, {})
        container_cost = sum(
            self.container_types[container_type]['cost_per_container'] * count
            for container_type, count in containers_needed.items()
        )
        
        # Orbit-specific cost multiplier
        orbit_multipliers = {
            StorageOrbitType.GEO_GRAVEYARD: 1.0,
            StorageOrbitType.SUPER_GEO: 1.3,
            StorageOrbitType.ELLIPTICAL_HEO: 0.8,
            StorageOrbitType.LAGRANGE_L4: 2.5,
            StorageOrbitType.LAGRANGE_L5: 2.5
        }
        
        multiplier = orbit_multipliers.get(orbit.orbit_type, 1.0)
        
        return (deployment_cost + operational_cost + container_cost) * multiplier
    
    def calculate_retrieval_cost(self, storage_location: StorageOrbit, mass_kg: float) -> float:
        """
        Calculate cost to retrieve materials from storage.
        
        Args:
            storage_location: Storage orbit location
            mass_kg: Mass of materials to retrieve
            
        Returns:
            Retrieval cost in USD
        """
        # Base retrieval cost
        retrieval_cost = mass_kg * self.retrieval_cost_per_kg
        
        # Delta-v cost component
        delta_v_cost = storage_location.delta_v_to_leo_ms * mass_kg * 1.27  # $1.27 per m/s per kg
        
        # Operational cost for retrieval mission
        retrieval_duration_days = 7.0  # Typical retrieval mission duration
        operational_cost = retrieval_duration_days * self.base_operational_cost
        
        return retrieval_cost + delta_v_cost + operational_cost
    
    def _calculate_containers_needed(self, total_mass_kg: float, 
                                   material_composition: Dict[str, float]) -> Dict[StorageContainerType, int]:
        """Calculate number of containers needed for given materials."""
        containers_needed = {}
        
        if not material_composition:
            # Default allocation for unknown composition
            containers_needed[StorageContainerType.STANDARD_MESH] = math.ceil(total_mass_kg / 1000.0)
            return containers_needed
        
        for material_type, percentage in material_composition.items():
            material_mass = total_mass_kg * percentage
            
            # Find appropriate container type
            suitable_container = None
            for container_type, specs in self.container_types.items():
                if material_type in specs['compatible_materials']:
                    suitable_container = container_type
                    break
            
            if suitable_container is None:
                suitable_container = StorageContainerType.STANDARD_MESH  # Default
            
            # Calculate containers needed for this material
            container_capacity = self.container_types[suitable_container]['capacity_kg']
            containers_for_material = math.ceil(material_mass / container_capacity)
            
            if suitable_container in containers_needed:
                containers_needed[suitable_container] += containers_for_material
            else:
                containers_needed[suitable_container] = containers_for_material
        
        return containers_needed
    
    def optimize_storage_allocation(self, satellites: List[Satellite], 
                                  storage_duration: timedelta) -> List[StorageAllocation]:
        """
        Optimize storage allocation for multiple satellites.
        
        Args:
            satellites: List of satellites to store
            storage_duration: Desired storage duration
            
        Returns:
            List of optimized storage allocations
        """
        allocations = []
        
        for satellite in satellites:
            # Calculate optimal orbit
            optimal_orbit = self.calculate_storage_orbit(satellite.mass, storage_duration)
            
            # Calculate storage cost
            storage_cost = self.calculate_storage_cost(satellite.mass, storage_duration, optimal_orbit)
            
            # Calculate containers needed
            containers_needed = self._calculate_containers_needed(
                satellite.mass, satellite.material_composition
            )
            
            # Create storage containers
            storage_containers = []
            container_id_counter = 1
            
            for container_type, count in containers_needed.items():
                for _ in range(count):
                    container = StorageContainer(
                        container_id=f"{satellite.id}-{container_type.value}-{container_id_counter}",
                        container_type=container_type,
                        capacity_kg=self.container_types[container_type]['capacity_kg'],
                        current_mass_kg=0.0,
                        materials_stored={},
                        deployment_date=datetime.utcnow(),
                        retrieval_date=None,
                        orbit_location=optimal_orbit
                    )
                    storage_containers.append(container)
                    container_id_counter += 1
            
            # Allocate materials to containers
            remaining_materials = satellite.material_composition.copy()
            for material_type, percentage in remaining_materials.items():
                material_mass = satellite.mass * percentage
                
                for container in storage_containers:
                    if container.can_store_material(material_type, material_mass):
                        container.add_material(material_type, material_mass)
                        break
            
            allocation = StorageAllocation(
                satellite_id=satellite.id,
                total_mass_kg=satellite.mass,
                storage_containers=storage_containers,
                total_storage_cost=storage_cost,
                deployment_delta_v=optimal_orbit.delta_v_from_leo_ms,
                retrieval_delta_v=optimal_orbit.delta_v_to_leo_ms,
                storage_duration=storage_duration
            )
            
            allocations.append(allocation)
        
        return allocations
    
    def get_storage_quote(self, satellites: List[Satellite], 
                         storage_duration: timedelta,
                         preferred_orbit: Optional[StorageOrbitType] = None) -> Dict:
        """
        Generate comprehensive storage quote for multiple satellites.
        
        Args:
            satellites: List of satellites to store
            storage_duration: Desired storage duration
            preferred_orbit: Preferred orbit type (optional)
            
        Returns:
            Comprehensive storage quote with costs and logistics
        """
        allocations = self.optimize_storage_allocation(satellites, storage_duration)
        
        total_cost = sum(allocation.total_storage_cost for allocation in allocations)
        total_mass = sum(allocation.total_mass_kg for allocation in allocations)
        total_containers = sum(allocation.get_total_containers() for allocation in allocations)
        
        # Calculate retrieval costs
        total_retrieval_cost = 0.0
        for allocation in allocations:
            if allocation.storage_containers:
                orbit = allocation.storage_containers[0].orbit_location
                retrieval_cost = self.calculate_retrieval_cost(orbit, allocation.total_mass_kg)
                total_retrieval_cost += retrieval_cost
        
        deployment_date = datetime.utcnow() + timedelta(days=30)  # 30 days lead time
        retrieval_date = deployment_date + storage_duration
        
        return {
            'quote_id': f"HEO-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            'service_type': 'HEO Storage',
            'satellite_count': len(satellites),
            'total_mass_kg': total_mass,
            'storage_duration_days': storage_duration.total_seconds() / 86400,
            'total_storage_cost_usd': total_cost,
            'total_retrieval_cost_usd': total_retrieval_cost,
            'total_lifecycle_cost_usd': total_cost + total_retrieval_cost,
            'cost_per_kg_per_day_usd': total_cost / (total_mass * storage_duration.total_seconds() / 86400) if total_mass > 0 else 0,
            'estimated_deployment_date': deployment_date.isoformat(),
            'estimated_retrieval_date': retrieval_date.isoformat(),
            'total_containers_required': total_containers,
            'storage_allocations': [
                {
                    'satellite_id': allocation.satellite_id,
                    'mass_kg': allocation.total_mass_kg,
                    'storage_cost_usd': allocation.total_storage_cost,
                    'containers_required': allocation.get_total_containers(),
                    'storage_efficiency': allocation.get_storage_efficiency(),
                    'orbit_type': allocation.storage_containers[0].orbit_location.orbit_type.value if allocation.storage_containers else None,
                    'deployment_delta_v_ms': allocation.deployment_delta_v,
                    'retrieval_delta_v_ms': allocation.retrieval_delta_v
                } for allocation in allocations
            ],
            'service_capabilities': {
                'available_orbits': [orbit.value for orbit in StorageOrbitType],
                'container_types': [container.value for container in StorageContainerType],
                'max_storage_capacity_kg': sum(orbit.storage_capacity_kg for orbit in self.storage_orbits.values()),
                'current_utilization': sum(orbit.current_utilization * orbit.storage_capacity_kg for orbit in self.storage_orbits.values()) / sum(orbit.storage_capacity_kg for orbit in self.storage_orbits.values())
            }
        }