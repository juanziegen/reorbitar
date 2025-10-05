"""
Solar Forge Processing Service Module

Handles advanced material refinement calculations, cost modeling, and timeline estimation
for satellite debris processing at deep solar forge stations.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

from ..models.satellite import Satellite
from ..models.cost import DetailedCost, MissionCost


class RefinementProcess(Enum):
    """Types of refinement processes available at solar forge stations."""
    METAL_PURIFICATION = "metal_purification"
    ALLOY_CREATION = "alloy_creation"
    CRYSTAL_GROWTH = "crystal_growth"
    COMPOSITE_SYNTHESIS = "composite_synthesis"
    RARE_EARTH_EXTRACTION = "rare_earth_extraction"
    SEMICONDUCTOR_FABRICATION = "semiconductor_fabrication"
    ADVANCED_CERAMICS = "advanced_ceramics"
    NANOSTRUCTURE_ASSEMBLY = "nanostructure_assembly"


class OutputMaterialGrade(Enum):
    """Quality grades for output materials."""
    INDUSTRIAL = "industrial"
    AEROSPACE = "aerospace"
    PRECISION = "precision"
    ULTRA_HIGH_PURITY = "ultra_high_purity"


@dataclass
class RefinedMaterial:
    """Refined material output from solar forge processing."""
    material_name: str
    mass_kg: float
    grade: OutputMaterialGrade
    purity_percentage: float
    market_value_per_kg: float
    processing_method: RefinementProcess
    
    def get_total_value(self) -> float:
        """Calculate total market value of the refined material."""
        return self.mass_kg * self.market_value_per_kg
    
    def get_quality_multiplier(self) -> float:
        """Get quality multiplier based on grade and purity."""
        grade_multipliers = {
            OutputMaterialGrade.INDUSTRIAL: 1.0,
            OutputMaterialGrade.AEROSPACE: 2.5,
            OutputMaterialGrade.PRECISION: 4.0,
            OutputMaterialGrade.ULTRA_HIGH_PURITY: 8.0
        }
        
        base_multiplier = grade_multipliers[self.grade]
        purity_bonus = (self.purity_percentage - 90.0) / 10.0  # Bonus for >90% purity
        return base_multiplier * (1.0 + max(0, purity_bonus))


@dataclass
class RefinementCapability:
    """Solar forge refinement capability for specific processes."""
    process_type: RefinementProcess
    input_materials: List[str]  # Materials that can be processed
    output_materials: List[str]  # Possible output materials
    energy_requirement_kwh_per_kg: float
    processing_time_per_kg: timedelta
    yield_efficiency: float  # 0.0 to 1.0, material yield
    max_batch_size_kg: float
    setup_time: timedelta
    achievable_grades: List[OutputMaterialGrade]
    
    def calculate_energy_cost(self, mass_kg: float, energy_cost_per_kwh: float = 0.05) -> float:
        """Calculate energy cost for processing given mass."""
        total_energy = mass_kg * self.energy_requirement_kwh_per_kg
        return total_energy * energy_cost_per_kwh
    
    def calculate_processing_time(self, mass_kg: float) -> timedelta:
        """Calculate total processing time including setup."""
        batches = max(1, mass_kg / self.max_batch_size_kg)
        batch_time = self.processing_time_per_kg * (mass_kg / batches)
        return self.setup_time + (batch_time * batches)
    
    def calculate_output_mass(self, input_mass_kg: float) -> float:
        """Calculate expected output mass based on yield efficiency."""
        return input_mass_kg * self.yield_efficiency


@dataclass
class ForgeStationStatus:
    """Current operational status of solar forge stations."""
    station_id: str
    location: str  # e.g., "L1 Lagrange Point", "Deep Solar Orbit"
    current_utilization: float  # 0.0 to 1.0
    available_energy_capacity_mw: float
    scheduled_maintenance: List[Tuple[datetime, datetime]]
    current_queue_size: int
    estimated_queue_time: timedelta
    operational_temperature_k: float
    
    def is_operational(self) -> bool:
        """Check if station is currently operational."""
        return (self.current_utilization < 0.95 and 
                self.available_energy_capacity_mw > 0 and
                self.operational_temperature_k > 1000)
    
    def get_next_available_slot(self, required_duration: timedelta) -> datetime:
        """Get next available processing slot."""
        current_time = datetime.utcnow()
        candidate_time = current_time + self.estimated_queue_time
        
        # Check maintenance windows
        for maint_start, maint_end in self.scheduled_maintenance:
            if candidate_time < maint_end and candidate_time + required_duration > maint_start:
                candidate_time = maint_end
        
        return candidate_time


class SolarForgeService:
    """
    Solar Forge Processing Service for advanced material refinement.
    
    Provides sophisticated material processing capabilities using concentrated
    solar energy at deep space forge stations.
    """
    
    def __init__(self):
        """Initialize solar forge service with processing capabilities."""
        self.refinement_capabilities = self._initialize_refinement_capabilities()
        self.forge_stations = self._initialize_forge_stations()
        self.base_operational_cost = 15000.0  # USD per day for deep space operations
        self.transport_cost_per_kg = 8500.0  # USD per kg to deep space forge
        self.energy_cost_per_kwh = 0.05  # USD per kWh (solar energy cost)
    
    def _initialize_refinement_capabilities(self) -> Dict[RefinementProcess, RefinementCapability]:
        """Initialize refinement capabilities for different processes."""
        return {
            RefinementProcess.METAL_PURIFICATION: RefinementCapability(
                process_type=RefinementProcess.METAL_PURIFICATION,
                input_materials=["aluminum", "titanium", "steel", "other_metals"],
                output_materials=["pure_aluminum", "pure_titanium", "pure_steel"],
                energy_requirement_kwh_per_kg=25.0,
                processing_time_per_kg=timedelta(hours=2),
                yield_efficiency=0.92,
                max_batch_size_kg=100.0,
                setup_time=timedelta(hours=6),
                achievable_grades=[OutputMaterialGrade.AEROSPACE, OutputMaterialGrade.PRECISION]
            ),
            RefinementProcess.ALLOY_CREATION: RefinementCapability(
                process_type=RefinementProcess.ALLOY_CREATION,
                input_materials=["aluminum", "titanium", "steel", "other_metals"],
                output_materials=["titanium_alloy", "aluminum_alloy", "superalloy"],
                energy_requirement_kwh_per_kg=45.0,
                processing_time_per_kg=timedelta(hours=4),
                yield_efficiency=0.88,
                max_batch_size_kg=50.0,
                setup_time=timedelta(hours=12),
                achievable_grades=[OutputMaterialGrade.AEROSPACE, OutputMaterialGrade.PRECISION, OutputMaterialGrade.ULTRA_HIGH_PURITY]
            ),
            RefinementProcess.CRYSTAL_GROWTH: RefinementCapability(
                process_type=RefinementProcess.CRYSTAL_GROWTH,
                input_materials=["electronics", "solar_panels"],
                output_materials=["silicon_crystals", "gallium_arsenide", "sapphire"],
                energy_requirement_kwh_per_kg=120.0,
                processing_time_per_kg=timedelta(hours=24),
                yield_efficiency=0.75,
                max_batch_size_kg=10.0,
                setup_time=timedelta(days=2),
                achievable_grades=[OutputMaterialGrade.PRECISION, OutputMaterialGrade.ULTRA_HIGH_PURITY]
            ),
            RefinementProcess.COMPOSITE_SYNTHESIS: RefinementCapability(
                process_type=RefinementProcess.COMPOSITE_SYNTHESIS,
                input_materials=["carbon_fiber", "composites"],
                output_materials=["carbon_nanotube_composite", "graphene_composite"],
                energy_requirement_kwh_per_kg=85.0,
                processing_time_per_kg=timedelta(hours=8),
                yield_efficiency=0.80,
                max_batch_size_kg=25.0,
                setup_time=timedelta(hours=18),
                achievable_grades=[OutputMaterialGrade.AEROSPACE, OutputMaterialGrade.PRECISION]
            ),
            RefinementProcess.RARE_EARTH_EXTRACTION: RefinementCapability(
                process_type=RefinementProcess.RARE_EARTH_EXTRACTION,
                input_materials=["electronics", "batteries", "solar_panels"],
                output_materials=["neodymium", "lithium", "rare_earth_elements"],
                energy_requirement_kwh_per_kg=200.0,
                processing_time_per_kg=timedelta(hours=12),
                yield_efficiency=0.65,
                max_batch_size_kg=15.0,
                setup_time=timedelta(days=1),
                achievable_grades=[OutputMaterialGrade.INDUSTRIAL, OutputMaterialGrade.PRECISION]
            ),
            RefinementProcess.SEMICONDUCTOR_FABRICATION: RefinementCapability(
                process_type=RefinementProcess.SEMICONDUCTOR_FABRICATION,
                input_materials=["electronics", "solar_panels"],
                output_materials=["silicon_wafers", "compound_semiconductors"],
                energy_requirement_kwh_per_kg=300.0,
                processing_time_per_kg=timedelta(days=2),
                yield_efficiency=0.60,
                max_batch_size_kg=5.0,
                setup_time=timedelta(days=3),
                achievable_grades=[OutputMaterialGrade.PRECISION, OutputMaterialGrade.ULTRA_HIGH_PURITY]
            ),
            RefinementProcess.ADVANCED_CERAMICS: RefinementCapability(
                process_type=RefinementProcess.ADVANCED_CERAMICS,
                input_materials=["composites", "other_metals"],
                output_materials=["ultra_high_temp_ceramics", "piezoelectric_ceramics"],
                energy_requirement_kwh_per_kg=150.0,
                processing_time_per_kg=timedelta(hours=16),
                yield_efficiency=0.85,
                max_batch_size_kg=20.0,
                setup_time=timedelta(hours=24),
                achievable_grades=[OutputMaterialGrade.AEROSPACE, OutputMaterialGrade.PRECISION]
            ),
            RefinementProcess.NANOSTRUCTURE_ASSEMBLY: RefinementCapability(
                process_type=RefinementProcess.NANOSTRUCTURE_ASSEMBLY,
                input_materials=["carbon_fiber", "electronics"],
                output_materials=["carbon_nanotubes", "quantum_dots", "metamaterials"],
                energy_requirement_kwh_per_kg=500.0,
                processing_time_per_kg=timedelta(days=3),
                yield_efficiency=0.45,
                max_batch_size_kg=2.0,
                setup_time=timedelta(days=5),
                achievable_grades=[OutputMaterialGrade.ULTRA_HIGH_PURITY]
            )
        }
    
    def _initialize_forge_stations(self) -> Dict[str, ForgeStationStatus]:
        """Initialize forge station status information."""
        return {
            "FORGE-L1": ForgeStationStatus(
                station_id="FORGE-L1",
                location="L1 Lagrange Point",
                current_utilization=0.45,
                available_energy_capacity_mw=500.0,
                scheduled_maintenance=[
                    (datetime(2025, 12, 1), datetime(2025, 12, 5))
                ],
                current_queue_size=2,
                estimated_queue_time=timedelta(days=14),
                operational_temperature_k=2500.0
            ),
            "FORGE-SOLAR": ForgeStationStatus(
                station_id="FORGE-SOLAR",
                location="Deep Solar Orbit (0.3 AU)",
                current_utilization=0.30,
                available_energy_capacity_mw=1200.0,
                scheduled_maintenance=[
                    (datetime(2025, 11, 20), datetime(2025, 11, 25))
                ],
                current_queue_size=1,
                estimated_queue_time=timedelta(days=8),
                operational_temperature_k=3200.0
            )
        }
    
    def calculate_refinement_cost(self, material_type: str, mass_kg: float, 
                                desired_grade: OutputMaterialGrade = OutputMaterialGrade.AEROSPACE) -> float:
        """
        Calculate refinement cost for a specific material type and mass.
        
        Args:
            material_type: Type of material to refine
            mass_kg: Mass of material in kilograms
            desired_grade: Desired output quality grade
            
        Returns:
            Refinement cost in USD
        """
        # Find appropriate refinement process
        suitable_process = None
        for process, capability in self.refinement_capabilities.items():
            if material_type.lower() in capability.input_materials and desired_grade in capability.achievable_grades:
                suitable_process = capability
                break
        
        if not suitable_process:
            # Default to metal purification for unknown materials
            suitable_process = self.refinement_capabilities[RefinementProcess.METAL_PURIFICATION]
        
        # Calculate energy cost
        energy_cost = suitable_process.calculate_energy_cost(mass_kg, self.energy_cost_per_kwh)
        
        # Calculate transport cost
        transport_cost = mass_kg * self.transport_cost_per_kg
        
        # Calculate operational cost based on processing time
        processing_time = suitable_process.calculate_processing_time(mass_kg)
        operational_days = max(1, processing_time.total_seconds() / 86400)
        operational_cost = operational_days * self.base_operational_cost
        
        # Grade premium
        grade_multipliers = {
            OutputMaterialGrade.INDUSTRIAL: 1.0,
            OutputMaterialGrade.AEROSPACE: 1.5,
            OutputMaterialGrade.PRECISION: 2.5,
            OutputMaterialGrade.ULTRA_HIGH_PURITY: 4.0
        }
        grade_premium = (energy_cost + operational_cost) * (grade_multipliers[desired_grade] - 1.0)
        
        return energy_cost + transport_cost + operational_cost + grade_premium
    
    def estimate_refinement_time(self, material_type: str, mass_kg: float,
                               desired_grade: OutputMaterialGrade = OutputMaterialGrade.AEROSPACE) -> timedelta:
        """
        Estimate refinement time for a specific material type and mass.
        
        Args:
            material_type: Type of material to refine
            mass_kg: Mass of material in kilograms
            desired_grade: Desired output quality grade
            
        Returns:
            Estimated refinement time
        """
        # Find appropriate refinement process
        suitable_process = None
        for process, capability in self.refinement_capabilities.items():
            if material_type.lower() in capability.input_materials and desired_grade in capability.achievable_grades:
                suitable_process = capability
                break
        
        if not suitable_process:
            suitable_process = self.refinement_capabilities[RefinementProcess.METAL_PURIFICATION]
        
        base_time = suitable_process.calculate_processing_time(mass_kg)
        
        # Add time premium for higher grades
        grade_time_multipliers = {
            OutputMaterialGrade.INDUSTRIAL: 1.0,
            OutputMaterialGrade.AEROSPACE: 1.2,
            OutputMaterialGrade.PRECISION: 1.8,
            OutputMaterialGrade.ULTRA_HIGH_PURITY: 3.0
        }
        
        return base_time * grade_time_multipliers[desired_grade]
    
    def get_output_materials(self, input_materials: List[Tuple[str, float]], 
                           desired_grade: OutputMaterialGrade = OutputMaterialGrade.AEROSPACE) -> List[RefinedMaterial]:
        """
        Predict output materials from input material composition.
        
        Args:
            input_materials: List of (material_type, mass_kg) tuples
            desired_grade: Desired output quality grade
            
        Returns:
            List of predicted refined materials
        """
        output_materials = []
        
        for material_type, mass_kg in input_materials:
            # Find suitable refinement process
            suitable_process = None
            capability = None
            
            for process, cap in self.refinement_capabilities.items():
                if material_type.lower() in cap.input_materials and desired_grade in cap.achievable_grades:
                    suitable_process = process
                    capability = cap
                    break
            
            if not capability:
                # Skip materials that can't be processed
                continue
            
            # Calculate output mass
            output_mass = capability.calculate_output_mass(mass_kg)
            
            # Determine output material type and market value
            if capability.output_materials:
                output_material_name = capability.output_materials[0]  # Take first option
                
                # Market values based on material type and grade (USD per kg)
                base_values = {
                    "pure_aluminum": 2.5,
                    "pure_titanium": 35.0,
                    "pure_steel": 1.2,
                    "titanium_alloy": 85.0,
                    "aluminum_alloy": 8.0,
                    "superalloy": 150.0,
                    "silicon_crystals": 500.0,
                    "gallium_arsenide": 2000.0,
                    "sapphire": 800.0,
                    "carbon_nanotube_composite": 5000.0,
                    "graphene_composite": 8000.0,
                    "neodymium": 120.0,
                    "lithium": 15.0,
                    "rare_earth_elements": 200.0,
                    "silicon_wafers": 1200.0,
                    "compound_semiconductors": 3000.0,
                    "ultra_high_temp_ceramics": 400.0,
                    "piezoelectric_ceramics": 600.0,
                    "carbon_nanotubes": 15000.0,
                    "quantum_dots": 25000.0,
                    "metamaterials": 50000.0
                }
                
                base_value = base_values.get(output_material_name, 10.0)
                
                # Calculate purity based on process and grade
                base_purity = 85.0 + (capability.yield_efficiency * 10.0)
                grade_purity_bonus = {
                    OutputMaterialGrade.INDUSTRIAL: 0.0,
                    OutputMaterialGrade.AEROSPACE: 5.0,
                    OutputMaterialGrade.PRECISION: 8.0,
                    OutputMaterialGrade.ULTRA_HIGH_PURITY: 12.0
                }
                purity = min(99.9, base_purity + grade_purity_bonus[desired_grade])
                
                refined_material = RefinedMaterial(
                    material_name=output_material_name,
                    mass_kg=output_mass,
                    grade=desired_grade,
                    purity_percentage=purity,
                    market_value_per_kg=base_value * RefinedMaterial(
                        material_name="temp", mass_kg=1, grade=desired_grade, 
                        purity_percentage=purity, market_value_per_kg=1, 
                        processing_method=suitable_process
                    ).get_quality_multiplier(),
                    processing_method=suitable_process
                )
                
                output_materials.append(refined_material)
        
        return output_materials
    
    def calculate_satellite_refinement_cost(self, satellite: Satellite, 
                                          desired_grade: OutputMaterialGrade = OutputMaterialGrade.AEROSPACE) -> DetailedCost:
        """
        Calculate detailed refinement cost for an entire satellite.
        
        Args:
            satellite: Satellite object with material composition
            desired_grade: Desired output quality grade
            
        Returns:
            Detailed cost breakdown
        """
        total_energy_cost = 0.0
        total_transport_cost = 0.0
        total_operational_cost = 0.0
        
        for material_type, percentage in satellite.material_composition.items():
            material_mass = satellite.mass * percentage
            refinement_cost = self.calculate_refinement_cost(material_type, material_mass, desired_grade)
            
            # Break down cost components (approximate)
            transport_cost = material_mass * self.transport_cost_per_kg
            energy_operational_cost = refinement_cost - transport_cost
            
            total_transport_cost += transport_cost
            total_energy_cost += energy_operational_cost * 0.4  # 40% energy
            total_operational_cost += energy_operational_cost * 0.6  # 60% operational
        
        total_cost = total_energy_cost + total_transport_cost + total_operational_cost
        
        return DetailedCost(
            propellant_cost=0.0,  # No propellant cost for solar forge
            operational_cost=total_operational_cost,
            processing_cost=total_energy_cost,
            storage_cost=total_transport_cost,  # Using storage_cost for transport
            total_cost=total_cost,
            cost_breakdown={
                'energy_processing': total_energy_cost,
                'transport_to_forge': total_transport_cost,
                'operational_overhead': total_operational_cost
            }
        )
    
    def get_refinement_quote(self, satellites: List[Satellite], 
                           desired_grade: OutputMaterialGrade = OutputMaterialGrade.AEROSPACE,
                           preferred_station: Optional[str] = None) -> Dict:
        """
        Generate a comprehensive refinement quote for multiple satellites.
        
        Args:
            satellites: List of satellites to refine
            desired_grade: Desired output quality grade
            preferred_station: Preferred forge station ID
            
        Returns:
            Comprehensive quote with costs, timeline, and output predictions
        """
        total_cost = 0.0
        total_mass = 0.0
        max_processing_time = timedelta()
        detailed_costs = []
        all_output_materials = []
        
        for satellite in satellites:
            satellite_cost = self.calculate_satellite_refinement_cost(satellite, desired_grade)
            satellite_time = self.estimate_refinement_time("aluminum", satellite.mass, desired_grade)  # Approximate
            
            # Get predicted output materials
            input_materials = [(mat_type, satellite.mass * percentage) 
                             for mat_type, percentage in satellite.material_composition.items()]
            output_materials = self.get_output_materials(input_materials, desired_grade)
            
            total_cost += satellite_cost.total_cost
            total_mass += satellite.mass
            
            if satellite_time > max_processing_time:
                max_processing_time = satellite_time
            
            detailed_costs.append({
                'satellite_id': satellite.id,
                'satellite_name': satellite.name,
                'mass_kg': satellite.mass,
                'refinement_cost': satellite_cost.total_cost,
                'processing_time_hours': satellite_time.total_seconds() / 3600,
                'cost_breakdown': satellite_cost.cost_breakdown,
                'output_materials': [
                    {
                        'material_name': mat.material_name,
                        'mass_kg': mat.mass_kg,
                        'grade': mat.grade.value,
                        'purity_percent': mat.purity_percentage,
                        'market_value_usd': mat.get_total_value()
                    } for mat in output_materials
                ]
            })
            
            all_output_materials.extend(output_materials)
        
        # Select best available station
        if preferred_station and preferred_station in self.forge_stations:
            station = self.forge_stations[preferred_station]
        else:
            # Select station with shortest queue time
            station = min(self.forge_stations.values(), key=lambda s: s.estimated_queue_time)
        
        start_time = station.get_next_available_slot(max_processing_time)
        completion_time = start_time + max_processing_time
        
        # Calculate total output value
        total_output_value = sum(mat.get_total_value() for mat in all_output_materials)
        
        return {
            'quote_id': f"FORGE-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            'service_type': 'Solar Forge Refinement',
            'selected_station': station.station_id,
            'station_location': station.location,
            'satellite_count': len(satellites),
            'total_input_mass_kg': total_mass,
            'total_refinement_cost_usd': total_cost,
            'cost_per_kg_usd': total_cost / total_mass if total_mass > 0 else 0,
            'estimated_start_time': start_time.isoformat(),
            'estimated_completion_time': completion_time.isoformat(),
            'processing_duration_hours': max_processing_time.total_seconds() / 3600,
            'output_grade': desired_grade.value,
            'total_output_value_usd': total_output_value,
            'value_added_usd': total_output_value - total_cost,
            'roi_percentage': ((total_output_value - total_cost) / total_cost * 100) if total_cost > 0 else 0,
            'detailed_satellite_costs': detailed_costs,
            'service_capabilities': {
                'available_processes': [process.value for process in RefinementProcess],
                'achievable_grades': [grade.value for grade in OutputMaterialGrade],
                'station_energy_capacity_mw': station.available_energy_capacity_mw,
                'queue_time_days': station.estimated_queue_time.total_seconds() / 86400
            }
        }