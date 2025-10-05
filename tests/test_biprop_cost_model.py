"""
Test suite for BipropCostModel implementation.
"""

import pytest
import math
import csv
from debris_removal_service.models.biprop_cost_model import BipropCostModel, BipropParameters
from debris_removal_service.models.cost import PropellantMass, DetailedCost, MissionCost


class TestBipropCostModel:
    """Test cases for BipropCostModel class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = BipropCostModel()
        self.custom_params = BipropParameters(
            isp=300.0,
            dry_mass=10.0,
            fuel_cost_per_kg=150.0,
            oxidizer_cost_per_kg=250.0,
            o_f_ratio=2.0
        )
        self.custom_model = BipropCostModel(self.custom_params)
    
    def test_initialization(self):
        """Test model initialization."""
        # Default parameters
        assert self.model.params.isp == 320.0
        assert self.model.params.dry_mass == 14.35
        assert self.model.params.o_f_ratio == 1.9
        
        # Custom parameters
        assert self.custom_model.params.isp == 300.0
        assert self.custom_model.params.dry_mass == 10.0
        assert self.custom_model.params.o_f_ratio == 2.0
    
    def test_invalid_parameters(self):
        """Test validation of invalid parameters."""
        with pytest.raises(ValueError, match="Specific impulse must be positive"):
            BipropCostModel(BipropParameters(isp=-100))
        
        with pytest.raises(ValueError, match="Dry mass must be positive"):
            BipropCostModel(BipropParameters(dry_mass=-5))
        
        with pytest.raises(ValueError, match="O/F ratio must be positive"):
            BipropCostModel(BipropParameters(o_f_ratio=-1))
    
    def test_delta_v_cost_calculation(self):
        """Test basic delta-v cost calculation."""
        # Test the $1.27 per m/s formula
        assert self.model.calculate_delta_v_cost(1.0) == pytest.approx(1.27, rel=1e-6)
        assert self.model.calculate_delta_v_cost(10.0) == pytest.approx(12.7, rel=1e-6)
        assert self.model.calculate_delta_v_cost(100.0) == pytest.approx(127.0, rel=1e-6)
        
        # Test zero delta-v
        assert self.model.calculate_delta_v_cost(0.0) == 0.0
        
        # Test negative delta-v
        with pytest.raises(ValueError, match="Delta-v cannot be negative"):
            self.model.calculate_delta_v_cost(-10.0)
    
    def test_propellant_requirements(self):
        """Test propellant mass calculations using rocket equation."""
        # Test zero delta-v
        prop_mass = self.model.get_propellant_requirements(0.0)
        assert prop_mass.fuel_kg == 0.0
        assert prop_mass.oxidizer_kg == 0.0
        assert prop_mass.total_kg == 0.0
        
        # Test small delta-v (1 m/s)
        prop_mass = self.model.get_propellant_requirements(1.0)
        assert prop_mass.fuel_kg > 0
        assert prop_mass.oxidizer_kg > 0
        assert prop_mass.total_kg > 0
        
        # Verify O/F ratio
        expected_ratio = self.model.params.o_f_ratio
        actual_ratio = prop_mass.oxidizer_kg / prop_mass.fuel_kg
        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-6)
        
        # Verify total mass
        assert prop_mass.total_kg == pytest.approx(
            prop_mass.fuel_kg + prop_mass.oxidizer_kg, rel=1e-6
        )
    
    def test_rocket_equation_accuracy(self):
        """Test accuracy of rocket equation implementation."""
        delta_v = 100.0  # m/s
        prop_mass = self.model.get_propellant_requirements(delta_v)
        
        # Calculate mass ratio
        initial_mass = self.model.params.dry_mass + prop_mass.total_kg
        mass_ratio = initial_mass / self.model.params.dry_mass
        
        # Verify rocket equation: Δv = Isp * g0 * ln(mass_ratio)
        ve = self.model.params.isp * self.model.G0
        calculated_delta_v = ve * math.log(mass_ratio)
        
        assert calculated_delta_v == pytest.approx(delta_v, rel=1e-6)
    
    def test_propellant_cost_calculation(self):
        """Test propellant cost calculation."""
        prop_mass = PropellantMass(fuel_kg=10.0, oxidizer_kg=19.0, total_kg=29.0)
        
        expected_cost = (10.0 * self.model.params.fuel_cost_per_kg + 
                        19.0 * self.model.params.oxidizer_cost_per_kg)
        actual_cost = self.model.calculate_propellant_cost(prop_mass)
        
        assert actual_cost == pytest.approx(expected_cost, rel=1e-6)
    
    def test_operational_overhead(self):
        """Test operational overhead calculation."""
        base_cost = 1000.0
        overhead = self.model.calculate_operational_overhead(base_cost)
        
        expected_overhead = base_cost * self.model.params.operational_overhead_factor
        assert overhead == pytest.approx(expected_overhead, rel=1e-6)
    
    def test_detailed_cost_calculation(self):
        """Test detailed cost breakdown."""
        delta_v = 50.0
        processing_cost = 100.0
        storage_cost = 50.0
        
        detailed_cost = self.model.calculate_detailed_cost(
            delta_v, processing_cost, storage_cost
        )
        
        # Verify cost components
        assert detailed_cost.propellant_cost > 0
        assert detailed_cost.operational_cost > 0
        assert detailed_cost.processing_cost == processing_cost
        assert detailed_cost.storage_cost == storage_cost
        
        # Verify total cost
        expected_total = (detailed_cost.propellant_cost + 
                         detailed_cost.operational_cost +
                         processing_cost + storage_cost)
        assert detailed_cost.total_cost == pytest.approx(expected_total, rel=1e-6)
    
    def test_mission_cost_calculation(self):
        """Test complete mission cost calculation."""
        delta_v = 100.0
        satellites_collected = 5
        processing_cost = 200.0
        storage_cost = 100.0
        
        mission_cost = self.model.calculate_mission_cost(
            delta_v, satellites_collected, processing_cost, storage_cost
        )
        
        # Verify cost components
        assert mission_cost.collection_cost > 0
        assert mission_cost.processing_cost == processing_cost
        assert mission_cost.storage_cost == storage_cost
        assert mission_cost.operational_overhead > 0
        assert mission_cost.total_cost > 0
        assert mission_cost.cost_per_satellite > 0
        
        # Verify cost per satellite
        expected_cost_per_satellite = mission_cost.total_cost / satellites_collected
        assert mission_cost.cost_per_satellite == pytest.approx(
            expected_cost_per_satellite, rel=1e-6
        )
        
        # Verify propellant mass is included
        assert mission_cost.propellant_mass is not None
        assert mission_cost.propellant_mass.total_kg > 0
        
        # Test invalid satellites count
        with pytest.raises(ValueError, match="Number of satellites collected must be positive"):
            self.model.calculate_mission_cost(delta_v, 0)
    
    def test_csv_validation(self):
        """Test validation against CSV reference data."""
        # Test a few known values from the CSV
        test_cases = [
            (1.0, 1.27),
            (10.0, 12.7),
            (50.0, 63.5),
            (100.0, 127.0)
        ]
        
        for delta_v, expected_cost in test_cases:
            assert self.model.validate_against_csv_data(delta_v, expected_cost, tolerance=0.01)
            
            # Test with tight tolerance should also pass for exact formula
            calculated_cost = self.model.calculate_delta_v_cost(delta_v)
            assert calculated_cost == pytest.approx(expected_cost, rel=1e-6)
    
    def test_cost_optimization_suggestions(self):
        """Test cost optimization suggestions."""
        # High delta-v mission
        suggestions = self.model.get_cost_optimization_suggestions(1500.0)
        assert 'high_delta_v' in suggestions
        
        # Low delta-v mission
        suggestions = self.model.get_cost_optimization_suggestions(10.0)
        assert 'high_delta_v' not in suggestions
        
        # Suggestions should be strings
        for key, value in suggestions.items():
            assert isinstance(value, str)
            assert len(value) > 0
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        delta_v = 100.0
        metrics = self.model.get_performance_metrics(delta_v)
        
        # Verify required metrics are present
        required_metrics = [
            'mass_ratio', 'propellant_fraction', 'cost_per_ms_usd',
            'cost_per_kg_propellant_usd', 'exhaust_velocity_ms', 'total_propellant_kg'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert metrics[metric] >= 0
        
        # Verify exhaust velocity calculation
        expected_ve = self.model.params.isp * self.model.G0
        assert metrics['exhaust_velocity_ms'] == pytest.approx(expected_ve, rel=1e-6)
        
        # Verify cost per m/s
        expected_cost_per_ms = self.model.calculate_delta_v_cost(delta_v) / delta_v
        assert metrics['cost_per_ms_usd'] == pytest.approx(expected_cost_per_ms, rel=1e-6)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very small delta-v
        small_dv = 0.001
        prop_mass = self.model.get_propellant_requirements(small_dv)
        assert prop_mass.total_kg > 0
        assert prop_mass.fuel_kg > 0
        assert prop_mass.oxidizer_kg > 0
        
        # Large delta-v
        large_dv = 5000.0
        prop_mass = self.model.get_propellant_requirements(large_dv)
        assert prop_mass.total_kg > 0
        
        # Verify mass ratio makes sense for large delta-v
        mass_ratio = (self.model.params.dry_mass + prop_mass.total_kg) / self.model.params.dry_mass
        assert mass_ratio > 1.0


class TestBipropParameters:
    """Test cases for BipropParameters dataclass."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        params = BipropParameters()
        
        assert params.isp == 320.0
        assert params.dry_mass == 14.35
        assert params.fuel_cost_per_kg == 200.0
        assert params.oxidizer_cost_per_kg == 320.0
        assert params.o_f_ratio == 1.9
        assert params.fuel_density == 0.789
        assert params.oxidizer_density == 1.45
        assert params.operational_overhead_factor == 0.15
        assert params.mission_complexity_factor == 1.0
    
    def test_custom_parameters(self):
        """Test custom parameter values."""
        params = BipropParameters(
            isp=350.0,
            dry_mass=20.0,
            fuel_cost_per_kg=180.0,
            oxidizer_cost_per_kg=300.0,
            o_f_ratio=2.2
        )
        
        assert params.isp == 350.0
        assert params.dry_mass == 20.0
        assert params.fuel_cost_per_kg == 180.0
        assert params.oxidizer_cost_per_kg == 300.0
        assert params.o_f_ratio == 2.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])