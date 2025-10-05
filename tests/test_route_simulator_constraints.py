"""
Test suite for RouteSimulator constraint handling and convergence detection.
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any

from debris_removal_service.models.satellite import Satellite, OrbitalElements
from debris_removal_service.models.service_request import (
    ServiceRequest, TimelineConstraints, BudgetConstraints, 
    ProcessingPreferences, ProcessingType
)
from debris_removal_service.services.route_simulator import RouteSimulator


class TestRouteSimulatorConstraints:
    """Test constraint handling in RouteSimulator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.route_simulator = RouteSimulator()
        
        # Create test satellites
        self.test_satellites = self._create_test_satellites()
        
        # Create test service request
        self.test_service_request = self._create_test_service_request()
    
    def _create_test_satellites(self) -> List[Satellite]:
        """Create test satellites with orbital elements."""
        satellites = []
        
        # Create satellites with different orbital characteristics
        for i in range(5):
            orbital_elements = OrbitalElements(
                semi_major_axis=7000 + i * 100,  # km
                eccentricity=0.01 + i * 0.005,
                inclination=45 + i * 10,  # degrees
                raan=i * 30,  # degrees
                argument_of_perigee=i * 45,  # degrees
                mean_anomaly=i * 60,  # degrees
                mean_motion=15.0 - i * 0.5,  # revolutions per day
                epoch=datetime.now()
            )
            
            satellite = Satellite(
                id=str(i + 1),
                name=f"TestSat-{i + 1}",
                tle_line1=f"1 {i+1:05d}U 21001A   21001.00000000  .00000000  00000-0  00000-0 0  999{i}",
                tle_line2=f"2 {i+1:05d}  45.000{i:03d} 000.0000 0010000 000.0000 000.0000 15.0000000{i:05d}{i}",
                mass=500.0 + i * 100,
                material_composition={"aluminum": 0.6, "steel": 0.3, "other": 0.1},
                decommission_date=datetime.now() + timedelta(days=30),
                orbital_elements=orbital_elements
            )
            satellites.append(satellite)
        
        return satellites
    
    def _create_test_service_request(self) -> ServiceRequest:
        """Create test service request."""
        timeline_constraints = TimelineConstraints(
            earliest_start=datetime.now() + timedelta(days=1),
            latest_completion=datetime.now() + timedelta(days=30),
            preferred_duration=timedelta(days=10)
        )
        
        budget_constraints = BudgetConstraints(
            max_total_cost=50000.0,
            preferred_cost=40000.0
        )
        
        processing_preferences = ProcessingPreferences(
            preferred_processing_types=[ProcessingType.ISS_RECYCLING],
            processing_timeline=timedelta(days=5)
        )
        
        return ServiceRequest(
            client_id="test_client",
            satellites=["1", "2", "3"],
            timeline_requirements=timeline_constraints,
            budget_constraints=budget_constraints,
            processing_preferences=processing_preferences
        )
    
    def test_constraint_analysis(self):
        """Test constraint analysis functionality."""
        analysis = self.route_simulator._analyze_constraints(
            self.test_service_request, self.test_satellites
        )
        
        assert isinstance(analysis, dict)
        assert 'feasible' in analysis
        assert 'constraint_violations' in analysis
        assert 'warnings' in analysis
        assert 'recommendations' in analysis
        
        # Should be feasible with reasonable test data
        assert analysis['feasible'] is True
    
    def test_budget_constraint_analysis(self):
        """Test budget constraint analysis."""
        analysis = self.route_simulator._analyze_budget_constraints(
            self.test_service_request, self.test_satellites[:3]
        )
        
        assert 'constraint_violations' in analysis
        assert 'warnings' in analysis
        assert 'recommendations' in analysis
        assert 'estimated_costs' in analysis
        
        estimated_costs = analysis['estimated_costs']
        assert 'collection_cost' in estimated_costs
        assert 'processing_cost' in estimated_costs
        assert 'total_cost' in estimated_costs
        assert 'budget_utilization' in estimated_costs
    
    def test_timeline_constraint_analysis(self):
        """Test timeline constraint analysis."""
        analysis = self.route_simulator._analyze_timeline_constraints(
            self.test_service_request, self.test_satellites[:3]
        )
        
        assert 'constraint_violations' in analysis
        assert 'warnings' in analysis
        assert 'recommendations' in analysis
        assert 'estimated_timeline' in analysis
        
        estimated_timeline = analysis['estimated_timeline']
        assert 'estimated_duration_hours' in estimated_timeline
        assert 'available_duration_hours' in estimated_timeline
        assert 'timeline_utilization' in estimated_timeline
        assert 'is_urgent' in estimated_timeline
    
    def test_orbital_feasibility_analysis(self):
        """Test orbital feasibility analysis."""
        analysis = self.route_simulator._analyze_orbital_feasibility(
            self.test_satellites[:3]
        )
        
        assert 'constraint_violations' in analysis
        assert 'warnings' in analysis
        assert 'recommendations' in analysis
        assert 'orbital_characteristics' in analysis
        
        orbital_chars = analysis['orbital_characteristics']
        assert 'satellites_with_elements' in orbital_chars
        assert 'inclination_range_deg' in orbital_chars
        assert 'altitude_range_km' in orbital_chars
    
    def test_processing_constraint_analysis(self):
        """Test processing constraint analysis."""
        analysis = self.route_simulator._analyze_processing_constraints(
            self.test_service_request, self.test_satellites[:3]
        )
        
        assert 'constraint_violations' in analysis
        assert 'warnings' in analysis
        assert 'recommendations' in analysis
        assert 'processing_characteristics' in analysis
        
        processing_chars = analysis['processing_characteristics']
        assert 'satellites_with_materials' in processing_chars
        assert 'preferred_processing_types' in processing_chars
    
    def test_constrained_optimization_setup(self):
        """Test setup of constrained optimization."""
        ga_config, route_constraints = self.route_simulator._setup_constrained_optimization(
            self.test_service_request, self.test_satellites
        )
        
        assert ga_config is not None
        assert route_constraints is not None
        
        # Check route constraints
        assert 'max_deltav_budget_ms' in route_constraints
        assert 'max_mission_duration_seconds' in route_constraints
        assert 'max_budget_usd' in route_constraints
        assert 'budget_violation_penalty' in route_constraints
        assert 'timeline_violation_penalty' in route_constraints
        assert 'constraint_tolerance' in route_constraints
    
    def test_route_constraints_creation(self):
        """Test creation of route constraints with penalties."""
        constraints = self.route_simulator._create_route_constraints_with_penalties(
            self.test_service_request, self.test_satellites
        )
        
        assert isinstance(constraints, dict)
        assert constraints['max_budget_usd'] == 50000.0
        assert constraints['max_deltav_budget_ms'] > 0
        assert constraints['max_mission_duration_seconds'] > 0
        assert constraints['required_satellites'] == {"1", "2", "3"}
        assert 'orbital_constraints' in constraints
    
    def test_orbital_constraints_creation(self):
        """Test creation of orbital constraints."""
        orbital_constraints = self.route_simulator._create_orbital_constraints(
            self.test_satellites
        )
        
        assert isinstance(orbital_constraints, dict)
        assert 'max_plane_change_dv_ms' in orbital_constraints
        assert 'max_altitude_change_km' in orbital_constraints
        assert 'min_transfer_time_hours' in orbital_constraints
        assert 'max_transfer_time_hours' in orbital_constraints
        assert 'inclination_range_deg' in orbital_constraints
        assert 'altitude_range_km' in orbital_constraints
    
    def test_constrained_fitness_evaluation(self):
        """Test constrained fitness evaluation."""
        # Create a mock route and cost
        class MockRoute:
            def __init__(self):
                self.total_cost = 45000.0
                self.total_delta_v = 2000.0
                self.mission_duration = timedelta(days=15)
        
        class MockCost:
            def __init__(self):
                self.total_cost = 45000.0
        
        route = MockRoute()
        cost = MockCost()
        
        constraints = self.route_simulator._create_route_constraints_with_penalties(
            self.test_service_request, self.test_satellites
        )
        
        fitness = self.route_simulator._evaluate_constrained_fitness(
            route, cost, self.test_service_request, constraints
        )
        
        assert isinstance(fitness, float)
        # Should be negative (cost minimization) but not extremely negative (no major violations)
        assert fitness < 0
        assert fitness > -100000
    
    def test_constraint_satisfaction_evaluation(self):
        """Test constraint satisfaction evaluation."""
        # Create a mock route within constraints
        class MockRoute:
            def __init__(self):
                self.total_cost = 40000.0  # Within budget
                self.total_delta_v = 1500.0  # Reasonable
                self.mission_duration = timedelta(days=20)  # Within timeline
        
        route = MockRoute()
        constraints = self.route_simulator._create_route_constraints_with_penalties(
            self.test_service_request, self.test_satellites
        )
        
        satisfaction = self.route_simulator._evaluate_constraint_satisfaction(
            route, self.test_service_request, constraints
        )
        
        assert isinstance(satisfaction, dict)
        assert 'budget_satisfied' in satisfaction
        assert 'timeline_satisfied' in satisfaction
        assert 'deltav_satisfied' in satisfaction
        assert 'overall_satisfied' in satisfaction
        assert 'violations' in satisfaction
        
        # Should satisfy all constraints with test data
        assert satisfaction['budget_satisfied'] is True
        assert satisfaction['timeline_satisfied'] is True
        assert satisfaction['overall_satisfied'] is True
    
    def test_optimization_progress_tracking(self):
        """Test optimization progress tracking."""
        optimization_id = "test_opt_123"
        
        # Initialize tracking
        self.route_simulator.active_simulations[optimization_id] = {
            'status': 'running',
            'start_time': datetime.now(),
            'progress': 0.0,
            'current_phase': 'initialization'
        }
        
        # Update progress
        self.route_simulator._update_optimization_progress(
            optimization_id, 50.0, 'genetic_optimization'
        )
        
        # Check status
        status = self.route_simulator.get_optimization_status(optimization_id)
        
        assert status['optimization_id'] == optimization_id
        assert status['status'] == 'running'
        assert status['progress'] == 50.0
        assert status['current_phase'] == 'genetic_optimization'
        assert 'elapsed_time_seconds' in status
        
        # Clean up
        del self.route_simulator.active_simulations[optimization_id]
    
    def test_budget_violation_handling(self):
        """Test handling of budget constraint violations."""
        # Create service request with very low budget
        low_budget_request = ServiceRequest(
            client_id="test_client",
            satellites=["1", "2", "3"],
            timeline_requirements=self.test_service_request.timeline_requirements,
            budget_constraints=BudgetConstraints(max_total_cost=1000.0),  # Very low budget
            processing_preferences=self.test_service_request.processing_preferences
        )
        
        analysis = self.route_simulator._analyze_budget_constraints(
            low_budget_request, self.test_satellites[:3]
        )
        
        # Should detect budget constraint violations
        assert len(analysis['constraint_violations']) > 0
        assert any('exceeds budget' in violation for violation in analysis['constraint_violations'])
    
    def test_timeline_violation_handling(self):
        """Test handling of timeline constraint violations."""
        # Create service request with very short timeline
        short_timeline_request = ServiceRequest(
            client_id="test_client",
            satellites=["1", "2", "3", "4", "5"],  # More satellites
            timeline_requirements=TimelineConstraints(
                earliest_start=datetime.now() + timedelta(days=1),
                latest_completion=datetime.now() + timedelta(days=2)  # Very short timeline
            ),
            budget_constraints=self.test_service_request.budget_constraints,
            processing_preferences=self.test_service_request.processing_preferences
        )
        
        analysis = self.route_simulator._analyze_timeline_constraints(
            short_timeline_request, self.test_satellites
        )
        
        # Should detect timeline constraint violations
        assert len(analysis['constraint_violations']) > 0
        assert any('exceeds available time' in violation for violation in analysis['constraint_violations'])


class TestRouteSimulatorIntegration:
    """Integration tests for RouteSimulator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.route_simulator = RouteSimulator()
        self.test_satellites = self._create_test_satellites()
        self.test_service_request = self._create_test_service_request()
    
    def _create_test_satellites(self) -> List[Satellite]:
        """Create test satellites."""
        satellites = []
        
        for i in range(3):
            orbital_elements = OrbitalElements(
                semi_major_axis=7000 + i * 50,
                eccentricity=0.01,
                inclination=45,
                raan=0,
                argument_of_perigee=0,
                mean_anomaly=0,
                mean_motion=15.0,
                epoch=datetime.now()
            )
            
            satellite = Satellite(
                id=str(i + 1),
                name=f"TestSat-{i + 1}",
                tle_line1=f"1 {i+1:05d}U 21001A   21001.00000000  .00000000  00000-0  00000-0 0  999{i}",
                tle_line2=f"2 {i+1:05d}  45.0000 000.0000 0010000 000.0000 000.0000 15.0000000{i:05d}{i}",
                mass=500.0,
                material_composition={"aluminum": 0.6, "steel": 0.3, "other": 0.1},
                decommission_date=datetime.now() + timedelta(days=30),
                orbital_elements=orbital_elements
            )
            satellites.append(satellite)
        
        return satellites
    
    def _create_test_service_request(self) -> ServiceRequest:
        """Create test service request."""
        timeline_constraints = TimelineConstraints(
            earliest_start=datetime.now() + timedelta(days=1),
            latest_completion=datetime.now() + timedelta(days=30)
        )
        
        budget_constraints = BudgetConstraints(max_total_cost=30000.0)
        
        processing_preferences = ProcessingPreferences(
            preferred_processing_types=[ProcessingType.ISS_RECYCLING]
        )
        
        return ServiceRequest(
            client_id="test_client",
            satellites=["1", "2", "3"],
            timeline_requirements=timeline_constraints,
            budget_constraints=budget_constraints,
            processing_preferences=processing_preferences
        )
    
    def test_full_constrained_optimization(self):
        """Test full constrained optimization workflow."""
        result = self.route_simulator.optimize_route_with_constraints(
            self.test_service_request, self.test_satellites
        )
        
        assert isinstance(result, dict)
        assert 'optimization_id' in result
        assert 'success' in result
        
        if result['success']:
            assert 'route' in result
            assert 'mission_cost' in result
            assert 'constraint_analysis' in result
            assert 'convergence_info' in result
            assert 'optimization_metadata' in result
            
            # Check convergence info
            convergence_info = result['convergence_info']
            assert 'converged' in convergence_info
            assert 'generations_run' in convergence_info
            assert 'final_fitness' in convergence_info
            assert 'convergence_reason' in convergence_info
            
            # Check optimization metadata
            metadata = result['optimization_metadata']
            assert 'execution_time_seconds' in metadata
            assert 'total_generations' in metadata
            assert 'constraint_violations' in metadata
        else:
            # If optimization failed, should have error information
            assert 'error' in result or 'message' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])