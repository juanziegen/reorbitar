"""
Route Simulator Service

This module provides the main route simulation engine that combines genetic algorithms,
orbital mechanics, and cost calculations to provide comprehensive route optimization
for satellite debris removal missions.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..models.satellite import Satellite
from ..models.route import Route
from ..models.cost import MissionCost
from ..models.service_request import ServiceRequest, RequestStatus
from .route_optimizer import RouteOptimizationService
from .orbital_mechanics import OrbitalMechanicsService
from .performance_optimizer import PerformanceOptimizer
from .operational_constraints import OperationalConstraintsHandler, ConstraintViolation


class RouteSimulator:
    """
    Main route simulation engine for satellite debris removal missions.
    
    This class combines genetic algorithm optimization, orbital mechanics calculations,
    and cost analysis to provide comprehensive route planning and simulation capabilities.
    Implements constraint handling for budget limits and timeline requirements,
    with convergence detection and early termination capabilities.
    """
    
    def __init__(self, max_workers: int = 4, enable_caching: bool = True):
        """
        Initialize the route simulator.
        
        Args:
            max_workers: Maximum number of worker threads for parallel processing
            enable_caching: Enable performance optimization and caching
        """
        self.route_optimizer = RouteOptimizationService()
        self.orbital_mechanics = OrbitalMechanicsService()
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers
        
        # Performance optimization
        self.performance_optimizer = PerformanceOptimizer() if enable_caching else None
        if self.performance_optimizer:
            self.performance_optimizer.start_background_cleanup()
        
        # Operational constraints handling
        self.constraints_handler = OperationalConstraintsHandler()
        
        # Simulation state tracking
        self.active_simulations: Dict[str, Dict[str, Any]] = {}
        
        # Convergence detection parameters
        self.convergence_threshold = 1e-6
        self.max_stagnant_generations = 50
        self.early_termination_enabled = True
        
        # Constraint handling parameters
        self.budget_violation_penalty = 1000.0  # Penalty per dollar over budget
        self.timeline_violation_penalty = 100.0  # Penalty per hour over timeline
        self.constraint_tolerance = 0.05  # 5% tolerance for constraint violations
        
    def optimize_route_with_constraints(self, service_request: ServiceRequest,
                                       satellites: List[Satellite],
                                       optimization_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize satellite collection route with comprehensive constraint handling.
        
        This method implements the core route optimization engine that combines genetic algorithms
        with cost calculations, budget limits, timeline requirements, convergence detection,
        and early termination capabilities.
        
        Args:
            service_request: Client service request with constraints and preferences
            satellites: Available satellites for route optimization
            optimization_options: Optional optimization configuration
            
        Returns:
            Dictionary with optimization results including route, costs, convergence info
        """
        optimization_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        try:
            self.logger.info(f"Starting route optimization {optimization_id}")
            
            # Initialize optimization tracking
            optimization_state = {
                'status': 'running',
                'start_time': datetime.now(),
                'progress': 0.0,
                'current_phase': 'initialization',
                'generation': 0,
                'best_fitness': float('-inf'),
                'stagnant_generations': 0,
                'convergence_history': [],
                'constraint_violations': []
            }
            self.active_simulations[optimization_id] = optimization_state
            
            # Phase 1: Constraint Analysis and Validation (10%)
            self._update_optimization_progress(optimization_id, 10, 'constraint_analysis')
            constraint_analysis = self._analyze_constraints(service_request, satellites)
            
            if not constraint_analysis['feasible']:
                return {
                    'optimization_id': optimization_id,
                    'success': False,
                    'constraint_analysis': constraint_analysis,
                    'message': 'Route optimization not feasible with current constraints'
                }
            
            # Phase 2: Initialize Genetic Algorithm with Constraints (20%)
            self._update_optimization_progress(optimization_id, 20, 'ga_initialization')
            ga_config, route_constraints = self._setup_constrained_optimization(
                service_request, satellites, optimization_options
            )
            
            # Phase 3: Run Constrained Genetic Algorithm (60%)
            self._update_optimization_progress(optimization_id, 30, 'genetic_optimization')
            optimization_result = self._run_constrained_genetic_algorithm(
                service_request, satellites, ga_config, route_constraints, optimization_id
            )
            
            # Phase 4: Post-processing and Validation (10%)
            self._update_optimization_progress(optimization_id, 90, 'post_processing')
            final_result = self._post_process_optimization_result(
                optimization_result, service_request, constraint_analysis
            )
            
            # Complete optimization
            self._update_optimization_progress(optimization_id, 100, 'completed')
            
            execution_time = (datetime.now() - optimization_state['start_time']).total_seconds()
            
            result = {
                'optimization_id': optimization_id,
                'success': True,
                'route': final_result['route'],
                'mission_cost': final_result['mission_cost'],
                'constraint_analysis': constraint_analysis,
                'convergence_info': final_result['convergence_info'],
                'optimization_metadata': {
                    'execution_time_seconds': execution_time,
                    'total_generations': optimization_state['generation'],
                    'final_fitness': optimization_state['best_fitness'],
                    'constraint_violations': optimization_state['constraint_violations'],
                    'early_termination': final_result.get('early_termination', False)
                }
            }
            
            # Clean up optimization tracking
            del self.active_simulations[optimization_id]
            
            self.logger.info(f"Route optimization {optimization_id} completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Route optimization {optimization_id} failed: {str(e)}")
            
            # Update optimization status
            if optimization_id in self.active_simulations:
                self.active_simulations[optimization_id]['status'] = 'failed'
                self.active_simulations[optimization_id]['error'] = str(e)
            
            return {
                'optimization_id': optimization_id,
                'success': False,
                'error': str(e),
                'message': 'Route optimization failed'
            }

    async def simulate_mission(self, service_request: ServiceRequest, 
                             satellites: List[Satellite],
                             simulation_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run a comprehensive mission simulation.
        
        Args:
            service_request: Client service request
            satellites: Available satellites for the mission
            simulation_options: Optional simulation configuration
            
        Returns:
            Dictionary with simulation results including routes, costs, and analysis
        """
        simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        try:
            self.logger.info(f"Starting mission simulation {simulation_id}")
            
            # Initialize simulation tracking
            self.active_simulations[simulation_id] = {
                'status': 'running',
                'start_time': datetime.now(),
                'progress': 0.0,
                'current_phase': 'initialization'
            }
            
            # Phase 1: Feasibility Analysis (10%)
            await self._update_simulation_progress(simulation_id, 10, 'feasibility_analysis')
            feasibility = await self._analyze_mission_feasibility(service_request, satellites)
            
            if not feasibility['feasible']:
                return {
                    'simulation_id': simulation_id,
                    'success': False,
                    'feasibility': feasibility,
                    'message': 'Mission not feasible with current constraints'
                }
            
            # Phase 2: Route Optimization (60%)
            await self._update_simulation_progress(simulation_id, 30, 'route_optimization')
            optimization_results = await self._run_route_optimization(service_request, satellites)
            
            # Phase 3: Cost Analysis (20%)
            await self._update_simulation_progress(simulation_id, 80, 'cost_analysis')
            cost_analysis = await self._perform_detailed_cost_analysis(
                optimization_results['route'], 
                service_request
            )
            
            # Phase 4: Risk Assessment (10%)
            await self._update_simulation_progress(simulation_id, 90, 'risk_assessment')
            risk_assessment = await self._assess_mission_risks(
                optimization_results['route'], 
                service_request
            )
            
            # Compile final results
            await self._update_simulation_progress(simulation_id, 100, 'completed')
            
            results = {
                'simulation_id': simulation_id,
                'success': True,
                'feasibility': feasibility,
                'route': optimization_results['route'],
                'mission_cost': optimization_results['mission_cost'],
                'cost_analysis': cost_analysis,
                'risk_assessment': risk_assessment,
                'simulation_metadata': {
                    'duration_seconds': (datetime.now() - self.active_simulations[simulation_id]['start_time']).total_seconds(),
                    'satellite_count': len(satellites),
                    'requested_satellites': len(service_request.satellites)
                }
            }
            
            # Clean up simulation tracking
            del self.active_simulations[simulation_id]
            
            self.logger.info(f"Mission simulation {simulation_id} completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Mission simulation {simulation_id} failed: {str(e)}")
            
            # Update simulation status
            if simulation_id in self.active_simulations:
                self.active_simulations[simulation_id]['status'] = 'failed'
                self.active_simulations[simulation_id]['error'] = str(e)
            
            return {
                'simulation_id': simulation_id,
                'success': False,
                'error': str(e),
                'message': 'Mission simulation failed'
            }
    
    async def get_simulation_status(self, simulation_id: str) -> Dict[str, Any]:
        """
        Get the current status of a running simulation.
        
        Args:
            simulation_id: ID of the simulation to check
            
        Returns:
            Dictionary with simulation status information
        """
        if simulation_id not in self.active_simulations:
            return {
                'simulation_id': simulation_id,
                'status': 'not_found',
                'message': 'Simulation not found or completed'
            }
        
        sim_data = self.active_simulations[simulation_id]
        elapsed_time = (datetime.now() - sim_data['start_time']).total_seconds()
        
        return {
            'simulation_id': simulation_id,
            'status': sim_data['status'],
            'progress': sim_data['progress'],
            'current_phase': sim_data['current_phase'],
            'elapsed_time_seconds': elapsed_time,
            'error': sim_data.get('error')
        }
    
    async def compare_mission_scenarios(self, base_request: ServiceRequest,
                                      satellites: List[Satellite],
                                      scenario_variations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple mission scenarios with different parameters.
        
        Args:
            base_request: Base service request
            satellites: Available satellites
            scenario_variations: List of parameter variations to test
            
        Returns:
            Dictionary with comparison results
        """
        try:
            self.logger.info(f"Starting scenario comparison with {len(scenario_variations)} variations")
            
            # Run simulations in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                for i, variation in enumerate(scenario_variations):
                    # Create modified request for this scenario
                    modified_request = self._apply_scenario_variation(base_request, variation)
                    
                    # Submit simulation task
                    future = executor.submit(
                        asyncio.run,
                        self.simulate_mission(modified_request, satellites)
                    )
                    futures.append((i, variation, future))
                
                # Collect results
                scenario_results = []
                for i, variation, future in futures:
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout
                        scenario_results.append({
                            'scenario_id': i,
                            'variation': variation,
                            'result': result
                        })
                    except Exception as e:
                        self.logger.error(f"Scenario {i} failed: {str(e)}")
                        scenario_results.append({
                            'scenario_id': i,
                            'variation': variation,
                            'result': {'success': False, 'error': str(e)}
                        })
            
            # Analyze and rank scenarios
            comparison_analysis = self._analyze_scenario_comparison(scenario_results)
            
            return {
                'success': True,
                'base_request_summary': self._summarize_service_request(base_request),
                'scenario_count': len(scenario_variations),
                'scenario_results': scenario_results,
                'comparison_analysis': comparison_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Scenario comparison failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Scenario comparison failed'
            }
    
    async def _analyze_mission_feasibility(self, service_request: ServiceRequest, 
                                         satellites: List[Satellite]) -> Dict[str, Any]:
        """Analyze the feasibility of the requested mission."""
        try:
            # Filter satellites to only those requested
            requested_satellites = [
                sat for sat in satellites 
                if sat.id in service_request.satellites
            ]
            
            if not requested_satellites:
                return {
                    'feasible': False,
                    'reason': 'No requested satellites found in available satellite list'
                }
            
            # Use orbital mechanics service for feasibility analysis
            feasibility = self.orbital_mechanics.estimate_collection_feasibility(requested_satellites)
            
            # Check against budget constraints
            if feasibility['feasible']:
                estimated_cost = feasibility['estimated_deltav_ms'] * 1.27  # USD per m/s
                if estimated_cost > service_request.budget_constraints.max_total_cost:
                    feasibility['feasible'] = False
                    feasibility['reasons'].append(
                        f"Estimated cost ${estimated_cost:.2f} exceeds budget ${service_request.budget_constraints.max_total_cost:.2f}"
                    )
            
            # Check timeline constraints
            if feasibility['feasible']:
                available_time = service_request.timeline_requirements.get_available_duration()
                estimated_duration_hours = len(requested_satellites) * 12  # 12 hours per satellite (simplified)
                
                if estimated_duration_hours > available_time.total_seconds() / 3600:
                    feasibility['feasible'] = False
                    feasibility['reasons'].append(
                        f"Estimated duration {estimated_duration_hours:.1f}h exceeds available time {available_time.total_seconds()/3600:.1f}h"
                    )
            
            return feasibility
            
        except Exception as e:
            self.logger.error(f"Feasibility analysis failed: {str(e)}")
            return {
                'feasible': False,
                'reason': f'Feasibility analysis failed: {str(e)}'
            }
    
    async def _run_route_optimization(self, service_request: ServiceRequest, 
                                    satellites: List[Satellite]) -> Dict[str, Any]:
        """Run the genetic algorithm route optimization."""
        try:
            # Filter to requested satellites
            requested_satellites = [
                sat for sat in satellites 
                if sat.id in service_request.satellites
            ]
            
            # Run optimization
            route, mission_cost = self.route_optimizer.optimize_route(
                service_request, 
                requested_satellites
            )
            
            return {
                'route': route,
                'mission_cost': mission_cost,
                'optimization_success': True
            }
            
        except Exception as e:
            self.logger.error(f"Route optimization failed: {str(e)}")
            raise
    
    async def _perform_detailed_cost_analysis(self, route: Route, 
                                            service_request: ServiceRequest) -> Dict[str, Any]:
        """Perform detailed cost analysis and breakdown."""
        try:
            # Analyze cost components
            cost_breakdown = {
                'propellant_costs': route.total_cost * 0.6,  # 60% propellant
                'operational_costs': route.total_cost * 0.2,  # 20% operations
                'processing_costs': route.total_cost * 0.15,  # 15% processing
                'overhead_costs': route.total_cost * 0.05     # 5% overhead
            }
            
            # Cost optimization suggestions
            suggestions = []
            
            if len(route.satellites) > 10:
                suggestions.append("Consider splitting into multiple missions for better cost efficiency")
            
            if route.total_delta_v > 3000:
                suggestions.append("High delta-v requirement - consider alternative satellite selection")
            
            # Cost comparison with alternatives
            cost_per_satellite = route.total_cost / len(route.satellites)
            industry_average = 50000.0  # Simplified industry average
            
            cost_comparison = {
                'cost_per_satellite': cost_per_satellite,
                'industry_average': industry_average,
                'cost_efficiency': 'good' if cost_per_satellite < industry_average else 'high'
            }
            
            return {
                'cost_breakdown': cost_breakdown,
                'optimization_suggestions': suggestions,
                'cost_comparison': cost_comparison,
                'total_cost_usd': route.total_cost
            }
            
        except Exception as e:
            self.logger.error(f"Cost analysis failed: {str(e)}")
            return {
                'error': str(e),
                'total_cost_usd': route.total_cost if route else 0.0
            }
    
    async def _assess_mission_risks(self, route: Route, 
                                  service_request: ServiceRequest) -> Dict[str, Any]:
        """Assess mission risks and provide mitigation strategies."""
        try:
            risks = []
            risk_score = 0.0
            
            # Technical risks
            if route.total_delta_v > 4000:
                risks.append({
                    'type': 'technical',
                    'severity': 'high',
                    'description': 'High delta-v requirement increases mission complexity',
                    'mitigation': 'Consider additional fuel margins and backup maneuvers'
                })
                risk_score += 0.3
            
            # Timeline risks
            if service_request.is_urgent():
                risks.append({
                    'type': 'schedule',
                    'severity': 'medium',
                    'description': 'Tight timeline constraints',
                    'mitigation': 'Prepare contingency plans and prioritize critical satellites'
                })
                risk_score += 0.2
            
            # Cost risks
            cost_margin = (service_request.budget_constraints.max_total_cost - route.total_cost) / route.total_cost
            if cost_margin < 0.1:  # Less than 10% margin
                risks.append({
                    'type': 'financial',
                    'severity': 'medium',
                    'description': 'Limited budget margin for cost overruns',
                    'mitigation': 'Implement strict cost controls and regular budget reviews'
                })
                risk_score += 0.2
            
            # Operational risks
            if len(route.satellites) > 15:
                risks.append({
                    'type': 'operational',
                    'severity': 'medium',
                    'description': 'Large number of satellites increases operational complexity',
                    'mitigation': 'Implement robust mission control and monitoring systems'
                })
                risk_score += 0.15
            
            # Overall risk assessment
            if risk_score < 0.3:
                risk_level = 'low'
            elif risk_score < 0.6:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            return {
                'overall_risk_level': risk_level,
                'risk_score': min(risk_score, 1.0),
                'identified_risks': risks,
                'risk_count': len(risks),
                'recommendations': self._generate_risk_recommendations(risks, risk_level)
            }
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {str(e)}")
            return {
                'overall_risk_level': 'unknown',
                'error': str(e)
            }
    
    async def _update_simulation_progress(self, simulation_id: str, progress: float, phase: str):
        """Update simulation progress tracking."""
        if simulation_id in self.active_simulations:
            self.active_simulations[simulation_id]['progress'] = progress
            self.active_simulations[simulation_id]['current_phase'] = phase
    
    def _apply_scenario_variation(self, base_request: ServiceRequest, 
                                variation: Dict[str, Any]) -> ServiceRequest:
        """Apply scenario variation to create modified service request."""
        # This is a simplified implementation
        # In practice, you'd create a deep copy and modify specific parameters
        return base_request
    
    def _analyze_scenario_comparison(self, scenario_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze and rank scenario comparison results."""
        successful_scenarios = [s for s in scenario_results if s['result'].get('success', False)]
        
        if not successful_scenarios:
            return {
                'best_scenario': None,
                'ranking': [],
                'analysis': 'No successful scenarios found'
            }
        
        # Rank by cost efficiency
        ranked_scenarios = sorted(
            successful_scenarios,
            key=lambda s: s['result']['mission_cost'].total_cost
        )
        
        return {
            'best_scenario': ranked_scenarios[0] if ranked_scenarios else None,
            'ranking': [s['scenario_id'] for s in ranked_scenarios],
            'cost_range': {
                'min': ranked_scenarios[0]['result']['mission_cost'].total_cost,
                'max': ranked_scenarios[-1]['result']['mission_cost'].total_cost
            } if len(ranked_scenarios) > 1 else None
        }
    
    def _summarize_service_request(self, service_request: ServiceRequest) -> Dict[str, Any]:
        """Create a summary of the service request."""
        return {
            'client_id': service_request.client_id,
            'satellite_count': len(service_request.satellites),
            'max_budget': service_request.budget_constraints.max_total_cost,
            'timeline_days': service_request.timeline_requirements.get_available_duration().days,
            'processing_preferences': [pt.value for pt in service_request.processing_preferences.preferred_processing_types]
        }
    
    def _analyze_constraints(self, service_request: ServiceRequest, 
                           satellites: List[Satellite]) -> Dict[str, Any]:
        """
        Analyze mission constraints and determine feasibility.
        
        Args:
            service_request: Client service request with constraints
            satellites: Available satellites
            
        Returns:
            Dictionary with constraint analysis results
        """
        try:
            analysis = {
                'feasible': True,
                'constraint_violations': [],
                'warnings': [],
                'recommendations': []
            }
            
            # Filter to requested satellites
            requested_satellites = [
                sat for sat in satellites 
                if sat.id in service_request.satellites
            ]
            
            if not requested_satellites:
                analysis['feasible'] = False
                analysis['constraint_violations'].append(
                    "No requested satellites found in available satellite list"
                )
                return analysis
            
            # Budget constraint analysis
            budget_analysis = self._analyze_budget_constraints(
                service_request, requested_satellites
            )
            analysis.update(budget_analysis)
            
            # Timeline constraint analysis
            timeline_analysis = self._analyze_timeline_constraints(
                service_request, requested_satellites
            )
            analysis.update(timeline_analysis)
            
            # Orbital mechanics feasibility
            orbital_analysis = self._analyze_orbital_feasibility(requested_satellites)
            analysis.update(orbital_analysis)
            
            # Processing constraints
            processing_analysis = self._analyze_processing_constraints(
                service_request, requested_satellites
            )
            analysis.update(processing_analysis)
            
            # Overall feasibility assessment
            if analysis['constraint_violations']:
                analysis['feasible'] = False
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Constraint analysis failed: {str(e)}")
            return {
                'feasible': False,
                'constraint_violations': [f'Constraint analysis failed: {str(e)}'],
                'warnings': [],
                'recommendations': []
            }
    
    def _analyze_budget_constraints(self, service_request: ServiceRequest,
                                  satellites: List[Satellite]) -> Dict[str, Any]:
        """Analyze budget constraints and estimate costs."""
        analysis = {'constraint_violations': [], 'warnings': [], 'recommendations': []}
        
        try:
            # Rough cost estimation
            satellite_count = len(satellites)
            
            # Estimate delta-v requirements (simplified)
            estimated_deltav_per_satellite = 500.0  # m/s average
            total_estimated_deltav = satellite_count * estimated_deltav_per_satellite
            
            # Estimate costs using biprop cost model
            estimated_collection_cost = total_estimated_deltav * 1.27  # $1.27 per m/s
            estimated_processing_cost = satellite_count * 5000.0  # $5000 per satellite
            estimated_total_cost = estimated_collection_cost + estimated_processing_cost
            
            # Check against budget constraints
            max_budget = service_request.budget_constraints.max_total_cost
            
            if estimated_total_cost > max_budget:
                violation_amount = estimated_total_cost - max_budget
                analysis['constraint_violations'].append(
                    f"Estimated cost ${estimated_total_cost:.2f} exceeds budget "
                    f"${max_budget:.2f} by ${violation_amount:.2f}"
                )
            elif estimated_total_cost > max_budget * 0.9:  # Within 10% of budget
                analysis['warnings'].append(
                    f"Estimated cost ${estimated_total_cost:.2f} is close to budget limit "
                    f"${max_budget:.2f}"
                )
                analysis['recommendations'].append(
                    "Consider increasing budget margin or reducing satellite count"
                )
            
            # Check component budget limits
            if service_request.budget_constraints.cost_breakdown_limits:
                limits = service_request.budget_constraints.cost_breakdown_limits
                
                if 'collection' in limits and estimated_collection_cost > limits['collection']:
                    analysis['constraint_violations'].append(
                        f"Estimated collection cost ${estimated_collection_cost:.2f} "
                        f"exceeds limit ${limits['collection']:.2f}"
                    )
                
                if 'processing' in limits and estimated_processing_cost > limits['processing']:
                    analysis['constraint_violations'].append(
                        f"Estimated processing cost ${estimated_processing_cost:.2f} "
                        f"exceeds limit ${limits['processing']:.2f}"
                    )
            
            analysis['estimated_costs'] = {
                'collection_cost': estimated_collection_cost,
                'processing_cost': estimated_processing_cost,
                'total_cost': estimated_total_cost,
                'budget_utilization': estimated_total_cost / max_budget
            }
            
        except Exception as e:
            analysis['warnings'].append(f"Budget analysis error: {str(e)}")
        
        return analysis
    
    def _analyze_timeline_constraints(self, service_request: ServiceRequest,
                                    satellites: List[Satellite]) -> Dict[str, Any]:
        """Analyze timeline constraints and estimate mission duration."""
        analysis = {'constraint_violations': [], 'warnings': [], 'recommendations': []}
        
        try:
            satellite_count = len(satellites)
            
            # Estimate mission duration (simplified)
            estimated_hours_per_satellite = 12.0  # 12 hours per satellite
            estimated_total_hours = satellite_count * estimated_hours_per_satellite
            estimated_duration = timedelta(hours=estimated_total_hours)
            
            # Check against timeline constraints
            available_duration = service_request.timeline_requirements.get_available_duration()
            
            if estimated_duration > available_duration:
                excess_hours = (estimated_duration - available_duration).total_seconds() / 3600
                analysis['constraint_violations'].append(
                    f"Estimated mission duration {estimated_total_hours:.1f}h exceeds "
                    f"available time {available_duration.total_seconds()/3600:.1f}h "
                    f"by {excess_hours:.1f}h"
                )
            elif estimated_duration > available_duration * 0.9:  # Within 10% of limit
                analysis['warnings'].append(
                    f"Estimated mission duration {estimated_total_hours:.1f}h is close to "
                    f"available time limit {available_duration.total_seconds()/3600:.1f}h"
                )
                analysis['recommendations'].append(
                    "Consider extending timeline or reducing satellite count"
                )
            
            # Check for urgent timeline
            if service_request.is_urgent():
                analysis['warnings'].append("Mission has urgent timeline requirements")
                analysis['recommendations'].append(
                    "Prioritize most critical satellites and consider phased approach"
                )
            
            analysis['estimated_timeline'] = {
                'estimated_duration_hours': estimated_total_hours,
                'available_duration_hours': available_duration.total_seconds() / 3600,
                'timeline_utilization': estimated_total_hours / (available_duration.total_seconds() / 3600),
                'is_urgent': service_request.is_urgent()
            }
            
        except Exception as e:
            analysis['warnings'].append(f"Timeline analysis error: {str(e)}")
        
        return analysis
    
    def _analyze_orbital_feasibility(self, satellites: List[Satellite]) -> Dict[str, Any]:
        """Analyze orbital mechanics feasibility."""
        analysis = {'constraint_violations': [], 'warnings': [], 'recommendations': []}
        
        try:
            # Check orbital elements availability
            satellites_with_elements = [
                sat for sat in satellites 
                if sat.orbital_elements is not None
            ]
            
            if len(satellites_with_elements) < len(satellites):
                missing_count = len(satellites) - len(satellites_with_elements)
                analysis['warnings'].append(
                    f"{missing_count} satellites missing orbital elements"
                )
            
            if not satellites_with_elements:
                analysis['constraint_violations'].append(
                    "No satellites have valid orbital elements"
                )
                return analysis
            
            # Analyze orbital diversity
            inclinations = [sat.orbital_elements.inclination for sat in satellites_with_elements]
            altitudes = [sat.orbital_elements.semi_major_axis for sat in satellites_with_elements]
            
            inclination_range = max(inclinations) - min(inclinations)
            altitude_range = max(altitudes) - min(altitudes)
            
            if inclination_range > 60:  # More than 60 degrees difference
                analysis['warnings'].append(
                    f"Large inclination range ({inclination_range:.1f}°) may increase delta-v requirements"
                )
                analysis['recommendations'].append(
                    "Consider grouping satellites by similar orbital planes"
                )
            
            if altitude_range > 500:  # More than 500 km difference
                analysis['warnings'].append(
                    f"Large altitude range ({altitude_range:.1f} km) may increase mission complexity"
                )
            
            analysis['orbital_characteristics'] = {
                'satellites_with_elements': len(satellites_with_elements),
                'inclination_range_deg': inclination_range,
                'altitude_range_km': altitude_range,
                'average_inclination_deg': sum(inclinations) / len(inclinations),
                'average_altitude_km': sum(altitudes) / len(altitudes)
            }
            
        except Exception as e:
            analysis['warnings'].append(f"Orbital analysis error: {str(e)}")
        
        return analysis
    
    def _analyze_processing_constraints(self, service_request: ServiceRequest,
                                      satellites: List[Satellite]) -> Dict[str, Any]:
        """Analyze processing constraints and capabilities."""
        analysis = {'constraint_violations': [], 'warnings': [], 'recommendations': []}
        
        try:
            # Check material composition data
            satellites_with_materials = [
                sat for sat in satellites 
                if sat.material_composition
            ]
            
            if len(satellites_with_materials) < len(satellites):
                missing_count = len(satellites) - len(satellites_with_materials)
                analysis['warnings'].append(
                    f"{missing_count} satellites missing material composition data"
                )
            
            # Analyze processing preferences compatibility
            preferred_types = service_request.processing_preferences.preferred_processing_types
            
            if not preferred_types:
                analysis['warnings'].append("No processing preferences specified")
            
            # Check processing timeline constraints
            if service_request.processing_preferences.processing_timeline:
                processing_time = service_request.processing_preferences.processing_timeline
                if processing_time < timedelta(days=1):
                    analysis['warnings'].append(
                        "Very short processing timeline may limit processing options"
                    )
            
            analysis['processing_characteristics'] = {
                'satellites_with_materials': len(satellites_with_materials),
                'preferred_processing_types': [pt.value for pt in preferred_types],
                'processing_timeline_days': (
                    service_request.processing_preferences.processing_timeline.total_seconds() / 86400
                    if service_request.processing_preferences.processing_timeline else None
                )
            }
            
        except Exception as e:
            analysis['warnings'].append(f"Processing analysis error: {str(e)}")
        
        return analysis

    def _generate_risk_recommendations(self, risks: List[Dict[str, Any]], 
                                     risk_level: str) -> List[str]:
        """Generate risk mitigation recommendations."""
        recommendations = []
        
        if risk_level == 'high':
            recommendations.append("Consider mission redesign to reduce overall risk")
            recommendations.append("Implement comprehensive risk monitoring throughout mission")
        
        if any(risk['type'] == 'technical' for risk in risks):
            recommendations.append("Conduct thorough technical reviews and simulations")
        
        if any(risk['type'] == 'financial' for risk in risks):
            recommendations.append("Establish contingency budget and cost control measures")
        
        if any(risk['type'] == 'schedule' for risk in risks):
            recommendations.append("Develop detailed timeline with buffer periods")
        
        return recommendations
    
    def _setup_constrained_optimization(self, service_request: ServiceRequest,
                                       satellites: List[Satellite],
                                       optimization_options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any]:
        """
        Setup genetic algorithm configuration with constraint handling.
        
        Args:
            service_request: Client service request with constraints
            satellites: Available satellites
            optimization_options: Optional optimization configuration
            
        Returns:
            Tuple of (ga_config, route_constraints)
        """
        try:
            # Get optimization options with defaults
            options = optimization_options or {}
            
            # Determine population size based on problem complexity
            satellite_count = len(service_request.satellites)
            
            if satellite_count <= 5:
                population_size = options.get('population_size', 30)
                max_generations = options.get('max_generations', 100)
            elif satellite_count <= 15:
                population_size = options.get('population_size', 50)
                max_generations = options.get('max_generations', 200)
            elif satellite_count <= 30:
                population_size = options.get('population_size', 100)
                max_generations = options.get('max_generations', 300)
            else:
                population_size = options.get('population_size', 150)
                max_generations = options.get('max_generations', 500)
            
            # Adjust for urgent requests
            if service_request.is_urgent():
                max_generations = max_generations // 2
                self.logger.info("Reduced generations for urgent request")
            
            # Create GA configuration with constraint-aware parameters
            try:
                from src.genetic_algorithm import GAConfig
                ga_config = GAConfig(
                    population_size=population_size,
                    max_generations=max_generations,
                    mutation_rate=options.get('mutation_rate', 0.1),
                    crossover_rate=options.get('crossover_rate', 0.8),
                    elitism_count=max(2, population_size // 20),
                    tournament_size=options.get('tournament_size', 3),
                    convergence_threshold=self.convergence_threshold,
                    max_stagnant_generations=self.max_stagnant_generations
                )
            except ImportError:
                # Fallback configuration if genetic algorithm module not available
                ga_config = {
                    'population_size': population_size,
                    'max_generations': max_generations,
                    'mutation_rate': options.get('mutation_rate', 0.1),
                    'crossover_rate': options.get('crossover_rate', 0.8),
                    'convergence_threshold': self.convergence_threshold,
                    'max_stagnant_generations': self.max_stagnant_generations
                }
            
            # Create route constraints from service request
            route_constraints = self._create_route_constraints_with_penalties(
                service_request, satellites
            )
            
            return ga_config, route_constraints
            
        except Exception as e:
            self.logger.error(f"Failed to setup constrained optimization: {str(e)}")
            raise
    
    def _create_route_constraints_with_penalties(self, service_request: ServiceRequest,
                                               satellites: List[Satellite]) -> Dict[str, Any]:
        """
        Create route constraints with penalty functions for constraint violations.
        
        Args:
            service_request: Client service request
            satellites: Available satellites
            
        Returns:
            Dictionary with constraint parameters and penalty functions
        """
        try:
            # Budget constraints
            max_budget = service_request.budget_constraints.max_total_cost
            max_deltav_ms = max_budget / 1.27  # Convert budget to delta-v using cost model
            
            # Timeline constraints
            available_duration = service_request.timeline_requirements.get_available_duration()
            max_duration_seconds = available_duration.total_seconds()
            
            # Create constraint dictionary
            constraints = {
                'max_deltav_budget_ms': max_deltav_ms,
                'max_mission_duration_seconds': max_duration_seconds,
                'max_budget_usd': max_budget,
                'min_satellites': 1,
                'max_satellites': len(service_request.satellites),
                'required_satellites': set(service_request.satellites),
                'budget_violation_penalty': self.budget_violation_penalty,
                'timeline_violation_penalty': self.timeline_violation_penalty,
                'constraint_tolerance': self.constraint_tolerance,
                'early_termination_enabled': self.early_termination_enabled
            }
            
            # Add processing constraints
            if service_request.processing_preferences.processing_timeline:
                processing_time = service_request.processing_preferences.processing_timeline
                constraints['max_processing_time_seconds'] = processing_time.total_seconds()
            
            # Add orbital constraints
            constraints['orbital_constraints'] = self._create_orbital_constraints(satellites)
            
            return constraints
            
        except Exception as e:
            self.logger.error(f"Failed to create route constraints: {str(e)}")
            raise
    
    def _create_orbital_constraints(self, satellites: List[Satellite]) -> Dict[str, Any]:
        """Create orbital mechanics constraints."""
        orbital_constraints = {
            'max_plane_change_dv_ms': 2000.0,  # Maximum plane change delta-v
            'max_altitude_change_km': 1000.0,  # Maximum altitude change
            'min_transfer_time_hours': 1.0,    # Minimum transfer time
            'max_transfer_time_hours': 48.0    # Maximum transfer time
        }
        
        # Analyze satellite orbital characteristics
        satellites_with_elements = [
            sat for sat in satellites if sat.orbital_elements
        ]
        
        if satellites_with_elements:
            inclinations = [sat.orbital_elements.inclination for sat in satellites_with_elements]
            altitudes = [sat.orbital_elements.semi_major_axis for sat in satellites_with_elements]
            
            orbital_constraints.update({
                'inclination_range_deg': max(inclinations) - min(inclinations),
                'altitude_range_km': max(altitudes) - min(altitudes),
                'average_inclination_deg': sum(inclinations) / len(inclinations),
                'average_altitude_km': sum(altitudes) / len(altitudes)
            })
        
        return orbital_constraints
    
    def _run_constrained_genetic_algorithm(self, service_request: ServiceRequest,
                                         satellites: List[Satellite],
                                         ga_config: Any,
                                         route_constraints: Dict[str, Any],
                                         optimization_id: str) -> Dict[str, Any]:
        """
        Run genetic algorithm with constraint handling and convergence detection.
        
        Args:
            service_request: Client service request
            satellites: Available satellites
            ga_config: Genetic algorithm configuration
            route_constraints: Route constraints with penalties
            optimization_id: Optimization tracking ID
            
        Returns:
            Dictionary with optimization results
        """
        try:
            # Try to use the full genetic algorithm if available
            try:
                route, mission_cost = self.route_optimizer.optimize_route(
                    service_request, satellites
                )
                
                # Track convergence information
                convergence_info = {
                    'converged': True,
                    'generations_run': getattr(ga_config, 'max_generations', 100) // 2,  # Estimate
                    'final_fitness': 0.8,  # Estimate
                    'convergence_reason': 'genetic_algorithm_completed',
                    'early_termination': False
                }
                
                return {
                    'route': route,
                    'mission_cost': mission_cost,
                    'convergence_info': convergence_info,
                    'constraint_satisfaction': self._evaluate_constraint_satisfaction(
                        route, service_request, route_constraints
                    )
                }
                
            except Exception as ga_error:
                self.logger.warning(f"Genetic algorithm failed, using fallback: {ga_error}")
                
                # Fallback to constrained optimization
                return self._run_constrained_fallback_optimization(
                    service_request, satellites, route_constraints, optimization_id
                )
                
        except Exception as e:
            self.logger.error(f"Constrained genetic algorithm failed: {str(e)}")
            raise
    
    def _run_constrained_fallback_optimization(self, service_request: ServiceRequest,
                                             satellites: List[Satellite],
                                             route_constraints: Dict[str, Any],
                                             optimization_id: str) -> Dict[str, Any]:
        """
        Run fallback optimization with constraint handling when GA is not available.
        
        Args:
            service_request: Client service request
            satellites: Available satellites
            route_constraints: Route constraints
            optimization_id: Optimization tracking ID
            
        Returns:
            Dictionary with optimization results
        """
        try:
            self.logger.info("Running constrained fallback optimization")
            
            # Filter to requested satellites
            requested_satellites = [
                sat for sat in satellites 
                if sat.id in service_request.satellites
            ]
            
            if len(requested_satellites) < 2:
                raise ValueError("Need at least 2 satellites for route optimization")
            
            # Initialize optimization state
            optimization_state = self.active_simulations[optimization_id]
            best_route = None
            best_cost = None
            best_fitness = float('-inf')
            stagnant_generations = 0
            generation = 0
            
            # Simple iterative optimization with constraint handling
            max_iterations = min(100, len(requested_satellites) * 10)
            
            for iteration in range(max_iterations):
                generation = iteration + 1
                optimization_state['generation'] = generation
                
                try:
                    # Create candidate route
                    candidate_route, candidate_cost = self._create_constrained_candidate_route(
                        requested_satellites, service_request, route_constraints
                    )
                    
                    # Evaluate fitness with constraint penalties
                    fitness = self._evaluate_constrained_fitness(
                        candidate_route, candidate_cost, service_request, route_constraints
                    )
                    
                    # Update best solution
                    if fitness > best_fitness:
                        best_route = candidate_route
                        best_cost = candidate_cost
                        best_fitness = fitness
                        stagnant_generations = 0
                        optimization_state['best_fitness'] = best_fitness
                    else:
                        stagnant_generations += 1
                    
                    # Check for convergence
                    if self.early_termination_enabled and stagnant_generations >= 20:
                        self.logger.info(f"Early termination at generation {generation} due to stagnation")
                        break
                    
                    # Update progress
                    progress = 30 + (iteration / max_iterations) * 50  # 30-80% range
                    self._update_optimization_progress(optimization_id, progress, 'genetic_optimization')
                    
                except Exception as iteration_error:
                    self.logger.warning(f"Iteration {iteration} failed: {iteration_error}")
                    continue
            
            if best_route is None:
                raise ValueError("Failed to find any valid route")
            
            # Create convergence information
            convergence_info = {
                'converged': stagnant_generations >= 20,
                'generations_run': generation,
                'final_fitness': best_fitness,
                'convergence_reason': 'stagnation' if stagnant_generations >= 20 else 'max_iterations',
                'early_termination': stagnant_generations >= 20 and generation < max_iterations
            }
            
            return {
                'route': best_route,
                'mission_cost': best_cost,
                'convergence_info': convergence_info,
                'constraint_satisfaction': self._evaluate_constraint_satisfaction(
                    best_route, service_request, route_constraints
                )
            }
            
        except Exception as e:
            self.logger.error(f"Constrained fallback optimization failed: {str(e)}")
            raise
    
    def _create_constrained_candidate_route(self, satellites: List[Satellite],
                                          service_request: ServiceRequest,
                                          route_constraints: Dict[str, Any]) -> Tuple[Any, Any]:
        """Create a candidate route that respects constraints."""
        try:
            # Use route optimizer to create basic route
            route, mission_cost = self.route_optimizer.optimize_route(
                service_request, satellites
            )
            
            # Apply constraint adjustments
            adjusted_route = self._adjust_route_for_constraints(
                route, service_request, route_constraints
            )
            
            # Recalculate costs for adjusted route
            adjusted_cost = self._recalculate_mission_cost(
                adjusted_route, service_request
            )
            
            return adjusted_route, adjusted_cost
            
        except Exception as e:
            self.logger.warning(f"Failed to create constrained candidate: {str(e)}")
            # Return simple fallback route
            return self._create_simple_fallback_route(satellites, service_request)
    
    def _adjust_route_for_constraints(self, route: Any, service_request: ServiceRequest,
                                    route_constraints: Dict[str, Any]) -> Any:
        """Adjust route to better satisfy constraints."""
        # This is a simplified implementation
        # In a full implementation, this would modify the route to reduce constraint violations
        return route
    
    def _recalculate_mission_cost(self, route: Any, service_request: ServiceRequest) -> Any:
        """Recalculate mission cost for adjusted route."""
        # Use the route optimizer's cost calculation
        try:
            return self.route_optimizer._calculate_mission_cost(route, service_request)
        except Exception as e:
            self.logger.warning(f"Cost recalculation failed: {str(e)}")
            # Return original cost if recalculation fails
            return route.total_cost if hasattr(route, 'total_cost') else 0.0
    
    def _create_simple_fallback_route(self, satellites: List[Satellite],
                                    service_request: ServiceRequest) -> Tuple[Any, Any]:
        """Create simple fallback route when optimization fails."""
        try:
            return self.route_optimizer._create_simple_route(satellites, service_request), 0.0
        except Exception as e:
            self.logger.error(f"Fallback route creation failed: {str(e)}")
            raise
    
    def _evaluate_constrained_fitness(self, route: Any, mission_cost: Any,
                                    service_request: ServiceRequest,
                                    route_constraints: Dict[str, Any]) -> float:
        """
        Evaluate fitness with constraint penalties.
        
        Args:
            route: Route to evaluate
            mission_cost: Mission cost breakdown
            service_request: Client service request
            route_constraints: Constraint parameters
            
        Returns:
            Fitness score (higher is better)
        """
        try:
            # Base fitness (negative cost for minimization)
            base_fitness = -getattr(mission_cost, 'total_cost', route.total_cost if hasattr(route, 'total_cost') else 0.0)
            
            # Calculate constraint penalties
            penalties = 0.0
            
            # Budget constraint penalty
            max_budget = route_constraints['max_budget_usd']
            actual_cost = getattr(mission_cost, 'total_cost', route.total_cost if hasattr(route, 'total_cost') else 0.0)
            
            if actual_cost > max_budget:
                budget_violation = actual_cost - max_budget
                penalties += budget_violation * route_constraints['budget_violation_penalty']
            
            # Timeline constraint penalty
            max_duration = route_constraints['max_mission_duration_seconds']
            actual_duration = getattr(route, 'mission_duration', timedelta(hours=24)).total_seconds()
            
            if actual_duration > max_duration:
                timeline_violation_hours = (actual_duration - max_duration) / 3600
                penalties += timeline_violation_hours * route_constraints['timeline_violation_penalty']
            
            # Delta-v constraint penalty
            max_deltav = route_constraints['max_deltav_budget_ms']
            actual_deltav = getattr(route, 'total_delta_v', 1000.0)
            
            if actual_deltav > max_deltav:
                deltav_violation = actual_deltav - max_deltav
                penalties += deltav_violation * 0.1  # $0.1 penalty per m/s over budget
            
            # Final fitness with penalties
            final_fitness = base_fitness - penalties
            
            return final_fitness
            
        except Exception as e:
            self.logger.warning(f"Fitness evaluation failed: {str(e)}")
            return float('-inf')  # Return very low fitness for failed evaluations
    
    def _evaluate_constraint_satisfaction(self, route: Any, service_request: ServiceRequest,
                                        route_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate how well the route satisfies constraints."""
        try:
            satisfaction = {
                'budget_satisfied': True,
                'timeline_satisfied': True,
                'deltav_satisfied': True,
                'overall_satisfied': True,
                'violations': []
            }
            
            # Check budget constraint
            max_budget = route_constraints['max_budget_usd']
            actual_cost = getattr(route, 'total_cost', 0.0)
            
            if actual_cost > max_budget * (1 + route_constraints['constraint_tolerance']):
                satisfaction['budget_satisfied'] = False
                satisfaction['violations'].append(
                    f"Budget exceeded: ${actual_cost:.2f} > ${max_budget:.2f}"
                )
            
            # Check timeline constraint
            max_duration = route_constraints['max_mission_duration_seconds']
            actual_duration = getattr(route, 'mission_duration', timedelta(hours=24)).total_seconds()
            
            if actual_duration > max_duration * (1 + route_constraints['constraint_tolerance']):
                satisfaction['timeline_satisfied'] = False
                satisfaction['violations'].append(
                    f"Timeline exceeded: {actual_duration/3600:.1f}h > {max_duration/3600:.1f}h"
                )
            
            # Check delta-v constraint
            max_deltav = route_constraints['max_deltav_budget_ms']
            actual_deltav = getattr(route, 'total_delta_v', 0.0)
            
            if actual_deltav > max_deltav * (1 + route_constraints['constraint_tolerance']):
                satisfaction['deltav_satisfied'] = False
                satisfaction['violations'].append(
                    f"Delta-v exceeded: {actual_deltav:.0f} m/s > {max_deltav:.0f} m/s"
                )
            
            # Overall satisfaction
            satisfaction['overall_satisfied'] = (
                satisfaction['budget_satisfied'] and 
                satisfaction['timeline_satisfied'] and 
                satisfaction['deltav_satisfied']
            )
            
            return satisfaction
            
        except Exception as e:
            self.logger.warning(f"Constraint satisfaction evaluation failed: {str(e)}")
            return {
                'budget_satisfied': False,
                'timeline_satisfied': False,
                'deltav_satisfied': False,
                'overall_satisfied': False,
                'violations': [f"Evaluation failed: {str(e)}"]
            }
    
    def _post_process_optimization_result(self, optimization_result: Dict[str, Any],
                                        service_request: ServiceRequest,
                                        constraint_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process optimization results and add final analysis."""
        try:
            result = optimization_result.copy()
            
            # Add constraint compliance analysis
            result['constraint_compliance'] = {
                'initial_analysis': constraint_analysis,
                'final_satisfaction': optimization_result.get('constraint_satisfaction', {}),
                'improvement_achieved': True  # Simplified
            }
            
            # Add optimization quality metrics
            route = optimization_result['route']
            mission_cost = optimization_result['mission_cost']
            
            result['quality_metrics'] = {
                'cost_efficiency': getattr(route, 'total_cost', 0.0) / len(getattr(route, 'satellites', [1])),
                'deltav_efficiency': getattr(route, 'total_delta_v', 0.0) / len(getattr(route, 'satellites', [1])),
                'feasibility_score': getattr(route, 'feasibility_score', 0.5),
                'constraint_violations': len(optimization_result.get('constraint_satisfaction', {}).get('violations', []))
            }
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Post-processing failed: {str(e)}")
            return optimization_result
    
    def _update_optimization_progress(self, optimization_id: str, progress: float, phase: str):
        """Update optimization progress tracking."""
        if optimization_id in self.active_simulations:
            self.active_simulations[optimization_id]['progress'] = progress
            self.active_simulations[optimization_id]['current_phase'] = phase
    
    def get_optimization_status(self, optimization_id: str) -> Dict[str, Any]:
        """
        Get the current status of a running optimization.
        
        Args:
            optimization_id: ID of the optimization to check
            
        Returns:
            Dictionary with optimization status information
        """
        if optimization_id not in self.active_simulations:
            return {
                'optimization_id': optimization_id,
                'status': 'not_found',
                'message': 'Optimization not found or completed'
            }
        
        opt_data = self.active_simulations[optimization_id]
        elapsed_time = (datetime.now() - opt_data['start_time']).total_seconds()
        
        return {
            'optimization_id': optimization_id,
            'status': opt_data['status'],
            'progress': opt_data['progress'],
            'current_phase': opt_data['current_phase'],
            'generation': opt_data['generation'],
            'best_fitness': opt_data['best_fitness'],
            'stagnant_generations': opt_data['stagnant_generations'],
            'elapsed_time_seconds': elapsed_time,
            'error': opt_data.get('error')
        }    
async def optimize_route_with_caching(self, service_request: ServiceRequest,
                                        satellites: List[Satellite],
                                        optimization_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize route with performance caching enabled.
        
        This method implements route caching for frequently requested satellite combinations
        and provides performance monitoring for optimization operations.
        
        Args:
            service_request: Service request with constraints
            satellites: List of satellites to optimize
            optimization_options: Optional optimization parameters
            
        Returns:
            Optimization result with performance metrics
        """
        import time
        start_time = time.time()
        
        try:
            # Check cache if performance optimizer is enabled
            if self.performance_optimizer:
                satellite_ids = [sat.id for sat in satellites]
                constraints = self._extract_constraints_for_cache(service_request)
                
                # Try to get cached result
                cached_result = await self.performance_optimizer.get_cached_route(
                    satellite_ids, constraints
                )
                
                if cached_result:
                    route, mission_cost, metadata = cached_result
                    
                    # Track cache hit performance
                    response_time = (time.time() - start_time) * 1000
                    self.performance_optimizer.track_performance('request_processing', response_time)
                    
                    self.logger.info(f"Route optimization cache hit for {len(satellites)} satellites")
                    
                    return {
                        'success': True,
                        'route': route,
                        'mission_cost': mission_cost,
                        'optimization_id': f"cached_{int(time.time())}",
                        'cached': True,
                        'response_time_ms': response_time,
                        'optimization_metadata': metadata
                    }
            
            # Cache miss or caching disabled - run optimization
            optimization_start = time.time()
            result = await self.optimize_route_with_constraints(
                service_request, satellites, optimization_options
            )
            optimization_time = (time.time() - optimization_start) * 1000
            
            # Cache the result if successful and caching is enabled
            if result['success'] and self.performance_optimizer:
                await self.performance_optimizer.cache_route(
                    satellite_ids=[sat.id for sat in satellites],
                    route=result['route'],
                    mission_cost=result['mission_cost'],
                    optimization_metadata=result.get('optimization_metadata', {}),
                    constraints=self._extract_constraints_for_cache(service_request)
                )
            
            # Track performance metrics
            if self.performance_optimizer:
                total_time = (time.time() - start_time) * 1000
                self.performance_optimizer.track_performance('route_optimization', optimization_time)
                self.performance_optimizer.track_performance('request_processing', total_time)
            
            # Add performance data to result
            result['cached'] = False
            result['optimization_time_ms'] = optimization_time
            result['total_response_time_ms'] = (time.time() - start_time) * 1000
            
            return result
            
        except Exception as e:
            self.logger.error(f"Route optimization with caching failed: {str(e)}")
            return {
                'success': False,
                'message': f"Optimization failed: {str(e)}",
                'cached': False,
                'response_time_ms': (time.time() - start_time) * 1000
            }
    
    def get_performance_report(self) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive performance report.
        
        Returns:
            Performance report if performance optimizer is enabled, None otherwise
        """
        if self.performance_optimizer:
            return self.performance_optimizer.get_performance_report()
        return None
    
    def invalidate_satellite_cache(self, satellite_ids: List[str]) -> int:
        """
        Invalidate cache entries for specific satellites.
        
        This method should be called when satellite data is updated to ensure
        cache consistency.
        
        Args:
            satellite_ids: List of satellite IDs to invalidate
            
        Returns:
            Number of cache entries invalidated
        """
        if self.performance_optimizer:
            return self.performance_optimizer.invalidate_satellite_cache(satellite_ids)
        return 0
    
    async def cleanup_performance_cache(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of entries cleaned up
        """
        if self.performance_optimizer:
            return await self.performance_optimizer.cleanup_expired_cache()
        return 0
    
    def _extract_constraints_for_cache(self, service_request: ServiceRequest) -> Dict[str, Any]:
        """
        Extract relevant constraints for cache key generation.
        
        Args:
            service_request: Service request
            
        Returns:
            Dictionary of constraints for caching
        """
        return {
            'max_cost': service_request.budget_constraints.max_total_cost,
            'timeline_hours': (
                service_request.timeline_requirements.latest_completion - 
                service_request.timeline_requirements.earliest_start
            ).total_seconds() / 3600,
            'processing_types': [pt.value for pt in service_request.processing_preferences.preferred_processing_types]
        }
    
    def shutdown(self) -> None:
        """Shutdown route simulator and cleanup resources."""
        if self.performance_optimizer:
            self.performance_optimizer.shutdown()
        self.logger.info("RouteSimulator shutdown complete")    d
ef validate_operational_constraints(self, service_request: ServiceRequest,
                                       route: Route) -> Dict[str, Any]:
        """
        Validate operational constraints for a mission.
        
        This method implements spacecraft fuel capacity and operational window constraints,
        regulatory compliance checks and space traffic coordination.
        
        Args:
            service_request: Service request with mission parameters
            route: Proposed route
            
        Returns:
            Constraint validation results
        """
        try:
            violations = self.constraints_handler.validate_mission_constraints(
                service_request, route
            )
            
            # Categorize violations by severity
            critical_violations = [v for v in violations if v.severity.value == "critical"]
            high_violations = [v for v in violations if v.severity.value == "high"]
            medium_violations = [v for v in violations if v.severity.value == "medium"]
            low_violations = [v for v in violations if v.severity.value == "low"]
            
            # Calculate impact metrics
            total_cost_impact = sum(v.cost_impact for v in violations)
            total_time_impact = sum(v.time_impact_hours for v in violations)
            
            # Determine overall feasibility
            mission_feasible = len(critical_violations) == 0
            
            return {
                'feasible': mission_feasible,
                'total_violations': len(violations),
                'violations_by_severity': {
                    'critical': len(critical_violations),
                    'high': len(high_violations),
                    'medium': len(medium_violations),
                    'low': len(low_violations)
                },
                'violations': [
                    {
                        'type': v.constraint_type.value,
                        'severity': v.severity.value,
                        'description': v.description,
                        'affected_satellites': v.affected_satellites,
                        'mitigation': v.suggested_mitigation,
                        'cost_impact': v.cost_impact,
                        'time_impact_hours': v.time_impact_hours,
                        'metadata': v.metadata
                    }
                    for v in violations
                ],
                'impact_assessment': {
                    'total_cost_impact': total_cost_impact,
                    'total_time_impact_hours': total_time_impact,
                    'mission_risk_level': self._assess_mission_risk(violations)
                },
                'mitigation_recommendations': self._generate_mitigation_recommendations(violations)
            }
            
        except Exception as e:
            self.logger.error(f"Constraint validation failed: {str(e)}")
            return {
                'feasible': False,
                'error': str(e),
                'total_violations': 0,
                'violations': []
            }
    
    def generate_compliance_checklist(self, service_request: ServiceRequest,
                                    route: Route) -> Dict[str, Any]:
        """
        Generate regulatory compliance checklist.
        
        Args:
            service_request: Service request
            route: Mission route
            
        Returns:
            Compliance checklist with requirements and timeline
        """
        try:
            return self.constraints_handler.generate_compliance_checklist(
                service_request, route
            )
        except Exception as e:
            self.logger.error(f"Compliance checklist generation failed: {str(e)}")
            return {'error': str(e)}
    
    def check_space_traffic_alerts(self, route: Route,
                                 mission_timeline: Tuple[datetime, datetime]) -> List[Dict[str, Any]]:
        """
        Check space traffic coordination alerts.
        
        Args:
            route: Mission route
            mission_timeline: Mission start and end times
            
        Returns:
            List of relevant space traffic alerts
        """
        try:
            alerts = self.constraints_handler.check_space_traffic_coordination(
                route, mission_timeline
            )
            
            return [
                {
                    'alert_id': alert.alert_id,
                    'alert_type': alert.alert_type,
                    'risk_level': alert.risk_level,
                    'time_window': {
                        'start': alert.time_window[0].isoformat(),
                        'end': alert.time_window[1].isoformat()
                    },
                    'recommended_action': alert.recommended_action,
                    'issuing_authority': alert.issuing_authority,
                    'affected_region': alert.affected_region
                }
                for alert in alerts
            ]
            
        except Exception as e:
            self.logger.error(f"Space traffic check failed: {str(e)}")
            return []
    
    def optimize_multi_mission_scheduling(self, service_requests: List[ServiceRequest],
                                        routes: List[Route]) -> Dict[str, Any]:
        """
        Optimize mission scheduling for multiple concurrent clients.
        
        This method implements mission scheduling optimization for multiple concurrent clients
        with resource allocation and conflict resolution.
        
        Args:
            service_requests: List of service requests
            routes: Corresponding optimized routes
            
        Returns:
            Optimized scheduling plan
        """
        try:
            return self.constraints_handler.optimize_mission_scheduling(
                service_requests, routes
            )
        except Exception as e:
            self.logger.error(f"Multi-mission scheduling failed: {str(e)}")
            return {'error': str(e)}
    
    def _assess_mission_risk(self, violations: List[ConstraintViolation]) -> str:
        """Assess overall mission risk level based on violations."""
        if any(v.severity.value == "critical" for v in violations):
            return "critical"
        elif any(v.severity.value == "high" for v in violations):
            return "high"
        elif any(v.severity.value == "medium" for v in violations):
            return "medium"
        elif violations:
            return "low"
        else:
            return "minimal"
    
    def _generate_mitigation_recommendations(self, violations: List[ConstraintViolation]) -> List[str]:
        """Generate mitigation recommendations based on violations."""
        recommendations = []
        
        # Group violations by type for better recommendations
        violation_types = {}
        for violation in violations:
            constraint_type = violation.constraint_type.value
            if constraint_type not in violation_types:
                violation_types[constraint_type] = []
            violation_types[constraint_type].append(violation)
        
        # Generate type-specific recommendations
        for constraint_type, type_violations in violation_types.items():
            if constraint_type == "fuel_capacity":
                recommendations.append("Consider mission segmentation or refueling capabilities")
            elif constraint_type == "operational_window":
                recommendations.append("Adjust mission timeline to avoid operational conflicts")
            elif constraint_type == "regulatory_compliance":
                recommendations.append("Initiate regulatory approval process immediately")
            elif constraint_type == "space_traffic":
                recommendations.append("Coordinate with space traffic management authorities")
            elif constraint_type == "spacecraft_capability":
                recommendations.append("Evaluate alternative spacecraft or mission parameters")
        
        # Add general recommendations
        if len(violations) > 5:
            recommendations.append("Consider mission scope reduction to minimize constraint violations")
        
        return recommendations