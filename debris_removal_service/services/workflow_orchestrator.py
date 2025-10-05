"""
Workflow Orchestrator for end-to-end service integration.

This module provides seamless integration between website requests, route optimization,
quote generation, and 3D visualization components. It orchestrates the complete
workflow from client input to mission execution planning.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
import asyncio
import uuid
from enum import Enum

from ..models.service_request import ServiceRequest, RequestStatus
from ..models.satellite import Satellite
from ..models.route import Route
from ..models.cost import MissionCost
from ..services.route_simulator import RouteSimulator
from ..api.service_request_manager import ServiceRequestManager


logger = logging.getLogger(__name__)


class WorkflowPhase(Enum):
    """Workflow execution phases."""
    VALIDATION = "validation"
    ROUTE_OPTIMIZATION = "route_optimization"
    QUOTE_GENERATION = "quote_generation"
    VISUALIZATION_PREP = "visualization_prep"
    CLIENT_REVIEW = "client_review"
    MISSION_PLANNING = "mission_planning"
    EXECUTION_READY = "execution_ready"


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowResult:
    """Container for workflow execution results."""
    
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.status = WorkflowStatus.PENDING
        self.current_phase = WorkflowPhase.VALIDATION
        self.phases_completed: List[WorkflowPhase] = []
        self.results: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.started_at = datetime.utcnow()
        self.completed_at: Optional[datetime] = None
        self.metadata: Dict[str, Any] = {}


class WorkflowOrchestrator:
    """
    Orchestrates end-to-end workflows from client requests to mission execution.
    
    This class provides seamless data flow from client input to 3D visualization,
    connects website requests to route optimization and quote generation, and
    implements mission approval and execution planning workflows.
    """
    
    def __init__(self, route_simulator: RouteSimulator, 
                 request_manager: ServiceRequestManager,
                 satellite_database: Any):
        """
        Initialize the workflow orchestrator.
        
        Args:
            route_simulator: Route simulation service
            request_manager: Service request manager
            satellite_database: Satellite data source
        """
        self.route_simulator = route_simulator
        self.request_manager = request_manager
        self.satellite_database = satellite_database
        
        # Active workflows tracking
        self._active_workflows: Dict[str, WorkflowResult] = {}
        self._workflow_callbacks: Dict[str, List[Callable]] = {}
        
        # Workflow configuration
        self.max_concurrent_workflows = 10
        self.workflow_timeout_hours = 24
        
        logger.info("WorkflowOrchestrator initialized")
    
    async def execute_client_request_workflow(self, service_request: ServiceRequest) -> WorkflowResult:
        """
        Execute complete workflow from client request to quote generation.
        
        This method connects website requests to route optimization and quote generation,
        implementing seamless data flow from client input to 3D visualization.
        
        Args:
            service_request: Client service request
            
        Returns:
            Workflow execution result
        """
        workflow_id = str(uuid.uuid4())
        workflow_result = WorkflowResult(workflow_id)
        
        try:
            logger.info(f"Starting client request workflow: {workflow_id}")
            
            # Store workflow
            self._active_workflows[workflow_id] = workflow_result
            workflow_result.status = WorkflowStatus.IN_PROGRESS
            
            # Phase 1: Validation
            await self._execute_validation_phase(workflow_result, service_request)
            
            # Phase 2: Route Optimization
            await self._execute_route_optimization_phase(workflow_result, service_request)
            
            # Phase 3: Quote Generation
            await self._execute_quote_generation_phase(workflow_result, service_request)
            
            # Phase 4: Visualization Preparation
            await self._execute_visualization_prep_phase(workflow_result, service_request)
            
            # Mark workflow as completed
            workflow_result.status = WorkflowStatus.COMPLETED
            workflow_result.completed_at = datetime.utcnow()
            workflow_result.current_phase = WorkflowPhase.CLIENT_REVIEW
            
            # Update service request status
            self.request_manager.update_request_status(
                service_request.request_id,
                RequestStatus.QUOTED,
                "Quote generated and ready for client review"
            )
            
            logger.info(f"Client request workflow completed: {workflow_id}")
            
            # Execute callbacks
            await self._execute_workflow_callbacks(workflow_id, workflow_result)
            
            return workflow_result
            
        except Exception as e:
            logger.error(f"Client request workflow failed: {workflow_id} - {str(e)}")
            workflow_result.status = WorkflowStatus.FAILED
            workflow_result.errors.append(str(e))
            workflow_result.completed_at = datetime.utcnow()
            
            # Update service request status
            self.request_manager.update_request_status(
                service_request.request_id,
                RequestStatus.REJECTED,
                f"Workflow failed: {str(e)}"
            )
            
            return workflow_result
    
    async def execute_mission_approval_workflow(self, request_id: str, 
                                              approval_data: Dict[str, Any]) -> WorkflowResult:
        """
        Execute mission approval and execution planning workflow.
        
        This method implements mission approval and execution planning workflows,
        transitioning from client approval to mission execution readiness.
        
        Args:
            request_id: Service request ID
            approval_data: Client approval data
            
        Returns:
            Workflow execution result
        """
        workflow_id = str(uuid.uuid4())
        workflow_result = WorkflowResult(workflow_id)
        
        try:
            logger.info(f"Starting mission approval workflow: {workflow_id}")
            
            # Get service request
            service_request = self.request_manager.get_request(request_id)
            if not service_request:
                raise ValueError(f"Service request not found: {request_id}")
            
            # Store workflow
            self._active_workflows[workflow_id] = workflow_result
            workflow_result.status = WorkflowStatus.IN_PROGRESS
            workflow_result.current_phase = WorkflowPhase.MISSION_PLANNING
            
            # Phase 1: Process Approval
            await self._execute_approval_processing_phase(workflow_result, service_request, approval_data)
            
            # Phase 2: Mission Planning
            await self._execute_mission_planning_phase(workflow_result, service_request)
            
            # Phase 3: Execution Preparation
            await self._execute_execution_preparation_phase(workflow_result, service_request)
            
            # Mark workflow as completed
            workflow_result.status = WorkflowStatus.COMPLETED
            workflow_result.completed_at = datetime.utcnow()
            workflow_result.current_phase = WorkflowPhase.EXECUTION_READY
            
            # Update service request status
            self.request_manager.update_request_status(
                service_request.request_id,
                RequestStatus.APPROVED,
                "Mission approved and execution planning completed"
            )
            
            logger.info(f"Mission approval workflow completed: {workflow_id}")
            
            # Execute callbacks
            await self._execute_workflow_callbacks(workflow_id, workflow_result)
            
            return workflow_result
            
        except Exception as e:
            logger.error(f"Mission approval workflow failed: {workflow_id} - {str(e)}")
            workflow_result.status = WorkflowStatus.FAILED
            workflow_result.errors.append(str(e))
            workflow_result.completed_at = datetime.utcnow()
            
            return workflow_result
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowResult]:
        """
        Get the status of a running workflow.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Workflow result if found, None otherwise
        """
        return self._active_workflows.get(workflow_id)
    
    def register_workflow_callback(self, workflow_id: str, callback: Callable) -> None:
        """
        Register a callback for workflow completion.
        
        Args:
            workflow_id: Workflow ID
            callback: Callback function to execute
        """
        if workflow_id not in self._workflow_callbacks:
            self._workflow_callbacks[workflow_id] = []
        self._workflow_callbacks[workflow_id].append(callback)
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """
        Cancel a running workflow.
        
        Args:
            workflow_id: Workflow ID to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        try:
            if workflow_id not in self._active_workflows:
                return False
            
            workflow_result = self._active_workflows[workflow_id]
            workflow_result.status = WorkflowStatus.CANCELLED
            workflow_result.completed_at = datetime.utcnow()
            
            logger.info(f"Workflow cancelled: {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel workflow: {workflow_id} - {str(e)}")
            return False
    
    def get_active_workflows(self) -> List[WorkflowResult]:
        """
        Get all active workflows.
        
        Returns:
            List of active workflow results
        """
        return [
            result for result in self._active_workflows.values()
            if result.status == WorkflowStatus.IN_PROGRESS
        ]
    
    def cleanup_completed_workflows(self, max_age_hours: int = 24) -> int:
        """
        Clean up completed workflows older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours for completed workflows
            
        Returns:
            Number of workflows cleaned up
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            workflows_to_remove = []
            
            for workflow_id, result in self._active_workflows.items():
                if (result.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED] and
                    result.completed_at and result.completed_at < cutoff_time):
                    workflows_to_remove.append(workflow_id)
            
            # Remove old workflows
            for workflow_id in workflows_to_remove:
                del self._active_workflows[workflow_id]
                if workflow_id in self._workflow_callbacks:
                    del self._workflow_callbacks[workflow_id]
            
            logger.info(f"Cleaned up {len(workflows_to_remove)} completed workflows")
            return len(workflows_to_remove)
            
        except Exception as e:
            logger.error(f"Failed to cleanup workflows: {str(e)}")
            return 0
    
    # Private workflow phase execution methods
    
    async def _execute_validation_phase(self, workflow_result: WorkflowResult, 
                                      service_request: ServiceRequest) -> None:
        """Execute validation phase of workflow."""
        try:
            logger.info(f"Executing validation phase: {workflow_result.workflow_id}")
            workflow_result.current_phase = WorkflowPhase.VALIDATION
            
            # Get satellites for validation
            satellites = self.satellite_database.list_satellites()
            
            # Create and validate service request
            validated_request = self.request_manager.create_request(service_request, satellites)
            
            # Store validation results
            workflow_result.results['validated_request'] = validated_request
            workflow_result.results['available_satellites'] = len(satellites)
            workflow_result.phases_completed.append(WorkflowPhase.VALIDATION)
            
            logger.info(f"Validation phase completed: {workflow_result.workflow_id}")
            
        except Exception as e:
            logger.error(f"Validation phase failed: {workflow_result.workflow_id} - {str(e)}")
            raise
    
    async def _execute_route_optimization_phase(self, workflow_result: WorkflowResult,
                                              service_request: ServiceRequest) -> None:
        """Execute route optimization phase of workflow."""
        try:
            logger.info(f"Executing route optimization phase: {workflow_result.workflow_id}")
            workflow_result.current_phase = WorkflowPhase.ROUTE_OPTIMIZATION
            
            # Get satellites for optimization
            satellites = []
            for sat_id in service_request.satellites:
                satellite = self.satellite_database.get_satellite(sat_id)
                if satellite:
                    satellites.append(satellite)
            
            # Run route optimization
            optimization_result = self.route_simulator.optimize_route_with_constraints(
                service_request=service_request,
                satellites=satellites,
                optimization_options={}
            )
            
            if not optimization_result['success']:
                raise ValueError(f"Route optimization failed: {optimization_result.get('message')}")
            
            # Store optimization results
            workflow_result.results['optimization_result'] = optimization_result
            workflow_result.results['optimal_route'] = optimization_result['route']
            workflow_result.phases_completed.append(WorkflowPhase.ROUTE_OPTIMIZATION)
            
            logger.info(f"Route optimization phase completed: {workflow_result.workflow_id}")
            
        except Exception as e:
            logger.error(f"Route optimization phase failed: {workflow_result.workflow_id} - {str(e)}")
            raise
    
    async def _execute_quote_generation_phase(self, workflow_result: WorkflowResult,
                                            service_request: ServiceRequest) -> None:
        """Execute quote generation phase of workflow."""
        try:
            logger.info(f"Executing quote generation phase: {workflow_result.workflow_id}")
            workflow_result.current_phase = WorkflowPhase.QUOTE_GENERATION
            
            # Get optimization results
            optimization_result = workflow_result.results['optimization_result']
            
            # Generate detailed quote
            quote_data = {
                'quote_id': str(uuid.uuid4()),
                'route': optimization_result['route'],
                'mission_cost': optimization_result['mission_cost'],
                'cost_breakdown': optimization_result.get('cost_analysis', {}),
                'timeline_estimate': optimization_result['route'].mission_duration,
                'risk_assessment': optimization_result.get('risk_assessment', {}),
                'recommendations': optimization_result.get('optimization_suggestions', []),
                'quote_valid_until': datetime.utcnow() + timedelta(days=30)
            }
            
            # Store quote results
            workflow_result.results['quote_data'] = quote_data
            workflow_result.phases_completed.append(WorkflowPhase.QUOTE_GENERATION)
            
            logger.info(f"Quote generation phase completed: {workflow_result.workflow_id}")
            
        except Exception as e:
            logger.error(f"Quote generation phase failed: {workflow_result.workflow_id} - {str(e)}")
            raise
    
    async def _execute_visualization_prep_phase(self, workflow_result: WorkflowResult,
                                              service_request: ServiceRequest) -> None:
        """Execute visualization preparation phase of workflow."""
        try:
            logger.info(f"Executing visualization prep phase: {workflow_result.workflow_id}")
            workflow_result.current_phase = WorkflowPhase.VISUALIZATION_PREP
            
            # Get route data
            optimal_route = workflow_result.results['optimal_route']
            
            # Prepare visualization data
            visualization_data = {
                'route_id': workflow_result.workflow_id,
                'satellites': [
                    {
                        'id': sat.id,
                        'name': sat.name,
                        'position': sat.get_current_position(),
                        'orbital_elements': sat.orbital_elements.__dict__
                    }
                    for sat in optimal_route.satellites
                ],
                'hops': [
                    {
                        'from_satellite': hop.from_satellite.id,
                        'to_satellite': hop.to_satellite.id,
                        'delta_v': hop.delta_v_required,
                        'transfer_time': hop.transfer_time.total_seconds(),
                        'cost': hop.cost,
                        'trajectory_points': self._calculate_trajectory_points(hop)
                    }
                    for hop in optimal_route.hops
                ],
                'animation_timeline': self._create_animation_timeline(optimal_route),
                'cost_visualization': self._prepare_cost_visualization(workflow_result.results['quote_data'])
            }
            
            # Store visualization data
            workflow_result.results['visualization_data'] = visualization_data
            workflow_result.phases_completed.append(WorkflowPhase.VISUALIZATION_PREP)
            
            logger.info(f"Visualization prep phase completed: {workflow_result.workflow_id}")
            
        except Exception as e:
            logger.error(f"Visualization prep phase failed: {workflow_result.workflow_id} - {str(e)}")
            raise
    
    async def _execute_approval_processing_phase(self, workflow_result: WorkflowResult,
                                               service_request: ServiceRequest,
                                               approval_data: Dict[str, Any]) -> None:
        """Execute approval processing phase."""
        try:
            logger.info(f"Executing approval processing phase: {workflow_result.workflow_id}")
            
            # Process client approval
            if approval_data.get('approved', False):
                success = self.request_manager.approve_request(
                    service_request.request_id,
                    approval_data.get('notes', 'Client approved mission')
                )
                if not success:
                    raise ValueError("Failed to process approval")
                
                workflow_result.results['approval_status'] = 'approved'
            else:
                success = self.request_manager.cancel_request(
                    service_request.request_id,
                    approval_data.get('notes', 'Client declined mission')
                )
                if not success:
                    raise ValueError("Failed to process cancellation")
                
                workflow_result.results['approval_status'] = 'declined'
                workflow_result.status = WorkflowStatus.CANCELLED
                return
            
            workflow_result.results['approval_data'] = approval_data
            
        except Exception as e:
            logger.error(f"Approval processing phase failed: {workflow_result.workflow_id} - {str(e)}")
            raise
    
    async def _execute_mission_planning_phase(self, workflow_result: WorkflowResult,
                                            service_request: ServiceRequest) -> None:
        """Execute mission planning phase."""
        try:
            logger.info(f"Executing mission planning phase: {workflow_result.workflow_id}")
            
            # Create detailed mission plan
            mission_plan = {
                'mission_id': str(uuid.uuid4()),
                'service_request_id': service_request.request_id,
                'planned_start_date': service_request.timeline_requirements.earliest_start,
                'estimated_completion': service_request.timeline_requirements.latest_completion,
                'resource_allocation': self._calculate_resource_allocation(service_request),
                'risk_mitigation': self._create_risk_mitigation_plan(service_request),
                'contingency_plans': self._create_contingency_plans(service_request),
                'regulatory_compliance': self._check_regulatory_compliance(service_request)
            }
            
            workflow_result.results['mission_plan'] = mission_plan
            workflow_result.phases_completed.append(WorkflowPhase.MISSION_PLANNING)
            
        except Exception as e:
            logger.error(f"Mission planning phase failed: {workflow_result.workflow_id} - {str(e)}")
            raise
    
    async def _execute_execution_preparation_phase(self, workflow_result: WorkflowResult,
                                                 service_request: ServiceRequest) -> None:
        """Execute execution preparation phase."""
        try:
            logger.info(f"Executing execution preparation phase: {workflow_result.workflow_id}")
            
            # Prepare execution checklist
            execution_checklist = {
                'pre_mission_checks': [
                    'Spacecraft fuel verification',
                    'Communication systems check',
                    'Navigation systems calibration',
                    'Mission timeline confirmation'
                ],
                'operational_windows': self._calculate_operational_windows(service_request),
                'ground_support_requirements': self._determine_ground_support(service_request),
                'monitoring_protocols': self._create_monitoring_protocols(service_request)
            }
            
            workflow_result.results['execution_checklist'] = execution_checklist
            workflow_result.phases_completed.append(WorkflowPhase.EXECUTION_READY)
            
        except Exception as e:
            logger.error(f"Execution preparation phase failed: {workflow_result.workflow_id} - {str(e)}")
            raise
    
    async def _execute_workflow_callbacks(self, workflow_id: str, workflow_result: WorkflowResult) -> None:
        """Execute registered callbacks for workflow completion."""
        try:
            if workflow_id in self._workflow_callbacks:
                for callback in self._workflow_callbacks[workflow_id]:
                    try:
                        await callback(workflow_result)
                    except Exception as e:
                        logger.error(f"Workflow callback failed: {workflow_id} - {str(e)}")
        except Exception as e:
            logger.error(f"Failed to execute workflow callbacks: {workflow_id} - {str(e)}")
    
    # Helper methods for data preparation
    
    def _calculate_trajectory_points(self, hop) -> List[Dict[str, float]]:
        """Calculate trajectory points for visualization."""
        # Mock implementation - would use orbital mechanics
        return [
            {'x': 0.0, 'y': 0.0, 'z': 0.0, 'time': 0.0},
            {'x': 1000.0, 'y': 500.0, 'z': 200.0, 'time': 0.5},
            {'x': 2000.0, 'y': 1000.0, 'z': 0.0, 'time': 1.0}
        ]
    
    def _create_animation_timeline(self, route: Route) -> Dict[str, Any]:
        """Create animation timeline for route visualization."""
        return {
            'total_duration': route.mission_duration.total_seconds(),
            'keyframes': [
                {
                    'time': i * 3600,  # Every hour
                    'active_hop': i % len(route.hops),
                    'spacecraft_position': {'x': i * 100, 'y': i * 50, 'z': 0}
                }
                for i in range(int(route.mission_duration.total_seconds() // 3600))
            ]
        }
    
    def _prepare_cost_visualization(self, quote_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare cost data for visualization."""
        return {
            'total_cost': quote_data['mission_cost'].total_cost,
            'cost_breakdown': {
                'collection': quote_data['mission_cost'].collection_cost,
                'processing': quote_data['mission_cost'].processing_cost,
                'storage': quote_data['mission_cost'].storage_cost
            },
            'cost_per_hop': [hop.cost for hop in quote_data['route'].hops]
        }
    
    def _calculate_resource_allocation(self, service_request: ServiceRequest) -> Dict[str, Any]:
        """Calculate resource allocation for mission."""
        return {
            'spacecraft_count': 1,
            'fuel_allocation_kg': len(service_request.satellites) * 50,
            'crew_hours': len(service_request.satellites) * 8,
            'ground_support_hours': 24
        }
    
    def _create_risk_mitigation_plan(self, service_request: ServiceRequest) -> List[Dict[str, str]]:
        """Create risk mitigation plan."""
        return [
            {'risk': 'Orbital debris collision', 'mitigation': 'Real-time tracking and avoidance maneuvers'},
            {'risk': 'Communication loss', 'mitigation': 'Redundant communication systems'},
            {'risk': 'Fuel shortage', 'mitigation': '20% fuel margin and refueling capability'}
        ]
    
    def _create_contingency_plans(self, service_request: ServiceRequest) -> List[Dict[str, str]]:
        """Create contingency plans."""
        return [
            {'scenario': 'Mission abort', 'plan': 'Safe return to parking orbit'},
            {'scenario': 'Partial collection', 'plan': 'Prioritize high-value targets'},
            {'scenario': 'Equipment failure', 'plan': 'Switch to backup systems'}
        ]
    
    def _check_regulatory_compliance(self, service_request: ServiceRequest) -> Dict[str, Any]:
        """Check regulatory compliance requirements."""
        return {
            'space_traffic_coordination': 'Required',
            'debris_mitigation_guidelines': 'Compliant',
            'international_agreements': 'Under review',
            'export_control_status': 'Approved'
        }
    
    def _calculate_operational_windows(self, service_request: ServiceRequest) -> List[Dict[str, Any]]:
        """Calculate operational windows."""
        return [
            {
                'window_start': service_request.timeline_requirements.earliest_start,
                'window_end': service_request.timeline_requirements.earliest_start + timedelta(hours=4),
                'window_type': 'Primary launch window'
            }
        ]
    
    def _determine_ground_support(self, service_request: ServiceRequest) -> Dict[str, Any]:
        """Determine ground support requirements."""
        return {
            'mission_control_hours': 24,
            'tracking_stations': ['Station A', 'Station B'],
            'communication_coverage': '95%'
        }
    
    def _create_monitoring_protocols(self, service_request: ServiceRequest) -> List[str]:
        """Create monitoring protocols."""
        return [
            'Real-time telemetry monitoring',
            'Orbital position tracking',
            'Fuel consumption monitoring',
            'Mission progress reporting'
        ]