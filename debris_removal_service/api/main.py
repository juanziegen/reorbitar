"""
FastAPI application for satellite debris removal service.

This module provides the main FastAPI application with route optimization endpoints,
satellite data retrieval and validation endpoints, and quote generation API with
cost breakdown responses.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import uuid

from ..models.satellite import Satellite
from ..models.service_request import (
    ServiceRequest, RequestStatus, TimelineConstraints, 
    BudgetConstraints, ProcessingPreferences, ProcessingType
)
from ..models.route import Route
from ..models.cost import MissionCost
from ..services.route_simulator import RouteSimulator
from ..services.workflow_orchestrator import WorkflowOrchestrator, WorkflowStatus
from .schemas import (
    RouteOptimizationRequest,
    RouteOptimizationResponse,
    SatelliteDataResponse,
    QuoteRequest,
    QuoteResponse,
    ServiceRequestCreate,
    ServiceRequestResponse,
    ErrorResponse,
    TimelineConstraintsSchema,
    BudgetConstraintsSchema,
    ProcessingPreferencesSchema,
    ProcessingTypeEnum
)
from .dependencies import get_route_simulator, get_satellite_database, get_service_request_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Satellite Debris Removal Service API",
    description="Commercial satellite debris removal service with genetic algorithm route optimization and accurate cost calculations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global workflow orchestrator instance
_workflow_orchestrator: Optional[WorkflowOrchestrator] = None

def get_workflow_orchestrator() -> WorkflowOrchestrator:
    """Get workflow orchestrator dependency."""
    global _workflow_orchestrator
    if _workflow_orchestrator is None:
        from .dependencies import get_route_simulator, get_satellite_database, get_service_request_manager
        _workflow_orchestrator = WorkflowOrchestrator(
            route_simulator=get_route_simulator(),
            request_manager=get_service_request_manager(),
            satellite_database=get_satellite_database()
        )
    return _workflow_orchestrator

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Satellite Debris Removal Service API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


# Route Optimization Endpoints
@app.post("/api/route/optimize", response_model=RouteOptimizationResponse)
async def optimize_route(
    request: RouteOptimizationRequest,
    background_tasks: BackgroundTasks,
    route_simulator: RouteSimulator = Depends(get_route_simulator),
    satellite_db = Depends(get_satellite_database)
):
    """
    Optimize satellite collection route using genetic algorithms.
    
    This endpoint implements the core route optimization functionality that combines
    genetic algorithms with cost calculations and constraint handling.
    """
    try:
        logger.info(f"Route optimization request received for {len(request.satellite_ids)} satellites")
        
        # Validate satellite IDs
        satellites = []
        for sat_id in request.satellite_ids:
            satellite = satellite_db.get_satellite(sat_id)
            if not satellite:
                raise HTTPException(
                    status_code=404,
                    detail=f"Satellite {sat_id} not found"
                )
            satellites.append(satellite)
        
        # Create service request from optimization request
        service_request = ServiceRequest(
            client_id=request.client_id or "api_client",
            satellites=request.satellite_ids,
            timeline_requirements=request.timeline_constraints,
            budget_constraints=request.budget_constraints,
            processing_preferences=request.processing_preferences,
            request_id=str(uuid.uuid4())
        )
        
        # Run route optimization
        optimization_result = route_simulator.optimize_route_with_constraints(
            service_request=service_request,
            satellites=satellites,
            optimization_options=request.optimization_options
        )
        
        if not optimization_result['success']:
            raise HTTPException(
                status_code=400,
                detail=f"Route optimization failed: {optimization_result.get('message', 'Unknown error')}"
            )
        
        # Format response
        response = RouteOptimizationResponse(
            optimization_id=optimization_result['optimization_id'],
            success=True,
            route=optimization_result['route'],
            mission_cost=optimization_result['mission_cost'],
            constraint_analysis=optimization_result['constraint_analysis'],
            convergence_info=optimization_result['convergence_info'],
            optimization_metadata=optimization_result['optimization_metadata']
        )
        
        logger.info(f"Route optimization completed successfully: {optimization_result['optimization_id']}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Route optimization failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Route optimization failed: {str(e)}"
        )


@app.get("/api/route/status/{optimization_id}", response_model=Dict[str, Any])
async def get_optimization_status(
    optimization_id: str,
    route_simulator: RouteSimulator = Depends(get_route_simulator)
):
    """Get the status of a running route optimization."""
    try:
        status = await route_simulator.get_simulation_status(optimization_id)
        return status
    except Exception as e:
        logger.error(f"Failed to get optimization status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get optimization status: {str(e)}"
        )


# Satellite Data Endpoints
@app.get("/api/satellite/{satellite_id}", response_model=SatelliteDataResponse)
async def get_satellite_data(
    satellite_id: str,
    satellite_db = Depends(get_satellite_database)
):
    """
    Retrieve satellite data and orbital parameters.
    
    This endpoint provides satellite data retrieval and validation functionality.
    """
    try:
        satellite = satellite_db.get_satellite(satellite_id)
        if not satellite:
            raise HTTPException(
                status_code=404,
                detail=f"Satellite {satellite_id} not found"
            )
        
        # Validate satellite data
        if not satellite.is_valid():
            logger.warning(f"Satellite {satellite_id} has invalid data")
        
        # Get additional orbital information
        perigee_alt, apogee_alt = satellite.get_altitude_km()
        
        response = SatelliteDataResponse(
            satellite=satellite,
            orbital_info={
                "perigee_altitude_km": perigee_alt,
                "apogee_altitude_km": apogee_alt,
                "orbital_period_minutes": 1440 / satellite.orbital_elements.mean_motion,
                "is_valid": satellite.is_valid()
            },
            last_updated=datetime.utcnow()
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve satellite data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve satellite data: {str(e)}"
        )


@app.get("/api/satellites", response_model=List[SatelliteDataResponse])
async def list_satellites(
    limit: int = 100,
    offset: int = 0,
    filter_valid: bool = True,
    satellite_db = Depends(get_satellite_database)
):
    """List available satellites with optional filtering."""
    try:
        satellites = satellite_db.list_satellites(
            limit=limit,
            offset=offset,
            filter_valid=filter_valid
        )
        
        responses = []
        for satellite in satellites:
            try:
                perigee_alt, apogee_alt = satellite.get_altitude_km()
                response = SatelliteDataResponse(
                    satellite=satellite,
                    orbital_info={
                        "perigee_altitude_km": perigee_alt,
                        "apogee_altitude_km": apogee_alt,
                        "orbital_period_minutes": 1440 / satellite.orbital_elements.mean_motion,
                        "is_valid": satellite.is_valid()
                    },
                    last_updated=datetime.utcnow()
                )
                responses.append(response)
            except Exception as e:
                logger.warning(f"Failed to process satellite {satellite.id}: {str(e)}")
                continue
        
        return responses
        
    except Exception as e:
        logger.error(f"Failed to list satellites: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list satellites: {str(e)}"
        )


@app.post("/api/satellite/validate", response_model=Dict[str, Any])
async def validate_satellite_data(satellite: Satellite):
    """Validate satellite data and TLE format."""
    try:
        validation_result = {
            "satellite_id": satellite.id,
            "is_valid": satellite.is_valid(),
            "validation_errors": []
        }
        
        # Detailed validation
        try:
            satellite._validate_tle_format()
        except ValueError as e:
            validation_result["validation_errors"].append(f"TLE format error: {str(e)}")
        
        try:
            perigee_alt, apogee_alt = satellite.get_altitude_km()
            if perigee_alt < 150:  # Below typical LEO
                validation_result["validation_errors"].append("Perigee altitude too low")
            if apogee_alt > 2000:  # Above typical LEO
                validation_result["validation_errors"].append("Apogee altitude too high for LEO")
        except Exception as e:
            validation_result["validation_errors"].append(f"Orbital calculation error: {str(e)}")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Satellite validation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Satellite validation failed: {str(e)}"
        )


# Quote Generation Endpoints
@app.post("/api/quote/generate", response_model=QuoteResponse)
async def generate_quote(
    request: QuoteRequest,
    background_tasks: BackgroundTasks,
    route_simulator: RouteSimulator = Depends(get_route_simulator),
    satellite_db = Depends(get_satellite_database)
):
    """
    Generate cost quote with detailed breakdown.
    
    This endpoint provides quote generation API with cost breakdown responses.
    """
    try:
        logger.info(f"Quote generation request received for {len(request.satellite_ids)} satellites")
        
        # Validate satellite IDs
        satellites = []
        for sat_id in request.satellite_ids:
            satellite = satellite_db.get_satellite(sat_id)
            if not satellite:
                raise HTTPException(
                    status_code=404,
                    detail=f"Satellite {sat_id} not found"
                )
            satellites.append(satellite)
        
        # Create service request
        service_request = ServiceRequest(
            client_id=request.client_id,
            satellites=request.satellite_ids,
            timeline_requirements=request.timeline_constraints,
            budget_constraints=request.budget_constraints,
            processing_preferences=request.processing_preferences,
            request_id=str(uuid.uuid4())
        )
        
        # Run mission simulation for quote generation
        simulation_result = await route_simulator.simulate_mission(
            service_request=service_request,
            satellites=satellites
        )
        
        if not simulation_result['success']:
            raise HTTPException(
                status_code=400,
                detail=f"Quote generation failed: {simulation_result.get('message', 'Unknown error')}"
            )
        
        # Generate quote ID
        quote_id = str(uuid.uuid4())
        
        # Calculate quote validity period (30 days)
        quote_expires = datetime.utcnow().replace(microsecond=0)
        quote_expires = quote_expires.replace(day=quote_expires.day + 30)
        
        # Format quote response
        response = QuoteResponse(
            quote_id=quote_id,
            client_id=request.client_id,
            satellite_ids=request.satellite_ids,
            route=simulation_result['route'],
            mission_cost=simulation_result['mission_cost'],
            cost_breakdown=simulation_result['cost_analysis'],
            processing_options=request.processing_preferences.preferred_processing_types,
            timeline_estimate=simulation_result['route'].mission_duration,
            quote_valid_until=quote_expires,
            risk_assessment=simulation_result['risk_assessment'],
            recommendations=simulation_result['cost_analysis'].get('optimization_suggestions', [])
        )
        
        logger.info(f"Quote generated successfully: {quote_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quote generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Quote generation failed: {str(e)}"
        )


@app.get("/api/quote/{quote_id}", response_model=QuoteResponse)
async def get_quote(quote_id: str):
    """Retrieve a previously generated quote."""
    # This would typically retrieve from a database
    # For now, return a not implemented response
    raise HTTPException(
        status_code=501,
        detail="Quote retrieval not yet implemented"
    )


# Service Request Management Endpoints
@app.post("/api/request/create", response_model=ServiceRequestResponse)
async def create_service_request(
    request: ServiceRequestCreate,
    background_tasks: BackgroundTasks,
    satellite_db = Depends(get_satellite_database),
    request_manager = Depends(get_service_request_manager)
):
    """
    Create a new service request for client request processing and validation.
    
    This endpoint implements client request processing and validation functionality.
    """
    try:
        logger.info(f"Creating service request for client {request.client_id}")
        
        # Get available satellites for validation
        satellites = satellite_db.list_satellites()
        
        # Convert schema to service request model
        service_request = ServiceRequest(
            client_id=request.client_id,
            satellites=request.satellite_ids,
            timeline_requirements=TimelineConstraints(
                earliest_start=request.timeline_constraints.earliest_start,
                latest_completion=request.timeline_constraints.latest_completion,
                preferred_duration=request.timeline_constraints.preferred_duration,
                blackout_periods=request.timeline_constraints.blackout_periods
            ),
            budget_constraints=BudgetConstraints(
                max_total_cost=request.budget_constraints.max_total_cost,
                preferred_cost=request.budget_constraints.preferred_cost,
                cost_breakdown_limits=request.budget_constraints.cost_breakdown_limits,
                payment_terms=request.budget_constraints.payment_terms
            ),
            processing_preferences=ProcessingPreferences(
                preferred_processing_types=[ProcessingType(pt.value) for pt in request.processing_preferences.preferred_processing_types],
                material_priorities=request.processing_preferences.material_priorities,
                processing_timeline=request.processing_preferences.processing_timeline,
                storage_duration=request.processing_preferences.storage_duration,
                special_requirements=request.processing_preferences.special_requirements
            ),
            notes=request.notes,
            contact_info=request.contact_info
        )
        
        # Create and validate the service request
        created_request = request_manager.create_request(service_request, satellites)
        
        # Create response
        response = ServiceRequestResponse(
            request_id=created_request.request_id,
            client_id=created_request.client_id,
            satellite_ids=created_request.satellites,
            status=created_request.status.value,
            timeline_constraints=request.timeline_constraints,
            budget_constraints=request.budget_constraints,
            processing_preferences=request.processing_preferences,
            created_at=created_request.created_at,
            updated_at=created_request.updated_at,
            notes=created_request.notes
        )
        
        logger.info(f"Service request created successfully: {created_request.request_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create service request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create service request: {str(e)}"
        )


@app.get("/api/request/{request_id}", response_model=ServiceRequestResponse)
async def get_service_request(
    request_id: str,
    request_manager = Depends(get_service_request_manager)
):
    """
    Retrieve a service request by ID.
    
    This endpoint provides request history and management functionality.
    """
    try:
        service_request = request_manager.get_request(request_id)
        if not service_request:
            raise HTTPException(
                status_code=404,
                detail=f"Service request {request_id} not found"
            )
        
        # Convert to response schema
        response = ServiceRequestResponse(
            request_id=service_request.request_id,
            client_id=service_request.client_id,
            satellite_ids=service_request.satellites,
            status=service_request.status.value,
            timeline_constraints=TimelineConstraintsSchema(
                earliest_start=service_request.timeline_requirements.earliest_start,
                latest_completion=service_request.timeline_requirements.latest_completion,
                preferred_duration=service_request.timeline_requirements.preferred_duration,
                blackout_periods=service_request.timeline_requirements.blackout_periods
            ),
            budget_constraints=BudgetConstraintsSchema(
                max_total_cost=service_request.budget_constraints.max_total_cost,
                preferred_cost=service_request.budget_constraints.preferred_cost,
                cost_breakdown_limits=service_request.budget_constraints.cost_breakdown_limits,
                payment_terms=service_request.budget_constraints.payment_terms
            ),
            processing_preferences=ProcessingPreferencesSchema(
                preferred_processing_types=[ProcessingTypeEnum(pt.value) for pt in service_request.processing_preferences.preferred_processing_types],
                material_priorities=service_request.processing_preferences.material_priorities,
                processing_timeline=service_request.processing_preferences.processing_timeline,
                storage_duration=service_request.processing_preferences.storage_duration,
                special_requirements=service_request.processing_preferences.special_requirements
            ),
            created_at=service_request.created_at,
            updated_at=service_request.updated_at,
            notes=service_request.notes
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve service request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve service request: {str(e)}"
        )


@app.put("/api/request/{request_id}/status", response_model=Dict[str, Any])
async def update_request_status(
    request_id: str,
    status_update: Dict[str, Any],
    background_tasks: BackgroundTasks,
    request_manager = Depends(get_service_request_manager)
):
    """
    Update service request status for mission status tracking and progress updates.
    
    This endpoint implements mission status tracking and progress updates functionality.
    """
    try:
        logger.info(f"Updating status for request {request_id}")
        
        # Validate status update data
        if "status" not in status_update:
            raise HTTPException(
                status_code=400,
                detail="Status field is required"
            )
        
        new_status = status_update["status"]
        notes = status_update.get("notes", "")
        
        # Validate status value
        try:
            status_enum = RequestStatus(new_status)
        except ValueError:
            valid_statuses = [status.value for status in RequestStatus]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {valid_statuses}"
            )
        
        # Get current request for previous status
        current_request = request_manager.get_request(request_id)
        if not current_request:
            raise HTTPException(
                status_code=404,
                detail=f"Service request {request_id} not found"
            )
        
        previous_status = current_request.status.value
        
        # Update the status
        success = request_manager.update_request_status(request_id, status_enum, notes)
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to update request status"
            )
        
        response = {
            "request_id": request_id,
            "previous_status": previous_status,
            "new_status": new_status,
            "updated_at": datetime.utcnow().isoformat(),
            "notes": notes,
            "success": True
        }
        
        logger.info(f"Status updated successfully for request {request_id}: {new_status}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update request status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update request status: {str(e)}"
        )


@app.get("/api/request/{request_id}/progress", response_model=Dict[str, Any])
async def get_request_progress(
    request_id: str,
    request_manager = Depends(get_service_request_manager)
):
    """
    Get detailed progress information for a service request.
    
    This endpoint provides mission status tracking and progress updates.
    """
    try:
        progress_data = request_manager.get_request_progress(request_id)
        if not progress_data:
            raise HTTPException(
                status_code=404,
                detail=f"Progress data not found for request {request_id}"
            )
        
        return progress_data
        
    except Exception as e:
        logger.error(f"Failed to get request progress: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get request progress: {str(e)}"
        )


@app.get("/api/client/{client_id}/requests", response_model=List[ServiceRequestResponse])
async def list_client_requests(
    client_id: str,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    request_manager = Depends(get_service_request_manager)
):
    """
    List service requests for a specific client.
    
    This endpoint provides request history and quote management functionality.
    """
    try:
        logger.info(f"Listing requests for client {client_id}")
        
        # Validate status filter if provided
        status_filter = None
        if status:
            try:
                status_filter = RequestStatus(status)
            except ValueError:
                valid_statuses = [s.value for s in RequestStatus]
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status filter. Must be one of: {valid_statuses}"
                )
        
        # Get requests from manager
        service_requests = request_manager.list_client_requests(
            client_id, status_filter, limit, offset
        )
        
        # Convert to response schemas
        responses = []
        for req in service_requests:
            response = ServiceRequestResponse(
                request_id=req.request_id,
                client_id=req.client_id,
                satellite_ids=req.satellites,
                status=req.status.value,
                timeline_constraints=TimelineConstraintsSchema(
                    earliest_start=req.timeline_requirements.earliest_start,
                    latest_completion=req.timeline_requirements.latest_completion,
                    preferred_duration=req.timeline_requirements.preferred_duration,
                    blackout_periods=req.timeline_requirements.blackout_periods
                ),
                budget_constraints=BudgetConstraintsSchema(
                    max_total_cost=req.budget_constraints.max_total_cost,
                    preferred_cost=req.budget_constraints.preferred_cost,
                    cost_breakdown_limits=req.budget_constraints.cost_breakdown_limits,
                    payment_terms=req.budget_constraints.payment_terms
                ),
                processing_preferences=ProcessingPreferencesSchema(
                    preferred_processing_types=[ProcessingTypeEnum(pt.value) for pt in req.processing_preferences.preferred_processing_types],
                    material_priorities=req.processing_preferences.material_priorities,
                    processing_timeline=req.processing_preferences.processing_timeline,
                    storage_duration=req.processing_preferences.storage_duration,
                    special_requirements=req.processing_preferences.special_requirements
                ),
                created_at=req.created_at,
                updated_at=req.updated_at,
                notes=req.notes
            )
            responses.append(response)
        
        return responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list client requests: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list client requests: {str(e)}"
        )


@app.post("/api/request/{request_id}/approve", response_model=Dict[str, Any])
async def approve_service_request(
    request_id: str,
    approval_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    request_manager = Depends(get_service_request_manager)
):
    """
    Approve a service request and initiate mission execution.
    
    This endpoint handles client approval and mission initiation.
    """
    try:
        logger.info(f"Processing approval for request {request_id}")
        
        # Validate approval data
        if "approved" not in approval_data:
            raise HTTPException(
                status_code=400,
                detail="Approval status is required"
            )
        
        approved = approval_data["approved"]
        client_notes = approval_data.get("notes", "")
        
        if approved:
            # Approve the request
            success = request_manager.approve_request(request_id, client_notes)
            if not success:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to approve request"
                )
            
            response = {
                "request_id": request_id,
                "status": "approved",
                "approved_at": datetime.utcnow().isoformat(),
                "mission_scheduled": True,
                "estimated_start_date": "2024-02-01T08:00:00Z",
                "notes": client_notes,
                "success": True
            }
        else:
            # Cancel the request
            success = request_manager.cancel_request(request_id, client_notes or "Client declined")
            if not success:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to cancel request"
                )
            
            response = {
                "request_id": request_id,
                "status": "cancelled",
                "cancelled_at": datetime.utcnow().isoformat(),
                "reason": client_notes or "Client declined",
                "success": True
            }
        
        logger.info(f"Request {request_id} {'approved' if approved else 'cancelled'}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process approval: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process approval: {str(e)}"
        )


@app.get("/api/requests/dashboard", response_model=Dict[str, Any])
async def get_requests_dashboard(
    request_manager = Depends(get_service_request_manager)
):
    """
    Get dashboard overview of all service requests.
    
    This endpoint provides request history and management overview.
    """
    try:
        dashboard_data = request_manager.get_dashboard_summary()
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get dashboard data: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# End-
to-End Workflow Integration Endpoints

@app.post("/api/workflow/client-request", response_model=Dict[str, Any])
async def execute_client_request_workflow(
    request: ServiceRequestCreate,
    background_tasks: BackgroundTasks,
    workflow_orchestrator: WorkflowOrchestrator = Depends(get_workflow_orchestrator),
    satellite_db = Depends(get_satellite_database)
):
    """
    Execute complete end-to-end workflow from client request to quote generation.
    
    This endpoint connects website requests to route optimization and quote generation,
    implementing seamless data flow from client input to 3D visualization.
    """
    try:
        logger.info(f"Starting end-to-end workflow for client {request.client_id}")
        
        # Convert schema to service request model
        service_request = ServiceRequest(
            client_id=request.client_id,
            satellites=request.satellite_ids,
            timeline_requirements=TimelineConstraints(
                earliest_start=request.timeline_constraints.earliest_start,
                latest_completion=request.timeline_constraints.latest_completion,
                preferred_duration=request.timeline_constraints.preferred_duration,
                blackout_periods=request.timeline_constraints.blackout_periods
            ),
            budget_constraints=BudgetConstraints(
                max_total_cost=request.budget_constraints.max_total_cost,
                preferred_cost=request.budget_constraints.preferred_cost,
                cost_breakdown_limits=request.budget_constraints.cost_breakdown_limits,
                payment_terms=request.budget_constraints.payment_terms
            ),
            processing_preferences=ProcessingPreferences(
                preferred_processing_types=[ProcessingType(pt.value) for pt in request.processing_preferences.preferred_processing_types],
                material_priorities=request.processing_preferences.material_priorities,
                processing_timeline=request.processing_preferences.processing_timeline,
                storage_duration=request.processing_preferences.storage_duration,
                special_requirements=request.processing_preferences.special_requirements
            ),
            notes=request.notes,
            contact_info=request.contact_info
        )
        
        # Execute workflow in background
        workflow_result = await workflow_orchestrator.execute_client_request_workflow(service_request)
        
        # Return workflow status and initial results
        response = {
            "workflow_id": workflow_result.workflow_id,
            "status": workflow_result.status.value,
            "current_phase": workflow_result.current_phase.value,
            "phases_completed": [phase.value for phase in workflow_result.phases_completed],
            "service_request_id": service_request.request_id,
            "estimated_completion": datetime.utcnow() + timedelta(minutes=30),
            "results": {
                "quote_available": "quote_data" in workflow_result.results,
                "visualization_ready": "visualization_data" in workflow_result.results,
                "route_optimized": "optimal_route" in workflow_result.results
            }
        }
        
        # Include results if workflow completed successfully
        if workflow_result.status == WorkflowStatus.COMPLETED:
            quote_data = workflow_result.results.get('quote_data', {})
            response["quote"] = {
                "quote_id": quote_data.get('quote_id'),
                "total_cost": quote_data.get('mission_cost', {}).get('total_cost', 0),
                "timeline_estimate_hours": quote_data.get('timeline_estimate', timedelta()).total_seconds() / 3600,
                "quote_valid_until": quote_data.get('quote_valid_until', datetime.utcnow()).isoformat()
            }
            
            visualization_data = workflow_result.results.get('visualization_data', {})
            response["visualization"] = {
                "route_id": visualization_data.get('route_id'),
                "satellite_count": len(visualization_data.get('satellites', [])),
                "hop_count": len(visualization_data.get('hops', [])),
                "animation_ready": bool(visualization_data.get('animation_timeline'))
            }
        
        logger.info(f"End-to-end workflow initiated: {workflow_result.workflow_id}")
        return response
        
    except Exception as e:
        logger.error(f"End-to-end workflow failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Workflow execution failed: {str(e)}"
        )


@app.get("/api/workflow/{workflow_id}/status", response_model=Dict[str, Any])
async def get_workflow_status(
    workflow_id: str,
    workflow_orchestrator: WorkflowOrchestrator = Depends(get_workflow_orchestrator)
):
    """
    Get the status of a running end-to-end workflow.
    
    This endpoint provides real-time status updates for workflow execution,
    enabling progress tracking and status monitoring.
    """
    try:
        workflow_result = await workflow_orchestrator.get_workflow_status(workflow_id)
        if not workflow_result:
            raise HTTPException(
                status_code=404,
                detail=f"Workflow {workflow_id} not found"
            )
        
        response = {
            "workflow_id": workflow_result.workflow_id,
            "status": workflow_result.status.value,
            "current_phase": workflow_result.current_phase.value,
            "phases_completed": [phase.value for phase in workflow_result.phases_completed],
            "started_at": workflow_result.started_at.isoformat(),
            "completed_at": workflow_result.completed_at.isoformat() if workflow_result.completed_at else None,
            "errors": workflow_result.errors,
            "progress_percentage": len(workflow_result.phases_completed) / 4 * 100,  # 4 main phases
            "results_available": {
                "validation": "validated_request" in workflow_result.results,
                "route_optimization": "optimal_route" in workflow_result.results,
                "quote_generation": "quote_data" in workflow_result.results,
                "visualization": "visualization_data" in workflow_result.results
            }
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get workflow status: {str(e)}"
        )


@app.get("/api/workflow/{workflow_id}/visualization", response_model=Dict[str, Any])
async def get_workflow_visualization_data(
    workflow_id: str,
    workflow_orchestrator: WorkflowOrchestrator = Depends(get_workflow_orchestrator)
):
    """
    Get visualization data for a completed workflow.
    
    This endpoint provides 3D visualization data prepared during workflow execution,
    enabling seamless integration with frontend visualization components.
    """
    try:
        workflow_result = await workflow_orchestrator.get_workflow_status(workflow_id)
        if not workflow_result:
            raise HTTPException(
                status_code=404,
                detail=f"Workflow {workflow_id} not found"
            )
        
        if "visualization_data" not in workflow_result.results:
            raise HTTPException(
                status_code=400,
                detail="Visualization data not available for this workflow"
            )
        
        visualization_data = workflow_result.results["visualization_data"]
        
        return {
            "workflow_id": workflow_id,
            "route_id": visualization_data["route_id"],
            "satellites": visualization_data["satellites"],
            "hops": visualization_data["hops"],
            "animation_timeline": visualization_data["animation_timeline"],
            "cost_visualization": visualization_data["cost_visualization"],
            "metadata": {
                "total_satellites": len(visualization_data["satellites"]),
                "total_hops": len(visualization_data["hops"]),
                "mission_duration_hours": visualization_data["animation_timeline"]["total_duration"] / 3600,
                "generated_at": workflow_result.completed_at.isoformat() if workflow_result.completed_at else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get visualization data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get visualization data: {str(e)}"
        )


@app.post("/api/workflow/mission-approval", response_model=Dict[str, Any])
async def execute_mission_approval_workflow(
    approval_request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    workflow_orchestrator: WorkflowOrchestrator = Depends(get_workflow_orchestrator)
):
    """
    Execute mission approval and execution planning workflow.
    
    This endpoint implements mission approval and execution planning workflows,
    transitioning approved requests to mission execution readiness.
    """
    try:
        request_id = approval_request.get("request_id")
        if not request_id:
            raise HTTPException(
                status_code=400,
                detail="Request ID is required"
            )
        
        logger.info(f"Starting mission approval workflow for request {request_id}")
        
        # Execute approval workflow
        workflow_result = await workflow_orchestrator.execute_mission_approval_workflow(
            request_id, approval_request
        )
        
        response = {
            "workflow_id": workflow_result.workflow_id,
            "status": workflow_result.status.value,
            "current_phase": workflow_result.current_phase.value,
            "request_id": request_id,
            "approval_status": workflow_result.results.get("approval_status"),
            "mission_ready": workflow_result.status == WorkflowStatus.COMPLETED
        }
        
        # Include mission plan if available
        if "mission_plan" in workflow_result.results:
            mission_plan = workflow_result.results["mission_plan"]
            response["mission_plan"] = {
                "mission_id": mission_plan["mission_id"],
                "planned_start_date": mission_plan["planned_start_date"].isoformat(),
                "estimated_completion": mission_plan["estimated_completion"].isoformat(),
                "resource_allocation": mission_plan["resource_allocation"]
            }
        
        # Include execution checklist if available
        if "execution_checklist" in workflow_result.results:
            response["execution_checklist"] = workflow_result.results["execution_checklist"]
        
        logger.info(f"Mission approval workflow completed: {workflow_result.workflow_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mission approval workflow failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Mission approval workflow failed: {str(e)}"
        )


@app.get("/api/workflow/active", response_model=List[Dict[str, Any]])
async def list_active_workflows(
    workflow_orchestrator: WorkflowOrchestrator = Depends(get_workflow_orchestrator)
):
    """
    List all active workflows for monitoring and management.
    
    This endpoint provides visibility into currently running workflows
    for operational monitoring and management purposes.
    """
    try:
        active_workflows = workflow_orchestrator.get_active_workflows()
        
        response = []
        for workflow in active_workflows:
            workflow_info = {
                "workflow_id": workflow.workflow_id,
                "status": workflow.status.value,
                "current_phase": workflow.current_phase.value,
                "started_at": workflow.started_at.isoformat(),
                "phases_completed": [phase.value for phase in workflow.phases_completed],
                "progress_percentage": len(workflow.phases_completed) / 4 * 100,
                "has_errors": len(workflow.errors) > 0,
                "error_count": len(workflow.errors)
            }
            response.append(workflow_info)
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to list active workflows: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list active workflows: {str(e)}"
        )


@app.delete("/api/workflow/{workflow_id}", response_model=Dict[str, Any])
async def cancel_workflow(
    workflow_id: str,
    workflow_orchestrator: WorkflowOrchestrator = Depends(get_workflow_orchestrator)
):
    """
    Cancel a running workflow.
    
    This endpoint allows cancellation of active workflows for operational
    management and resource cleanup.
    """
    try:
        success = await workflow_orchestrator.cancel_workflow(workflow_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Workflow {workflow_id} not found or cannot be cancelled"
            )
        
        return {
            "workflow_id": workflow_id,
            "status": "cancelled",
            "cancelled_at": datetime.utcnow().isoformat(),
            "success": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel workflow: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel workflow: {str(e)}"
        )


@app.post("/api/workflow/cleanup", response_model=Dict[str, Any])
async def cleanup_completed_workflows(
    max_age_hours: int = 24,
    workflow_orchestrator: WorkflowOrchestrator = Depends(get_workflow_orchestrator)
):
    """
    Clean up completed workflows older than specified age.
    
    This endpoint provides maintenance functionality for workflow cleanup
    and resource management.
    """
    try:
        cleaned_count = workflow_orchestrator.cleanup_completed_workflows(max_age_hours)
        
        return {
            "cleaned_workflows": cleaned_count,
            "max_age_hours": max_age_hours,
            "cleanup_timestamp": datetime.utcnow().isoformat(),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup workflows: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cleanup workflows: {str(e)}"
        )
# Per
formance Optimization and Monitoring Endpoints

@app.get("/api/performance/report", response_model=Dict[str, Any])
async def get_performance_report(
    route_simulator: RouteSimulator = Depends(get_route_simulator)
):
    """
    Get comprehensive performance report with optimization suggestions.
    
    This endpoint provides performance monitoring and optimization suggestions
    for route optimization and database operations.
    """
    try:
        performance_report = route_simulator.get_performance_report()
        
        if not performance_report:
            return {
                "message": "Performance monitoring not enabled",
                "caching_enabled": False,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return performance_report
        
    except Exception as e:
        logger.error(f"Failed to get performance report: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance report: {str(e)}"
        )


@app.post("/api/performance/cache/invalidate", response_model=Dict[str, Any])
async def invalidate_satellite_cache(
    satellite_ids: List[str],
    route_simulator: RouteSimulator = Depends(get_route_simulator)
):
    """
    Invalidate cache entries for specific satellites.
    
    This endpoint allows manual cache invalidation when satellite data is updated
    to ensure cache consistency and data accuracy.
    """
    try:
        if not satellite_ids:
            raise HTTPException(
                status_code=400,
                detail="At least one satellite ID must be provided"
            )
        
        invalidated_count = route_simulator.invalidate_satellite_cache(satellite_ids)
        
        return {
            "satellite_ids": satellite_ids,
            "invalidated_entries": invalidated_count,
            "timestamp": datetime.utcnow().isoformat(),
            "success": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to invalidate cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to invalidate cache: {str(e)}"
        )


@app.post("/api/performance/cache/cleanup", response_model=Dict[str, Any])
async def cleanup_performance_cache(
    route_simulator: RouteSimulator = Depends(get_route_simulator)
):
    """
    Clean up expired cache entries for performance optimization.
    
    This endpoint provides manual cache cleanup functionality for
    performance maintenance and memory management.
    """
    try:
        cleaned_count = await route_simulator.cleanup_performance_cache()
        
        return {
            "cleaned_entries": cleaned_count,
            "timestamp": datetime.utcnow().isoformat(),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cleanup cache: {str(e)}"
        )


@app.get("/api/performance/metrics", response_model=Dict[str, Any])
async def get_performance_metrics(
    route_simulator: RouteSimulator = Depends(get_route_simulator)
):
    """
    Get real-time performance metrics for monitoring.
    
    This endpoint provides current performance metrics including
    cache hit rates, response times, and system resource usage.
    """
    try:
        performance_report = route_simulator.get_performance_report()
        
        if not performance_report:
            return {
                "message": "Performance monitoring not enabled",
                "metrics": {},
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Extract key metrics for dashboard display
        metrics = performance_report.get('metrics', {})
        cache_stats = performance_report.get('cache_statistics', {})
        
        return {
            "cache_hit_rate": metrics.get('cache_hit_rate', 0.0),
            "cache_miss_rate": metrics.get('cache_miss_rate', 0.0),
            "average_response_time_ms": metrics.get('average_response_time_ms', 0.0),
            "total_requests": metrics.get('total_requests', 0),
            "cache_size": cache_stats.get('cache_size', 0),
            "memory_usage_mb": metrics.get('memory_usage_mb', 0.0),
            "database_query_time_ms": metrics.get('database_query_time_ms', 0.0),
            "optimization_time_ms": metrics.get('optimization_time_ms', 0.0),
            "performance_grade": performance_report.get('performance_grade', 'N/A'),
            "last_updated": metrics.get('last_updated', datetime.utcnow().isoformat()),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance metrics: {str(e)}"
        )


@app.get("/api/performance/suggestions", response_model=List[str])
async def get_optimization_suggestions(
    route_simulator: RouteSimulator = Depends(get_route_simulator)
):
    """
    Get performance optimization suggestions.
    
    This endpoint provides actionable optimization suggestions based on
    current system performance and usage patterns.
    """
    try:
        performance_report = route_simulator.get_performance_report()
        
        if not performance_report:
            return ["Performance monitoring not enabled - consider enabling caching for better performance"]
        
        suggestions = performance_report.get('optimization_suggestions', [])
        
        # Add general suggestions if none are available
        if not suggestions:
            suggestions = [
                "System performance is optimal",
                "Continue monitoring for performance trends",
                "Consider periodic cache cleanup for memory management"
            ]
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Failed to get optimization suggestions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get optimization suggestions: {str(e)}"
        )


# Enhanced route optimization endpoint with caching
@app.post("/api/route/optimize-cached", response_model=RouteOptimizationResponse)
async def optimize_route_with_caching(
    request: RouteOptimizationRequest,
    background_tasks: BackgroundTasks,
    route_simulator: RouteSimulator = Depends(get_route_simulator),
    satellite_db = Depends(get_satellite_database)
):
    """
    Optimize satellite collection route with performance caching.
    
    This endpoint implements route caching for frequently requested satellite combinations
    and provides enhanced performance monitoring for optimization operations.
    """
    try:
        logger.info(f"Cached route optimization request received for {len(request.satellite_ids)} satellites")
        
        # Validate satellite IDs
        satellites = []
        for sat_id in request.satellite_ids:
            satellite = satellite_db.get_satellite(sat_id)
            if not satellite:
                raise HTTPException(
                    status_code=404,
                    detail=f"Satellite {sat_id} not found"
                )
            satellites.append(satellite)
        
        # Create service request from optimization request
        service_request = ServiceRequest(
            client_id=request.client_id or "api_client",
            satellites=request.satellite_ids,
            timeline_requirements=request.timeline_constraints,
            budget_constraints=request.budget_constraints,
            processing_preferences=request.processing_preferences,
            request_id=str(uuid.uuid4())
        )
        
        # Run cached route optimization
        optimization_result = await route_simulator.optimize_route_with_caching(
            service_request=service_request,
            satellites=satellites,
            optimization_options=request.optimization_options
        )
        
        if not optimization_result['success']:
            raise HTTPException(
                status_code=400,
                detail=f"Route optimization failed: {optimization_result.get('message', 'Unknown error')}"
            )
        
        # Format response with performance data
        response = RouteOptimizationResponse(
            optimization_id=optimization_result.get('optimization_id', str(uuid.uuid4())),
            success=True,
            route=optimization_result['route'],
            mission_cost=optimization_result['mission_cost'],
            constraint_analysis=optimization_result.get('constraint_analysis', {}),
            convergence_info=optimization_result.get('convergence_info', {}),
            optimization_metadata={
                **optimization_result.get('optimization_metadata', {}),
                'cached_result': optimization_result.get('cached', False),
                'optimization_time_ms': optimization_result.get('optimization_time_ms', 0),
                'total_response_time_ms': optimization_result.get('total_response_time_ms', 0),
                'performance_optimized': True
            }
        )
        
        cache_status = "hit" if optimization_result.get('cached', False) else "miss"
        logger.info(f"Cached route optimization completed: {optimization_result.get('optimization_id')} (cache {cache_status})")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cached route optimization failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Route optimization failed: {str(e)}"
        )#
 Operational Constraints and Compliance Endpoints

@app.post("/api/constraints/validate", response_model=Dict[str, Any])
async def validate_operational_constraints(
    validation_request: Dict[str, Any],
    route_simulator: RouteSimulator = Depends(get_route_simulator),
    satellite_db = Depends(get_satellite_database)
):
    """
    Validate operational constraints for a mission.
    
    This endpoint implements spacecraft fuel capacity and operational window constraints,
    regulatory compliance checks and space traffic coordination.
    """
    try:
        # Extract request data
        request_id = validation_request.get("request_id")
        satellite_ids = validation_request.get("satellite_ids", [])
        
        if not request_id or not satellite_ids:
            raise HTTPException(
                status_code=400,
                detail="Request ID and satellite IDs are required"
            )
        
        # Get service request (mock for now - would come from database)
        service_request = ServiceRequest(
            client_id=validation_request.get("client_id", "test_client"),
            satellites=satellite_ids,
            timeline_requirements=TimelineConstraints(
                earliest_start=datetime.fromisoformat(validation_request.get("earliest_start", datetime.utcnow().isoformat())),
                latest_completion=datetime.fromisoformat(validation_request.get("latest_completion", (datetime.utcnow() + timedelta(days=30)).isoformat())),
                preferred_duration=timedelta(hours=validation_request.get("preferred_duration_hours", 168))
            ),
            budget_constraints=BudgetConstraints(
                max_total_cost=validation_request.get("max_total_cost", 1000000.0),
                preferred_cost=validation_request.get("preferred_cost", 800000.0)
            ),
            processing_preferences=ProcessingPreferences(
                preferred_processing_types=[ProcessingType.ISS_RECYCLING]
            ),
            request_id=request_id
        )
        
        # Get satellites
        satellites = []
        for sat_id in satellite_ids:
            satellite = satellite_db.get_satellite(sat_id)
            if satellite:
                satellites.append(satellite)
        
        # Create mock route for validation (in real implementation, this would be optimized route)
        from ..models.route import Route, Hop
        mock_route = Route(
            satellites=satellites,
            hops=[],
            total_delta_v=sum(100.0 for _ in satellites),  # Mock delta-v
            total_cost=len(satellites) * 50000.0,
            mission_duration=timedelta(hours=len(satellites) * 24),
            feasibility_score=0.8
        )
        
        # Validate constraints
        validation_result = route_simulator.validate_operational_constraints(
            service_request, mock_route
        )
        
        return {
            "request_id": request_id,
            "validation_timestamp": datetime.utcnow().isoformat(),
            "constraint_validation": validation_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Constraint validation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Constraint validation failed: {str(e)}"
        )


@app.post("/api/compliance/checklist", response_model=Dict[str, Any])
async def generate_compliance_checklist(
    checklist_request: Dict[str, Any],
    route_simulator: RouteSimulator = Depends(get_route_simulator),
    satellite_db = Depends(get_satellite_database)
):
    """
    Generate regulatory compliance checklist for a mission.
    
    This endpoint provides regulatory compliance checks and documentation requirements
    for space debris removal missions.
    """
    try:
        # Extract request data
        request_id = checklist_request.get("request_id")
        satellite_ids = checklist_request.get("satellite_ids", [])
        
        if not request_id or not satellite_ids:
            raise HTTPException(
                status_code=400,
                detail="Request ID and satellite IDs are required"
            )
        
        # Create service request
        service_request = ServiceRequest(
            client_id=checklist_request.get("client_id", "test_client"),
            satellites=satellite_ids,
            timeline_requirements=TimelineConstraints(
                earliest_start=datetime.fromisoformat(checklist_request.get("earliest_start", datetime.utcnow().isoformat())),
                latest_completion=datetime.fromisoformat(checklist_request.get("latest_completion", (datetime.utcnow() + timedelta(days=30)).isoformat()))
            ),
            budget_constraints=BudgetConstraints(
                max_total_cost=checklist_request.get("max_total_cost", 1000000.0)
            ),
            processing_preferences=ProcessingPreferences(
                preferred_processing_types=[ProcessingType.ISS_RECYCLING]
            ),
            request_id=request_id
        )
        
        # Get satellites and create mock route
        satellites = []
        for sat_id in satellite_ids:
            satellite = satellite_db.get_satellite(sat_id)
            if satellite:
                satellites.append(satellite)
        
        from ..models.route import Route
        mock_route = Route(
            satellites=satellites,
            hops=[],
            total_delta_v=sum(100.0 for _ in satellites),
            total_cost=len(satellites) * 50000.0,
            mission_duration=timedelta(hours=len(satellites) * 24),
            feasibility_score=0.8
        )
        
        # Generate compliance checklist
        checklist = route_simulator.generate_compliance_checklist(
            service_request, mock_route
        )
        
        return {
            "request_id": request_id,
            "generated_at": datetime.utcnow().isoformat(),
            "compliance_checklist": checklist
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compliance checklist generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Compliance checklist generation failed: {str(e)}"
        )


@app.get("/api/space-traffic/alerts", response_model=List[Dict[str, Any]])
async def get_space_traffic_alerts(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    risk_level: Optional[str] = None,
    route_simulator: RouteSimulator = Depends(get_route_simulator)
):
    """
    Get space traffic coordination alerts for mission planning.
    
    This endpoint provides space traffic coordination information
    for safe mission planning and execution.
    """
    try:
        # Parse time parameters
        if start_time:
            start_dt = datetime.fromisoformat(start_time)
        else:
            start_dt = datetime.utcnow()
        
        if end_time:
            end_dt = datetime.fromisoformat(end_time)
        else:
            end_dt = start_dt + timedelta(days=30)
        
        # Create mock route for alert checking
        from ..models.route import Route
        mock_route = Route(
            satellites=[],
            hops=[],
            total_delta_v=0.0,
            total_cost=0.0,
            mission_duration=timedelta(hours=24),
            feasibility_score=1.0
        )
        
        # Get space traffic alerts
        alerts = route_simulator.check_space_traffic_alerts(
            mock_route, (start_dt, end_dt)
        )
        
        # Filter by risk level if specified
        if risk_level:
            alerts = [alert for alert in alerts if alert.get('risk_level') == risk_level]
        
        return alerts
        
    except Exception as e:
        logger.error(f"Space traffic alerts retrieval failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Space traffic alerts retrieval failed: {str(e)}"
        )


@app.post("/api/scheduling/optimize", response_model=Dict[str, Any])
async def optimize_multi_mission_scheduling(
    scheduling_request: Dict[str, Any],
    route_simulator: RouteSimulator = Depends(get_route_simulator),
    satellite_db = Depends(get_satellite_database)
):
    """
    Optimize mission scheduling for multiple concurrent clients.
    
    This endpoint implements mission scheduling optimization for multiple concurrent clients
    with resource allocation and conflict resolution.
    """
    try:
        missions = scheduling_request.get("missions", [])
        
        if not missions:
            raise HTTPException(
                status_code=400,
                detail="At least one mission must be provided"
            )
        
        # Create service requests and routes
        service_requests = []
        routes = []
        
        for mission in missions:
            # Create service request
            service_request = ServiceRequest(
                client_id=mission.get("client_id", "test_client"),
                satellites=mission.get("satellite_ids", []),
                timeline_requirements=TimelineConstraints(
                    earliest_start=datetime.fromisoformat(mission.get("earliest_start", datetime.utcnow().isoformat())),
                    latest_completion=datetime.fromisoformat(mission.get("latest_completion", (datetime.utcnow() + timedelta(days=30)).isoformat()))
                ),
                budget_constraints=BudgetConstraints(
                    max_total_cost=mission.get("max_total_cost", 1000000.0)
                ),
                processing_preferences=ProcessingPreferences(
                    preferred_processing_types=[ProcessingType.ISS_RECYCLING]
                ),
                request_id=mission.get("request_id", f"mission_{len(service_requests)}")
            )
            service_requests.append(service_request)
            
            # Create mock route
            satellites = []
            for sat_id in mission.get("satellite_ids", []):
                satellite = satellite_db.get_satellite(sat_id)
                if satellite:
                    satellites.append(satellite)
            
            from ..models.route import Route
            route = Route(
                satellites=satellites,
                hops=[],
                total_delta_v=len(satellites) * 100.0,
                total_cost=len(satellites) * 50000.0,
                mission_duration=timedelta(hours=len(satellites) * 24),
                feasibility_score=0.8
            )
            routes.append(route)
        
        # Optimize scheduling
        scheduling_result = route_simulator.optimize_multi_mission_scheduling(
            service_requests, routes
        )
        
        return {
            "scheduling_request_id": str(uuid.uuid4()),
            "optimization_timestamp": datetime.utcnow().isoformat(),
            "scheduling_result": scheduling_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-mission scheduling failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Multi-mission scheduling failed: {str(e)}"
        )


@app.get("/api/constraints/spacecraft-capabilities", response_model=Dict[str, Any])
async def get_spacecraft_capabilities(
    route_simulator: RouteSimulator = Depends(get_route_simulator)
):
    """
    Get current spacecraft capabilities and operational constraints.
    
    This endpoint provides information about spacecraft fuel capacity,
    operational windows, and capability constraints.
    """
    try:
        # Get default spacecraft capabilities
        capabilities = route_simulator.constraints_handler.default_spacecraft
        
        return {
            "spacecraft_capabilities": {
                "max_fuel_capacity_kg": capabilities.max_fuel_capacity_kg,
                "fuel_consumption_rate_kg_per_ms": capabilities.fuel_consumption_rate_kg_per_ms,
                "max_delta_v_per_maneuver": capabilities.max_delta_v_per_maneuver,
                "max_mission_duration_hours": capabilities.max_mission_duration_hours,
                "operational_altitude_range_km": capabilities.operational_altitude_range,
                "max_payload_capacity_kg": capabilities.max_payload_capacity_kg,
                "communication_range_km": capabilities.communication_range_km,
                "power_generation_watts": capabilities.power_generation_watts,
                "thermal_limits": capabilities.thermal_limits
            },
            "operational_windows_count": len(route_simulator.constraints_handler.operational_windows),
            "regulatory_requirements_count": len(route_simulator.constraints_handler.regulatory_requirements),
            "active_space_traffic_alerts": len(route_simulator.constraints_handler.space_traffic_alerts),
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Spacecraft capabilities retrieval failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Spacecraft capabilities retrieval failed: {str(e)}"
        )