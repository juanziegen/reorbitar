"""
Service Request Manager for handling client request processing and validation.

This module provides comprehensive service request management including
client request processing, validation, mission status tracking, progress updates,
request history, and quote management functionality.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import uuid
from enum import Enum

from ..models.service_request import ServiceRequest, RequestStatus
from ..models.satellite import Satellite
from ..models.route import Route
from ..models.cost import MissionCost


logger = logging.getLogger(__name__)


class RequestValidationError(Exception):
    """Exception raised when request validation fails."""
    pass


class ServiceRequestManager:
    """
    Manager for service request lifecycle and operations.
    
    This class handles client request processing and validation,
    mission status tracking and progress updates, and request
    history and quote management functionality.
    """
    
    def __init__(self):
        """Initialize the service request manager."""
        # In production, this would connect to a database
        self._requests: Dict[str, ServiceRequest] = {}
        self._request_history: Dict[str, List[Dict[str, Any]]] = {}
        self._progress_tracking: Dict[str, Dict[str, Any]] = {}
        
        logger.info("ServiceRequestManager initialized")
    
    def create_request(self, service_request: ServiceRequest, 
                      satellites: List[Satellite]) -> ServiceRequest:
        """
        Create and validate a new service request.
        
        Args:
            service_request: The service request to create
            satellites: Available satellites for validation
            
        Returns:
            Created and validated service request
            
        Raises:
            RequestValidationError: If validation fails
        """
        try:
            logger.info(f"Creating service request for client {service_request.client_id}")
            
            # Validate the request
            self._validate_service_request(service_request, satellites)
            
            # Generate request ID if not provided
            if not service_request.request_id:
                service_request.request_id = str(uuid.uuid4())
            
            # Store the request
            self._requests[service_request.request_id] = service_request
            
            # Initialize request history
            self._request_history[service_request.request_id] = [{
                "timestamp": datetime.utcnow(),
                "action": "created",
                "status": service_request.status.value,
                "notes": "Service request created"
            }]
            
            # Initialize progress tracking
            self._progress_tracking[service_request.request_id] = {
                "overall_progress": 0.0,
                "current_phase": "validation",
                "phases": {
                    "validation": {"status": "completed", "progress": 100.0},
                    "route_optimization": {"status": "pending", "progress": 0.0},
                    "mission_planning": {"status": "pending", "progress": 0.0},
                    "execution": {"status": "pending", "progress": 0.0}
                },
                "last_updated": datetime.utcnow()
            }
            
            logger.info(f"Service request created successfully: {service_request.request_id}")
            return service_request
            
        except Exception as e:
            logger.error(f"Failed to create service request: {str(e)}")
            raise RequestValidationError(f"Failed to create service request: {str(e)}")
    
    def get_request(self, request_id: str) -> Optional[ServiceRequest]:
        """
        Retrieve a service request by ID.
        
        Args:
            request_id: The request ID to retrieve
            
        Returns:
            Service request if found, None otherwise
        """
        return self._requests.get(request_id)
    
    def update_request_status(self, request_id: str, new_status: RequestStatus,
                            notes: Optional[str] = None) -> bool:
        """
        Update service request status with tracking.
        
        Args:
            request_id: The request ID to update
            new_status: New status to set
            notes: Optional notes for the status change
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            if request_id not in self._requests:
                logger.warning(f"Request not found: {request_id}")
                return False
            
            request = self._requests[request_id]
            old_status = request.status
            
            # Update the request
            request.update_status(new_status, notes)
            
            # Add to history
            if request_id not in self._request_history:
                self._request_history[request_id] = []
            
            self._request_history[request_id].append({
                "timestamp": datetime.utcnow(),
                "action": "status_updated",
                "old_status": old_status.value,
                "new_status": new_status.value,
                "notes": notes or ""
            })
            
            # Update progress tracking based on status
            self._update_progress_for_status(request_id, new_status)
            
            logger.info(f"Request {request_id} status updated: {old_status.value} -> {new_status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update request status: {str(e)}")
            return False
    
    def get_request_progress(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed progress information for a request.
        
        Args:
            request_id: The request ID
            
        Returns:
            Progress information dictionary or None if not found
        """
        if request_id not in self._progress_tracking:
            return None
        
        progress_data = self._progress_tracking[request_id].copy()
        progress_data["request_id"] = request_id
        
        # Add current request status
        if request_id in self._requests:
            progress_data["current_status"] = self._requests[request_id].status.value
        
        return progress_data
    
    def list_client_requests(self, client_id: str, status_filter: Optional[RequestStatus] = None,
                           limit: int = 50, offset: int = 0) -> List[ServiceRequest]:
        """
        List service requests for a specific client.
        
        Args:
            client_id: Client ID to filter by
            status_filter: Optional status filter
            limit: Maximum number of requests to return
            offset: Offset for pagination
            
        Returns:
            List of service requests
        """
        try:
            # Filter requests by client
            client_requests = [
                req for req in self._requests.values()
                if req.client_id == client_id
            ]
            
            # Apply status filter if provided
            if status_filter:
                client_requests = [
                    req for req in client_requests
                    if req.status == status_filter
                ]
            
            # Sort by creation date (newest first)
            client_requests.sort(key=lambda x: x.created_at, reverse=True)
            
            # Apply pagination
            start_idx = offset
            end_idx = offset + limit
            
            return client_requests[start_idx:end_idx]
            
        except Exception as e:
            logger.error(f"Failed to list client requests: {str(e)}")
            return []
    
    def get_request_history(self, request_id: str) -> List[Dict[str, Any]]:
        """
        Get the history of actions for a request.
        
        Args:
            request_id: The request ID
            
        Returns:
            List of history entries
        """
        return self._request_history.get(request_id, [])
    
    def approve_request(self, request_id: str, approval_notes: Optional[str] = None) -> bool:
        """
        Approve a service request and initiate mission planning.
        
        Args:
            request_id: The request ID to approve
            approval_notes: Optional approval notes
            
        Returns:
            True if approval successful, False otherwise
        """
        try:
            if request_id not in self._requests:
                logger.warning(f"Request not found for approval: {request_id}")
                return False
            
            request = self._requests[request_id]
            
            # Validate current status allows approval
            if request.status not in [RequestStatus.QUOTED, RequestStatus.PENDING]:
                logger.warning(f"Request {request_id} cannot be approved in status {request.status.value}")
                return False
            
            # Update status to approved
            success = self.update_request_status(
                request_id, 
                RequestStatus.APPROVED, 
                approval_notes or "Request approved by client"
            )
            
            if success:
                # Add approval-specific history entry
                self._request_history[request_id].append({
                    "timestamp": datetime.utcnow(),
                    "action": "approved",
                    "notes": approval_notes or "Request approved",
                    "next_steps": "Mission planning initiated"
                })
                
                logger.info(f"Request {request_id} approved successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to approve request: {str(e)}")
            return False
    
    def cancel_request(self, request_id: str, cancellation_reason: Optional[str] = None) -> bool:
        """
        Cancel a service request.
        
        Args:
            request_id: The request ID to cancel
            cancellation_reason: Optional cancellation reason
            
        Returns:
            True if cancellation successful, False otherwise
        """
        try:
            if request_id not in self._requests:
                logger.warning(f"Request not found for cancellation: {request_id}")
                return False
            
            request = self._requests[request_id]
            
            # Validate current status allows cancellation
            if request.status in [RequestStatus.COMPLETED, RequestStatus.CANCELLED]:
                logger.warning(f"Request {request_id} cannot be cancelled in status {request.status.value}")
                return False
            
            # Update status to cancelled
            success = self.update_request_status(
                request_id,
                RequestStatus.CANCELLED,
                cancellation_reason or "Request cancelled"
            )
            
            if success:
                logger.info(f"Request {request_id} cancelled successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cancel request: {str(e)}")
            return False
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """
        Get dashboard summary of all requests.
        
        Returns:
            Dashboard summary data
        """
        try:
            total_requests = len(self._requests)
            
            # Count by status
            status_counts = {}
            for status in RequestStatus:
                status_counts[status.value] = sum(
                    1 for req in self._requests.values()
                    if req.status == status
                )
            
            # Calculate metrics
            completed_requests = status_counts.get("completed", 0)
            success_rate = (completed_requests / total_requests * 100) if total_requests > 0 else 0.0
            
            # Get recent activity
            recent_activity = []
            for request_id, history in self._request_history.items():
                if history:
                    recent_activity.extend([
                        {
                            "request_id": request_id,
                            "timestamp": entry["timestamp"],
                            "action": entry["action"],
                            "details": entry.get("notes", "")
                        }
                        for entry in history[-3:]  # Last 3 entries per request
                    ])
            
            # Sort by timestamp and take most recent
            recent_activity.sort(key=lambda x: x["timestamp"], reverse=True)
            recent_activity = recent_activity[:10]  # Top 10 most recent
            
            return {
                "summary": {
                    "total_requests": total_requests,
                    "active_requests": sum(
                        status_counts.get(status, 0) 
                        for status in ["processing", "approved", "in_progress"]
                    ),
                    "completed_requests": completed_requests,
                    "pending_approval": status_counts.get("quoted", 0)
                },
                "status_breakdown": status_counts,
                "recent_activity": recent_activity,
                "performance_metrics": {
                    "success_rate": success_rate,
                    "average_processing_time_hours": 24.0,  # Mock data
                    "client_satisfaction": 4.2  # Mock data
                },
                "last_updated": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard summary: {str(e)}")
            return {
                "summary": {"total_requests": 0, "active_requests": 0, "completed_requests": 0, "pending_approval": 0},
                "status_breakdown": {},
                "recent_activity": [],
                "performance_metrics": {"success_rate": 0.0, "average_processing_time_hours": 0.0, "client_satisfaction": 0.0},
                "last_updated": datetime.utcnow()
            }
    
    def _validate_service_request(self, service_request: ServiceRequest, 
                                satellites: List[Satellite]) -> None:
        """
        Validate a service request against business rules.
        
        Args:
            service_request: The service request to validate
            satellites: Available satellites for validation
            
        Raises:
            RequestValidationError: If validation fails
        """
        # Validate satellite IDs exist
        available_satellite_ids = {sat.id for sat in satellites}
        for sat_id in service_request.satellites:
            if sat_id not in available_satellite_ids:
                raise RequestValidationError(f"Satellite {sat_id} not found in available satellites")
        
        # Validate timeline constraints
        timeline = service_request.timeline_requirements
        if timeline.earliest_start >= timeline.latest_completion:
            raise RequestValidationError("Timeline earliest start must be before latest completion")
        
        # Check if timeline is reasonable (not too short)
        duration = timeline.latest_completion - timeline.earliest_start
        if duration.total_seconds() < 24 * 3600:  # Less than 24 hours
            raise RequestValidationError("Mission timeline too short (minimum 24 hours required)")
        
        # Validate budget constraints
        budget = service_request.budget_constraints
        if budget.max_total_cost <= 0:
            raise RequestValidationError("Maximum budget must be positive")
        
        # Estimate minimum cost and check feasibility
        min_estimated_cost = len(service_request.satellites) * 10000  # $10k per satellite minimum
        if budget.max_total_cost < min_estimated_cost:
            raise RequestValidationError(
                f"Budget too low. Minimum estimated cost: ${min_estimated_cost:,.2f}"
            )
        
        # Validate processing preferences
        processing = service_request.processing_preferences
        if not processing.preferred_processing_types:
            raise RequestValidationError("At least one processing type must be specified")
        
        logger.info(f"Service request validation passed for client {service_request.client_id}")
    
    def _update_progress_for_status(self, request_id: str, status: RequestStatus) -> None:
        """
        Update progress tracking based on status change.
        
        Args:
            request_id: The request ID
            status: New status
        """
        if request_id not in self._progress_tracking:
            return
        
        progress = self._progress_tracking[request_id]
        
        # Update progress based on status
        status_progress_map = {
            RequestStatus.PENDING: 10.0,
            RequestStatus.PROCESSING: 25.0,
            RequestStatus.QUOTED: 50.0,
            RequestStatus.APPROVED: 60.0,
            RequestStatus.IN_PROGRESS: 80.0,
            RequestStatus.COMPLETED: 100.0,
            RequestStatus.CANCELLED: 0.0,
            RequestStatus.REJECTED: 0.0
        }
        
        progress["overall_progress"] = status_progress_map.get(status, 0.0)
        progress["last_updated"] = datetime.utcnow()
        
        # Update phase status based on overall progress
        if progress["overall_progress"] >= 25.0:
            progress["phases"]["route_optimization"]["status"] = "completed"
            progress["phases"]["route_optimization"]["progress"] = 100.0
            progress["current_phase"] = "mission_planning"
        
        if progress["overall_progress"] >= 60.0:
            progress["phases"]["mission_planning"]["status"] = "completed"
            progress["phases"]["mission_planning"]["progress"] = 100.0
            progress["current_phase"] = "execution"
        
        if progress["overall_progress"] >= 80.0:
            progress["phases"]["execution"]["status"] = "in_progress"
            progress["phases"]["execution"]["progress"] = 50.0
        
        if progress["overall_progress"] >= 100.0:
            progress["phases"]["execution"]["status"] = "completed"
            progress["phases"]["execution"]["progress"] = 100.0
            progress["current_phase"] = "completed"