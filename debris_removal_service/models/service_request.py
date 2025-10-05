"""
Service request data models for client interactions.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any


class RequestStatus(Enum):
    """Status of a service request."""
    PENDING = "pending"
    PROCESSING = "processing"
    QUOTED = "quoted"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class ProcessingType(Enum):
    """Types of material processing services."""
    ISS_RECYCLING = "iss_recycling"
    SOLAR_FORGE = "solar_forge"
    HEO_STORAGE = "heo_storage"
    IMMEDIATE_DISPOSAL = "immediate_disposal"


@dataclass
class TimelineConstraints:
    """Timeline requirements and constraints for a mission."""
    earliest_start: datetime
    latest_completion: datetime
    preferred_duration: Optional[timedelta] = None
    blackout_periods: Optional[List[tuple[datetime, datetime]]] = None
    
    def __post_init__(self):
        """Validate timeline constraints."""
        if self.earliest_start >= self.latest_completion:
            raise ValueError("Earliest start must be before latest completion")
        
        if self.preferred_duration and self.preferred_duration.total_seconds() <= 0:
            raise ValueError("Preferred duration must be positive")
        
        # Validate blackout periods don't overlap and are within mission timeframe
        if self.blackout_periods:
            for start, end in self.blackout_periods:
                if start >= end:
                    raise ValueError("Blackout period start must be before end")
                if start < self.earliest_start or end > self.latest_completion:
                    raise ValueError("Blackout periods must be within mission timeframe")
    
    def get_available_duration(self) -> timedelta:
        """Calculate total available time excluding blackout periods."""
        total_duration = self.latest_completion - self.earliest_start
        
        if not self.blackout_periods:
            return total_duration
        
        blackout_duration = sum(
            (end - start for start, end in self.blackout_periods),
            timedelta()
        )
        
        return total_duration - blackout_duration
    
    def is_time_available(self, start: datetime, duration: timedelta) -> bool:
        """Check if a time period is available (not in blackout)."""
        end = start + duration
        
        # Check if within overall constraints
        if start < self.earliest_start or end > self.latest_completion:
            return False
        
        # Check against blackout periods
        if self.blackout_periods:
            for blackout_start, blackout_end in self.blackout_periods:
                # Check for overlap
                if not (end <= blackout_start or start >= blackout_end):
                    return False
        
        return True


@dataclass
class BudgetConstraints:
    """Budget constraints and preferences for a mission."""
    max_total_cost: float  # USD
    preferred_cost: Optional[float] = None  # USD
    cost_breakdown_limits: Optional[Dict[str, float]] = None  # component -> max cost
    payment_terms: Optional[str] = None
    
    def __post_init__(self):
        """Validate budget constraints."""
        if self.max_total_cost <= 0:
            raise ValueError("Maximum total cost must be positive")
        
        if self.preferred_cost and self.preferred_cost > self.max_total_cost:
            raise ValueError("Preferred cost cannot exceed maximum cost")
        
        if self.cost_breakdown_limits:
            total_component_limits = sum(self.cost_breakdown_limits.values())
            if total_component_limits > self.max_total_cost:
                raise ValueError("Sum of component limits cannot exceed total budget")
    
    def is_within_budget(self, total_cost: float, cost_breakdown: Optional[Dict[str, float]] = None) -> bool:
        """Check if a cost estimate is within budget constraints."""
        if total_cost > self.max_total_cost:
            return False
        
        if self.cost_breakdown_limits and cost_breakdown:
            for component, limit in self.cost_breakdown_limits.items():
                if component in cost_breakdown and cost_breakdown[component] > limit:
                    return False
        
        return True


@dataclass
class ProcessingPreferences:
    """Client preferences for material processing."""
    preferred_processing_types: List[ProcessingType]
    material_priorities: Optional[Dict[str, float]] = None  # material -> priority (0-1)
    processing_timeline: Optional[timedelta] = None
    storage_duration: Optional[timedelta] = None
    special_requirements: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate processing preferences."""
        if not self.preferred_processing_types:
            raise ValueError("At least one processing type must be specified")
        
        if self.material_priorities:
            for material, priority in self.material_priorities.items():
                if not (0.0 <= priority <= 1.0):
                    raise ValueError("Material priorities must be between 0 and 1")
        
        if self.processing_timeline and self.processing_timeline.total_seconds() <= 0:
            raise ValueError("Processing timeline must be positive")
        
        if self.storage_duration and self.storage_duration.total_seconds() <= 0:
            raise ValueError("Storage duration must be positive")
    
    def get_processing_priority(self, processing_type: ProcessingType) -> float:
        """Get priority for a processing type (higher index = higher priority)."""
        try:
            return len(self.preferred_processing_types) - self.preferred_processing_types.index(processing_type)
        except ValueError:
            return 0.0  # Not in preferred list


@dataclass
class ServiceRequest:
    """Complete service request from a client."""
    client_id: str
    satellites: List[str]  # Satellite IDs to be collected
    timeline_requirements: TimelineConstraints
    budget_constraints: BudgetConstraints
    processing_preferences: ProcessingPreferences
    status: RequestStatus = RequestStatus.PENDING
    request_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    notes: Optional[str] = None
    contact_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Set default values and validate request."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        
        if self.updated_at is None:
            self.updated_at = self.created_at
        
        if not self.satellites:
            raise ValueError("At least one satellite must be specified")
        
        if not self.client_id:
            raise ValueError("Client ID is required")
    
    def update_status(self, new_status: RequestStatus, notes: Optional[str] = None):
        """Update request status with timestamp."""
        self.status = new_status
        self.updated_at = datetime.utcnow()
        if notes:
            if self.notes:
                self.notes += f"\n{datetime.utcnow().isoformat()}: {notes}"
            else:
                self.notes = f"{datetime.utcnow().isoformat()}: {notes}"
    
    def get_satellite_count(self) -> int:
        """Get number of satellites in the request."""
        return len(self.satellites)
    
    def is_urgent(self, urgency_threshold_days: float = 30.0) -> bool:
        """Check if request has urgent timeline requirements."""
        available_time = self.timeline_requirements.get_available_duration()
        return available_time.total_seconds() < urgency_threshold_days * 86400
    
    def get_request_summary(self) -> Dict[str, Any]:
        """Get a summary of the service request."""
        return {
            'request_id': self.request_id,
            'client_id': self.client_id,
            'satellite_count': self.get_satellite_count(),
            'status': self.status.value,
            'max_budget_usd': self.budget_constraints.max_total_cost,
            'available_duration_days': self.timeline_requirements.get_available_duration().total_seconds() / 86400,
            'is_urgent': self.is_urgent(),
            'preferred_processing': [pt.value for pt in self.processing_preferences.preferred_processing_types],
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }