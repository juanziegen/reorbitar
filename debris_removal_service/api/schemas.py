"""
Pydantic schemas for API request and response models.

This module defines the data validation and serialization schemas
for the satellite debris removal service API endpoints.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

from ..models.satellite import Satellite, OrbitalElements
from ..models.route import Route, Hop
from ..models.cost import MissionCost, DetailedCost
from ..models.service_request import (
    TimelineConstraints, BudgetConstraints, ProcessingPreferences,
    ProcessingType, RequestStatus
)


class ProcessingTypeEnum(str, Enum):
    """Processing type enumeration for API."""
    ISS_RECYCLING = "iss_recycling"
    SOLAR_FORGE = "solar_forge"
    HEO_STORAGE = "heo_storage"
    IMMEDIATE_DISPOSAL = "immediate_disposal"


class TimelineConstraintsSchema(BaseModel):
    """Timeline constraints schema."""
    earliest_start: datetime
    latest_completion: datetime
    preferred_duration: Optional[timedelta] = None
    blackout_periods: Optional[List[tuple[datetime, datetime]]] = None
    
    @validator('latest_completion')
    def validate_completion_after_start(cls, v, values):
        if 'earliest_start' in values and v <= values['earliest_start']:
            raise ValueError('Latest completion must be after earliest start')
        return v


class BudgetConstraintsSchema(BaseModel):
    """Budget constraints schema."""
    max_total_cost: float = Field(..., gt=0, description="Maximum total cost in USD")
    preferred_cost: Optional[float] = Field(None, gt=0, description="Preferred cost in USD")
    cost_breakdown_limits: Optional[Dict[str, float]] = None
    payment_terms: Optional[str] = None
    
    @validator('preferred_cost')
    def validate_preferred_cost(cls, v, values):
        if v is not None and 'max_total_cost' in values and v > values['max_total_cost']:
            raise ValueError('Preferred cost cannot exceed maximum cost')
        return v


class ProcessingPreferencesSchema(BaseModel):
    """Processing preferences schema."""
    preferred_processing_types: List[ProcessingTypeEnum]
    material_priorities: Optional[Dict[str, float]] = None
    processing_timeline: Optional[timedelta] = None
    storage_duration: Optional[timedelta] = None
    special_requirements: Optional[List[str]] = None
    
    @validator('preferred_processing_types')
    def validate_processing_types(cls, v):
        if not v:
            raise ValueError('At least one processing type must be specified')
        return v


class RouteOptimizationRequest(BaseModel):
    """Request schema for route optimization."""
    satellite_ids: List[str] = Field(..., min_items=2, description="List of satellite IDs to collect")
    client_id: Optional[str] = None
    timeline_constraints: TimelineConstraintsSchema
    budget_constraints: BudgetConstraintsSchema
    processing_preferences: ProcessingPreferencesSchema
    optimization_options: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "satellite_ids": ["SAT001", "SAT002", "SAT003"],
                "client_id": "client_123",
                "timeline_constraints": {
                    "earliest_start": "2024-01-01T00:00:00Z",
                    "latest_completion": "2024-06-01T00:00:00Z"
                },
                "budget_constraints": {
                    "max_total_cost": 100000.0,
                    "preferred_cost": 80000.0
                },
                "processing_preferences": {
                    "preferred_processing_types": ["iss_recycling", "heo_storage"]
                }
            }
        }


class OrbitalElementsSchema(BaseModel):
    """Orbital elements schema."""
    semi_major_axis: float
    eccentricity: float
    inclination: float
    raan: float
    argument_of_perigee: float
    mean_anomaly: float
    mean_motion: float
    epoch: datetime
    
    class Config:
        from_attributes = True


class SatelliteSchema(BaseModel):
    """Satellite data schema."""
    id: str
    name: str
    tle_line1: str
    tle_line2: str
    mass: float
    material_composition: Dict[str, float]
    decommission_date: datetime
    orbital_elements: Optional[OrbitalElementsSchema] = None
    
    class Config:
        from_attributes = True


class HopSchema(BaseModel):
    """Hop data schema."""
    from_satellite_id: str
    to_satellite_id: str
    delta_v_required: float
    transfer_time: timedelta
    cost: float
    hop_number: int
    
    class Config:
        from_attributes = True


class RouteSchema(BaseModel):
    """Route data schema."""
    satellite_ids: List[str]
    hops: List[HopSchema]
    total_delta_v: float
    total_cost: float
    mission_duration: timedelta
    feasibility_score: float
    route_id: Optional[str] = None
    
    class Config:
        from_attributes = True


class MissionCostSchema(BaseModel):
    """Mission cost schema."""
    collection_cost: float
    processing_cost: float
    storage_cost: float
    operational_overhead: float
    total_cost: float
    cost_per_satellite: float
    
    class Config:
        from_attributes = True


class RouteOptimizationResponse(BaseModel):
    """Response schema for route optimization."""
    optimization_id: str
    success: bool
    route: Optional[RouteSchema] = None
    mission_cost: Optional[MissionCostSchema] = None
    constraint_analysis: Optional[Dict[str, Any]] = None
    convergence_info: Optional[Dict[str, Any]] = None
    optimization_metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None


class SatelliteDataResponse(BaseModel):
    """Response schema for satellite data."""
    satellite: SatelliteSchema
    orbital_info: Dict[str, Any]
    last_updated: datetime


class QuoteRequest(BaseModel):
    """Request schema for quote generation."""
    client_id: str
    satellite_ids: List[str] = Field(..., min_items=1, description="List of satellite IDs")
    timeline_constraints: TimelineConstraintsSchema
    budget_constraints: BudgetConstraintsSchema
    processing_preferences: ProcessingPreferencesSchema
    additional_requirements: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "client_id": "client_123",
                "satellite_ids": ["SAT001", "SAT002"],
                "timeline_constraints": {
                    "earliest_start": "2024-01-01T00:00:00Z",
                    "latest_completion": "2024-06-01T00:00:00Z"
                },
                "budget_constraints": {
                    "max_total_cost": 100000.0
                },
                "processing_preferences": {
                    "preferred_processing_types": ["iss_recycling"]
                }
            }
        }


class QuoteResponse(BaseModel):
    """Response schema for quote generation."""
    quote_id: str
    client_id: str
    satellite_ids: List[str]
    route: RouteSchema
    mission_cost: MissionCostSchema
    cost_breakdown: Dict[str, Any]
    processing_options: List[ProcessingTypeEnum]
    timeline_estimate: timedelta
    quote_valid_until: datetime
    risk_assessment: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ServiceRequestCreate(BaseModel):
    """Schema for creating service requests."""
    client_id: str
    satellite_ids: List[str] = Field(..., min_items=1)
    timeline_constraints: TimelineConstraintsSchema
    budget_constraints: BudgetConstraintsSchema
    processing_preferences: ProcessingPreferencesSchema
    notes: Optional[str] = None
    contact_info: Optional[Dict[str, Any]] = None


class ServiceRequestResponse(BaseModel):
    """Response schema for service requests."""
    request_id: str
    client_id: str
    satellite_ids: List[str]
    status: str
    timeline_constraints: TimelineConstraintsSchema
    budget_constraints: BudgetConstraintsSchema
    processing_preferences: ProcessingPreferencesSchema
    created_at: datetime
    updated_at: datetime
    notes: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    timestamp: datetime
    version: str = "1.0.0"


class OptimizationStatusResponse(BaseModel):
    """Optimization status response schema."""
    optimization_id: str
    status: str
    progress: float = Field(..., ge=0.0, le=100.0)
    current_phase: str
    elapsed_time_seconds: float
    error: Optional[str] = None


class SatelliteValidationResponse(BaseModel):
    """Satellite validation response schema."""
    satellite_id: str
    is_valid: bool
    validation_errors: List[str]
    warnings: Optional[List[str]] = None


class CostBreakdownResponse(BaseModel):
    """Cost breakdown response schema."""
    propellant_cost: float
    operational_cost: float
    processing_cost: float
    storage_cost: float
    total_cost: float
    cost_percentages: Dict[str, float]
    largest_component: str
    optimization_suggestions: Optional[List[str]] = None