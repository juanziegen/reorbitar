"""
REST API endpoints and handlers for the satellite debris removal service.

This module provides FastAPI application with route optimization endpoints,
satellite data retrieval and validation endpoints, and quote generation API
with cost breakdown responses.
"""

# Import only when FastAPI is available
try:
    from .main import app
    from .schemas import (
        RouteOptimizationRequest,
        RouteOptimizationResponse,
        SatelliteDataResponse,
        QuoteRequest,
        QuoteResponse,
        ServiceRequestCreate,
        ServiceRequestResponse,
        ErrorResponse
    )
    from .dependencies import (
        get_route_simulator,
        get_satellite_database,
        get_api_config,
        check_service_health
    )
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    app = None

if _FASTAPI_AVAILABLE:
    __all__ = [
        "app",
        "RouteOptimizationRequest",
        "RouteOptimizationResponse", 
        "SatelliteDataResponse",
        "QuoteRequest",
        "QuoteResponse",
        "ServiceRequestCreate",
        "ServiceRequestResponse",
        "ErrorResponse",
        "get_route_simulator",
        "get_satellite_database",
        "get_api_config",
        "check_service_health"
    ]
else:
    __all__ = []