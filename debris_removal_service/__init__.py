"""
Satellite Debris Removal Service

A comprehensive commercial platform for satellite debris removal services
combining genetic algorithm route optimization, biprop cost calculations,
and professional client interfaces.
"""

from .models import (
    Satellite, OrbitalElements, Route, Hop, ManeuverDetails,
    PropellantMass, MissionCost, DetailedCost,
    ServiceRequest, TimelineConstraints, BudgetConstraints, 
    ProcessingPreferences, RequestStatus, ProcessingType
)
from .utils import SatelliteDataValidator, TLEParser
from .services import RouteOptimizationService, OrbitalMechanicsService, RouteSimulator

__version__ = "0.1.0"

__all__ = [
    # Core models
    'Satellite', 'OrbitalElements', 'Route', 'Hop', 'ManeuverDetails',
    'PropellantMass', 'MissionCost', 'DetailedCost',
    'ServiceRequest', 'TimelineConstraints', 'BudgetConstraints', 
    'ProcessingPreferences', 'RequestStatus', 'ProcessingType',
    # Utilities
    'SatelliteDataValidator', 'TLEParser',
    # Services
    'RouteOptimizationService', 'OrbitalMechanicsService', 'RouteSimulator'
]