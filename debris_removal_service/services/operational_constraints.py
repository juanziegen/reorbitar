"""
Operational Constraints Handler for spacecraft and mission constraints.

This module implements spacecraft fuel capacity and operational window constraints,
regulatory compliance checks and space traffic coordination, and mission scheduling
optimization for multiple concurrent clients.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import math
from collections import defaultdict

from ..models.satellite import Satellite
from ..models.route import Route, Hop
from ..models.service_request import ServiceRequest


logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Types of operational constraints."""
    FUEL_CAPACITY = "fuel_capacity"
    OPERATIONAL_WINDOW = "operational_window"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    SPACE_TRAFFIC = "space_traffic"
    MISSION_SCHEDULING = "mission_scheduling"
    SPACECRAFT_CAPABILITY = "spacecraft_capability"


class ConstraintSeverity(Enum):
    """Severity levels for constraint violations."""
    CRITICAL = "critical"  # Mission cannot proceed
    HIGH = "high"         # Significant risk or cost impact
    MEDIUM = "medium"     # Moderate impact, workarounds available
    LOW = "low"          # Minor impact, easily mitigated


@dataclass
class ConstraintViolation:
    """Represents a constraint violation."""
    constraint_type: ConstraintType
    severity: ConstraintSeverity
    description: str
    affected_satellites: List[str]
    suggested_mitigation: str
    cost_impact: float = 0.0
    time_impact_hours: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpacecraftCapabilities:
    """Spacecraft operational capabilities and constraints."""
    max_fuel_capacity_kg: float
    fuel_consumption_rate_kg_per_ms: float  # kg per m/s delta-v
    max_delta_v_per_maneuver: float  # m/s
    max_mission_duration_hours: float
    operational_altitude_range: Tuple[float, float]  # km
    max_payload_capacity_kg: float
    communication_range_km: float
    power_generation_watts: float
    thermal_limits: Dict[str, float]  # temperature ranges
    
    def calculate_fuel_required(self, total_delta_v: float) -> float:
        """Calculate fuel required for given delta-v."""
        return total_delta_v * self.fuel_consumption_rate_kg_per_ms
    
    def can_perform_maneuver(self, delta_v: float) -> bool:
        """Check if spacecraft can perform a single maneuver."""
        return delta_v <= self.max_delta_v_per_maneuver
    
    def can_complete_mission(self, route: Route) -> Tuple[bool, str]:
        """Check if spacecraft can complete entire mission."""
        total_fuel_needed = self.calculate_fuel_required(route.total_delta_v)
        
        if total_fuel_needed > self.max_fuel_capacity_kg:
            return False, f"Insufficient fuel capacity: {total_fuel_needed:.1f}kg needed, {self.max_fuel_capacity_kg:.1f}kg available"
        
        if route.mission_duration.total_seconds() / 3600 > self.max_mission_duration_hours:
            return False, f"Mission duration exceeds limit: {route.mission_duration.total_seconds()/3600:.1f}h vs {self.max_mission_duration_hours:.1f}h"
        
        return True, "Mission feasible"


@dataclass
class OperationalWindow:
    """Represents an operational time window."""
    start_time: datetime
    end_time: datetime
    window_type: str  # "launch", "maneuver", "communication", etc.
    priority: int  # 1 = highest priority
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def duration_hours(self) -> float:
        """Get window duration in hours."""
        return (self.end_time - self.start_time).total_seconds() / 3600
    
    def overlaps_with(self, other: 'OperationalWindow') -> bool:
        """Check if this window overlaps with another."""
        return not (self.end_time <= other.start_time or self.start_time >= other.end_time)
    
    def contains_time(self, time: datetime) -> bool:
        """Check if a specific time falls within this window."""
        return self.start_time <= time <= self.end_time


@dataclass
class RegulatoryRequirement:
    """Regulatory compliance requirement."""
    requirement_id: str
    description: str
    applicable_regions: List[str]
    compliance_deadline: Optional[datetime]
    required_documentation: List[str]
    approval_authority: str
    estimated_approval_time_days: int
    mandatory: bool = True


@dataclass
class SpaceTrafficAlert:
    """Space traffic coordination alert."""
    alert_id: str
    alert_type: str  # "conjunction", "debris", "launch", etc.
    affected_region: Dict[str, Any]  # orbital region definition
    time_window: Tuple[datetime, datetime]
    risk_level: str  # "low", "medium", "high", "critical"
    recommended_action: str
    issuing_authority: str


class OperationalConstraintsHandler:
    """
    Handles all operational constraints for mission planning and execution.
    
    This class implements spacecraft fuel capacity and operational window constraints,
    regulatory compliance checks and space traffic coordination, and mission scheduling
    optimization for multiple concurrent clients.
    """
    
    def __init__(self):
        """Initialize operational constraints handler."""
        # Default spacecraft capabilities (can be customized per mission)
        self.default_spacecraft = SpacecraftCapabilities(
            max_fuel_capacity_kg=500.0,
            fuel_consumption_rate_kg_per_ms=0.1,  # 100g per m/s delta-v
            max_delta_v_per_maneuver=1000.0,  # 1 km/s per maneuver
            max_mission_duration_hours=720.0,  # 30 days
            operational_altitude_range=(200.0, 2000.0),  # LEO range
            max_payload_capacity_kg=1000.0,
            communication_range_km=50000.0,
            power_generation_watts=2000.0,
            thermal_limits={"min_temp_c": -150, "max_temp_c": 120}
        )
        
        # Operational windows database
        self.operational_windows: List[OperationalWindow] = []
        
        # Regulatory requirements database
        self.regulatory_requirements: List[RegulatoryRequirement] = []
        
        # Space traffic alerts
        self.space_traffic_alerts: List[SpaceTrafficAlert] = []
        
        # Mission scheduling
        self.scheduled_missions: Dict[str, Dict[str, Any]] = {}
        
        # Initialize with default constraints
        self._initialize_default_constraints()
        
        logger.info("OperationalConstraintsHandler initialized")
    
    def validate_mission_constraints(self, service_request: ServiceRequest, 
                                   route: Route, 
                                   spacecraft_capabilities: Optional[SpacecraftCapabilities] = None) -> List[ConstraintViolation]:
        """
        Validate all operational constraints for a mission.
        
        Args:
            service_request: Service request with mission parameters
            route: Proposed route
            spacecraft_capabilities: Optional custom spacecraft capabilities
            
        Returns:
            List of constraint violations
        """
        violations = []
        spacecraft = spacecraft_capabilities or self.default_spacecraft
        
        # Validate fuel capacity constraints
        fuel_violations = self._validate_fuel_constraints(route, spacecraft)
        violations.extend(fuel_violations)
        
        # Validate operational window constraints
        window_violations = self._validate_operational_windows(service_request, route)
        violations.extend(window_violations)
        
        # Validate regulatory compliance
        regulatory_violations = self._validate_regulatory_compliance(service_request, route)
        violations.extend(regulatory_violations)
        
        # Validate space traffic coordination
        traffic_violations = self._validate_space_traffic(service_request, route)
        violations.extend(traffic_violations)
        
        # Validate spacecraft capabilities
        capability_violations = self._validate_spacecraft_capabilities(route, spacecraft)
        violations.extend(capability_violations)
        
        logger.info(f"Mission constraint validation completed: {len(violations)} violations found")
        return violations
    
    def optimize_mission_scheduling(self, service_requests: List[ServiceRequest],
                                  routes: List[Route]) -> Dict[str, Any]:
        """
        Optimize mission scheduling for multiple concurrent clients.
        
        Args:
            service_requests: List of service requests
            routes: Corresponding optimized routes
            
        Returns:
            Optimized scheduling plan
        """
        try:
            logger.info(f"Optimizing scheduling for {len(service_requests)} missions")
            
            # Create mission scheduling data
            missions = []
            for i, (request, route) in enumerate(zip(service_requests, routes)):
                mission = {
                    'mission_id': f"mission_{i}",
                    'request_id': request.request_id,
                    'client_id': request.client_id,
                    'route': route,
                    'earliest_start': request.timeline_requirements.earliest_start,
                    'latest_completion': request.timeline_requirements.latest_completion,
                    'priority': self._calculate_mission_priority(request),
                    'resource_requirements': self._calculate_resource_requirements(route),
                    'constraints': self.validate_mission_constraints(request, route)
                }
                missions.append(mission)
            
            # Sort by priority and timeline constraints
            missions.sort(key=lambda m: (m['priority'], m['earliest_start']))
            
            # Schedule missions with conflict resolution
            scheduled_missions = []
            resource_allocation = defaultdict(list)
            
            for mission in missions:
                # Find optimal scheduling slot
                optimal_slot = self._find_optimal_scheduling_slot(
                    mission, scheduled_missions, resource_allocation
                )
                
                if optimal_slot:
                    mission['scheduled_start'] = optimal_slot['start_time']
                    mission['scheduled_completion'] = optimal_slot['end_time']
                    mission['resource_allocation'] = optimal_slot['resources']
                    mission['scheduling_conflicts'] = optimal_slot.get('conflicts', [])
                    
                    scheduled_missions.append(mission)
                    
                    # Update resource allocation
                    for resource, allocation in optimal_slot['resources'].items():
                        resource_allocation[resource].append({
                            'mission_id': mission['mission_id'],
                            'start_time': optimal_slot['start_time'],
                            'end_time': optimal_slot['end_time'],
                            'allocation': allocation
                        })
                else:
                    mission['scheduling_status'] = 'deferred'
                    mission['deferral_reason'] = 'No available scheduling slot found'
            
            # Generate scheduling optimization report
            optimization_report = {
                'total_missions': len(missions),
                'scheduled_missions': len(scheduled_missions),
                'deferred_missions': len(missions) - len(scheduled_missions),
                'resource_utilization': self._calculate_resource_utilization(resource_allocation),
                'scheduling_efficiency': len(scheduled_missions) / len(missions) * 100,
                'timeline_optimization': self._analyze_timeline_optimization(scheduled_missions),
                'conflict_resolution': self._analyze_scheduling_conflicts(scheduled_missions)
            }
            
            return {
                'scheduled_missions': scheduled_missions,
                'resource_allocation': dict(resource_allocation),
                'optimization_report': optimization_report,
                'scheduling_timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Mission scheduling optimization failed: {str(e)}")
            return {
                'scheduled_missions': [],
                'resource_allocation': {},
                'optimization_report': {'error': str(e)},
                'scheduling_timestamp': datetime.utcnow()
            }
    
    def check_space_traffic_coordination(self, route: Route, 
                                       mission_timeline: Tuple[datetime, datetime]) -> List[SpaceTrafficAlert]:
        """
        Check space traffic coordination requirements.
        
        Args:
            route: Mission route
            mission_timeline: Mission start and end times
            
        Returns:
            List of relevant space traffic alerts
        """
        relevant_alerts = []
        
        for alert in self.space_traffic_alerts:
            # Check if alert time window overlaps with mission
            alert_start, alert_end = alert.time_window
            mission_start, mission_end = mission_timeline
            
            if not (alert_end <= mission_start or alert_start >= mission_end):
                # Check if route intersects with alert region
                if self._route_intersects_alert_region(route, alert.affected_region):
                    relevant_alerts.append(alert)
        
        return relevant_alerts
    
    def generate_compliance_checklist(self, service_request: ServiceRequest,
                                    route: Route) -> Dict[str, Any]:
        """
        Generate regulatory compliance checklist.
        
        Args:
            service_request: Service request
            route: Mission route
            
        Returns:
            Compliance checklist with requirements and status
        """
        checklist = {
            'mission_id': service_request.request_id,
            'compliance_requirements': [],
            'documentation_needed': [],
            'approval_timeline': {},
            'compliance_status': 'pending',
            'estimated_approval_time_days': 0
        }
        
        # Check applicable regulatory requirements
        for requirement in self.regulatory_requirements:
            if self._requirement_applies_to_mission(requirement, service_request, route):
                checklist['compliance_requirements'].append({
                    'requirement_id': requirement.requirement_id,
                    'description': requirement.description,
                    'mandatory': requirement.mandatory,
                    'approval_authority': requirement.approval_authority,
                    'estimated_time_days': requirement.estimated_approval_time_days,
                    'documentation': requirement.required_documentation
                })
                
                checklist['documentation_needed'].extend(requirement.required_documentation)
                checklist['estimated_approval_time_days'] = max(
                    checklist['estimated_approval_time_days'],
                    requirement.estimated_approval_time_days
                )
        
        # Remove duplicate documentation
        checklist['documentation_needed'] = list(set(checklist['documentation_needed']))
        
        # Generate approval timeline
        checklist['approval_timeline'] = self._generate_approval_timeline(
            checklist['compliance_requirements']
        )
        
        return checklist
    
    def add_operational_window(self, window: OperationalWindow) -> None:
        """Add operational window constraint."""
        self.operational_windows.append(window)
        logger.debug(f"Added operational window: {window.window_type} from {window.start_time} to {window.end_time}")
    
    def add_space_traffic_alert(self, alert: SpaceTrafficAlert) -> None:
        """Add space traffic alert."""
        self.space_traffic_alerts.append(alert)
        logger.info(f"Added space traffic alert: {alert.alert_type} - {alert.risk_level} risk")
    
    def update_spacecraft_capabilities(self, capabilities: SpacecraftCapabilities) -> None:
        """Update default spacecraft capabilities."""
        self.default_spacecraft = capabilities
        logger.info("Updated default spacecraft capabilities")
    
    # Private helper methods
    
    def _validate_fuel_constraints(self, route: Route, 
                                 spacecraft: SpacecraftCapabilities) -> List[ConstraintViolation]:
        """Validate fuel capacity constraints."""
        violations = []
        
        # Check total fuel requirement
        total_fuel_needed = spacecraft.calculate_fuel_required(route.total_delta_v)
        
        if total_fuel_needed > spacecraft.max_fuel_capacity_kg:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.FUEL_CAPACITY,
                severity=ConstraintSeverity.CRITICAL,
                description=f"Insufficient fuel capacity: {total_fuel_needed:.1f}kg needed, {spacecraft.max_fuel_capacity_kg:.1f}kg available",
                affected_satellites=[sat.id for sat in route.satellites],
                suggested_mitigation="Reduce mission scope, add refueling stop, or use multiple spacecraft",
                cost_impact=50000.0  # Estimated additional cost
            ))
        
        # Check individual maneuver constraints
        for hop in route.hops:
            if not spacecraft.can_perform_maneuver(hop.delta_v_required):
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.FUEL_CAPACITY,
                    severity=ConstraintSeverity.HIGH,
                    description=f"Maneuver exceeds spacecraft capability: {hop.delta_v_required:.1f} m/s vs {spacecraft.max_delta_v_per_maneuver:.1f} m/s limit",
                    affected_satellites=[hop.from_satellite.id, hop.to_satellite.id],
                    suggested_mitigation="Split maneuver into multiple burns or use different trajectory",
                    cost_impact=10000.0
                ))
        
        return violations
    
    def _validate_operational_windows(self, service_request: ServiceRequest,
                                    route: Route) -> List[ConstraintViolation]:
        """Validate operational window constraints."""
        violations = []
        
        mission_start = service_request.timeline_requirements.earliest_start
        mission_end = service_request.timeline_requirements.latest_completion
        
        # Check for conflicting operational windows
        for window in self.operational_windows:
            if window.overlaps_with(OperationalWindow(mission_start, mission_end, "mission", 1)):
                if window.window_type in ["maintenance", "blackout", "restricted"]:
                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.OPERATIONAL_WINDOW,
                        severity=ConstraintSeverity.HIGH,
                        description=f"Mission conflicts with {window.window_type} window from {window.start_time} to {window.end_time}",
                        affected_satellites=[sat.id for sat in route.satellites],
                        suggested_mitigation="Adjust mission timeline to avoid conflict",
                        time_impact_hours=window.duration_hours()
                    ))
        
        return violations
    
    def _validate_regulatory_compliance(self, service_request: ServiceRequest,
                                      route: Route) -> List[ConstraintViolation]:
        """Validate regulatory compliance requirements."""
        violations = []
        
        # Check for missing regulatory approvals
        for requirement in self.regulatory_requirements:
            if (requirement.mandatory and 
                self._requirement_applies_to_mission(requirement, service_request, route)):
                
                if requirement.compliance_deadline and requirement.compliance_deadline < datetime.utcnow():
                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.REGULATORY_COMPLIANCE,
                        severity=ConstraintSeverity.CRITICAL,
                        description=f"Regulatory requirement expired: {requirement.description}",
                        affected_satellites=[sat.id for sat in route.satellites],
                        suggested_mitigation=f"Obtain approval from {requirement.approval_authority}",
                        time_impact_hours=requirement.estimated_approval_time_days * 24
                    ))
        
        return violations
    
    def _validate_space_traffic(self, service_request: ServiceRequest,
                              route: Route) -> List[ConstraintViolation]:
        """Validate space traffic coordination."""
        violations = []
        
        mission_timeline = (
            service_request.timeline_requirements.earliest_start,
            service_request.timeline_requirements.latest_completion
        )
        
        alerts = self.check_space_traffic_coordination(route, mission_timeline)
        
        for alert in alerts:
            severity = ConstraintSeverity.CRITICAL if alert.risk_level == "critical" else ConstraintSeverity.HIGH
            
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.SPACE_TRAFFIC,
                severity=severity,
                description=f"Space traffic alert: {alert.alert_type} - {alert.recommended_action}",
                affected_satellites=[sat.id for sat in route.satellites],
                suggested_mitigation=alert.recommended_action,
                metadata={'alert_id': alert.alert_id, 'issuing_authority': alert.issuing_authority}
            ))
        
        return violations
    
    def _validate_spacecraft_capabilities(self, route: Route,
                                        spacecraft: SpacecraftCapabilities) -> List[ConstraintViolation]:
        """Validate spacecraft capability constraints."""
        violations = []
        
        # Check mission duration
        mission_duration_hours = route.mission_duration.total_seconds() / 3600
        if mission_duration_hours > spacecraft.max_mission_duration_hours:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.SPACECRAFT_CAPABILITY,
                severity=ConstraintSeverity.HIGH,
                description=f"Mission duration exceeds spacecraft limit: {mission_duration_hours:.1f}h vs {spacecraft.max_mission_duration_hours:.1f}h",
                affected_satellites=[sat.id for sat in route.satellites],
                suggested_mitigation="Reduce mission scope or use multiple spacecraft"
            ))
        
        # Check altitude constraints
        for satellite in route.satellites:
            try:
                perigee, apogee = satellite.get_altitude_km()
                min_alt, max_alt = spacecraft.operational_altitude_range
                
                if perigee < min_alt or apogee > max_alt:
                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.SPACECRAFT_CAPABILITY,
                        severity=ConstraintSeverity.MEDIUM,
                        description=f"Satellite {satellite.id} altitude outside operational range: {perigee:.1f}-{apogee:.1f}km vs {min_alt:.1f}-{max_alt:.1f}km",
                        affected_satellites=[satellite.id],
                        suggested_mitigation="Use specialized spacecraft or modify operational parameters"
                    ))
            except Exception as e:
                logger.warning(f"Could not validate altitude for satellite {satellite.id}: {str(e)}")
        
        return violations
    
    def _initialize_default_constraints(self) -> None:
        """Initialize default operational constraints."""
        # Add default operational windows
        now = datetime.utcnow()
        
        # Daily maintenance windows
        for i in range(30):  # Next 30 days
            maintenance_start = now.replace(hour=2, minute=0, second=0, microsecond=0) + timedelta(days=i)
            maintenance_end = maintenance_start + timedelta(hours=2)
            
            self.operational_windows.append(OperationalWindow(
                start_time=maintenance_start,
                end_time=maintenance_end,
                window_type="maintenance",
                priority=1
            ))
        
        # Add default regulatory requirements
        self.regulatory_requirements.extend([
            RegulatoryRequirement(
                requirement_id="SPACE_DEBRIS_MITIGATION",
                description="Space debris mitigation guidelines compliance",
                applicable_regions=["global"],
                compliance_deadline=None,
                required_documentation=["debris_mitigation_plan", "mission_analysis"],
                approval_authority="National Space Agency",
                estimated_approval_time_days=30
            ),
            RegulatoryRequirement(
                requirement_id="ORBITAL_SAFETY",
                description="Orbital safety coordination and approval",
                applicable_regions=["LEO"],
                compliance_deadline=None,
                required_documentation=["safety_analysis", "collision_assessment"],
                approval_authority="Space Traffic Coordination Center",
                estimated_approval_time_days=14
            )
        ])
        
        # Add sample space traffic alerts
        alert_start = now + timedelta(days=7)
        alert_end = alert_start + timedelta(hours=6)
        
        self.space_traffic_alerts.append(SpaceTrafficAlert(
            alert_id="ALERT_001",
            alert_type="conjunction",
            affected_region={"altitude_range": [400, 600], "inclination_range": [50, 70]},
            time_window=(alert_start, alert_end),
            risk_level="medium",
            recommended_action="Avoid maneuvers in affected region during time window",
            issuing_authority="Space Surveillance Network"
        ))
    
    def _calculate_mission_priority(self, service_request: ServiceRequest) -> int:
        """Calculate mission priority score."""
        priority = 5  # Default priority
        
        # Higher priority for urgent timelines
        timeline_days = (service_request.timeline_requirements.latest_completion - 
                        service_request.timeline_requirements.earliest_start).days
        if timeline_days < 7:
            priority -= 2
        elif timeline_days < 30:
            priority -= 1
        
        # Higher priority for higher budget
        if service_request.budget_constraints.max_total_cost > 1000000:
            priority -= 1
        
        return max(1, priority)  # Minimum priority of 1
    
    def _calculate_resource_requirements(self, route: Route) -> Dict[str, float]:
        """Calculate resource requirements for a route."""
        return {
            'fuel_kg': self.default_spacecraft.calculate_fuel_required(route.total_delta_v),
            'mission_duration_hours': route.mission_duration.total_seconds() / 3600,
            'spacecraft_count': 1,
            'ground_support_hours': route.mission_duration.total_seconds() / 3600 * 0.5
        }
    
    def _find_optimal_scheduling_slot(self, mission: Dict[str, Any],
                                    scheduled_missions: List[Dict[str, Any]],
                                    resource_allocation: Dict[str, List]) -> Optional[Dict[str, Any]]:
        """Find optimal scheduling slot for a mission."""
        # Simple scheduling algorithm - can be enhanced with more sophisticated optimization
        earliest_start = mission['earliest_start']
        latest_completion = mission['latest_completion']
        mission_duration = mission['route'].mission_duration
        
        # Try to schedule at earliest possible time
        candidate_start = earliest_start
        
        while candidate_start + mission_duration <= latest_completion:
            candidate_end = candidate_start + mission_duration
            
            # Check for conflicts with existing missions
            conflicts = []
            for scheduled in scheduled_missions:
                if not (candidate_end <= scheduled['scheduled_start'] or 
                       candidate_start >= scheduled['scheduled_completion']):
                    conflicts.append(scheduled['mission_id'])
            
            if not conflicts:
                return {
                    'start_time': candidate_start,
                    'end_time': candidate_end,
                    'resources': mission['resource_requirements'],
                    'conflicts': []
                }
            
            # Move to next possible slot (1 hour increment)
            candidate_start += timedelta(hours=1)
        
        return None  # No suitable slot found
    
    def _calculate_resource_utilization(self, resource_allocation: Dict[str, List]) -> Dict[str, float]:
        """Calculate resource utilization statistics."""
        utilization = {}
        
        for resource, allocations in resource_allocation.items():
            if allocations:
                total_allocated_hours = sum(
                    (alloc['end_time'] - alloc['start_time']).total_seconds() / 3600
                    for alloc in allocations
                )
                # Assume 24/7 availability for 30 days
                total_available_hours = 24 * 30
                utilization[resource] = (total_allocated_hours / total_available_hours) * 100
            else:
                utilization[resource] = 0.0
        
        return utilization
    
    def _analyze_timeline_optimization(self, scheduled_missions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze timeline optimization results."""
        if not scheduled_missions:
            return {}
        
        total_slack_hours = 0
        on_time_missions = 0
        
        for mission in scheduled_missions:
            # Calculate slack time
            preferred_completion = mission['latest_completion']
            actual_completion = mission['scheduled_completion']
            
            if actual_completion <= preferred_completion:
                on_time_missions += 1
                slack_hours = (preferred_completion - actual_completion).total_seconds() / 3600
                total_slack_hours += slack_hours
        
        return {
            'on_time_percentage': (on_time_missions / len(scheduled_missions)) * 100,
            'average_slack_hours': total_slack_hours / len(scheduled_missions),
            'total_missions_analyzed': len(scheduled_missions)
        }
    
    def _analyze_scheduling_conflicts(self, scheduled_missions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze scheduling conflicts and resolutions."""
        total_conflicts = sum(len(mission.get('scheduling_conflicts', [])) for mission in scheduled_missions)
        
        return {
            'total_conflicts_resolved': total_conflicts,
            'missions_with_conflicts': sum(1 for mission in scheduled_missions if mission.get('scheduling_conflicts')),
            'conflict_resolution_rate': 100.0  # All conflicts resolved in current implementation
        }
    
    def _route_intersects_alert_region(self, route: Route, alert_region: Dict[str, Any]) -> bool:
        """Check if route intersects with alert region."""
        # Simplified intersection check - can be enhanced with more precise orbital mechanics
        altitude_range = alert_region.get('altitude_range', [0, 10000])
        
        for satellite in route.satellites:
            try:
                perigee, apogee = satellite.get_altitude_km()
                if (perigee <= altitude_range[1] and apogee >= altitude_range[0]):
                    return True
            except Exception:
                continue
        
        return False
    
    def _requirement_applies_to_mission(self, requirement: RegulatoryRequirement,
                                      service_request: ServiceRequest, route: Route) -> bool:
        """Check if regulatory requirement applies to mission."""
        # Simplified applicability check
        if "global" in requirement.applicable_regions:
            return True
        
        # Check if mission operates in LEO (most common case)
        if "LEO" in requirement.applicable_regions:
            for satellite in route.satellites:
                try:
                    perigee, apogee = satellite.get_altitude_km()
                    if perigee < 2000:  # LEO threshold
                        return True
                except Exception:
                    continue
        
        return False
    
    def _generate_approval_timeline(self, requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate approval timeline for compliance requirements."""
        if not requirements:
            return {}
        
        # Sort by estimated approval time
        sorted_requirements = sorted(requirements, key=lambda r: r['estimated_time_days'])
        
        timeline = {}
        current_date = datetime.utcnow()
        
        for req in sorted_requirements:
            approval_date = current_date + timedelta(days=req['estimated_time_days'])
            timeline[req['requirement_id']] = {
                'start_date': current_date.isoformat(),
                'estimated_completion': approval_date.isoformat(),
                'authority': req['approval_authority'],
                'critical_path': req['estimated_time_days'] == max(r['estimated_time_days'] for r in requirements)
            }
        
        return timeline