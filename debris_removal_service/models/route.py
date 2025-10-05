"""
Route and hop data models for satellite collection missions.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional
from .satellite import Satellite


@dataclass
class ManeuverDetails:
    """Details of orbital maneuvers required for a hop."""
    departure_burn_dv: float  # m/s
    arrival_burn_dv: float  # m/s
    plane_change_dv: float  # m/s
    total_dv: float  # m/s
    transfer_type: str  # "hohmann", "bi-elliptic", "plane_change", etc.
    phase_angle: float  # degrees
    wait_time: timedelta  # time to wait for proper phasing


@dataclass
class Hop:
    """Single hop between two satellites in a collection route."""
    from_satellite: Satellite
    to_satellite: Satellite
    delta_v_required: float  # m/s
    transfer_time: timedelta
    cost: float  # USD
    maneuver_details: ManeuverDetails
    hop_number: int  # Position in the route sequence
    
    def __post_init__(self):
        """Validate hop data after initialization."""
        if self.delta_v_required < 0:
            raise ValueError("Delta-v cannot be negative")
        if self.cost < 0:
            raise ValueError("Cost cannot be negative")
        if self.hop_number < 1:
            raise ValueError("Hop number must be positive")
    
    def get_efficiency_ratio(self) -> float:
        """Calculate cost efficiency as cost per m/s of delta-v."""
        if self.delta_v_required == 0:
            return float('inf')
        return self.cost / self.delta_v_required
    
    def is_feasible(self, max_dv_per_hop: float = 1000.0) -> bool:
        """Check if hop is feasible given constraints."""
        return (self.delta_v_required <= max_dv_per_hop and 
                self.transfer_time.total_seconds() > 0)


@dataclass
class Route:
    """Complete satellite collection route with multiple hops."""
    satellites: List[Satellite]
    hops: List[Hop]
    total_delta_v: float  # m/s
    total_cost: float  # USD
    mission_duration: timedelta
    feasibility_score: float  # 0.0 to 1.0, higher is better
    route_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate route consistency after initialization."""
        self._validate_route_consistency()
    
    def _validate_route_consistency(self):
        """Ensure route data is internally consistent."""
        if len(self.satellites) < 2:
            raise ValueError("Route must contain at least 2 satellites")
        
        if len(self.hops) != len(self.satellites) - 1:
            raise ValueError("Number of hops must be one less than number of satellites")
        
        # Verify hop sequence matches satellite sequence
        for i, hop in enumerate(self.hops):
            if hop.from_satellite != self.satellites[i]:
                raise ValueError(f"Hop {i} from_satellite doesn't match route sequence")
            if hop.to_satellite != self.satellites[i + 1]:
                raise ValueError(f"Hop {i} to_satellite doesn't match route sequence")
        
        # Verify totals are consistent with individual hops
        calculated_dv = sum(hop.delta_v_required for hop in self.hops)
        calculated_cost = sum(hop.cost for hop in self.hops)
        
        if abs(calculated_dv - self.total_delta_v) > 0.1:  # Allow small floating point errors
            raise ValueError("Total delta-v doesn't match sum of hop delta-vs")
        
        if abs(calculated_cost - self.total_cost) > 0.01:  # Allow small floating point errors
            raise ValueError("Total cost doesn't match sum of hop costs")
    
    def get_satellite_count(self) -> int:
        """Get number of satellites in the route."""
        return len(self.satellites)
    
    def get_average_cost_per_satellite(self) -> float:
        """Calculate average cost per satellite collected."""
        if len(self.satellites) <= 1:
            return self.total_cost
        return self.total_cost / (len(self.satellites) - 1)  # Exclude starting satellite
    
    def get_cost_efficiency(self) -> float:
        """Calculate cost efficiency as cost per m/s of delta-v."""
        if self.total_delta_v == 0:
            return float('inf')
        return self.total_cost / self.total_delta_v
    
    def get_time_efficiency(self) -> float:
        """Calculate time efficiency as satellites collected per day."""
        if self.mission_duration.total_seconds() == 0:
            return float('inf')
        satellites_collected = len(self.satellites) - 1  # Exclude starting satellite
        days = self.mission_duration.total_seconds() / 86400
        return satellites_collected / days
    
    def is_feasible(self, max_total_dv: float = 5000.0, max_duration_days: float = 365.0) -> bool:
        """Check if entire route is feasible given mission constraints."""
        if self.total_delta_v > max_total_dv:
            return False
        
        if self.mission_duration.total_seconds() > max_duration_days * 86400:
            return False
        
        if not (0.0 <= self.feasibility_score <= 1.0):
            return False
        
        # Check all individual hops are feasible
        return all(hop.is_feasible() for hop in self.hops)
    
    def get_route_summary(self) -> dict:
        """Get a summary of route characteristics."""
        return {
            'satellite_count': self.get_satellite_count(),
            'total_delta_v_ms': self.total_delta_v,
            'total_cost_usd': self.total_cost,
            'mission_duration_days': self.mission_duration.total_seconds() / 86400,
            'cost_per_satellite': self.get_average_cost_per_satellite(),
            'cost_efficiency_usd_per_ms': self.get_cost_efficiency(),
            'time_efficiency_sats_per_day': self.get_time_efficiency(),
            'feasibility_score': self.feasibility_score,
            'is_feasible': self.is_feasible()
        }