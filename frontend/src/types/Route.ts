import { Satellite } from './Satellite';

export interface Route {
  id: string;
  satellites: Satellite[];
  hops: Hop[];
  totalDeltaV: number;
  totalCost: number;
  missionDuration: number; // hours
  feasibilityScore: number;
  optimizationMetrics: OptimizationMetrics;
}

export interface Hop {
  id: string;
  fromSatellite: Satellite;
  toSatellite: Satellite;
  deltaVRequired: number; // m/s
  transferTime: number; // hours
  cost: number; // USD
  maneuverDetails: ManeuverDetails;
  fuelConsumption: FuelConsumption;
}

export interface ManeuverDetails {
  burnDuration: number; // seconds
  burnDirection: {
    x: number;
    y: number;
    z: number;
  };
  maneuverType: 'hohmann' | 'bielliptic' | 'plane_change' | 'combined';
  phaseAngle: number; // degrees
  waitTime: number; // hours
}

export interface FuelConsumption {
  fuel: number; // kg
  oxidizer: number; // kg
  total: number; // kg
  remainingCapacity: number; // kg
}

export interface OptimizationMetrics {
  convergenceGeneration: number;
  fitnessScore: number;
  constraintViolations: ConstraintViolation[];
  alternativeRoutes: number;
  computationTime: number; // seconds
}

export interface ConstraintViolation {
  type: 'fuel_limit' | 'time_limit' | 'orbital_mechanics' | 'collision_risk';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  suggestedFix?: string;
}

export interface RouteRequest {
  targetSatellites: string[]; // Satellite IDs
  startingPosition?: {
    latitude: number;
    longitude: number;
    altitude: number;
  };
  constraints: RouteConstraints;
  preferences: RoutePreferences;
}

export interface RouteConstraints {
  maxDeltaV: number; // m/s
  maxMissionDuration: number; // hours
  maxCost: number; // USD
  fuelCapacity: number; // kg
  launchWindow: {
    start: Date;
    end: Date;
  };
  avoidanceZones: AvoidanceZone[];
}

export interface AvoidanceZone {
  center: {
    latitude: number;
    longitude: number;
    altitude: number;
  };
  radius: number; // km
  reason: string;
}

export interface RoutePreferences {
  prioritizeSpeed: boolean;
  prioritizeCost: boolean;
  prioritizeFuel: boolean;
  allowPlaneChanges: boolean;
  preferHohmannTransfers: boolean;
  maxHopsPerRoute: number;
}

export interface RouteVisualizationState {
  selectedRoute: Route | null;
  isAnimating: boolean;
  animationSpeed: number;
  currentHop: number;
  showTrajectories: boolean;
  showDeltaVInfo: boolean;
  showCostInfo: boolean;
  showPropulsionEffects: boolean;
  cameraMode: 'free' | 'follow' | 'overview' | 'side';
}