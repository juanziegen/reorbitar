import { Satellite, Position3D, OrbitalElements } from '../types/Satellite';

// Earth constants
export const EARTH_RADIUS = 6371; // km
export const EARTH_MU = 398600.4418; // km³/s² (Earth's gravitational parameter)

/**
 * Calculate satellite position at a given time
 * This is a simplified implementation for demonstration
 * In production, use a proper orbital propagation library like satellite.js
 */
export function calculateSatellitePosition(satellite: Satellite, time: Date): Position3D {
  const elements = satellite.orbitalElements;
  
  // Calculate mean motion (rad/s)
  const n = Math.sqrt(EARTH_MU / Math.pow(elements.semiMajorAxis, 3));
  
  // Time since epoch (simplified - using current time)
  const timeSinceEpoch = (time.getTime() - Date.now()) / 1000; // seconds
  
  // Mean anomaly at current time
  const M = (elements.meanAnomaly * Math.PI / 180) + n * timeSinceEpoch;
  
  // Solve Kepler's equation (simplified - ignoring eccentricity for demo)
  const E = M; // Eccentric anomaly (simplified for circular orbits)
  
  // True anomaly
  const nu = E; // (simplified for circular orbits)
  
  // Distance from Earth center
  const r = elements.semiMajorAxis * (1 - elements.eccentricity * Math.cos(E));
  
  // Position in orbital plane
  const x_orb = r * Math.cos(nu);
  const y_orb = r * Math.sin(nu);
  const z_orb = 0;
  
  // Convert to Earth-Centered Inertial (ECI) coordinates
  const position = orbitalToECI(
    { x: x_orb, y: y_orb, z: z_orb },
    elements.inclination,
    elements.raan,
    elements.argumentOfPerigee
  );
  
  return position;
}

/**
 * Convert orbital plane coordinates to Earth-Centered Inertial (ECI)
 */
function orbitalToECI(
  orbitalPos: Position3D,
  inclination: number,
  raan: number,
  argumentOfPerigee: number
): Position3D {
  // Convert degrees to radians
  const i = inclination * Math.PI / 180;
  const Ω = raan * Math.PI / 180;
  const ω = argumentOfPerigee * Math.PI / 180;
  
  // Rotation matrices
  const cosΩ = Math.cos(Ω);
  const sinΩ = Math.sin(Ω);
  const cosi = Math.cos(i);
  const sini = Math.sin(i);
  const cosω = Math.cos(ω);
  const sinω = Math.sin(ω);
  
  // Transformation matrix elements
  const P11 = cosΩ * cosω - sinΩ * sinω * cosi;
  const P12 = -cosΩ * sinω - sinΩ * cosω * cosi;
  const P13 = sinΩ * sini;
  
  const P21 = sinΩ * cosω + cosΩ * sinω * cosi;
  const P22 = -sinΩ * sinω + cosΩ * cosω * cosi;
  const P23 = -cosΩ * sini;
  
  const P31 = sinω * sini;
  const P32 = cosω * sini;
  const P33 = cosi;
  
  // Transform to ECI
  const x = P11 * orbitalPos.x + P12 * orbitalPos.y + P13 * orbitalPos.z;
  const y = P21 * orbitalPos.x + P22 * orbitalPos.y + P23 * orbitalPos.z;
  const z = P31 * orbitalPos.x + P32 * orbitalPos.y + P33 * orbitalPos.z;
  
  return { x, y, z };
}

/**
 * Calculate orbital period in seconds
 */
export function calculateOrbitalPeriod(semiMajorAxis: number): number {
  return 2 * Math.PI * Math.sqrt(Math.pow(semiMajorAxis, 3) / EARTH_MU);
}

/**
 * Calculate altitude from semi-major axis
 */
export function calculateAltitude(semiMajorAxis: number): number {
  return semiMajorAxis - EARTH_RADIUS;
}

/**
 * Determine if orbit is in LEO (Low Earth Orbit)
 */
export function isLEO(altitude: number): boolean {
  return altitude >= 160 && altitude <= 2000;
}

/**
 * Calculate approximate delta-v between two orbital positions
 * This is a simplified calculation for demonstration
 */
export function calculateDeltaV(pos1: Position3D, pos2: Position3D): number {
  const distance = Math.sqrt(
    Math.pow(pos2.x - pos1.x, 2) +
    Math.pow(pos2.y - pos1.y, 2) +
    Math.pow(pos2.z - pos1.z, 2)
  );
  
  // Simplified delta-v approximation based on distance
  // In reality, this would require complex orbital mechanics calculations
  const r1 = Math.sqrt(pos1.x * pos1.x + pos1.y * pos1.y + pos1.z * pos1.z);
  const r2 = Math.sqrt(pos2.x * pos2.x + pos2.y * pos2.y + pos2.z * pos2.z);
  
  const v1 = Math.sqrt(EARTH_MU / r1);
  const v2 = Math.sqrt(EARTH_MU / r2);
  
  // Simplified Hohmann transfer approximation
  const deltaV = Math.abs(v2 - v1) + Math.abs(v2 - v1) * 0.1; // rough approximation
  
  return deltaV;
}