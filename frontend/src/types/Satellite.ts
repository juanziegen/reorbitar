export interface Satellite {
  id: string;
  name: string;
  tle: {
    line1: string;
    line2: string;
  };
  mass: number;
  status: 'active' | 'debris' | 'decommissioned';
  orbitalElements: OrbitalElements;
  materialComposition?: MaterialComposition;
  decommissionDate?: Date;
}

export interface OrbitalElements {
  semiMajorAxis: number;      // km
  eccentricity: number;       // dimensionless
  inclination: number;        // degrees
  raan: number;              // Right Ascension of Ascending Node (degrees)
  argumentOfPerigee: number; // degrees
  meanAnomaly: number;       // degrees
  epoch?: Date;
}

export interface MaterialComposition {
  aluminum: number;   // percentage
  steel: number;      // percentage
  titanium: number;   // percentage
  electronics: number; // percentage
  other: number;      // percentage
}

export interface Position3D {
  x: number;
  y: number;
  z: number;
}

export interface Velocity3D {
  vx: number;
  vy: number;
  vz: number;
}

export interface SatelliteState {
  position: Position3D;
  velocity: Velocity3D;
  timestamp: Date;
}