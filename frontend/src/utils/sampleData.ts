import { Route, Hop, ManeuverDetails, FuelConsumption, OptimizationMetrics } from '../types/Route';
import { Satellite } from '../types/Satellite';

export function createSampleRoute(): Route {
  const satellites = createSampleSatellites();
  const hops = createSampleHops(satellites);
  
  return {
    id: 'sample-route-1',
    satellites: satellites,
    hops: hops,
    totalDeltaV: hops.reduce((sum, hop) => sum + hop.deltaVRequired, 0),
    totalCost: hops.reduce((sum, hop) => sum + hop.cost, 0),
    missionDuration: hops.reduce((sum, hop) => sum + hop.transferTime, 0),
    feasibilityScore: 0.85,
    optimizationMetrics: {
      convergenceGeneration: 45,
      fitnessScore: 0.92,
      constraintViolations: [],
      alternativeRoutes: 12,
      computationTime: 2.3
    }
  };
}

function createSampleSatellites(): Satellite[] {
  return [
    {
      id: 'debris-sat-1',
      name: 'Defunct Communication Satellite',
      tle: {
        line1: '1 25544U 98067A   21001.00000000  .00002182  00000-0  40864-4 0  9990',
        line2: '2 25544  51.6461 339.2971 0002829  86.3372 273.8338 15.48919103123456'
      },
      mass: 2200,
      status: 'debris',
      orbitalElements: {
        semiMajorAxis: 6793,
        eccentricity: 0.0002829,
        inclination: 51.6461,
        raan: 339.2971,
        argumentOfPerigee: 86.3372,
        meanAnomaly: 273.8338
      },
      materialComposition: {
        aluminum: 45,
        steel: 25,
        titanium: 15,
        electronics: 10,
        other: 5
      }
    },
    {
      id: 'debris-sat-2',
      name: 'Old Weather Satellite',
      tle: {
        line1: '1 25545U 98067B   21001.00000000  .00001500  00000-0  30000-4 0  9991',
        line2: '2 25545  98.2000 100.0000 0010000 120.0000 240.0000 14.20000000123457'
      },
      mass: 850,
      status: 'debris',
      orbitalElements: {
        semiMajorAxis: 7200,
        eccentricity: 0.001,
        inclination: 98.2,
        raan: 100.0,
        argumentOfPerigee: 120.0,
        meanAnomaly: 240.0
      },
      materialComposition: {
        aluminum: 50,
        steel: 20,
        titanium: 12,
        electronics: 15,
        other: 3
      }
    },
    {
      id: 'debris-sat-3',
      name: 'Decommissioned Navigation Satellite',
      tle: {
        line1: '1 25546U 98067C   21001.00000000  .00001200  00000-0  25000-4 0  9992',
        line2: '2 25546  55.0000 200.0000 0005000 45.0000 315.0000 14.50000000123458'
      },
      mass: 1500,
      status: 'decommissioned',
      orbitalElements: {
        semiMajorAxis: 6950,
        eccentricity: 0.0005,
        inclination: 55.0,
        raan: 200.0,
        argumentOfPerigee: 45.0,
        meanAnomaly: 315.0
      },
      materialComposition: {
        aluminum: 40,
        steel: 30,
        titanium: 18,
        electronics: 8,
        other: 4
      }
    },
    {
      id: 'debris-sat-4',
      name: 'Abandoned Research Satellite',
      tle: {
        line1: '1 25547U 98067D   21001.00000000  .00001800  00000-0  35000-4 0  9993',
        line2: '2 25547  82.5000 150.0000 0008000 90.0000 270.0000 14.80000000123459'
      },
      mass: 1100,
      status: 'debris',
      orbitalElements: {
        semiMajorAxis: 7100,
        eccentricity: 0.0008,
        inclination: 82.5,
        raan: 150.0,
        argumentOfPerigee: 90.0,
        meanAnomaly: 270.0
      },
      materialComposition: {
        aluminum: 42,
        steel: 28,
        titanium: 16,
        electronics: 11,
        other: 3
      }
    },
    {
      id: 'debris-sat-5',
      name: 'Failed Imaging Satellite',
      tle: {
        line1: '1 25548U 98067E   21001.00000000  .00001600  00000-0  32000-4 0  9994',
        line2: '2 25548  65.0000 280.0000 0012000 135.0000 225.0000 14.60000000123460'
      },
      mass: 1350,
      status: 'debris',
      orbitalElements: {
        semiMajorAxis: 7050,
        eccentricity: 0.0012,
        inclination: 65.0,
        raan: 280.0,
        argumentOfPerigee: 135.0,
        meanAnomaly: 225.0
      },
      materialComposition: {
        aluminum: 48,
        steel: 22,
        titanium: 14,
        electronics: 13,
        other: 3
      }
    },
    {
      id: 'debris-sat-6',
      name: 'Defunct Military Satellite',
      tle: {
        line1: '1 25549U 98067F   21001.00000000  .00001400  00000-0  28000-4 0  9995',
        line2: '2 25549  97.8000 320.0000 0015000 180.0000 180.0000 14.40000000123461'
      },
      mass: 1850,
      status: 'debris',
      orbitalElements: {
        semiMajorAxis: 7150,
        eccentricity: 0.0015,
        inclination: 97.8,
        raan: 320.0,
        argumentOfPerigee: 180.0,
        meanAnomaly: 180.0
      },
      materialComposition: {
        aluminum: 38,
        steel: 32,
        titanium: 20,
        electronics: 7,
        other: 3
      }
    }
  ];
}

function createSampleHops(satellites: Satellite[]): Hop[] {
  return [
    {
      id: 'hop-1',
      fromSatellite: satellites[0],
      toSatellite: satellites[1],
      deltaVRequired: 1250, // m/s
      transferTime: 8.5, // hours
      cost: 1587.5, // USD ($1.27 per m/s)
      maneuverDetails: {
        burnDuration: 180, // seconds
        burnDirection: { x: 0.6, y: 0.8, z: 0.0 },
        maneuverType: 'hohmann',
        phaseAngle: 45,
        waitTime: 2.5
      },
      fuelConsumption: {
        fuel: 85.2,
        oxidizer: 161.9,
        total: 247.1,
        remainingCapacity: 752.9
      }
    },
    {
      id: 'hop-2',
      fromSatellite: satellites[1],
      toSatellite: satellites[2],
      deltaVRequired: 980, // m/s
      transferTime: 6.2, // hours
      cost: 1244.6, // USD
      maneuverDetails: {
        burnDuration: 145,
        burnDirection: { x: -0.4, y: 0.3, z: 0.85 },
        maneuverType: 'plane_change',
        phaseAngle: 72,
        waitTime: 1.8
      },
      fuelConsumption: {
        fuel: 66.8,
        oxidizer: 127.0,
        total: 193.8,
        remainingCapacity: 559.1
      }
    },
    {
      id: 'hop-3',
      fromSatellite: satellites[2],
      toSatellite: satellites[3],
      deltaVRequired: 1100, // m/s
      transferTime: 7.3, // hours
      cost: 1397.0, // USD
      maneuverDetails: {
        burnDuration: 165,
        burnDirection: { x: 0.5, y: -0.6, z: 0.64 },
        maneuverType: 'combined',
        phaseAngle: 58,
        waitTime: 2.1
      },
      fuelConsumption: {
        fuel: 75.0,
        oxidizer: 142.5,
        total: 217.5,
        remainingCapacity: 341.6
      }
    },
    {
      id: 'hop-4',
      fromSatellite: satellites[3],
      toSatellite: satellites[4],
      deltaVRequired: 850, // m/s
      transferTime: 5.8, // hours
      cost: 1079.5, // USD
      maneuverDetails: {
        burnDuration: 128,
        burnDirection: { x: -0.7, y: 0.5, z: 0.5 },
        maneuverType: 'hohmann',
        phaseAngle: 82,
        waitTime: 1.5
      },
      fuelConsumption: {
        fuel: 58.0,
        oxidizer: 110.2,
        total: 168.2,
        remainingCapacity: 173.4
      }
    },
    {
      id: 'hop-5',
      fromSatellite: satellites[4],
      toSatellite: satellites[5],
      deltaVRequired: 1320, // m/s
      transferTime: 9.1, // hours
      cost: 1676.4, // USD
      maneuverDetails: {
        burnDuration: 195,
        burnDirection: { x: 0.8, y: 0.4, z: -0.45 },
        maneuverType: 'bielliptic',
        phaseAngle: 38,
        waitTime: 3.0
      },
      fuelConsumption: {
        fuel: 90.0,
        oxidizer: 171.0,
        total: 261.0,
        remainingCapacity: -87.6
      }
    },
    {
      id: 'hop-6',
      fromSatellite: satellites[5],
      toSatellite: satellites[0], // Return to base
      deltaVRequired: 1450, // m/s
      transferTime: 10.2, // hours
      cost: 1841.5, // USD
      maneuverDetails: {
        burnDuration: 210,
        burnDirection: { x: -0.55, y: -0.75, z: 0.35 },
        maneuverType: 'hohmann',
        phaseAngle: 95,
        waitTime: 3.5
      },
      fuelConsumption: {
        fuel: 99.0,
        oxidizer: 188.1,
        total: 287.1,
        remainingCapacity: -374.7
      }
    }
  ];
}

export function createSampleRoutes(): Route[] {
  return [
    createSampleRoute(),
    {
      id: 'sample-route-2',
      satellites: createSampleSatellites().slice(0, 2),
      hops: createSampleHops(createSampleSatellites()).slice(0, 1),
      totalDeltaV: 1250,
      totalCost: 1587.5,
      missionDuration: 8.5,
      feasibilityScore: 0.92,
      optimizationMetrics: {
        convergenceGeneration: 32,
        fitnessScore: 0.88,
        constraintViolations: [],
        alternativeRoutes: 8,
        computationTime: 1.7
      }
    }
  ];
}