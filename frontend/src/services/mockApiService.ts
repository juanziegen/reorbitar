import { Route, RouteRequest } from '../types/Route';
import { Satellite } from '../types/Satellite';
import {
  RouteOptimizationStatus,
  SatelliteFilters,
  SatellitePosition,
  MissionStatus
} from './apiService';

// Mock satellite data generator
function generateMockSatellites(count: number = 50): Satellite[] {
  const satellites: Satellite[] = [];
  const names = [
    'COSMOS', 'FENGYUN', 'ENVISAT', 'IRIDIUM', 'METEOR',
    'NOAA', 'TERRA', 'AQUA', 'AURA', 'DEBRIS'
  ];

  for (let i = 0; i < count; i++) {
    const name = `${names[i % names.length]}-${1000 + i}`;
    const altitude = 400 + Math.random() * 1600; // 400-2000 km
    const inclination = Math.random() * 180;
    const mass = 100 + Math.random() * 5000; // 100-5100 kg

    const eccentricity = 0.0001 + Math.random() * 0.002;
    const semiMajorAxis = 6371 + altitude; // Earth radius + altitude

    satellites.push({
      id: `SAT-${10000 + i}`,
      name,
      tle: {
        line1: `1 ${10000 + i}U 98067A   21001.00000000  .00002182  00000-0  40768-4 0  9992`,
        line2: `2 ${10000 + i}  ${inclination.toFixed(4)} 339.2911 0002829 242.9350 117.0717 15.48919103123456`,
      },
      mass,
      status: i % 3 === 0 ? 'active' : 'debris',
      orbitalElements: {
        semiMajorAxis,
        eccentricity,
        inclination,
        raan: Math.random() * 360,
        argumentOfPerigee: Math.random() * 360,
        meanAnomaly: Math.random() * 360,
        epoch: new Date(),
      },
      materialComposition: {
        aluminum: 0.3 + Math.random() * 0.2,
        steel: 0.2 + Math.random() * 0.2,
        titanium: 0.1 + Math.random() * 0.1,
        electronics: 0.1 + Math.random() * 0.1,
        other: 0.2 + Math.random() * 0.2,
      },
      decommissionDate: new Date(2020 + Math.floor(Math.random() * 5), Math.floor(Math.random() * 12), 1),
    });
  }

  return satellites;
}

// Mock position calculator
function calculateMockPosition(satellite: Satellite, timestamp?: Date): SatellitePosition {
  const time = timestamp || new Date();
  const t = time.getTime() / 1000;

  // Calculate orbital period using Kepler's third law
  const mu = 398600.4418; // Earth's gravitational parameter (km^3/s^2)
  const a = satellite.orbitalElements.semiMajorAxis;
  const period = 2 * Math.PI * Math.sqrt(Math.pow(a, 3) / mu); // seconds

  const angle = (2 * Math.PI * t) / period;
  const radius = a * (1 - satellite.orbitalElements.eccentricity * Math.cos(angle));
  const inclination = satellite.orbitalElements.inclination;
  const altitude = radius - 6371; // Subtract Earth radius

  return {
    satelliteId: satellite.id,
    timestamp: time.toISOString(),
    position: {
      x: radius * Math.cos(angle),
      y: radius * Math.sin(angle),
      z: radius * Math.sin(inclination * Math.PI / 180) * Math.sin(angle),
    },
    velocity: {
      vx: -(2 * Math.PI * radius / period) * Math.sin(angle),
      vy: (2 * Math.PI * radius / period) * Math.cos(angle),
      vz: (2 * Math.PI * radius / period) * Math.cos(inclination * Math.PI / 180),
    },
    altitude,
    latitude: Math.asin((radius * Math.sin(inclination * Math.PI / 180) * Math.sin(angle)) / radius) * 180 / Math.PI,
    longitude: angle * 180 / Math.PI,
  };
}

export class MockApiService {
  private static instance: MockApiService;
  private satellites: Satellite[];
  private wsListeners: Set<(data: any) => void>;
  private wsIntervalId: NodeJS.Timeout | null = null;

  private constructor() {
    this.satellites = generateMockSatellites(50);
    this.wsListeners = new Set();
  }

  public static getInstance(): MockApiService {
    if (!MockApiService.instance) {
      MockApiService.instance = new MockApiService();
    }
    return MockApiService.instance;
  }

  // Route optimization endpoints
  async optimizeRoute(request: RouteRequest): Promise<Route> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500));

    const targetSatellites = this.satellites
      .filter(s => s.status === 'debris')
      .slice(0, request.targetSatellites?.length || 5);

    const hops = targetSatellites.map((sat, index) => {
      const deltaVRequired = 100 + Math.random() * 400; // m/s
      const transferTime = 1 + Math.random() * 5; // hours
      const fuelNeeded = deltaVRequired * 0.5; // Simplified fuel calculation

      return {
        id: `HOP-${Date.now()}-${index}`,
        fromSatellite: index === 0 ?
          { ...this.satellites[0], id: 'BASE', name: 'Launch Base' } :
          targetSatellites[index - 1],
        toSatellite: sat,
        deltaVRequired,
        transferTime,
        cost: deltaVRequired * 1.27, // USD
        maneuverDetails: {
          burnDuration: 10 + Math.random() * 50,
          burnDirection: { x: Math.random(), y: Math.random(), z: Math.random() },
          maneuverType: 'hohmann' as const,
          phaseAngle: Math.random() * 360,
          waitTime: Math.random() * 2,
        },
        fuelConsumption: {
          fuel: fuelNeeded * 0.65,
          oxidizer: fuelNeeded * 0.35,
          total: fuelNeeded,
          remainingCapacity: 1000 - fuelNeeded * (index + 1),
        },
      };
    });

    const totalDeltaV = hops.reduce((sum, hop) => sum + hop.deltaVRequired, 0);
    const totalCost = totalDeltaV * 1.27;
    const missionDuration = hops.reduce((sum, hop) => sum + hop.transferTime, 0);

    return {
      id: `ROUTE-${Date.now()}`,
      satellites: targetSatellites,
      hops,
      totalDeltaV,
      totalCost,
      missionDuration,
      feasibilityScore: 0.85 + Math.random() * 0.14,
      optimizationMetrics: {
        convergenceGeneration: Math.floor(50 + Math.random() * 50),
        fitnessScore: 0.85 + Math.random() * 0.14,
        constraintViolations: [],
        alternativeRoutes: Math.floor(3 + Math.random() * 7),
        computationTime: 0.5 + Math.random() * 2,
      },
    };
  }

  async getRouteStatus(routeId: string): Promise<RouteOptimizationStatus> {
    await new Promise(resolve => setTimeout(resolve, 200));

    return {
      id: routeId,
      status: 'completed',
      progress: 100,
      currentGeneration: 100,
      bestFitness: 0.95,
      estimatedTimeRemaining: 0,
    };
  }

  // Satellite data endpoints
  async getSatellites(filters?: SatelliteFilters): Promise<Satellite[]> {
    await new Promise(resolve => setTimeout(resolve, 300));

    let filtered = [...this.satellites];

    if (filters?.status) {
      filtered = filtered.filter(s => s.status === filters.status);
    }

    if (filters?.minAltitude) {
      filtered = filtered.filter(s => {
        const altitude = s.orbitalElements.semiMajorAxis - 6371;
        return altitude >= filters.minAltitude!;
      });
    }

    if (filters?.maxAltitude) {
      filtered = filtered.filter(s => {
        const altitude = s.orbitalElements.semiMajorAxis - 6371;
        return altitude <= filters.maxAltitude!;
      });
    }

    return filtered;
  }

  async getSatelliteById(id: string): Promise<Satellite> {
    await new Promise(resolve => setTimeout(resolve, 200));

    const satellite = this.satellites.find(s => s.id === id);
    if (!satellite) {
      throw new Error(`Satellite ${id} not found`);
    }
    return satellite;
  }

  async getSatellitePosition(id: string, timestamp?: Date): Promise<SatellitePosition> {
    await new Promise(resolve => setTimeout(resolve, 100));

    const satellite = this.satellites.find(s => s.id === id);
    if (!satellite) {
      throw new Error(`Satellite ${id} not found`);
    }

    return calculateMockPosition(satellite, timestamp);
  }

  // Mock WebSocket connection
  createWebSocketConnection(onMessage: (data: any) => void, onError?: (error: Event) => void): any {
    this.wsListeners.add(onMessage);

    // Simulate initial connection
    setTimeout(() => {
      onMessage({
        type: 'satellite_list_update',
        satellites: this.satellites.filter(s => s.status === 'debris').slice(0, 10),
      });
    }, 500);

    // Simulate periodic position updates
    if (!this.wsIntervalId) {
      this.wsIntervalId = setInterval(() => {
        const randomSatellite = this.satellites[Math.floor(Math.random() * this.satellites.length)];
        const position = calculateMockPosition(randomSatellite);

        this.wsListeners.forEach(listener => {
          listener({
            type: 'satellite_position_update',
            satelliteId: randomSatellite.id,
            position,
          });
        });
      }, 5000);
    }

    // Return mock WebSocket object
    return {
      readyState: 1, // OPEN
      onopen: null as any,
      onmessage: null as any,
      onerror: null as any,
      onclose: null as any,
      close: () => {
        this.wsListeners.delete(onMessage);
        if (this.wsListeners.size === 0 && this.wsIntervalId) {
          clearInterval(this.wsIntervalId);
          this.wsIntervalId = null;
        }
      },
      send: () => {},
    };
  }

  // Mission tracking endpoints
  async getMissionStatus(missionId: string): Promise<MissionStatus> {
    await new Promise(resolve => setTimeout(resolve, 200));

    return {
      id: missionId,
      routeId: `ROUTE-${Date.now()}`,
      status: 'active',
      currentHop: 2,
      progress: 40,
      startTime: new Date(Date.now() - 3600000).toISOString(),
      estimatedCompletion: new Date(Date.now() + 7200000).toISOString(),
      collectedSatellites: ['SAT-10001', 'SAT-10002'],
      fuelRemaining: 75,
      totalCost: 25000,
      issues: [],
    };
  }

  async startMission(routeId: string): Promise<MissionStatus> {
    await new Promise(resolve => setTimeout(resolve, 300));

    return {
      id: `MISSION-${Date.now()}`,
      routeId,
      status: 'planned',
      currentHop: 0,
      progress: 0,
      startTime: new Date().toISOString(),
      collectedSatellites: [],
      fuelRemaining: 100,
      totalCost: 0,
      issues: [],
    };
  }
}

export default MockApiService;
