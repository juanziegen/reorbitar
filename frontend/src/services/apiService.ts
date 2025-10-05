import { Route, RouteRequest } from '../types/Route';
import { Satellite } from '../types/Satellite';
import { MockApiService } from './mockApiService';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';
const USE_MOCK_API = process.env.REACT_APP_USE_MOCK === 'true' || true; // Default to mock mode

export class ApiService {
  private static instance: ApiService;
  private baseUrl: string;
  private mockService: MockApiService | null = null;
  private useMock: boolean = USE_MOCK_API;

  private constructor() {
    this.baseUrl = API_BASE_URL;
    if (this.useMock) {
      this.mockService = MockApiService.getInstance();
      console.log('🎭 Running in MOCK MODE - No backend required');
    }
  }

  public static getInstance(): ApiService {
    if (!ApiService.instance) {
      ApiService.instance = new ApiService();
    }
    return ApiService.instance;
  }

  private async tryRealApiOrMock<T>(
    realApiFn: () => Promise<T>,
    mockApiFn: () => Promise<T>
  ): Promise<T> {
    if (this.useMock && this.mockService) {
      return mockApiFn();
    }

    try {
      return await realApiFn();
    } catch (error) {
      // If real API fails, fall back to mock
      console.warn('API call failed, falling back to mock data:', error);
      if (!this.mockService) {
        this.mockService = MockApiService.getInstance();
      }
      this.useMock = true;
      return mockApiFn();
    }
  }

  // Route optimization endpoints
  async optimizeRoute(request: RouteRequest): Promise<Route> {
    return this.tryRealApiOrMock(
      async () => {
        const response = await fetch(`${this.baseUrl}/route/optimize`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(request),
        });

        if (!response.ok) {
          throw new Error(`Route optimization failed: ${response.statusText}`);
        }

        return await response.json();
      },
      () => this.mockService!.optimizeRoute(request)
    );
  }

  async getRouteStatus(routeId: string): Promise<RouteOptimizationStatus> {
    return this.tryRealApiOrMock(
      async () => {
        const response = await fetch(`${this.baseUrl}/route/${routeId}/status`);

        if (!response.ok) {
          throw new Error(`Failed to get route status: ${response.statusText}`);
        }

        return await response.json();
      },
      () => this.mockService!.getRouteStatus(routeId)
    );
  }

  // Satellite data endpoints
  async getSatellites(filters?: SatelliteFilters): Promise<Satellite[]> {
    return this.tryRealApiOrMock(
      async () => {
        const queryParams = new URLSearchParams();
        if (filters) {
          Object.entries(filters).forEach(([key, value]) => {
            if (value !== undefined) {
              queryParams.append(key, value.toString());
            }
          });
        }

        const url = `${this.baseUrl}/satellites${queryParams.toString() ? '?' + queryParams.toString() : ''}`;
        const response = await fetch(url);

        if (!response.ok) {
          throw new Error(`Failed to fetch satellites: ${response.statusText}`);
        }

        return await response.json();
      },
      () => this.mockService!.getSatellites(filters)
    );
  }

  async getSatelliteById(id: string): Promise<Satellite> {
    return this.tryRealApiOrMock(
      async () => {
        const response = await fetch(`${this.baseUrl}/satellite/${id}`);

        if (!response.ok) {
          throw new Error(`Failed to fetch satellite: ${response.statusText}`);
        }

        return await response.json();
      },
      () => this.mockService!.getSatelliteById(id)
    );
  }

  async getSatellitePosition(id: string, timestamp?: Date): Promise<SatellitePosition> {
    return this.tryRealApiOrMock(
      async () => {
        const queryParams = new URLSearchParams();
        if (timestamp) {
          queryParams.append('timestamp', timestamp.toISOString());
        }

        const url = `${this.baseUrl}/satellite/${id}/position${queryParams.toString() ? '?' + queryParams.toString() : ''}`;
        const response = await fetch(url);

        if (!response.ok) {
          throw new Error(`Failed to fetch satellite position: ${response.statusText}`);
        }

        return await response.json();
      },
      () => this.mockService!.getSatellitePosition(id, timestamp)
    );
  }

  // Real-time updates via WebSocket
  createWebSocketConnection(onMessage: (data: any) => void, onError?: (error: Event) => void): any {
    if (this.useMock && this.mockService) {
      return this.mockService.createWebSocketConnection(onMessage, onError);
    }

    try {
      const wsUrl = this.baseUrl.replace('http', 'ws') + '/ws';
      const ws = new WebSocket(wsUrl);

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch (error) {
          console.error('WebSocket message parsing error:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        // Fall back to mock on WebSocket error
        if (!this.mockService) {
          this.mockService = MockApiService.getInstance();
          this.useMock = true;
        }
        if (onError) {
          onError(error);
        }
      };

      ws.onclose = () => {
        console.log('WebSocket connection closed');
      };

      return ws;
    } catch (error) {
      console.error('Failed to create WebSocket, using mock:', error);
      if (!this.mockService) {
        this.mockService = MockApiService.getInstance();
        this.useMock = true;
      }
      return this.mockService.createWebSocketConnection(onMessage, onError);
    }
  }

  // Mission tracking endpoints
  async getMissionStatus(missionId: string): Promise<MissionStatus> {
    return this.tryRealApiOrMock(
      async () => {
        const response = await fetch(`${this.baseUrl}/mission/${missionId}/status`);

        if (!response.ok) {
          throw new Error(`Failed to get mission status: ${response.statusText}`);
        }

        return await response.json();
      },
      () => this.mockService!.getMissionStatus(missionId)
    );
  }

  async startMission(routeId: string): Promise<MissionStatus> {
    return this.tryRealApiOrMock(
      async () => {
        const response = await fetch(`${this.baseUrl}/mission/start`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ routeId }),
        });

        if (!response.ok) {
          throw new Error(`Failed to start mission: ${response.statusText}`);
        }

        return await response.json();
      },
      () => this.mockService!.startMission(routeId)
    );
  }
}

// Types for API responses
export interface RouteOptimizationStatus {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number; // 0-100
  currentGeneration?: number;
  bestFitness?: number;
  estimatedTimeRemaining?: number; // seconds
  error?: string;
}

export interface SatelliteFilters {
  status?: 'active' | 'debris' | 'decommissioned';
  minAltitude?: number;
  maxAltitude?: number;
  inclination?: number;
  mass?: number;
}

export interface SatellitePosition {
  satelliteId: string;
  timestamp: string;
  position: {
    x: number;
    y: number;
    z: number;
  };
  velocity: {
    vx: number;
    vy: number;
    vz: number;
  };
  altitude: number;
  latitude: number;
  longitude: number;
}

export interface MissionStatus {
  id: string;
  routeId: string;
  status: 'planned' | 'active' | 'completed' | 'aborted';
  currentHop: number;
  progress: number; // 0-100
  startTime?: string;
  estimatedCompletion?: string;
  actualCompletion?: string;
  collectedSatellites: string[];
  fuelRemaining: number;
  totalCost: number;
  issues: MissionIssue[];
}

export interface MissionIssue {
  type: 'warning' | 'error' | 'info';
  message: string;
  timestamp: string;
  resolved: boolean;
}

export default ApiService;