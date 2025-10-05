import { useState, useEffect, useCallback, useRef } from 'react';
import { ApiService, SatellitePosition, RouteOptimizationStatus, MissionStatus } from '../services/apiService';
import { Satellite } from '../types/Satellite';
import { Route } from '../types/Route';

export interface RealTimeDataState {
  satellites: Satellite[];
  satellitePositions: Map<string, SatellitePosition>;
  routeOptimizationStatus: RouteOptimizationStatus | null;
  missionStatus: MissionStatus | null;
  isConnected: boolean;
  lastUpdate: Date | null;
  error: string | null;
}

export function useRealTimeData() {
  const [state, setState] = useState<RealTimeDataState>({
    satellites: [],
    satellitePositions: new Map(),
    routeOptimizationStatus: null,
    missionStatus: null,
    isConnected: false,
    lastUpdate: null,
    error: null,
  });

  const apiService = ApiService.getInstance();
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const updateIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // WebSocket message handler
  const handleWebSocketMessage = useCallback((data: any) => {
    setState(prevState => {
      const newState = { ...prevState, lastUpdate: new Date(), error: null };

      switch (data.type) {
        case 'satellite_position_update':
          const newPositions = new Map(prevState.satellitePositions);
          newPositions.set(data.satelliteId, data.position);
          newState.satellitePositions = newPositions;
          break;

        case 'route_optimization_progress':
          newState.routeOptimizationStatus = data.status;
          break;

        case 'mission_status_update':
          newState.missionStatus = data.status;
          break;

        case 'satellite_list_update':
          newState.satellites = data.satellites;
          break;

        default:
          console.log('Unknown WebSocket message type:', data.type);
      }

      return newState;
    });
  }, []);

  // WebSocket error handler
  const handleWebSocketError = useCallback((error: Event) => {
    setState(prevState => ({
      ...prevState,
      isConnected: false,
      error: 'WebSocket connection error',
    }));

    // Attempt to reconnect after 5 seconds
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    reconnectTimeoutRef.current = setTimeout(() => {
      connectWebSocket();
    }, 5000);
  }, []);

  // Connect to WebSocket
  const connectWebSocket = useCallback(() => {
    try {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        return;
      }

      wsRef.current = apiService.createWebSocketConnection(
        handleWebSocketMessage,
        handleWebSocketError
      );

      wsRef.current.onopen = () => {
        setState(prevState => ({
          ...prevState,
          isConnected: true,
          error: null,
        }));
      };

      wsRef.current.onclose = () => {
        setState(prevState => ({
          ...prevState,
          isConnected: false,
        }));
      };
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      setState(prevState => ({
        ...prevState,
        error: 'Failed to establish real-time connection',
      }));
    }
  }, [handleWebSocketMessage, handleWebSocketError, apiService]);

  // Periodic updates for satellite positions
  const updateSatellitePositions = useCallback(async () => {
    try {
      const satellites = state.satellites;
      if (satellites.length === 0) return;

      const positionPromises = satellites.map(async (satellite) => {
        try {
          const position = await apiService.getSatellitePosition(satellite.id);
          return { satelliteId: satellite.id, position };
        } catch (error) {
          console.error(`Failed to update position for satellite ${satellite.id}:`, error);
          return null;
        }
      });

      const positions = await Promise.all(positionPromises);
      const validPositions = positions.filter(p => p !== null) as Array<{
        satelliteId: string;
        position: SatellitePosition;
      }>;

      if (validPositions.length > 0) {
        setState(prevState => {
          const newPositions = new Map(prevState.satellitePositions);
          validPositions.forEach(({ satelliteId, position }) => {
            newPositions.set(satelliteId, position);
          });
          return {
            ...prevState,
            satellitePositions: newPositions,
            lastUpdate: new Date(),
          };
        });
      }
    } catch (error) {
      console.error('Failed to update satellite positions:', error);
    }
  }, [state.satellites, apiService]);

  // Load initial satellite data
  const loadSatellites = useCallback(async () => {
    try {
      const satellites = await apiService.getSatellites({
        status: 'debris', // Focus on debris for removal service
      });
      setState(prevState => ({
        ...prevState,
        satellites,
        error: null,
      }));
    } catch (error) {
      console.error('Failed to load satellites:', error);
      setState(prevState => ({
        ...prevState,
        error: 'Failed to load satellite data',
      }));
    }
  }, [apiService]);

  // Start route optimization
  const startRouteOptimization = useCallback(async (routeRequest: any): Promise<string | null> => {
    try {
      const route = await apiService.optimizeRoute(routeRequest);
      return route.id;
    } catch (error) {
      console.error('Failed to start route optimization:', error);
      setState(prevState => ({
        ...prevState,
        error: 'Failed to start route optimization',
      }));
      return null;
    }
  }, [apiService]);

  // Start mission
  const startMission = useCallback(async (routeId: string): Promise<boolean> => {
    try {
      const missionStatus = await apiService.startMission(routeId);
      setState(prevState => ({
        ...prevState,
        missionStatus,
        error: null,
      }));
      return true;
    } catch (error) {
      console.error('Failed to start mission:', error);
      setState(prevState => ({
        ...prevState,
        error: 'Failed to start mission',
      }));
      return false;
    }
  }, [apiService]);

  // Initialize real-time data connection
  useEffect(() => {
    loadSatellites();
    connectWebSocket();

    // Set up periodic position updates (fallback if WebSocket fails)
    updateIntervalRef.current = setInterval(updateSatellitePositions, 30000); // Every 30 seconds

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (updateIntervalRef.current) {
        clearInterval(updateIntervalRef.current);
      }
    };
  }, [loadSatellites, connectWebSocket, updateSatellitePositions]);

  return {
    ...state,
    startRouteOptimization,
    startMission,
    reconnect: connectWebSocket,
    refreshSatellites: loadSatellites,
  };
}

export default useRealTimeData;