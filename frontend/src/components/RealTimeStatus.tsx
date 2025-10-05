import React from 'react';
import { RealTimeDataState } from '../hooks/useRealTimeData';
import './RealTimeStatus.css';

interface RealTimeStatusProps {
  realTimeData: RealTimeDataState;
  onReconnect: () => void;
  onRefresh: () => void;
}

const RealTimeStatus: React.FC<RealTimeStatusProps> = ({
  realTimeData,
  onReconnect,
  onRefresh
}) => {
  const {
    isConnected,
    lastUpdate,
    error,
    satellites,
    satellitePositions,
    routeOptimizationStatus,
    missionStatus
  } = realTimeData;

  const formatLastUpdate = (date: Date | null) => {
    if (!date) return 'Never';
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSeconds = Math.floor(diffMs / 1000);
    
    if (diffSeconds < 60) return `${diffSeconds}s ago`;
    if (diffSeconds < 3600) return `${Math.floor(diffSeconds / 60)}m ago`;
    return `${Math.floor(diffSeconds / 3600)}h ago`;
  };

  return (
    <div className="real-time-status">
      <div className="status-header">
        <h3>Real-Time Status</h3>
        <div className={`connection-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
          <div className="status-dot" />
          <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
        </div>
      </div>

      {error && (
        <div className="error-message">
          <span className="error-icon">⚠️</span>
          <span>{error}</span>
          <button onClick={onReconnect} className="retry-button">
            Retry
          </button>
        </div>
      )}

      <div className="status-sections">
        {/* Data Status */}
        <div className="status-section">
          <h4>Data Status</h4>
          <div className="status-item">
            <span>Satellites Tracked:</span>
            <span className="status-value">{satellites.length}</span>
          </div>
          <div className="status-item">
            <span>Position Updates:</span>
            <span className="status-value">{satellitePositions.size}</span>
          </div>
          <div className="status-item">
            <span>Last Update:</span>
            <span className="status-value">{formatLastUpdate(lastUpdate)}</span>
          </div>
          <button onClick={onRefresh} className="refresh-button">
            🔄 Refresh
          </button>
        </div>

        {/* Route Optimization Status */}
        {routeOptimizationStatus && (
          <div className="status-section">
            <h4>Route Optimization</h4>
            <div className="status-item">
              <span>Status:</span>
              <span className={`status-badge ${routeOptimizationStatus.status}`}>
                {routeOptimizationStatus.status}
              </span>
            </div>
            <div className="status-item">
              <span>Progress:</span>
              <span className="status-value">{routeOptimizationStatus.progress}%</span>
            </div>
            {routeOptimizationStatus.currentGeneration && (
              <div className="status-item">
                <span>Generation:</span>
                <span className="status-value">{routeOptimizationStatus.currentGeneration}</span>
              </div>
            )}
            {routeOptimizationStatus.estimatedTimeRemaining && (
              <div className="status-item">
                <span>ETA:</span>
                <span className="status-value">
                  {Math.ceil(routeOptimizationStatus.estimatedTimeRemaining / 60)}m
                </span>
              </div>
            )}
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${routeOptimizationStatus.progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Mission Status */}
        {missionStatus && (
          <div className="status-section">
            <h4>Mission Status</h4>
            <div className="status-item">
              <span>Status:</span>
              <span className={`status-badge ${missionStatus.status}`}>
                {missionStatus.status}
              </span>
            </div>
            <div className="status-item">
              <span>Current Hop:</span>
              <span className="status-value">{missionStatus.currentHop + 1}</span>
            </div>
            <div className="status-item">
              <span>Progress:</span>
              <span className="status-value">{missionStatus.progress}%</span>
            </div>
            <div className="status-item">
              <span>Collected:</span>
              <span className="status-value">{missionStatus.collectedSatellites.length}</span>
            </div>
            <div className="status-item">
              <span>Fuel Remaining:</span>
              <span className="status-value">{missionStatus.fuelRemaining.toFixed(1)} kg</span>
            </div>
            <div className="progress-bar">
              <div
                className="progress-fill mission"
                style={{ width: `${missionStatus.progress}%` }}
              />
            </div>
            
            {missionStatus.issues.length > 0 && (
              <div className="mission-issues">
                <h5>Issues:</h5>
                {missionStatus.issues.map((issue, index) => (
                  <div key={index} className={`issue ${issue.type} ${issue.resolved ? 'resolved' : ''}`}>
                    <span className="issue-icon">
                      {issue.type === 'error' ? '❌' : issue.type === 'warning' ? '⚠️' : 'ℹ️'}
                    </span>
                    <span className="issue-message">{issue.message}</span>
                    {issue.resolved && <span className="resolved-badge">✓</span>}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Satellite List */}
        <div className="status-section">
          <h4>Tracked Satellites</h4>
          <div className="satellite-list">
            {satellites.slice(0, 5).map((satellite) => {
              const position = satellitePositions.get(satellite.id);
              return (
                <div key={satellite.id} className="satellite-item">
                  <div className="satellite-info">
                    <span className="satellite-name">{satellite.name}</span>
                    <span className={`satellite-status ${satellite.status}`}>
                      {satellite.status}
                    </span>
                  </div>
                  {position && (
                    <div className="satellite-position">
                      <span>Alt: {position.altitude.toFixed(0)} km</span>
                      <span>Lat: {position.latitude.toFixed(2)}°</span>
                      <span>Lon: {position.longitude.toFixed(2)}°</span>
                    </div>
                  )}
                </div>
              );
            })}
            {satellites.length > 5 && (
              <div className="satellite-item more">
                <span>... and {satellites.length - 5} more</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RealTimeStatus;