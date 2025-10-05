import React from 'react';
import { RouteOptimizationStatus, MissionStatus } from '../services/apiService';
import './ProgressTracker.css';

interface ProgressTrackerProps {
  routeOptimization?: RouteOptimizationStatus | null;
  mission?: MissionStatus | null;
  onCancel?: () => void;
}

const ProgressTracker: React.FC<ProgressTrackerProps> = ({
  routeOptimization,
  mission,
  onCancel
}) => {
  if (!routeOptimization && !mission) {
    return null;
  }

  return (
    <div className="progress-tracker">
      {routeOptimization && (
        <RouteOptimizationProgress 
          status={routeOptimization} 
          onCancel={onCancel}
        />
      )}
      
      {mission && (
        <MissionProgress 
          status={mission} 
          onCancel={onCancel}
        />
      )}
    </div>
  );
};

interface RouteOptimizationProgressProps {
  status: RouteOptimizationStatus;
  onCancel?: () => void;
}

const RouteOptimizationProgress: React.FC<RouteOptimizationProgressProps> = ({
  status,
  onCancel
}) => {
  const getStatusIcon = () => {
    switch (status.status) {
      case 'pending': return '⏳';
      case 'running': return '🔄';
      case 'completed': return '✅';
      case 'failed': return '❌';
      default: return '❓';
    }
  };

  const getStatusColor = () => {
    switch (status.status) {
      case 'pending': return '#ffa500';
      case 'running': return '#4a9eff';
      case 'completed': return '#44ff44';
      case 'failed': return '#ff4444';
      default: return '#888888';
    }
  };

  return (
    <div className="progress-section">
      <div className="progress-header">
        <div className="progress-title">
          <span className="progress-icon">{getStatusIcon()}</span>
          <h3>Route Optimization</h3>
        </div>
        {onCancel && status.status === 'running' && (
          <button className="cancel-button" onClick={onCancel}>
            Cancel
          </button>
        )}
      </div>

      <div className="progress-content">
        <div className="progress-info">
          <div className="info-row">
            <span>Status:</span>
            <span style={{ color: getStatusColor() }}>
              {status.status.toUpperCase()}
            </span>
          </div>
          
          <div className="info-row">
            <span>Progress:</span>
            <span>{status.progress}%</span>
          </div>

          {status.currentGeneration && (
            <div className="info-row">
              <span>Generation:</span>
              <span>{status.currentGeneration}</span>
            </div>
          )}

          {status.bestFitness && (
            <div className="info-row">
              <span>Best Fitness:</span>
              <span>{status.bestFitness.toFixed(4)}</span>
            </div>
          )}

          {status.estimatedTimeRemaining && (
            <div className="info-row">
              <span>Time Remaining:</span>
              <span>{Math.ceil(status.estimatedTimeRemaining / 60)} minutes</span>
            </div>
          )}
        </div>

        <div className="progress-bar-container">
          <div className="progress-bar">
            <div 
              className="progress-fill optimization"
              style={{ 
                width: `${status.progress}%`,
                backgroundColor: getStatusColor()
              }}
            />
          </div>
          <div className="progress-percentage">{status.progress}%</div>
        </div>

        {status.error && (
          <div className="error-message">
            <span className="error-icon">⚠️</span>
            <span>{status.error}</span>
          </div>
        )}
      </div>
    </div>
  );
};

interface MissionProgressProps {
  status: MissionStatus;
  onCancel?: () => void;
}

const MissionProgress: React.FC<MissionProgressProps> = ({
  status,
  onCancel
}) => {
  const getStatusIcon = () => {
    switch (status.status) {
      case 'planned': return '📋';
      case 'active': return '🚀';
      case 'completed': return '🎉';
      case 'aborted': return '🛑';
      default: return '❓';
    }
  };

  const getStatusColor = () => {
    switch (status.status) {
      case 'planned': return '#888888';
      case 'active': return '#44ff44';
      case 'completed': return '#00ff88';
      case 'aborted': return '#ff4444';
      default: return '#888888';
    }
  };

  const formatTime = (timeString?: string) => {
    if (!timeString) return 'N/A';
    return new Date(timeString).toLocaleString();
  };

  const formatDuration = (start?: string, end?: string) => {
    if (!start || !end) return 'N/A';
    const duration = new Date(end).getTime() - new Date(start).getTime();
    const hours = Math.floor(duration / (1000 * 60 * 60));
    const minutes = Math.floor((duration % (1000 * 60 * 60)) / (1000 * 60));
    return `${hours}h ${minutes}m`;
  };

  return (
    <div className="progress-section">
      <div className="progress-header">
        <div className="progress-title">
          <span className="progress-icon">{getStatusIcon()}</span>
          <h3>Mission Execution</h3>
        </div>
        {onCancel && status.status === 'active' && (
          <button className="cancel-button abort" onClick={onCancel}>
            Abort Mission
          </button>
        )}
      </div>

      <div className="progress-content">
        <div className="progress-info">
          <div className="info-row">
            <span>Status:</span>
            <span style={{ color: getStatusColor() }}>
              {status.status.toUpperCase()}
            </span>
          </div>
          
          <div className="info-row">
            <span>Current Hop:</span>
            <span>{status.currentHop + 1}</span>
          </div>

          <div className="info-row">
            <span>Progress:</span>
            <span>{status.progress}%</span>
          </div>

          <div className="info-row">
            <span>Collected:</span>
            <span>{status.collectedSatellites.length} satellites</span>
          </div>

          <div className="info-row">
            <span>Fuel Remaining:</span>
            <span>{status.fuelRemaining.toFixed(1)} kg</span>
          </div>

          <div className="info-row">
            <span>Total Cost:</span>
            <span>${status.totalCost.toLocaleString()}</span>
          </div>

          {status.startTime && (
            <div className="info-row">
              <span>Started:</span>
              <span>{formatTime(status.startTime)}</span>
            </div>
          )}

          {status.estimatedCompletion && (
            <div className="info-row">
              <span>Est. Completion:</span>
              <span>{formatTime(status.estimatedCompletion)}</span>
            </div>
          )}

          {status.actualCompletion && (
            <div className="info-row">
              <span>Completed:</span>
              <span>{formatTime(status.actualCompletion)}</span>
            </div>
          )}

          {status.startTime && status.estimatedCompletion && (
            <div className="info-row">
              <span>Duration:</span>
              <span>{formatDuration(status.startTime, status.estimatedCompletion)}</span>
            </div>
          )}
        </div>

        <div className="progress-bar-container">
          <div className="progress-bar">
            <div 
              className="progress-fill mission"
              style={{ 
                width: `${status.progress}%`,
                backgroundColor: getStatusColor()
              }}
            />
          </div>
          <div className="progress-percentage">{status.progress}%</div>
        </div>

        {status.issues.length > 0 && (
          <div className="mission-issues">
            <h4>Mission Issues:</h4>
            <div className="issues-list">
              {status.issues.slice(0, 3).map((issue, index) => (
                <div key={index} className={`issue ${issue.type} ${issue.resolved ? 'resolved' : ''}`}>
                  <span className="issue-icon">
                    {issue.type === 'error' ? '❌' : issue.type === 'warning' ? '⚠️' : 'ℹ️'}
                  </span>
                  <span className="issue-message">{issue.message}</span>
                  <span className="issue-time">
                    {new Date(issue.timestamp).toLocaleTimeString()}
                  </span>
                  {issue.resolved && <span className="resolved-badge">✓</span>}
                </div>
              ))}
              {status.issues.length > 3 && (
                <div className="more-issues">
                  ... and {status.issues.length - 3} more issues
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ProgressTracker;