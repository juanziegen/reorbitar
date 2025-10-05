import React from 'react';
import './RouteControls.css';

interface RouteControlsProps {
  isAnimating: boolean;
  onPlay: () => void;
  onPause: () => void;
  onReset: () => void;
  onSpeedChange: (speed: number) => void;
  animationSpeed: number;
  currentHop: number;
  totalHops: number;
  onHopSelect: (hopIndex: number) => void;
}

const RouteControls: React.FC<RouteControlsProps> = ({
  isAnimating,
  onPlay,
  onPause,
  onReset,
  onSpeedChange,
  animationSpeed,
  currentHop,
  totalHops,
  onHopSelect
}) => {
  return (
    <div className="route-controls">
      <div className="controls-section">
        <h3>Animation Controls</h3>
        
        <div className="playback-controls">
          <button
            className={`control-button ${isAnimating ? 'active' : ''}`}
            onClick={isAnimating ? onPause : onPlay}
            title={isAnimating ? 'Pause Animation' : 'Play Animation'}
          >
            {isAnimating ? '⏸️' : '▶️'}
          </button>
          
          <button
            className="control-button"
            onClick={onReset}
            title="Reset Animation"
          >
            ⏹️
          </button>
        </div>
        
        <div className="speed-control">
          <label>Animation Speed:</label>
          <input
            type="range"
            min="0.1"
            max="3.0"
            step="0.1"
            value={animationSpeed}
            onChange={(e) => onSpeedChange(parseFloat(e.target.value))}
            className="speed-slider"
          />
          <span className="speed-value">{animationSpeed.toFixed(1)}x</span>
        </div>
      </div>
      
      <div className="controls-section">
        <h3>Route Progress</h3>
        
        <div className="progress-info">
          <span>Hop {currentHop + 1} of {totalHops}</span>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${((currentHop + 1) / totalHops) * 100}%` }}
            />
          </div>
        </div>
        
        <div className="hop-selector">
          <label>Jump to Hop:</label>
          <select
            value={currentHop}
            onChange={(e) => onHopSelect(parseInt(e.target.value))}
            className="hop-select"
          >
            {Array.from({ length: totalHops }, (_, i) => (
              <option key={i} value={i}>
                Hop {i + 1}
              </option>
            ))}
          </select>
        </div>
      </div>
      
      <div className="controls-section">
        <h3>View Options</h3>
        
        <div className="view-controls">
          <label className="checkbox-label">
            <input type="checkbox" defaultChecked />
            Show Trajectories
          </label>
          
          <label className="checkbox-label">
            <input type="checkbox" defaultChecked />
            Show Delta-V Info
          </label>
          
          <label className="checkbox-label">
            <input type="checkbox" defaultChecked />
            Show Cost Info
          </label>
          
          <label className="checkbox-label">
            <input type="checkbox" />
            Show Propulsion Effects
          </label>
        </div>
      </div>
      
      <div className="controls-section">
        <h3>Camera Controls</h3>
        
        <div className="camera-presets">
          <button className="preset-button">Earth View</button>
          <button className="preset-button">Follow Spacecraft</button>
          <button className="preset-button">Overview</button>
          <button className="preset-button">Side View</button>
        </div>
        
        <div className="camera-info">
          <small>
            Use mouse to orbit, zoom, and pan the view
          </small>
        </div>
      </div>
    </div>
  );
};

export default RouteControls;