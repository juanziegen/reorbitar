import React, { Suspense, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import Earth from './Earth';
import SatelliteRenderer from './SatelliteRenderer';
import CoordinateSystem from './CoordinateSystem';
import RouteVisualization from './RouteVisualization';
import RouteControls from './RouteControls';
import RealTimeStatus from './RealTimeStatus';
import { Route } from '../types/Route';
import { createSampleRoute } from '../utils/sampleData';
import { useRealTimeData } from '../hooks/useRealTimeData';
import './OrbitalVisualization.css';

const OrbitalVisualization: React.FC = () => {
  const [selectedRoute, setSelectedRoute] = useState<Route | null>(createSampleRoute());
  const [isAnimating, setIsAnimating] = useState(true); // Start with autoplay enabled
  const [animationSpeed, setAnimationSpeed] = useState(1.0);
  const [currentHop, setCurrentHop] = useState(0);
  const [showControls, setShowControls] = useState(true);
  const [showRealTimeStatus, setShowRealTimeStatus] = useState(true);

  // Real-time data hook
  const realTimeData = useRealTimeData();

  // Auto-start animation on mount
  useEffect(() => {
    // Small delay to ensure everything is loaded
    const timer = setTimeout(() => {
      setIsAnimating(true);
    }, 500);
    return () => clearTimeout(timer);
  }, []);

  const handleAnimationToggle = () => {
    setIsAnimating(!isAnimating);
  };

  const handleRouteReset = () => {
    setIsAnimating(false);
    setCurrentHop(0);
  };

  const handleSpeedChange = (speed: number) => {
    setAnimationSpeed(speed);
  };

  const handleHopSelect = (hopIndex: number) => {
    setCurrentHop(hopIndex);
    setIsAnimating(false);
  };

  // Update satellite data when real-time data changes
  useEffect(() => {
    if (realTimeData.satellites.length > 0) {
      // Update satellite positions in the visualization
      // This would typically trigger a re-render of satellite positions
    }
  }, [realTimeData.satellites, realTimeData.satellitePositions]);

  return (
    <div className="orbital-visualization">
      <Canvas
        camera={{
          position: [0, 0, 15000],
          fov: 60,
          near: 1,
          far: 100000
        }}
        gl={{ antialias: true, alpha: true }}
      >
        <Suspense fallback={null}>
          {/* Lighting */}
          <ambientLight intensity={0.2} />
          <directionalLight
            position={[10000, 10000, 5000]}
            intensity={1}
            castShadow
            shadow-mapSize-width={2048}
            shadow-mapSize-height={2048}
          />
          
          {/* Background stars */}
          <Stars
            radius={50000}
            depth={50}
            count={5000}
            factor={4}
            saturation={0}
            fade
          />
          
          {/* Earth-centered coordinate system */}
          <CoordinateSystem />
          
          {/* Earth */}
          <Earth />
          
          {/* Satellite renderer */}
          <SatelliteRenderer 
            satellites={realTimeData.satellites}
            satellitePositions={realTimeData.satellitePositions}
          />
          
          {/* Route visualization */}
          <RouteVisualization
            route={selectedRoute}
            isAnimating={isAnimating}
            onAnimationToggle={handleAnimationToggle}
            onRouteReset={handleRouteReset}
          />
          
          {/* Camera controls */}
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            zoomSpeed={0.6}
            panSpeed={0.8}
            rotateSpeed={0.4}
            minDistance={7000}
            maxDistance={50000}
          />
        </Suspense>
      </Canvas>
      
      {/* UI Overlay - Left Side */}
      <div className="visualization-overlay">
        <div className="controls-panel">
          <h3>Visualization Controls</h3>
          <div className="control-group">
            <label>
              <input type="checkbox" defaultChecked />
              Show Satellites
            </label>
            <label>
              <input type="checkbox" defaultChecked />
              Show Orbits
            </label>
            <label>
              <input type="checkbox" />
              Show Coordinate Grid
            </label>
          </div>
        </div>
      </div>

      {/* Route Information - Bottom Right */}
      <div className="info-panel-bottom">
        <h3>Route Information</h3>
        {selectedRoute && (
          <>
            <div className="info-item">
              <span>Total Hops:</span>
              <span>{selectedRoute.hops.length}</span>
            </div>
            <div className="info-item">
              <span>Total ΔV:</span>
              <span>{selectedRoute.totalDeltaV.toFixed(0)} m/s</span>
            </div>
            <div className="info-item">
              <span>Total Cost:</span>
              <span>${(selectedRoute.totalCost / 1000).toFixed(1)}k</span>
            </div>
            <div className="info-item">
              <span>Duration:</span>
              <span>{selectedRoute.missionDuration.toFixed(1)} hrs</span>
            </div>
            <div className="info-item">
              <span>Status:</span>
              <span className={isAnimating ? 'status-playing' : 'status-paused'}>
                {isAnimating ? '▶ Playing' : '⏸ Paused'}
              </span>
            </div>
          </>
        )}
      </div>
      
      {/* Route Controls */}
      {showControls && selectedRoute && (
        <RouteControls
          isAnimating={isAnimating}
          onPlay={() => setIsAnimating(true)}
          onPause={() => setIsAnimating(false)}
          onReset={handleRouteReset}
          onSpeedChange={handleSpeedChange}
          animationSpeed={animationSpeed}
          currentHop={currentHop}
          totalHops={selectedRoute.hops.length}
          onHopSelect={handleHopSelect}
        />
      )}
      
      {/* Real-Time Status Panel */}
      {showRealTimeStatus && (
        <RealTimeStatus
          realTimeData={realTimeData}
          onReconnect={realTimeData.reconnect}
          onRefresh={realTimeData.refreshSatellites}
        />
      )}
      
      {/* Toggle Controls Button */}
      <button
        className="toggle-controls-btn"
        onClick={() => setShowControls(!showControls)}
        title={showControls ? 'Hide Controls' : 'Show Controls'}
      >
        {showControls ? '🎛️' : '⚙️'}
      </button>
      
      {/* Toggle Real-Time Status Button */}
      <button
        className="toggle-status-btn"
        onClick={() => setShowRealTimeStatus(!showRealTimeStatus)}
        title={showRealTimeStatus ? 'Hide Status' : 'Show Status'}
      >
        {showRealTimeStatus ? '📊' : '📈'}
      </button>
    </div>
  );
};

export default OrbitalVisualization;