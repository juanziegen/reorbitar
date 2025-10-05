import React, { useState, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Route } from '../types/Route';
import TrajectoryLine from './TrajectoryLine';
import HopAnimation from './HopAnimation';

interface RouteVisualizationProps {
  route: Route | null;
  isAnimating: boolean;
  onAnimationToggle: () => void;
  onRouteReset: () => void;
}

const RouteVisualization: React.FC<RouteVisualizationProps> = ({
  route,
  isAnimating,
  onAnimationToggle,
  onRouteReset
}) => {
  const [currentHopIndex, setCurrentHopIndex] = useState(0);
  const [animationProgress, setAnimationProgress] = useState(0);
  const animationSpeed = 0.5; // Animation speed multiplier
  
  const animationRef = useRef<number>(0);
  
  // Animation loop
  useFrame((state, delta) => {
    if (!isAnimating || !route || route.hops.length === 0) return;
    
    animationRef.current += delta * animationSpeed;
    
    // Each hop takes 2 seconds to complete
    const hopDuration = 2.0;
    const totalDuration = route.hops.length * hopDuration;
    
    if (animationRef.current >= totalDuration) {
      // Animation complete, reset
      animationRef.current = 0;
      setCurrentHopIndex(0);
      setAnimationProgress(0);
      return;
    }
    
    const currentTime = animationRef.current;
    const hopIndex = Math.floor(currentTime / hopDuration);
    const hopProgress = (currentTime % hopDuration) / hopDuration;
    
    setCurrentHopIndex(hopIndex);
    setAnimationProgress(hopProgress);
  });
  
  if (!route) {
    return null;
  }
  
  return (
    <group>
      {/* Render all trajectory lines */}
      {route.hops.map((hop, index) => (
        <TrajectoryLine
          key={`trajectory-${index}`}
          hop={hop}
          isActive={index <= currentHopIndex}
          progress={index === currentHopIndex ? animationProgress : 1}
        />
      ))}
      
      {/* Render hop animations */}
      {route.hops.map((hop, index) => (
        <HopAnimation
          key={`hop-${index}`}
          hop={hop}
          isActive={index === currentHopIndex && isAnimating}
          progress={index === currentHopIndex ? animationProgress : 0}
        />
      ))}
      
      {/* Route information display */}
      <RouteInfoDisplay route={route} currentHop={currentHopIndex} />
    </group>
  );
};

interface RouteInfoDisplayProps {
  route: Route;
  currentHop: number;
}

const RouteInfoDisplay: React.FC<RouteInfoDisplayProps> = ({ route, currentHop }) => {
  // Info now displayed in UI panels, no need for 3D overlays
  return null;
};

export default RouteVisualization;