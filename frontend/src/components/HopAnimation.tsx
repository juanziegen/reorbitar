import React, { useMemo } from 'react';
import { Text } from '@react-three/drei';
import * as THREE from 'three';
import { Hop } from '../types/Route';
import { calculateSatellitePosition } from '../utils/orbitalMechanics';

interface HopAnimationProps {
  hop: Hop;
  isActive: boolean;
  progress: number;
}

const HopAnimation: React.FC<HopAnimationProps> = ({ hop, isActive, progress }) => {
  // Calculate current position along the trajectory
  const currentPosition = useMemo(() => {
    if (!isActive || progress === 0) return null;
    
    const fromPos = calculateSatellitePosition(hop.fromSatellite, new Date());
    const toPos = calculateSatellitePosition(hop.toSatellite, new Date());
    
    // Create curved trajectory
    const midPoint = new THREE.Vector3(
      (fromPos.x + toPos.x) / 2,
      (fromPos.y + toPos.y) / 2,
      (fromPos.z + toPos.z) / 2
    );
    
    // Add curvature
    const distance = Math.sqrt(
      Math.pow(toPos.x - fromPos.x, 2) +
      Math.pow(toPos.y - fromPos.y, 2) +
      Math.pow(toPos.z - fromPos.z, 2)
    );
    
    const curvature = distance * 0.2;
    const earthCenter = new THREE.Vector3(0, 0, 0);
    const midToEarth = midPoint.clone().sub(earthCenter).normalize();
    midPoint.add(midToEarth.multiplyScalar(curvature));
    
    // Calculate position along Bezier curve
    const t = progress;
    const position = new THREE.Vector3();
    position.x = (1 - t) * (1 - t) * fromPos.x + 2 * (1 - t) * t * midPoint.x + t * t * toPos.x;
    position.y = (1 - t) * (1 - t) * fromPos.y + 2 * (1 - t) * t * midPoint.y + t * t * toPos.y;
    position.z = (1 - t) * (1 - t) * fromPos.z + 2 * (1 - t) * t * midPoint.z + t * t * toPos.z;
    
    return position;
  }, [hop, isActive, progress]);
  
  if (!isActive || !currentPosition) return null;
  
  return (
    <group>
      {/* Animated spacecraft/debris collector */}
      <SpacecraftModel position={currentPosition} progress={progress} />
      
      {/* Propulsion effects */}
      <PropulsionEffects position={currentPosition} progress={progress} />
      
      {/* Progress trail */}
      <ProgressTrail hop={hop} progress={progress} />
      
      {/* Maneuver information */}
      <ManeuverInfo hop={hop} position={currentPosition} progress={progress} />
    </group>
  );
};

interface SpacecraftModelProps {
  position: THREE.Vector3;
  progress: number;
}

const SpacecraftModel: React.FC<SpacecraftModelProps> = ({ position, progress }) => {
  return (
    <group position={[position.x, position.y, position.z]}>
      {/* Main spacecraft body */}
      <mesh>
        <boxGeometry args={[200, 100, 300]} />
        <meshPhongMaterial
          color="#cccccc"
          emissive="#004400"
          emissiveIntensity={0.2}
        />
      </mesh>
      
      {/* Solar panels */}
      <mesh position={[-150, 0, 0]}>
        <boxGeometry args={[20, 400, 200]} />
        <meshPhongMaterial
          color="#000080"
          emissive="#000020"
          emissiveIntensity={0.3}
        />
      </mesh>
      
      <mesh position={[150, 0, 0]}>
        <boxGeometry args={[20, 400, 200]} />
        <meshPhongMaterial
          color="#000080"
          emissive="#000020"
          emissiveIntensity={0.3}
        />
      </mesh>
      
      {/* Thruster nozzles */}
      <mesh position={[0, 0, -180]}>
        <cylinderGeometry args={[30, 40, 60, 8]} />
        <meshPhongMaterial
          color="#444444"
          emissive="#440000"
          emissiveIntensity={progress > 0.1 && progress < 0.9 ? 0.5 : 0.1}
        />
      </mesh>
    </group>
  );
};

interface PropulsionEffectsProps {
  position: THREE.Vector3;
  progress: number;
}

const PropulsionEffects: React.FC<PropulsionEffectsProps> = ({ position, progress }) => {
  // Show propulsion effects during maneuvers (start and end of hop)
  const showEffects = progress < 0.2 || progress > 0.8;
  
  if (!showEffects) return null;
  
  return (
    <group position={[position.x, position.y, position.z - 250]}>
      {/* Thruster plume */}
      <mesh>
        <coneGeometry args={[80, 300, 8]} />
        <meshBasicMaterial
          color="#ff6600"
          transparent
          opacity={0.7}
        />
      </mesh>
      
      {/* Inner flame */}
      <mesh>
        <coneGeometry args={[40, 200, 8]} />
        <meshBasicMaterial
          color="#ffff00"
          transparent
          opacity={0.9}
        />
      </mesh>
      
      {/* Particle effects */}
      <ParticleEffects />
    </group>
  );
};

const ParticleEffects: React.FC = () => {
  const particleCount = 20;
  const positions = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 100;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 100;
      pos[i * 3 + 2] = -Math.random() * 200;
    }
    return pos;
  }, []);
  
  return (
    <points>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particleCount}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        color="#ffaa00"
        size={20}
        transparent
        opacity={0.8}
      />
    </points>
  );
};

interface ProgressTrailProps {
  hop: Hop;
  progress: number;
}

const ProgressTrail: React.FC<ProgressTrailProps> = ({ hop, progress }) => {
  // Create a fading trail behind the spacecraft
  const trailPoints = useMemo(() => {
    const fromPos = calculateSatellitePosition(hop.fromSatellite, new Date());
    const toPos = calculateSatellitePosition(hop.toSatellite, new Date());
    
    const points: THREE.Vector3[] = [];
    const trailLength = 10;
    
    for (let i = 0; i < trailLength; i++) {
      const t = Math.max(0, progress - (i * 0.05));
      if (t <= 0) break;
      
      // Calculate position along trajectory
      const midPoint = new THREE.Vector3(
        (fromPos.x + toPos.x) / 2,
        (fromPos.y + toPos.y) / 2,
        (fromPos.z + toPos.z) / 2
      );
      
      const distance = Math.sqrt(
        Math.pow(toPos.x - fromPos.x, 2) +
        Math.pow(toPos.y - fromPos.y, 2) +
        Math.pow(toPos.z - fromPos.z, 2)
      );
      
      const curvature = distance * 0.2;
      const earthCenter = new THREE.Vector3(0, 0, 0);
      const midToEarth = midPoint.clone().sub(earthCenter).normalize();
      midPoint.add(midToEarth.multiplyScalar(curvature));
      
      const position = new THREE.Vector3();
      position.x = (1 - t) * (1 - t) * fromPos.x + 2 * (1 - t) * t * midPoint.x + t * t * toPos.x;
      position.y = (1 - t) * (1 - t) * fromPos.y + 2 * (1 - t) * t * midPoint.y + t * t * toPos.y;
      position.z = (1 - t) * (1 - t) * fromPos.z + 2 * (1 - t) * t * midPoint.z + t * t * toPos.z;
      
      points.push(position);
    }
    
    return points;
  }, [hop, progress]);
  
  if (trailPoints.length < 2) return null;
  
  const geometry = new THREE.BufferGeometry().setFromPoints(trailPoints);
  
  return (
    <primitive object={new THREE.Line(geometry, new THREE.LineBasicMaterial({
      color: "#00ffff",
      transparent: true,
      opacity: 0.6,
      linewidth: 2
    }))} />
  );
};

interface ManeuverInfoProps {
  hop: Hop;
  position: THREE.Vector3;
  progress: number;
}

const ManeuverInfo: React.FC<ManeuverInfoProps> = ({ hop, position, progress }) => {
  const progressPercent = (progress * 100).toFixed(0);
  const deltaVRemaining = (hop.deltaVRequired * (1 - progress)).toFixed(0);

  return (
    <group position={[position.x, position.y + 600, position.z]}>
      {/* Progress text */}
      <Text
        position={[0, 0, 0]}
        fontSize={150}
        color="#00ff00"
        anchorX="center"
        anchorY="middle"
        outlineWidth={3}
        outlineColor="#000000"
      >
        {progressPercent}% Complete
      </Text>

      {/* Delta-v remaining text */}
      <Text
        position={[0, -250, 0]}
        fontSize={120}
        color={progress < 0.5 ? "#ffff00" : "#00ff00"}
        anchorX="center"
        anchorY="middle"
        outlineWidth={2}
        outlineColor="#000000"
      >
        ΔV: {deltaVRemaining} m/s
      </Text>
    </group>
  );
};

export default HopAnimation;