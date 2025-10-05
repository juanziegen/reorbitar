import React, { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';
import { Hop } from '../types/Route';
import { calculateSatellitePosition } from '../utils/orbitalMechanics';

interface TrajectoryLineProps {
  hop: Hop;
  isActive: boolean;
  progress: number;
}

const TrajectoryLine: React.FC<TrajectoryLineProps> = ({ hop, isActive, progress }) => {
  // Calculate trajectory points using elliptical orbital mechanics
  const trajectoryPoints = useMemo(() => {
    const fromPos = calculateSatellitePosition(hop.fromSatellite, new Date());
    const toPos = calculateSatellitePosition(hop.toSatellite, new Date());

    const points: THREE.Vector3[] = [];
    const numPoints = 100; // More points for smoother ellipse

    // Calculate distances from Earth center
    const r1 = Math.sqrt(fromPos.x ** 2 + fromPos.y ** 2 + fromPos.z ** 2);
    const r2 = Math.sqrt(toPos.x ** 2 + toPos.y ** 2 + toPos.z ** 2);

    // For Hohmann transfer: semi-major axis is average of radii
    const a_transfer = (r1 + r2) / 2;

    // Periapsis (closest to Earth) and apoapsis (farthest)
    const r_peri = Math.min(r1, r2);
    const r_apo = Math.max(r1, r2);

    // Ensure transfer orbit doesn't intersect Earth
    const earthRadius = 6371; // km
    const minSafeRadius = earthRadius + 300; // 300km minimum altitude
    const safe_r_peri = Math.max(r_peri, minSafeRadius);

    // Recalculate semi-major axis if needed
    const safe_a = (safe_r_peri + r_apo) / 2;

    // Eccentricity of transfer ellipse
    const e_transfer = (r_apo - safe_r_peri) / (r_apo + safe_r_peri);

    // Create vectors
    const fromVec = new THREE.Vector3(fromPos.x, fromPos.y, fromPos.z);
    const toVec = new THREE.Vector3(toPos.x, toPos.y, toPos.z);

    // Find the plane of transfer
    const transferAxis = new THREE.Vector3().crossVectors(fromVec, toVec).normalize();

    // If parallel or antiparallel, use arbitrary perpendicular axis
    if (transferAxis.length() < 0.001) {
      // Choose perpendicular axis
      if (Math.abs(fromVec.x) < 0.9) {
        transferAxis.set(1, 0, 0);
      } else {
        transferAxis.set(0, 1, 0);
      }
      transferAxis.crossVectors(transferAxis, fromVec).normalize();
    }

    // Angle to sweep through
    const dotProduct = fromVec.dot(toVec) / (fromVec.length() * toVec.length());
    const sweepAngle = Math.acos(Math.max(-1, Math.min(1, dotProduct)));

    // Use the shorter arc (more realistic for orbital transfers)
    const totalAngle = Math.min(sweepAngle, Math.PI);

    // Generate elliptical path
    for (let i = 0; i <= numPoints; i++) {
      const t = i / numPoints;

      // True anomaly along the ellipse
      const nu = t * Math.PI; // Half orbit for transfer

      // Calculate radius at this true anomaly using vis-viva equation
      // r = a(1-e²)/(1+e*cos(ν))
      const r = safe_a * (1 - e_transfer * e_transfer) / (1 + e_transfer * Math.cos(nu));

      // Ensure radius is always safe
      const safeR = Math.max(r, minSafeRadius);

      // Rotate start vector around transfer axis
      const angle = t * totalAngle;
      const quaternion = new THREE.Quaternion().setFromAxisAngle(transferAxis, angle);

      const point = fromVec.clone().normalize().multiplyScalar(safeR).applyQuaternion(quaternion);

      points.push(point);
    }

    return points;
  }, [hop]);
  
  // Create visible portion of trajectory based on progress
  const visiblePoints = useMemo(() => {
    if (!isActive) return [];
    
    const numVisiblePoints = Math.floor(trajectoryPoints.length * progress);
    return trajectoryPoints.slice(0, Math.max(1, numVisiblePoints));
  }, [trajectoryPoints, isActive, progress]);
  
  const geometry = useMemo(() => {
    if (visiblePoints.length < 2) return null;
    return new THREE.BufferGeometry().setFromPoints(visiblePoints);
  }, [visiblePoints]);
  
  if (!geometry) return null;
  
  return (
    <group>
      {/* Outer glow line for better visibility */}
      <primitive object={new THREE.Line(geometry, new THREE.LineBasicMaterial({
        color: isActive ? "#8b5cf6" : "#4a5568",
        linewidth: 8,
        transparent: true,
        opacity: isActive ? 0.3 : 0.15
      }))} />

      {/* Main trajectory line */}
      <primitive object={new THREE.Line(geometry, new THREE.LineBasicMaterial({
        color: isActive ? "#06b6d4" : "#888888",
        linewidth: 4,
        transparent: true,
        opacity: isActive ? 0.95 : 0.4
      }))} />

      {/* Animated particles along trajectory */}
      {isActive && <TrajectoryParticles points={visiblePoints} progress={progress} />}

      {/* Delta-v indicators along the path */}
      {isActive && (
        <DeltaVIndicators hop={hop} trajectoryPoints={visiblePoints} />
      )}

      {/* Cost display at midpoint */}
      {isActive && progress > 0.5 && (
        <CostIndicator hop={hop} trajectoryPoints={trajectoryPoints} />
      )}
    </group>
  );
};

interface DeltaVIndicatorsProps {
  hop: Hop;
  trajectoryPoints: THREE.Vector3[];
}

// Animated particles along trajectory
interface TrajectoryParticlesProps {
  points: THREE.Vector3[];
  progress: number;
}

const TrajectoryParticles: React.FC<TrajectoryParticlesProps> = ({ points, progress }) => {
  const particleRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (particleRef.current && points.length > 0) {
      // Pulsing animation
      const scale = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.3;
      particleRef.current.scale.set(scale, scale, scale);
    }
  });

  if (points.length === 0) return null;

  const currentIndex = Math.floor(points.length * progress);
  const currentPoint = points[Math.min(currentIndex, points.length - 1)];

  return (
    <mesh ref={particleRef} position={[currentPoint.x, currentPoint.y, currentPoint.z]}>
      <sphereGeometry args={[200, 16, 16]} />
      <meshBasicMaterial
        color="#06b6d4"
        transparent
        opacity={0.9}
      />
      {/* Glow effect */}
      <pointLight color="#06b6d4" intensity={2} distance={1000} />
    </mesh>
  );
};

const DeltaVIndicators: React.FC<DeltaVIndicatorsProps> = ({ hop, trajectoryPoints }) => {
  const sphereRef1 = useRef<THREE.Mesh>(null);
  const sphereRef2 = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    // Pulsing animation for indicators
    const scale = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.2;
    if (sphereRef1.current) sphereRef1.current.scale.set(scale, scale, scale);
    if (sphereRef2.current) sphereRef2.current.scale.set(scale, scale, scale);
  });

  if (trajectoryPoints.length < 10) return null;

  // Show delta-v indicators at start and end of trajectory
  const startPoint = trajectoryPoints[0];
  const endPoint = trajectoryPoints[trajectoryPoints.length - 1];

  return (
    <group>
      {/* Start maneuver indicator */}
      <group position={[startPoint.x, startPoint.y, startPoint.z]}>
        <mesh ref={sphereRef1}>
          <sphereGeometry args={[250, 16, 16]} />
          <meshBasicMaterial
            color="#8b5cf6"
            transparent
            opacity={0.7}
          />
        </mesh>
        <pointLight color="#8b5cf6" intensity={3} distance={1500} />
        {/* Outer glow ring */}
        <mesh>
          <ringGeometry args={[300, 400, 32]} />
          <meshBasicMaterial
            color="#8b5cf6"
            transparent
            opacity={0.3}
            side={THREE.DoubleSide}
          />
        </mesh>
        {/* Start burn label */}
        <Text
          position={[0, 600, 0]}
          fontSize={140}
          color="#8b5cf6"
          anchorX="center"
          anchorY="middle"
          outlineWidth={3}
          outlineColor="#000000"
        >
          START BURN
        </Text>
        <Text
          position={[0, 400, 0]}
          fontSize={100}
          color="#ffffff"
          anchorX="center"
          anchorY="middle"
          outlineWidth={2}
          outlineColor="#000000"
        >
          {hop.fromSatellite.name}
        </Text>
      </group>

      {/* End maneuver indicator */}
      <group position={[endPoint.x, endPoint.y, endPoint.z]}>
        <mesh ref={sphereRef2}>
          <sphereGeometry args={[250, 16, 16]} />
          <meshBasicMaterial
            color="#06b6d4"
            transparent
            opacity={0.7}
          />
        </mesh>
        <pointLight color="#06b6d4" intensity={3} distance={1500} />
        {/* Outer glow ring */}
        <mesh>
          <ringGeometry args={[300, 400, 32]} />
          <meshBasicMaterial
            color="#06b6d4"
            transparent
            opacity={0.3}
            side={THREE.DoubleSide}
          />
        </mesh>
        {/* End burn label */}
        <Text
          position={[0, 600, 0]}
          fontSize={140}
          color="#06b6d4"
          anchorX="center"
          anchorY="middle"
          outlineWidth={3}
          outlineColor="#000000"
        >
          END BURN
        </Text>
        <Text
          position={[0, 400, 0]}
          fontSize={100}
          color="#ffffff"
          anchorX="center"
          anchorY="middle"
          outlineWidth={2}
          outlineColor="#000000"
        >
          {hop.toSatellite.name}
        </Text>
      </group>
    </group>
  );
};

interface CostIndicatorProps {
  hop: Hop;
  trajectoryPoints: THREE.Vector3[];
}

const CostIndicator: React.FC<CostIndicatorProps> = ({ hop, trajectoryPoints }) => {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    // Gentle floating animation
    if (groupRef.current) {
      const midIndex = Math.floor(trajectoryPoints.length / 2);
      const midPoint = trajectoryPoints[midIndex];
      if (midPoint) {
        groupRef.current.position.y = midPoint.y + Math.sin(state.clock.elapsedTime) * 100;
      }
    }
  });

  const midIndex = Math.floor(trajectoryPoints.length / 2);
  const midPoint = trajectoryPoints[midIndex];

  if (!midPoint) return null;

  const costColor = hop.cost > 50000 ? "#ef4444" : "#10b981";
  const deltaVColor = hop.deltaVRequired > 300 ? "#f59e0b" : "#06b6d4";

  return (
    <group ref={groupRef} position={[midPoint.x, midPoint.y, midPoint.z]}>
      {/* Cost text */}
      <Text
        position={[0, 700, 0]}
        fontSize={180}
        color={costColor}
        anchorX="center"
        anchorY="middle"
        outlineWidth={4}
        outlineColor="#000000"
      >
        ${(hop.cost / 1000).toFixed(1)}k
      </Text>

      {/* Delta-V text */}
      <Text
        position={[0, 400, 0]}
        fontSize={150}
        color={deltaVColor}
        anchorX="center"
        anchorY="middle"
        outlineWidth={3}
        outlineColor="#000000"
      >
        ΔV: {hop.deltaVRequired.toFixed(0)} m/s
      </Text>

      {/* Transfer time text */}
      <Text
        position={[0, 150, 0]}
        fontSize={120}
        color="#06b6d4"
        anchorX="center"
        anchorY="middle"
        outlineWidth={2}
        outlineColor="#000000"
      >
        {hop.transferTime.toFixed(1)} hrs
      </Text>

      {/* Connecting line */}
      <mesh position={[0, -200, 0]}>
        <cylinderGeometry args={[15, 15, 800]} />
        <meshBasicMaterial
          color="#ffffff"
          transparent
          opacity={0.3}
        />
      </mesh>
    </group>
  );
};

export default TrajectoryLine;