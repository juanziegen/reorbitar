import React, { useState } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';
import { Satellite } from '../types/Satellite';
import { calculateSatellitePosition } from '../utils/orbitalMechanics';

interface SatelliteRendererProps {
  satellites?: Satellite[];
  satellitePositions?: Map<string, any>;
  showOrbits?: boolean;
}

const SatelliteRenderer: React.FC<SatelliteRendererProps> = ({
  satellites = [],
  satellitePositions = new Map(),
  showOrbits = true
}) => {
  const [currentTime, setCurrentTime] = useState(new Date());
  
  // Update time for orbital calculations
  useFrame((state, delta) => {
    setCurrentTime(prev => new Date(prev.getTime() + delta * 1000 * 60)); // 1 minute per second
  });
  
  // Sample satellites for demonstration
  const sampleSatellites: Satellite[] = [
    {
      id: 'sat-1',
      name: 'Demo Satellite 1',
      tle: {
        line1: '1 25544U 98067A   21001.00000000  .00002182  00000-0  40864-4 0  9990',
        line2: '2 25544  51.6461 339.2971 0002829  86.3372 273.8338 15.48919103123456'
      },
      mass: 420,
      status: 'active',
      orbitalElements: {
        semiMajorAxis: 6793,
        eccentricity: 0.0002829,
        inclination: 51.6461,
        raan: 339.2971,
        argumentOfPerigee: 86.3372,
        meanAnomaly: 273.8338
      }
    },
    {
      id: 'sat-2',
      name: 'Demo Satellite 2',
      tle: {
        line1: '1 25545U 98067B   21001.00000000  .00001500  00000-0  30000-4 0  9991',
        line2: '2 25545  98.2000 100.0000 0010000 120.0000 240.0000 14.20000000123457'
      },
      mass: 150,
      status: 'debris',
      orbitalElements: {
        semiMajorAxis: 7200,
        eccentricity: 0.001,
        inclination: 98.2,
        raan: 100.0,
        argumentOfPerigee: 120.0,
        meanAnomaly: 240.0
      }
    }
  ];
  
  const activeSatellites = satellites.length > 0 ? satellites : sampleSatellites;
  
  return (
    <group>
      {activeSatellites.map((satellite) => (
        <SatelliteObject
          key={satellite.id}
          satellite={satellite}
          currentTime={currentTime}
          showOrbit={showOrbits}
          realTimePosition={satellitePositions.get(satellite.id)}
        />
      ))}
    </group>
  );
};

interface SatelliteObjectProps {
  satellite: Satellite;
  currentTime: Date;
  showOrbit: boolean;
  realTimePosition?: any;
}

const SatelliteObject: React.FC<SatelliteObjectProps> = ({
  satellite,
  currentTime,
  showOrbit,
  realTimePosition
}) => {
  // Use real-time position if available, otherwise calculate
  const position = realTimePosition 
    ? realTimePosition.position 
    : calculateSatellitePosition(satellite, currentTime);
  const isDebris = satellite.status === 'debris';
  
  return (
    <group>
      {/* Satellite object */}
      <mesh position={[position.x, position.y, position.z]}>
        <boxGeometry args={[100, 100, 100]} />
        <meshPhongMaterial
          color={isDebris ? '#ff4444' : '#44ff44'}
          emissive={isDebris ? '#440000' : '#004400'}
          emissiveIntensity={0.3}
        />
      </mesh>
      
      {/* Satellite label */}
      <Text
        position={[position.x, position.y + 300, position.z]}
        fontSize={150}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
        outlineWidth={3}
        outlineColor="#000000"
      >
        {satellite.name}
      </Text>
      
      {/* Orbit path */}
      {showOrbit && (
        <OrbitPath satellite={satellite} />
      )}
    </group>
  );
};

interface OrbitPathProps {
  satellite: Satellite;
}

const OrbitPath: React.FC<OrbitPathProps> = ({ satellite }) => {
  const orbitPoints: THREE.Vector3[] = [];
  const numPoints = 100;
  
  // Calculate orbit points
  for (let i = 0; i < numPoints; i++) {
    const meanAnomaly = (i / numPoints) * 2 * Math.PI;
    const position = calculateOrbitPosition(satellite.orbitalElements, meanAnomaly);
    orbitPoints.push(new THREE.Vector3(position.x, position.y, position.z));
  }
  
  const orbitGeometry = new THREE.BufferGeometry().setFromPoints(orbitPoints);
  
  return (
    <primitive object={new THREE.Line(orbitGeometry, new THREE.LineBasicMaterial({
      color: satellite.status === 'debris' ? '#ff8888' : '#88ff88',
      transparent: true,
      opacity: 0.6,
      linewidth: 2
    }))} />
  );
};

// Helper function to calculate position on orbit
const calculateOrbitPosition = (elements: any, meanAnomaly: number) => {
  const { 
    semiMajorAxis, 
    inclination, 
    raan, 
    argumentOfPerigee = 0 // Default value if not provided
  } = elements;
  
  // Simplified orbital mechanics calculation
  const E = meanAnomaly; // Simplified: ignoring eccentricity for demo
  const r = semiMajorAxis;
  
  // Position in orbital plane
  const x_orb = r * Math.cos(E);
  const y_orb = r * Math.sin(E);
  // z_orb = 0 for circular orbit in orbital plane
  
  // Rotate to Earth-centered inertial frame
  const cosRaan = Math.cos(raan * Math.PI / 180);
  const sinRaan = Math.sin(raan * Math.PI / 180);
  const cosInc = Math.cos(inclination * Math.PI / 180);
  const sinInc = Math.sin(inclination * Math.PI / 180);
  const cosArg = Math.cos(argumentOfPerigee * Math.PI / 180);
  const sinArg = Math.sin(argumentOfPerigee * Math.PI / 180);
  
  const x = (cosRaan * cosArg - sinRaan * sinArg * cosInc) * x_orb +
            (-cosRaan * sinArg - sinRaan * cosArg * cosInc) * y_orb;
  const y = (sinRaan * cosArg + cosRaan * sinArg * cosInc) * x_orb +
            (-sinRaan * sinArg + cosRaan * cosArg * cosInc) * y_orb;
  const z = (sinInc * sinArg) * x_orb + (sinInc * cosArg) * y_orb;
  
  return { x, y, z };
};

export default SatelliteRenderer;