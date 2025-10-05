import React from 'react';
import * as THREE from 'three';

const CoordinateSystem: React.FC = () => {
  const AXIS_LENGTH = 15000;
  const EARTH_RADIUS = 6371;
  
  // Create axis geometries
  const axisGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(AXIS_LENGTH, 0, 0)
  ]);
  
  return (
    <group>
      {/* X-axis (Red) - Points towards vernal equinox */}
      <primitive object={new THREE.Line(axisGeometry, new THREE.LineBasicMaterial({ color: "#ff4444", linewidth: 2 }))} />
      <mesh position={[AXIS_LENGTH + 500, 0, 0]}>
        <sphereGeometry args={[200, 8, 6]} />
        <meshBasicMaterial color="#ff4444" />
      </mesh>
      
      {/* Y-axis (Green) - Points 90° ahead in equatorial plane */}
      <primitive object={new THREE.Line(axisGeometry, new THREE.LineBasicMaterial({ color: "#44ff44", linewidth: 2 }))} rotation={[0, 0, Math.PI / 2]} />
      <mesh position={[0, AXIS_LENGTH + 500, 0]}>
        <sphereGeometry args={[200, 8, 6]} />
        <meshBasicMaterial color="#44ff44" />
      </mesh>
      
      {/* Z-axis (Blue) - Points towards North Pole */}
      <primitive object={new THREE.Line(axisGeometry, new THREE.LineBasicMaterial({ color: "#4444ff", linewidth: 2 }))} rotation={[0, -Math.PI / 2, 0]} />
      <mesh position={[0, 0, AXIS_LENGTH + 500]}>
        <sphereGeometry args={[200, 8, 6]} />
        <meshBasicMaterial color="#4444ff" />
      </mesh>
      
      {/* Orbital altitude reference circles */}
      {/* LEO boundary at 2000 km */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[EARTH_RADIUS + 2000, EARTH_RADIUS + 2010, 64]} />
        <meshBasicMaterial
          color="#ffff00"
          transparent
          opacity={0.4}
          side={THREE.DoubleSide}
        />
      </mesh>
      
      {/* ISS orbit at ~400 km */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[EARTH_RADIUS + 400, EARTH_RADIUS + 405, 64]} />
        <meshBasicMaterial
          color="#00ff00"
          transparent
          opacity={0.6}
          side={THREE.DoubleSide}
        />
      </mesh>
      
      {/* Grid lines for reference */}
      <GridLines />
    </group>
  );
};

const GridLines: React.FC = () => {
  const GRID_SIZE = 20000;
  const GRID_DIVISIONS = 20;
  
  return (
    <group>
      {/* XY plane grid */}
      <gridHelper
        args={[GRID_SIZE, GRID_DIVISIONS, '#444444', '#222222']}
        rotation={[Math.PI / 2, 0, 0]}
      />
      
      {/* XZ plane grid */}
      <gridHelper
        args={[GRID_SIZE, GRID_DIVISIONS, '#444444', '#222222']}
      />
    </group>
  );
};

export default CoordinateSystem;