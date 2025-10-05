import React, { useRef } from 'react';
import { useFrame, useLoader } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import { TextureLoader } from 'three';
import * as THREE from 'three';

const Earth: React.FC = () => {
  const meshRef = useRef<THREE.Mesh>(null);
  
  // Earth radius in kilometers (scaled for visualization)
  const EARTH_RADIUS = 6371;
  
  // Load Earth texture (using a simple color for now, can be replaced with actual texture)
  const earthTexture = useLoader(TextureLoader, 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAyNCIgaGVpZ2h0PSI1MTIiIHZpZXdCb3g9IjAgMCAxMDI0IDUxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjEwMjQiIGhlaWdodD0iNTEyIiBmaWxsPSIjMjE0NTg1Ii8+CjxjaXJjbGUgY3g9IjIwMCIgY3k9IjE1MCIgcj0iNDAiIGZpbGw9IiM0Yjc2ODgiLz4KPGNpcmNsZSBjeD0iNDAwIiBjeT0iMjAwIiByPSI2MCIgZmlsbD0iIzRiNzY4OCIvPgo8Y2lyY2xlIGN4PSI3MDAiIGN5PSIzMDAiIHI9IjUwIiBmaWxsPSIjNGI3Njg4Ii8+CjxjaXJjbGUgY3g9IjMwMCIgY3k9IjQwMCIgcj0iMzAiIGZpbGw9IiM0Yjc2ODgiLz4KPC9zdmc+');
  
  // Rotate Earth
  useFrame((state, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.1; // Slow rotation
    }
  });

  return (
    <group>
      {/* Earth sphere */}
      <mesh ref={meshRef} castShadow receiveShadow>
        <sphereGeometry args={[EARTH_RADIUS, 64, 32]} />
        <meshPhongMaterial
          map={earthTexture}
          color="#4a90e2"
          shininess={100}
          specular="#ffffff"
        />
      </mesh>
      
      {/* Earth atmosphere glow */}
      <mesh scale={[1.02, 1.02, 1.02]}>
        <sphereGeometry args={[EARTH_RADIUS, 32, 16]} />
        <meshBasicMaterial
          color="#87ceeb"
          transparent
          opacity={0.1}
          side={THREE.BackSide}
        />
      </mesh>
      
      {/* Equatorial plane indicator */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[EARTH_RADIUS * 1.1, EARTH_RADIUS * 1.15, 64]} />
        <meshBasicMaterial
          color="#ffffff"
          transparent
          opacity={0.3}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Earth label */}
      <Text
        position={[0, EARTH_RADIUS + 1500, 0]}
        fontSize={300}
        color="#4a90e2"
        anchorX="center"
        anchorY="middle"
        outlineWidth={5}
        outlineColor="#000000"
      >
        EARTH
      </Text>
    </group>
  );
};

export default Earth;