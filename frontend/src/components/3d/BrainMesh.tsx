"use client";
import React, { useRef, useState } from 'react';
import { useFrame, useLoader } from '@react-three/fiber';
import { useTexture } from '@react-three/drei';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import * as THREE from 'three';
import { BrainMeshProps } from './types';
import { useTheme } from '@/store/theme-store';

export const BrainMesh: React.FC<BrainMeshProps> = ({
  color = "#10b981",
  emissiveIntensity = 0.1
}) => {
  const meshRef = useRef<THREE.Group>(null);
  const [hovered, setHovered] = useState(false);
  const theme = useTheme();
  const isLightMode = theme === 'light';

  // Use React Three Fiber's useLoader hook for better integration
  const brainObject = useLoader(OBJLoader, '/obj/freesurff.Obj');
  const texture = useTexture('/obj/brain.jpg');

  // Animate the brain rotation with interactive effects
  useFrame((state) => {
    if (meshRef.current) {
      // Slower rotation when hovered to allow better inspection
      const rotationSpeed = hovered ? 0.002 : 0.005;
      meshRef.current.rotation.y += rotationSpeed;
      meshRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.3) * 0.1;

      // Subtle pulsing effect when hovered
      if (hovered) {
        const scale = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.02;
        meshRef.current.scale.setScalar(scale);
      } else {
        meshRef.current.scale.setScalar(1);
      }
    }
  });

  // Clone and prepare the brain object
  const brainClone = brainObject.clone();

  // Apply materials to all meshes in the object with theme-aware properties
  brainClone.traverse((child: THREE.Object3D) => {
    if (child instanceof THREE.Mesh) {
      // Enhanced material properties for better light mode visibility
      const baseOpacity = isLightMode ? 0.85 : 0.9;
      const hoverOpacity = isLightMode ? 0.95 : 1.0;
      const baseShininess = isLightMode ? 80 : 100;
      const hoverShininess = isLightMode ? 120 : 150;
      
      child.material = new THREE.MeshPhongMaterial({
        map: texture,
        color: color,
        transparent: true,
        opacity: hovered ? hoverOpacity : baseOpacity,
        emissive: color,
        emissiveIntensity: hovered ? emissiveIntensity * 1.5 : emissiveIntensity,
        shininess: hovered ? hoverShininess : baseShininess,
        // Enhanced properties for light mode
        specular: isLightMode ? new THREE.Color(0x444444) : new THREE.Color(0x222222),
        reflectivity: isLightMode ? 0.3 : 0.2,
      });
    }
  });

  // Create wireframe version with theme-aware styling
  const wireframeClone = brainObject.clone();
  wireframeClone.traverse((child: THREE.Object3D) => {
    if (child instanceof THREE.Mesh) {
      // Better wireframe visibility in light mode
      const wireframeOpacity = isLightMode ? 0.15 : 0.2;
      const wireframeColor = isLightMode ? 
        new THREE.Color(color).multiplyScalar(0.7) : // Darker in light mode
        new THREE.Color(color);
      
      child.material = new THREE.MeshBasicMaterial({
        color: wireframeColor,
        transparent: true,
        opacity: wireframeOpacity,
        wireframe: true,
      });
    }
  });

  return (
    <group
      ref={meshRef}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
      onPointerDown={(e) => e.stopPropagation()}
    >
      <primitive object={brainClone} scale={[1, 1, 1]} />
      <primitive object={wireframeClone} scale={[1.02, 1.02, 1.02]} />
    </group>
  );
};