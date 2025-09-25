import React from 'react';
import { LoadingFallbackProps } from './types';

export const LoadingFallback: React.FC<LoadingFallbackProps> = ({ color }) => (
  <mesh>
    <sphereGeometry args={[1, 32, 32]} />
    <meshPhongMaterial
      color={color}
      transparent
      opacity={0.4}
      emissive={color}
      emissiveIntensity={0.1}
    />
  </mesh>
);