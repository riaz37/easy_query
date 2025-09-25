import React from "react";
import { useTheme } from "@/store/theme-store";

interface SceneLightingProps {
  color: string;
}

export const SceneLighting: React.FC<SceneLightingProps> = ({ color }) => {
  const theme = useTheme();
  const isLightMode = theme === "light";

  return (
    <>
      {/* Ambient light - brighter in light mode for better visibility */}
      <ambientLight
        intensity={isLightMode ? 0.7 : 0.4}
        color={isLightMode ? "#f8fafc" : "#404040"}
      />

      {/* Main directional light - stronger and warmer in light mode */}
      <directionalLight
        position={[5, 5, 5]}
        intensity={isLightMode ? 1.2 : 0.6}
        color={isLightMode ? "#ffffff" : "#e2e8f0"}
        castShadow
      />

      {/* Secondary directional light for light mode to reduce harsh shadows */}
      {isLightMode && (
        <directionalLight
          position={[-3, -2, 4]}
          intensity={0.4}
          color="#f1f5f9"
        />
      )}

      {/* Accent point light - adjusted for theme */}
      <pointLight
        position={[0, 0, 0]}
        intensity={isLightMode ? 0.5 : 0.8}
        color={color}
        distance={15}
      />

      {/* Additional rim lighting for light mode */}
      {isLightMode && (
        <pointLight
          position={[0, 8, -5]}
          intensity={0.3}
          color="#cbd5e1"
          distance={20}
        />
      )}
    </>
  );
};
