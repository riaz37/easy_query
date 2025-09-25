"use client";
import React, { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { BrainMesh } from "./BrainMesh";
import { LoadingFallback } from "./LoadingFallback";
import { SceneLighting } from "./SceneLighting";
import { BrainModelProps } from "./types";

export const BrainModel: React.FC<BrainModelProps> = ({
  className = "",
  color = "#10b981",
  emissiveIntensity = 0.1,
  enableControls = false,
}) => {
  return (
    <div className={`w-full h-full ${className}`}>
      <Canvas
        camera={{ position: [0, 0, 8], fov: 45 }}
        gl={{ antialias: true, alpha: true }}
        style={{
          background: "transparent",
          cursor: enableControls ? "grab" : "default",
        }}
        onPointerDown={(e) => {
          if (enableControls) {
            e.currentTarget.style.cursor = "grabbing";
          }
        }}
        onPointerUp={(e) => {
          if (enableControls) {
            e.currentTarget.style.cursor = "grab";
          }
        }}
      >
        <SceneLighting color={color} />

        {/* Brain model with Suspense */}
        <Suspense fallback={<LoadingFallback color={color} />}>
          <BrainMesh color={color} emissiveIntensity={emissiveIntensity} />
        </Suspense>

        {/* Interactive orbit controls */}
        {enableControls && (
          <OrbitControls
            minDistance={3}
            maxDistance={20}
            rotateSpeed={1.0}
            zoomSpeed={0.8}
            panSpeed={0.5}
            maxPolarAngle={Math.PI}
            minPolarAngle={0}
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            autoRotate={false}
            dampingFactor={0.05}
            enableDamping={true}
          />
        )}
      </Canvas>
    </div>
  );
};

export default BrainModel;
