import React, { useRef, useState, useEffect } from "react";
import { BrainModel } from "../3d/BrainModel";
import { SystemCard } from "./SystemCard";
import { Spotlight } from "../ui/spotlight";
import { useDragAndDrop } from "./hooks/useDragAndDrop";
import { useThreeScene } from "./hooks/useThreeScene";
import { SYSTEM_NODES, INITIAL_CARD_POSITIONS } from "./constants";
import { useResolvedTheme } from "@/store/theme-store";

const Dashboard: React.FC = () => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [activeNode, setActiveNode] = useState<string | null>(null);
  const theme = useResolvedTheme();

  // Use custom hooks for drag and drop functionality
  const { cardPositions, dragging, handleMouseDown } = useDragAndDrop({
    initialPositions: INITIAL_CARD_POSITIONS,
    containerRef: mountRef,
  });

  // Use custom hook for Three.js scene management
  const { updateConnections } = useThreeScene({
    mountRef,
    cardPositions,
  });

  // Update connections when positions change
  useEffect(() => {
    updateConnections();
  }, [cardPositions, updateConnections]);

  const handleNodeMouseEnter = (nodeId: string): void => {
    if (!dragging) {
      setActiveNode(nodeId);
    }
  };

  const handleNodeMouseLeave = (): void => {
    if (!dragging) {
      setActiveNode(null);
    }
  };

  return (
    <div
      className="relative w-full h-screen overflow-hidden pt-28"
      style={{
        background:
          theme === "dark"
            ? "linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f172a 100%)"
            : "linear-gradient(135deg, #ffffff 0%, #f0f9f5 30%, #e6f7ff 70%, #f0f9f5 100%)",
      }}
    >
      {/* Enhanced Background Effects for Light Mode */}
      {theme === "light" && (
        <>
          {/* Subtle gradient overlays */}
          <div className="absolute inset-0 bg-gradient-to-br from-emerald-50/30 via-transparent to-blue-50/20 pointer-events-none" />
          <div className="absolute inset-0 bg-gradient-to-tl from-green-50/40 via-transparent to-emerald-50/30 pointer-events-none" />

          {/* Floating light particles */}
          <div className="absolute inset-0 pointer-events-none overflow-hidden">
            {[...Array(12)].map((_, i) => (
              <div
                key={i}
                className="absolute w-1 h-1 bg-emerald-400/20 rounded-full animate-pulse"
                style={{
                  left: `${Math.random() * 100}%`,
                  top: `${Math.random() * 100}%`,
                  animation: `float ${
                    4 + Math.random() * 4
                  }s ease-in-out infinite`,
                  animationDelay: `${Math.random() * 2}s`,
                  boxShadow: "0 0 8px rgba(16, 185, 129, 0.3)",
                }}
              />
            ))}
          </div>
        </>
      )}

      {/* Theme-aware Spotlight Effect */}
      <div className="absolute inset-0 z-[5] pointer-events-none overflow-visible">
        <Spotlight
          className="!opacity-100 animate-spotlight absolute -top-20 left-1/2 transform -translate-x-1/2 w-[140%] h-[140%]"
          fill={theme === "dark" ? "#10b981" : "#059669"}
        />
      </div>

      {/* 3D Scene */}
      <div ref={mountRef} className="absolute inset-0" />

      {/* 3D Brain Model - Interactive with theme-aware colors */}
      <div className="absolute inset-0 flex items-center justify-center z-10">
        <div className="w-96 h-96">
          <BrainModel
            color={theme === "dark" ? "#10b981" : "#047857"}
            emissiveIntensity={theme === "dark" ? 0.2 : 0.08}
            enableControls={true}
          />
        </div>
      </div>

      {/* Overlay Content with theme-aware text colors */}
      <div
        className={`relative z-20 flex flex-col items-center justify-center h-full px-4 pointer-events-none ${
          theme === "dark" ? "text-white" : "text-gray-800"
        }`}
      >
        {/* System Cards - Now draggable with theme support */}
        <div className="absolute inset-0 pointer-events-none">
          {SYSTEM_NODES.map((node, index) => (
            <SystemCard
              key={node.id}
              node={node}
              position={cardPositions[index]}
              isActive={activeNode === node.id}
              isDragging={dragging === node.id}
              onMouseDown={handleMouseDown}
              onMouseEnter={handleNodeMouseEnter}
              onMouseLeave={handleNodeMouseLeave}
              theme={theme}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
