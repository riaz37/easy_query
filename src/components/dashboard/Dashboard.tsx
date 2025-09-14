import React, { useRef, useState } from "react";
import { SystemCard } from "./SystemCard";
import { Spotlight } from "../ui/spotlight";
import { useDragAndDrop } from "./hooks/useDragAndDrop";
import { SYSTEM_NODES, INITIAL_CARD_POSITIONS } from "./constants";
import { useResolvedTheme } from "@/store/theme-store";
import Image from "next/image";

const Dashboard: React.FC = () => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [activeNode, setActiveNode] = useState<string | null>(null);
  const theme = useResolvedTheme();

  // Use custom hooks for drag and drop functionality
  const { cardPositions, dragging, handleMouseDown } = useDragAndDrop({
    initialPositions: INITIAL_CARD_POSITIONS,
    containerRef: mountRef,
  });

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
            ? "var(--BG-2-Primary-Color, #000000D9)"
            : "linear-gradient(135deg, #ffffff 0%, #f0f9f5 30%, #e6f7ff 70%, #f0f9f5 100%)",
      }}
    >
      {/* Frame SVG Background */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <Image
          src="/dashboard/frame.svg"
          alt="Background Frame"
          fill
          className="object-cover object-top"
          priority
          style={{
            opacity: 0.6,
            mixBlendMode: 'lighten',
          }}
        />
      </div>

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

      {/* Brain Model SVG - Centered with circle.svg borders */}
      <div className="absolute inset-0 flex items-center justify-center z-10">
        <div className="relative w-[1100px] h-[1100px]">
          {/* Circle Border - Positioned lower to center the brain model */}
          <div 
            className="absolute"
            style={{ 
              width: '1100px',
              height: '1100px',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -45%)', // Moved up by 5% to center brain
              zIndex: 1
            }}
          >
            <Image
              src="/dashboard/circle.svg"
              alt="Circle Border"
              fill
              className="object-contain transition-all duration-300 opacity-70"
              priority
            />
          </div>
          
          {/* Brain Model Container - Centered in the viewport */}
          <div 
            className="absolute"
            style={{ 
              width: '276px', 
              height: '208px',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              zIndex: 2
            }}
          >
            <Image
              src="/dashboard/brainmodel.svg"
              alt="Brain Model"
              fill
              className={`object-contain transition-all duration-300 ${
                theme === "dark"
                  ? "brightness-110 contrast-110"
                  : "brightness-90 contrast-105"
              }`}
              priority
            />
          </div>
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
