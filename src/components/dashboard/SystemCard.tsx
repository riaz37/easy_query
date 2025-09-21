import React, { useState, useRef, useCallback } from "react";
import { SystemNode, CardPosition } from "./types";
import Image from "next/image";

interface SystemCardProps {
  node: SystemNode;
  position: CardPosition;
  isActive: boolean;
  isDragging: boolean;
  onMouseDown: (e: React.MouseEvent, nodeId: string) => void;
  onMouseEnter: (nodeId: string) => void;
  onMouseLeave: () => void;
  onPositionChange?: (nodeId: string, newPosition: CardPosition) => void;
  theme?: "light" | "dark";
}

export const SystemCard: React.FC<SystemCardProps> = ({
  node,
  position,
  isActive,
  isDragging,
  onMouseDown,
  onMouseEnter,
  onMouseLeave,
  onPositionChange,
  theme = "dark",
}) => {
  const [isDraggingLocal, setIsDraggingLocal] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const cardRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    setIsDraggingLocal(true);
    setDragStart({ x: e.clientX, y: e.clientY });
    setDragOffset({ x: 0, y: 0 });
    
    onMouseDown(e, node.id);
  }, [node.id, onMouseDown]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDraggingLocal) return;
    
    e.preventDefault();
    
    const deltaX = e.clientX - dragStart.x;
    const deltaY = e.clientY - dragStart.y;
    
    setDragOffset({ x: deltaX, y: deltaY });
  }, [isDraggingLocal, dragStart]);

  const handleMouseUp = useCallback(() => {
    if (!isDraggingLocal) return;
    
    setIsDraggingLocal(false);
    
    // Calculate new position based on drag offset
    if (onPositionChange && (dragOffset.x !== 0 || dragOffset.y !== 0)) {
      const containerWidth = window.innerWidth;
      const containerHeight = window.innerHeight;
      
      const newX = Math.max(0, Math.min(100, position.x + (dragOffset.x / containerWidth) * 100));
      const newY = Math.max(0, Math.min(100, position.y + (dragOffset.y / containerHeight) * 100));
      
      onPositionChange(node.id, { x: newX, y: newY });
    }
    
    setDragOffset({ x: 0, y: 0 });
  }, [isDraggingLocal, dragOffset, position, onPositionChange, node.id]);

  // Add global mouse event listeners
  React.useEffect(() => {
    if (isDraggingLocal) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDraggingLocal, handleMouseMove, handleMouseUp]);

  // Calculate current position with drag offset
  const currentPosition = {
    x: position.x + (dragOffset.x / window.innerWidth) * 100,
    y: position.y + (dragOffset.y / window.innerHeight) * 100
  };
  return (
    <div
      ref={cardRef}
      className="absolute transform -translate-x-1/2 -translate-y-1/2 pointer-events-auto"
      style={{
        left: `${currentPosition.x}%`,
        top: `${currentPosition.y}%`,
        zIndex: (isDragging || isDraggingLocal) ? 1000 : 10,
        transition: (isDragging || isDraggingLocal) ? "none" : "all 0.3s ease-out",
      }}
      onMouseEnter={() => onMouseEnter(node.id)}
      onMouseLeave={onMouseLeave}
    >
      <div
        className={`relative group cursor-grab active:cursor-grabbing transition-all duration-300 ${
          isActive || isDragging || isDraggingLocal ? "scale-105" : "hover:scale-105"
        } ${(isDragging || isDraggingLocal) ? "z-50 shadow-2xl" : ""}`}
        onMouseDown={handleMouseDown}
        style={{
          cursor: (isDragging || isDraggingLocal) ? "grabbing" : "grab",
          filter:
            isActive || isDragging || isDraggingLocal
              ? "drop-shadow(0 0 30px rgba(16, 185, 129, 0.6))"
              : undefined,
        }}
        onMouseEnter={(e) => {
          if (!isDraggingLocal) {
            e.currentTarget.style.filter =
              "drop-shadow(0 0 40px rgba(16, 185, 129, 0.8))";
          }
        }}
        onMouseLeave={(e) => {
          if (!isActive && !isDragging && !isDraggingLocal) {
            e.currentTarget.style.filter = "";
          }
        }}
      >
        {/* Glass Card - Theme-aware Design */}
        <div
          className={`relative overflow-hidden ${
            (isDragging || isDraggingLocal) ? "shadow-2xl shadow-emerald-500/20" : ""
          }`}
          style={{
            width: "371.2px",
            height: "200px",
            borderRadius: "25.6px",
            background:
              theme === "light"
                ? "rgba(255, 255, 255, 0.95)"
                : "rgba(255, 255, 255, 0.03)",
            border:
              theme === "light"
                ? "1.2px solid rgba(16, 185, 129, 0.2)"
                : "1.2px solid transparent",
            backgroundImage:
              theme === "light"
                ? "linear-gradient(158.39deg, rgba(255, 255, 255, 0.98) 14.19%, rgba(240, 249, 245, 0.95) 50.59%, rgba(255, 255, 255, 0.98) 68.79%, rgba(240, 249, 245, 0.95) 105.18%)"
                : "linear-gradient(158.39deg, rgba(255, 255, 255, 0.06) 14.19%, rgba(255, 255, 255, 0.000015) 50.59%, rgba(255, 255, 255, 0.000015) 68.79%, rgba(255, 255, 255, 0.015) 105.18%)",
            backgroundOrigin: "border-box",
            backgroundClip: "padding-box, border-box",
            backdropFilter: "blur(20px)",
            WebkitBackdropFilter: "blur(20px)",
            boxShadow:
              theme === "light"
                ? "0 8px 32px rgba(16, 185, 129, 0.12), 0 1px 0 rgba(255, 255, 255, 0.8) inset, 0 0 0 1px rgba(16, 185, 129, 0.08)"
                : undefined,
          }}
        >
          {/* Animated glow dots in corners */}
          <div
            className="absolute top-4 right-4 w-1 h-1 bg-emerald-400 rounded-full animate-pulse opacity-60"
            style={{
              boxShadow:
                theme === "light" ? "0 0 6px #10b981" : "0 0 8px #10b981",
            }}
          />
          <div
            className="absolute top-8 right-2 w-0.5 h-0.5 bg-emerald-300 rounded-full animate-pulse opacity-40"
            style={{
              boxShadow:
                theme === "light" ? "0 0 3px #6ee7b7" : "0 0 4px #6ee7b7",
            }}
          />
          <div
            className="absolute bottom-12 right-6 w-0.5 h-0.5 bg-emerald-400 rounded-full animate-pulse opacity-50"
            style={{
              boxShadow:
                theme === "light" ? "0 0 4px #10b981" : "0 0 6px #10b981",
            }}
          />
          <div
            className="absolute bottom-4 right-2 w-1 h-1 bg-emerald-300 rounded-full animate-pulse opacity-30"
            style={{
              boxShadow:
                theme === "light" ? "0 0 6px #6ee7b7" : "0 0 8px #6ee7b7",
            }}
          />

          {/* Main content layout */}
          <div className="relative z-10 h-full">
            {/* Dynamic Icon - Background */}
            <div className="absolute left-0 top-0 w-80 h-full">
              <Image
                src={node.iconPath}
                alt={node.title}
                fill
                className="object-cover rounded-l-[25.6px]"
                priority
                style={{
                  objectPosition: "left center",
                }}
              />
            </div>

            {/* Dynamic Text content - Overlapping */}
            <div className="absolute left-48 top-0 right-0 h-full px-6 py-6 flex flex-col justify-center overflow-hidden z-10">
              {/* Gradient overlay to blend with the image */}
              <div
                className="absolute inset-0 opacity-20"
                style={{
                  background:
                    "linear-gradient(0deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.03)), linear-gradient(246.02deg, rgba(19, 245, 132, 0) 91.9%, rgba(19, 245, 132, 0.2) 114.38%), linear-gradient(59.16deg, rgba(19, 245, 132, 0) 71.78%, rgba(19, 245, 132, 0.2) 124.92%)",
                }}
              />

              <div className="relative z-10">
                <h2 className="modal-title-enhanced text-xl font-semibold mb-4 tracking-wide">
                  {node.title}
                </h2>
                <p className="modal-description-enhanced text-sm leading-relaxed break-words">
                  {node.description}
                </p>
              </div>
            </div>
          </div>

          {/* Hover glow effect */}
          <div
            className={`absolute inset-0 rounded-[25.6px] transition-all duration-500 pointer-events-none ${
              isDragging || isDraggingLocal || isActive
                ? "opacity-30"
                : "opacity-0 group-hover:opacity-20"
            }`}
            style={{
              background:
                theme === "light"
                  ? "radial-gradient(circle at 30% 50%, rgba(16, 185, 129, 0.15) 0%, transparent 70%)"
                  : "radial-gradient(circle at 30% 50%, rgba(16, 185, 129, 0.2) 0%, transparent 70%)",
            }}
          />
        </div>
      </div>
    </div>
  );
};
