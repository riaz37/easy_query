"use client";
import { cn } from "@/lib/utils";
import { motion } from "motion/react";
import React, { useEffect, useState } from "react";
import { useResolvedTheme } from "@/store/theme-store";

export const EnhancedBackground = ({
  children,
  className,
  intensity = "medium",
}: {
  children: React.ReactNode;
  className?: string;
  intensity?: "low" | "medium" | "high";
}) => {
  const resolvedTheme = useResolvedTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return <div className="w-full h-full" />;
  }

  const currentTheme = resolvedTheme;
  const particleCount =
    intensity === "high" ? 15 : intensity === "low" ? 5 : 8; // Reduced particle count
  const gridSize = intensity === "high" ? 60 : intensity === "low" ? 80 : 70; // Larger grid for less clutter

  return (
    <div className={cn("relative w-full h-full min-h-screen overflow-hidden", className)}>
      {/* Theme-aware gradient background with smooth transitions */}
      <div
        className="absolute inset-0 bg-gradient-to-br transition-colors duration-700 ease-in-out"
        style={{
          backgroundImage: currentTheme === "dark"
            ? "linear-gradient(to bottom right, #0a0a0a, #1a1a1a, #2a2a2a)"
            : "linear-gradient(135deg, #ffffff 0%, #f8fafc 25%, #f0f9f5 50%, #e6f7ff 75%, #f0f9f5 100%)"
        }}
      />

      {/* Additional light mode enhancements */}
      {currentTheme === "light" && (
        <>
          {/* Subtle radial gradients for depth */}
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-emerald-100/30 rounded-full blur-3xl" />
          <div className="absolute bottom-0 right-1/4 w-80 h-80 bg-blue-100/25 rounded-full blur-3xl" />
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[600px] h-[400px] bg-green-50/20 rounded-full blur-3xl" />
        </>
      )}

      {/* Subtle animated grid */}
      <div className="absolute inset-0 opacity-5">
        <div
          className="absolute inset-0"
          style={{
            backgroundImage: `
              linear-gradient(rgba(34,197,94,0.05) 1px, transparent 1px),
              linear-gradient(90deg, rgba(34,197,94,0.05) 1px, transparent 1px)
            `,
            backgroundSize: `${gridSize}px ${gridSize}px`,
          }}
        />
      </div>

      {/* Floating particles */}
      <div className="absolute inset-0">
        {[...Array(particleCount)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-green-400/40 rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
            animate={{
              y: [0, -100, 0],
              opacity: [0, 1, 0],
              scale: [0, 1, 0],
            }}
            transition={{
              duration: 4 + Math.random() * 3,
              repeat: Infinity,
              delay: Math.random() * 4,
            }}
          />
        ))}
      </div>

      {/* Subtle animated lines */}
      <div className="absolute inset-0">
        {[...Array(intensity === "high" ? 4 : intensity === "low" ? 1 : 2)].map(
          (_, i) => (
            <motion.div
              key={i}
              className="absolute h-px bg-gradient-to-r from-transparent via-green-400/30 to-transparent"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                width: `${100 + Math.random() * 200}px`,
              }}
              animate={{
                x: [0, 100, 0],
                opacity: [0, 0.3, 0],
              }}
              transition={{
                duration: 8 + Math.random() * 4,
                repeat: Infinity,
                delay: Math.random() * 8,
              }}
            />
          )
        )}
      </div>

      {/* Subtle glowing orbs */}
      <div className="absolute inset-0">
        {[...Array(intensity === "high" ? 3 : intensity === "low" ? 1 : 2)].map(
          (_, i) => (
            <motion.div
              key={i}
              className="absolute w-48 h-48 bg-green-400/5 rounded-full blur-3xl"
              style={{
                left: `${20 + i * 25}%`,
                top: `${30 + i * 15}%`,
              }}
              animate={{
                scale: [1, 1.2, 1],
                opacity: [0.1, 0.2, 0.1],
              }}
              transition={{
                duration: 12 + Math.random() * 6,
                repeat: Infinity,
                delay: Math.random() * 6,
              }}
            />
          )
        )}
      </div>

      {/* Content */}
      <div className="relative z-10">{children}</div>
    </div>
  );
};

export const AnimatedGradient = ({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) => {
  return (
    <div className={cn("relative", className)}>
      <div className="absolute inset-0 bg-gradient-to-r from-green-500/20 via-blue-500/20 to-purple-500/20 animate-pulse" />
      <div className="relative z-10">{children}</div>
    </div>
  );
};

export const FloatingElements = ({
  children,
  className,
  count = 5,
}: {
  children: React.ReactNode;
  className?: string;
  count?: number;
}) => {
  return (
    <div className={cn("relative", className)}>
      {/* Floating elements */}
      <div className="absolute inset-0 pointer-events-none">
        {[...Array(count)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-2 h-2 bg-green-400/30 rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
            animate={{
              y: [0, -30, 0],
              opacity: [0.3, 0.8, 0.3],
              scale: [1, 1.2, 1],
            }}
            transition={{
              duration: 4 + Math.random() * 2,
              repeat: Infinity,
              delay: Math.random() * 2,
            }}
          />
        ))}
      </div>

      <div className="relative z-10">{children}</div>
    </div>
  );
};
