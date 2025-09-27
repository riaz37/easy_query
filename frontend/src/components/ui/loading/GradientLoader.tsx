"use client";

import { cn } from "@/lib/utils";
import { LoadingProps } from "./types";

const sizeClasses = {
  xs: "w-8 h-8",
  sm: "w-12 h-12",
  md: "w-16 h-16",
  lg: "w-20 h-20",
  xl: "w-24 h-24"
};

export function GradientLoader({ 
  size = "md", 
  className 
}: Omit<LoadingProps, 'variant'>) {
  return (
    <div className={cn("relative", sizeClasses[size], className)}>
      {/* Outer ring with emerald gradient */}
      <div
        className="absolute inset-0 rounded-full border-4 border-transparent animate-spin"
        style={{ 
          background: 'conic-gradient(from 0deg, transparent, #10b981, #34d399, #10b981, transparent)',
          animationDuration: "2s"
        }}
      />
      
      {/* Inner ring with subtle gradient */}
      <div
        className="absolute inset-2 rounded-full border-2 border-transparent animate-spin"
        style={{ 
          background: 'conic-gradient(from 180deg, transparent, #10b981/30, #34d399/30, transparent)',
          animationDuration: "1.5s",
          animationDirection: "reverse"
        }}
      />
      
      {/* Center dot with emerald gradient */}
      <div
        className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-gradient-to-r from-[#10b981] to-[#34d399] animate-pulse shadow-lg"
        style={{ 
          animationDuration: "1.2s",
          boxShadow: "0 0 10px rgba(16, 185, 129, 0.5)"
        }}
      />
    </div>
  );
}
