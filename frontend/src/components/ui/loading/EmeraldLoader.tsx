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

export function EmeraldLoader({ 
  size = "md", 
  className 
}: Omit<LoadingProps, 'variant'>) {
  return (
    <div className={cn("relative", sizeClasses[size], className)}>
      {/* Main spinning ring */}
      <div
        className="absolute inset-0 rounded-full border-4 border-[#10b981]/20 border-t-[#10b981] animate-spin"
        style={{ animationDuration: "1.5s" }}
      />
      
      {/* Inner pulsing ring */}
      <div
        className="absolute inset-2 rounded-full border-2 border-[#34d399]/40 animate-pulse"
        style={{ animationDuration: "2s" }}
      />
      
      {/* Center dot with glow */}
      <div
        className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-2 h-2 bg-[#10b981] rounded-full animate-pulse"
        style={{ 
          animationDuration: "1s",
          boxShadow: "0 0 8px rgba(16, 185, 129, 0.6)"
        }}
      />
      
      {/* Subtle background glow */}
      <div
        className="absolute inset-0 rounded-full bg-gradient-to-r from-[#10b981]/5 to-[#34d399]/5 animate-pulse blur-sm"
        style={{ animationDuration: "3s" }}
      />
    </div>
  );
}
