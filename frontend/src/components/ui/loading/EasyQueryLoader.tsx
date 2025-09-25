"use client";

import { cn } from "@/lib/utils";
import { LoadingProps } from "./types";
import Image from "next/image";

const sizeClasses = {
  xs: "w-12 h-12",
  sm: "w-16 h-16",
  md: "w-20 h-20",
  lg: "w-24 h-24",
  xl: "w-32 h-32"
};

const logoSizeClasses = {
  xs: "w-6 h-6",
  sm: "w-8 h-8",
  md: "w-10 h-10",
  lg: "w-12 h-12",
  xl: "w-16 h-16"
};

export function EasyQueryLoader({ 
  size = "md", 
  className 
}: Omit<LoadingProps, 'variant'>) {
  return (
    <div className={cn("relative flex items-center justify-center", sizeClasses[size], className)}>
      {/* Animated background ring */}
      <div
        className="absolute inset-0 rounded-full border-4 border-[#10b981]/20 animate-spin"
        style={{ animationDuration: "3s" }}
      />
      
      {/* Pulsing ring */}
      <div
        className="absolute inset-2 rounded-full border-2 border-[#10b981] animate-pulse"
        style={{ animationDuration: "2s" }}
      />
      
      {/* Logo with subtle rotation */}
      <div
        className="relative z-10 animate-pulse"
        style={{ animationDuration: "1.5s" }}
      >
        <Image
          src="/logo/logo.svg"
          alt="Easy Query Logo"
          width={64}
          height={64}
          className={cn(
            "object-contain",
            logoSizeClasses[size]
          )}
          priority
        />
      </div>
      
      {/* Glow effect */}
      <div
        className="absolute inset-0 rounded-full bg-gradient-to-r from-[#10b981]/10 to-[#34d399]/10 animate-pulse blur-sm"
        style={{ animationDuration: "2.5s" }}
      />
    </div>
  );
}
