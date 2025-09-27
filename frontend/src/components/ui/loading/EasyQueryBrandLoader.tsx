"use client";

import { cn } from "@/lib/utils";
import { LoadingProps } from "./types";
import Image from "next/image";

const sizeClasses = {
  xs: "w-16 h-16",
  sm: "w-20 h-20",
  md: "w-24 h-24",
  lg: "w-32 h-32",
  xl: "w-40 h-40"
};

const logoSizeClasses = {
  xs: "w-8 h-8",
  sm: "w-10 h-10",
  md: "w-12 h-12",
  lg: "w-16 h-16",
  xl: "w-20 h-20"
};

export function EasyQueryBrandLoader({ 
  size = "md", 
  className 
}: Omit<LoadingProps, 'variant'>) {
  return (
    <div className={cn("relative flex items-center justify-center", sizeClasses[size], className)}>
      {/* Outer rotating ring */}
      <div
        className="absolute inset-0 rounded-full border-2 border-[#10b981]/30 animate-spin"
        style={{ animationDuration: "4s" }}
      />
      
      {/* Middle pulsing ring */}
      <div
        className="absolute inset-3 rounded-full border border-[#10b981]/50 animate-pulse"
        style={{ animationDuration: "2s" }}
      />
      
      {/* Inner glow ring */}
      <div
        className="absolute inset-6 rounded-full bg-gradient-to-r from-[#10b981]/20 to-[#34d399]/20 animate-pulse"
        style={{ animationDuration: "1.5s" }}
      />
      
      {/* Logo container with subtle bounce */}
      <div
        className="relative z-10 flex items-center justify-center"
        style={{
          animation: "logoBounce 2s ease-in-out infinite"
        }}
      >
        <Image
          src="/logo/logo.svg"
          alt="Easy Query Logo"
          width={80}
          height={80}
          className={cn(
            "object-contain drop-shadow-lg",
            logoSizeClasses[size]
          )}
          priority
        />
      </div>
      
      {/* Floating particles */}
      <div className="absolute inset-0">
        <div
          className="absolute w-1 h-1 bg-[#10b981] rounded-full animate-ping"
          style={{
            top: "10%",
            left: "20%",
            animationDelay: "0s",
            animationDuration: "2s"
          }}
        />
        <div
          className="absolute w-1 h-1 bg-[#34d399] rounded-full animate-ping"
          style={{
            top: "20%",
            right: "15%",
            animationDelay: "0.5s",
            animationDuration: "2s"
          }}
        />
        <div
          className="absolute w-1 h-1 bg-[#10b981] rounded-full animate-ping"
          style={{
            bottom: "15%",
            left: "25%",
            animationDelay: "1s",
            animationDuration: "2s"
          }}
        />
        <div
          className="absolute w-1 h-1 bg-[#34d399] rounded-full animate-ping"
          style={{
            bottom: "25%",
            right: "20%",
            animationDelay: "1.5s",
            animationDuration: "2s"
          }}
        />
      </div>
      
      <style jsx>{`
        @keyframes logoBounce {
          0%, 100% {
            transform: translateY(0px) scale(1);
          }
          50% {
            transform: translateY(-2px) scale(1.05);
          }
        }
      `}</style>
    </div>
  );
}
