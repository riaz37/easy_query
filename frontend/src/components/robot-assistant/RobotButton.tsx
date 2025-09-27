"use client";

import React from "react";
import Image from "next/image";
import { cn } from "@/lib/utils";

interface RobotButtonProps {
  robotState: "idle" | "selecting" | "chatting" | "talking";
  isInConversation: boolean;
  onClick: () => void;
}

export function RobotButton({ robotState, isInConversation, onClick }: RobotButtonProps) {
  return (
    <div className="relative">
      <button
        onClick={onClick}
        className={cn(
          "transition-all duration-300 hover:scale-110 focus:outline-none active:outline-none",
          "p-2 rounded-full relative cursor-pointer",
          robotState === "talking" && "animate-pulse"
        )}
        title={
          robotState === "talking"
            ? "Click to disconnect voice"
            : "Click to interact"
        }
      >
        {/* Pulsing Glow Ring */}
        <div
          className={cn(
            "absolute inset-0 rounded-full transition-all duration-1000",
            "bg-gradient-to-r from-emerald-400/20 to-emerald-600/20",
            "animate-pulse",
            robotState === "talking" &&
              "from-emerald-400/40 to-emerald-600/40",
            "hover:from-emerald-400/30 hover:to-emerald-600/30"
          )}
          style={{
            animationDuration: "2s",
            animationTimingFunction: "ease-in-out",
          }}
        />
        
        <Image
          src="/autopilot.svg"
          alt="AI Assistant"
          width={48}
          height={48}
          className="w-12 h-12 relative z-10"
        />
      </button>

      {/* Talking indicator dots */}
      {robotState === "talking" && isInConversation && (
        <div className="absolute -top-2 -right-2 flex gap-1">
          <div
            className="w-1 h-1 bg-emerald-400 rounded-full animate-bounce"
            style={{ animationDelay: "0ms" }}
          ></div>
          <div
            className="w-1 h-1 bg-emerald-400 rounded-full animate-bounce"
            style={{ animationDelay: "150ms" }}
          ></div>
          <div
            className="w-1 h-1 bg-emerald-400 rounded-full animate-bounce"
            style={{ animationDelay: "300ms" }}
          ></div>
        </div>
      )}

      {/* Expanding rings for active states */}
      {robotState === "talking" && isInConversation && (
        <div className="absolute inset-0 pointer-events-none">
          <div
            className="absolute inset-0 border-2 border-emerald-400/30 rounded-full animate-ping"
            style={{ animationDuration: "1.5s" }}
          ></div>
          <div
            className="absolute inset-0 border-2 border-emerald-400/20 rounded-full animate-ping"
            style={{ animationDuration: "1.5s", animationDelay: "0.5s" }}
          ></div>
        </div>
      )}

      {/* Attention-seeking pulse for idle state */}
      {robotState === "idle" && (
        <div className="absolute inset-0 pointer-events-none">
          <div
            className="absolute inset-0 border-2 border-emerald-400/20 rounded-full animate-pulse"
            style={{ animationDuration: "3s" }}
          ></div>
        </div>
      )}
    </div>
  );
}
