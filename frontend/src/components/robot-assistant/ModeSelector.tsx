"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { useTheme } from "@/store/theme-store";

interface ModeSelectorProps {
  onModeSelect: (mode: "voice" | "text") => void;
}

export function ModeSelector({ onModeSelect }: ModeSelectorProps) {
  const theme = useTheme();

  return (
    <div className="absolute bottom-6 right-20 flex items-center animate-in slide-in-from-right-2 duration-300">
      {/* Speech bubble */}
      <div
        className={cn(
          "px-4 py-2 rounded-2xl shadow-lg backdrop-blur-xl relative whitespace-nowrap",
          theme === "dark" 
            ? "bg-white/10 border border-emerald-500/20" 
            : "bg-white/95 border border-emerald-500/30 shadow-emerald-500/20"
        )}
        style={{
          background:
            theme === "dark"
              ? "rgba(255, 255, 255, 0.1)"
              : "linear-gradient(158.39deg, rgba(255, 255, 255, 0.98) 14.19%, rgba(240, 249, 245, 0.95) 50.59%, rgba(255, 255, 255, 0.98) 68.79%, rgba(240, 249, 245, 0.95) 105.18%)",
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
          boxShadow:
            theme === "dark"
              ? "0 8px 32px rgba(0, 0, 0, 0.3)"
              : "0 8px 32px rgba(16, 185, 129, 0.12), 0 1px 0 rgba(255, 255, 255, 0.8) inset",
        }}
      >
        {/* Speech bubble tail pointing to robot */}
        <div
          className={cn(
            "absolute -right-2 top-1/2 transform -translate-y-1/2 w-0 h-0",
            "border-l-[12px] border-l-emerald-500/30 border-t-[8px] border-t-transparent border-b-[8px] border-b-transparent"
          )}
        ></div>
        
        <div className="flex flex-col gap-2">
          <span
            className={cn(
              "text-sm font-medium",
              theme === "dark" ? "text-emerald-400" : "text-emerald-600"
            )}
          >
            What would you like to do?
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => onModeSelect("voice")}
              className={cn(
                "px-3 py-1 text-xs rounded-md transition-all duration-200",
                theme === "dark"
                  ? "bg-emerald-500 hover:bg-emerald-600 text-white"
                  : "bg-emerald-500 hover:bg-emerald-600 text-white shadow-sm hover:shadow-md"
              )}
            >
              Talk
            </button>
            <button
              onClick={() => onModeSelect("text")}
              className={cn(
                "px-3 py-1 text-xs rounded-md transition-all duration-200",
                theme === "dark"
                  ? "bg-emerald-500 hover:bg-emerald-600 text-white"
                  : "bg-emerald-500 hover:bg-emerald-600 text-white shadow-sm hover:shadow-md"
              )}
            >
              Chat
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
