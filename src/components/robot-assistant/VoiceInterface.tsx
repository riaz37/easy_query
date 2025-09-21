"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { useTheme } from "@/store/theme-store";

interface VoiceMessage {
  id: string;
  content: string;
  timestamp: Date | string | number;
  type: "user" | "assistant";
}

interface VoiceInterfaceProps {
  messages: VoiceMessage[];
  isInConversation: boolean;
}

export function VoiceInterface({ messages, isInConversation }: VoiceInterfaceProps) {
  const theme = useTheme();

  const formatTime = (date: Date | string | number) => {
    try {
      const dateObj = date instanceof Date ? date : new Date(date);
      if (isNaN(dateObj.getTime())) {
        return "Invalid time";
      }
      return dateObj.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch (error) {
      console.error("Error formatting time:", error, "Input:", date);
      return "Invalid time";
    }
  };

  return (
    <div className="absolute bottom-6 right-20 animate-in slide-in-from-right-2 duration-300">
      {/* Speech bubble */}
      <div
        className={cn(
          "px-4 py-2 rounded-2xl shadow-lg backdrop-blur-xl relative w-80",
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
        
        {messages.length > 0 ? (
          <div className="flex flex-col gap-2">
            <span
              className={cn(
                "text-sm font-medium break-words",
                theme === "dark" ? "text-emerald-400" : "text-emerald-600"
              )}
            >
              {messages[messages.length - 1]?.content}
            </span>
            <span
              className={cn(
                "text-xs self-end",
                theme === "dark" ? "text-gray-400" : "text-gray-500"
              )}
            >
              {formatTime(
                messages[messages.length - 1]?.timestamp
              )}
            </span>
          </div>
        ) : (
          <span
            className={cn(
              "text-sm",
              theme === "dark" ? "text-gray-300" : "text-gray-600"
            )}
          >
            {isInConversation ? "Listening..." : "Ready to talk"}
          </span>
        )}
      </div>
    </div>
  );
}
