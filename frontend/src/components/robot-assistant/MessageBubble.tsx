"use client";

import React from "react";
import Image from "next/image";
import { User } from "lucide-react";
import { cn } from "@/lib/utils";

interface Message {
  id: string;
  content: string;
  timestamp: Date | string | number;
  type: "user" | "assistant";
}

interface MessageBubbleProps {
  message: Message;
  formatTime: (date: Date | string | number) => string;
}

export function MessageBubble({ message, formatTime }: MessageBubbleProps) {
  return (
    <div
      className={cn(
        "flex gap-2 mb-4",
        message.type === "user" ? "justify-end" : "justify-start"
      )}
    >
      {/* AI Avatar */}
      {message.type !== "user" && (
        <div className="flex-shrink-0 -mt-6">
          <Image
            src="/autopilot.svg"
            alt="AI Assistant"
            width={30}
            height={30}
          />
        </div>
      )}

      <div className="flex flex-col max-w-[80%]">
        <div className="relative">
          <div
            className={cn(
              "px-4 py-3 text-sm",
              message.type === "user"
                ? "text-white rounded-2xl rounded-tr-sm"
                : "text-white/72 rounded-2xl rounded-tl-sm"
            )}
            style={message.type === "user" ? {
              background: "var(--primary-12, rgba(19, 245, 132, 0.12))",
              boxShadow: "0px 1px 1px -0.5px elevation-shadow"
            } : {
              background: "linear-gradient(121.65deg, rgba(63, 67, 70, 0.3) 55.58%, rgba(76, 81, 85, 0.3) 97.52%)",
              border: "1px solid",
              borderImageSource: "linear-gradient(90deg, rgba(65, 69, 72, 0.5) 12.35%, rgba(109, 115, 120, 0.5) 42.59%, rgba(65, 69, 72, 0.5) 72.84%)",
              color: "rgba(255, 255, 255, 0.72)"
            }}
          >
            <div className="whitespace-pre-wrap break-words">
              {message.content}
            </div>
          </div>
          
          {/* Speech bubble tail - pointing upward */}
          <div
            className={cn(
              "absolute w-0 h-0",
              message.type === "user"
                ? "right-0 -top-3 border-l-[12px] border-l-transparent border-b-[12px]"
                : "left-0 -top-3 border-r-[12px] border-r-transparent border-b-[12px]"
            )}
            style={message.type === "user" ? {
              borderBottomColor: "var(--primary-12, rgba(19, 245, 132, 0.12))"
            } : {
              borderBottomColor: "rgba(63, 67, 70, 0.3)"
            }}
          />
        </div>
        <div
          className={cn(
            "flex items-center gap-1 mt-1 text-xs text-slate-400",
            message.type === "user" ? "justify-end" : "justify-start"
          )}
        >
          {message.type === "user" && (
            <>
              <div className="w-3 h-3 text-slate-400">
                ✓
              </div>
              <span>You</span>
            </>
          )}
          <span>{formatTime(message.timestamp)}</span>
        </div>
      </div>

      {/* User Avatar */}
      {message.type === "user" && (
        <div className="flex-shrink-0 -mt-6">
          <div 
            className="w-8 h-8 rounded-full flex items-center justify-center"
            style={{
              background: "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
              border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))"
            }}
          >
            <User className="w-4 h-4 text-emerald-400" />
          </div>
        </div>
      )}
    </div>
  );
}
