"use client";

import React from "react";
import Image from "next/image";

interface ModeSelectorProps {
  onModeSelect: (mode: "voice" | "text") => void;
}

export function ModeSelector({ onModeSelect }: ModeSelectorProps) {
  return (
    <div className="absolute bottom-20 right-0 w-16 animate-in slide-in-from-right-2 duration-300">
      {/* Voice Assistant Options */}
      <div 
        className="rounded-[30px] p-1 space-y-1"
        style={{
          background: "rgba(255, 255, 255, 0.08)"
        }}
      >
        {/* Talk Option */}
        <button
          onClick={() => onModeSelect("voice")}
          className="w-full flex flex-col items-center justify-center p-1 rounded-lg cursor-pointer"
        >
          <div 
            className="w-10 h-10 mb-1 flex items-center justify-center rounded-full transition-all duration-300 hover:scale-110 hover:bg-emerald-500/20 hover:shadow-lg hover:shadow-emerald-500/30"
            style={{
              background: "rgba(255, 255, 255, 0.08)"
            }}
          >
            <Image
              src="/ai-assistant/voice.svg"
              alt="Voice Assistant"
              width={24}
              height={24}
              className="w-6 h-6 transition-all duration-300 hover:brightness-110"
            />
          </div>
          <span className="text-gray-300 font-medium text-xs">Talk</span>
        </button>

        {/* Chat Option */}
        <button
          onClick={() => onModeSelect("text")}
          className="w-full flex flex-col items-center justify-center p-1 rounded-lg cursor-pointer"
        >
          <div 
            className="w-10 h-10 mb-1 flex items-center justify-center rounded-full transition-all duration-300 hover:scale-110 hover:bg-emerald-500/20 hover:shadow-lg hover:shadow-emerald-500/30"
            style={{
              background: "rgba(255, 255, 255, 0.08)"
            }}
          >
            <Image
              src="/ai-assistant/chat.svg"
              alt="Chat Assistant"
              width={24}
              height={24}
              className="w-6 h-6 transition-all duration-300 hover:brightness-110"
            />
          </div>
          <span className="text-gray-300 font-medium text-xs">Chat</span>
        </button>
      </div>
    </div>
  );
}
