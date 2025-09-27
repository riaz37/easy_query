"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { History } from "lucide-react";

interface FileQueryPageHeaderProps {
  onHistoryClick: () => void;
  username?: string;
  className?: string;
}

export function FileQueryPageHeader({ 
  onHistoryClick, 
  username = "",
  className = "" 
}: FileQueryPageHeaderProps) {
  return (
    <div className={`flex items-center justify-between mb-8 ${className}`}>
      <div>
        <h1 
          className="text-4xl font-bold mb-2 block"
          style={{
            background:
              "radial-gradient(70.83% 118.23% at 55.46% 50%, #0DAC5C 0%, #FFFFFF 84.18%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            backgroundClip: "text",
            color: "transparent",
            display: "block",
            backgroundSize: "100% 100%",
            backgroundRepeat: "no-repeat",
          }}
        >
          Hi there, {username}
        </h1>
        <p 
          className="text-xl block"
          style={{
            background:
              "radial-gradient(70.83% 118.23% at 55.46% 50%, #0DAC5C 0%, #FFFFFF 84.18%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            backgroundClip: "text",
            color: "transparent",
            display: "block",
            backgroundSize: "100% 100%",
            backgroundRepeat: "no-repeat",
          }}
        >
          What would you like to know?
        </p>
      </div>
      <Button
        onClick={onHistoryClick}
        className="px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg flex items-center gap-2"
      >
        <History className="h-4 w-4" />
        History
      </Button>
    </div>
  );
}
