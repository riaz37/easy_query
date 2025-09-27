"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/loading/Spinner";

interface FileQueryFormProps {
  query: string;
  setQuery: (query: string) => void;
  isExecuting: boolean;
  onUploadClick: () => void;
  onExecuteClick: () => void;
  className?: string;
}

export function FileQueryForm({
  query,
  setQuery,
  isExecuting,
  onUploadClick,
  onExecuteClick,
  className = "",
}: FileQueryFormProps) {
  return (
    <div className={`relative -mt-16 px-0.5 z-10 ${className}`}>
      <div className="relative">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about your uploaded files..."
          className="w-full h-16 px-4 pr-40 bg-slate-800/50 text-white placeholder-slate-400 focus:outline-none border-0"
          style={{
            background:
              "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
            borderRadius: "16px",
            outline: "none",
            border: "none",
          }}
        />
        
        <div className="absolute right-2 top-1/2 transform -translate-y-1/2 flex gap-2">
          <Button
            variant="outline"
            onClick={() => setQuery("")}
            className="text-xs cursor-pointer"
            style={{
              background:
                "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
              border:
                "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
              color: "white",
              borderRadius: "99px",
              height: "40px",
              minWidth: "60px",
            }}
          >
            Clear
          </Button>
          <Button
            onClick={onExecuteClick}
            disabled={isExecuting || !query.trim()}
            className="text-xs cursor-pointer"
            style={{
              background:
                "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
              border:
                "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
              color: "var(--p-main, rgba(19, 245, 132, 1))",
              borderRadius: "99px",
              height: "40px",
              minWidth: "60px",
            }}
          >
            {isExecuting ? (
              <Spinner size="sm" variant="accent-green" />
            ) : (
              "Ask"
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}
