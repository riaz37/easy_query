"use client";

import React from "react";
import { Button } from "@/components/ui/button";

interface FileQueryFormProps {
  query: string;
  setQuery: (query: string) => void;
  isExecuting: boolean;
  onUploadClick: () => void;
  onClearClick: () => void;
  onExecuteClick: () => void;
  onSaveClick: () => void;
  className?: string;
}

export function FileQueryForm({
  query,
  setQuery,
  isExecuting,
  onUploadClick,
  onClearClick,
  onExecuteClick,
  onSaveClick,
  className = "",
}: FileQueryFormProps) {
  return (
    <div className={`relative -mt-16 px-4 z-10 ${className}`}>
      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask a question about your uploaded files..."
        className="w-full h-48 p-4 pr-32 bg-slate-800/50 text-white placeholder-slate-400 focus:outline-none resize-none border-0"
        style={{
          background:
            "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
          borderRadius: "16px",
          outline: "none",
          border: "none",
        }}
      />

      <div className="absolute bottom-3 left-7 right-7 flex justify-between">
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={onUploadClick}
            className="text-xs"
            style={{
              background:
                "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
              border:
                "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
              color: "white",
              borderRadius: "99px",
              height: "48px",
              minWidth: "64px",
            }}
          >
            Upload
          </Button>
          <Button
            variant="outline"
            onClick={onClearClick}
            className="text-xs"
            style={{
              background:
                "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
              border:
                "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
              color: "white",
              borderRadius: "99px",
              height: "48px",
              minWidth: "64px",
            }}
          >
            Clear
          </Button>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={onExecuteClick}
            disabled={isExecuting || !query.trim()}
            className="text-xs"
            style={{
              background:
                "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
              border:
                "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
              color: "var(--p-main, rgba(19, 245, 132, 1))",
              borderRadius: "99px",
              height: "48px",
              minWidth: "64px",
            }}
          >
            {isExecuting ? "Executing..." : "Execute"}
          </Button>
          <Button
            variant="outline"
            onClick={onSaveClick}
            disabled={!query.trim()}
            className="text-xs"
            style={{
              background:
                "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
              border:
                "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
              color: "white",
              borderRadius: "99px",
              height: "48px",
              minWidth: "64px",
            }}
          >
            Save
          </Button>
        </div>
      </div>
    </div>
  );
}
