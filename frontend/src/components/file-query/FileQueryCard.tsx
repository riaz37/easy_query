"use client";

import React from "react";
import { FileQueryHeader } from "./FileQueryHeader";
import { QueryForm } from "@/components/shared/QueryForm";

interface FileQueryCardProps {
  query: string;
  setQuery: (query: string) => void;
  isExecuting: boolean;
  onUploadClick: () => void;
  onExecuteClick: () => void;
  className?: string;
}

export function FileQueryCard({
  query,
  setQuery,
  isExecuting,
  onUploadClick,
  onExecuteClick,
  className = "",
}: FileQueryCardProps) {
  return (
    <div
      className={`px-2 py-2 flex flex-col h-full ${className}`}
      style={{
        background:
          "linear-gradient(0deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.03)), linear-gradient(246.02deg, rgba(19, 245, 132, 0) 91.9%, rgba(19, 245, 132, 0.2) 114.38%), linear-gradient(59.16deg, rgba(19, 245, 132, 0) 71.78%, rgba(19, 245, 132, 0.2) 124.92%)",
        border: "1.5px solid",
        borderImageSource:
          "linear-gradient(158.39deg, rgba(255, 255, 255, 0.06) 14.19%, rgba(255, 255, 255, 1.5e-05) 50.59%, rgba(255, 255, 255, 1.5e-05) 68.79%, rgba(255, 255, 255, 0.015) 105.18%)",
        borderRadius: "30px",
        backdropFilter: "blur(30px)",
      }}
    >
      <FileQueryHeader />
      <QueryForm
        query={query}
        setQuery={setQuery}
        isExecuting={isExecuting}
        onExecuteClick={onExecuteClick}
        onUploadClick={onUploadClick}
        placeholder="Ask a question about your uploaded files..."
        placeholderType="file"
        buttonText="Ask"
        showUploadButton={true}
        showClearButton={false}
        enableTypewriter={true}
      />
    </div>
  );
}
