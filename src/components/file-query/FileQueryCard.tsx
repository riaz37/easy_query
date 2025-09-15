"use client";

import React from "react";
import { FileQueryHeader } from "./FileQueryHeader";
import { FileQueryForm } from "./FileQueryForm";

interface FileQueryCardProps {
  query: string;
  setQuery: (query: string) => void;
  isExecuting: boolean;
  onUploadClick: () => void;
  onClearClick: () => void;
  onExecuteClick: () => void;
  onSaveClick: () => void;
  className?: string;
}

export function FileQueryCard({
  query,
  setQuery,
  isExecuting,
  onUploadClick,
  onClearClick,
  onExecuteClick,
  onSaveClick,
  className = "",
}: FileQueryCardProps) {
  return (
    <div
      className={`p-6 ${className}`}
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
      <FileQueryForm
        query={query}
        setQuery={setQuery}
        isExecuting={isExecuting}
        onUploadClick={onUploadClick}
        onClearClick={onClearClick}
        onExecuteClick={onExecuteClick}
        onSaveClick={onSaveClick}
      />
    </div>
  );
}
