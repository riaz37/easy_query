"use client";

import React, { useRef, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";

interface QueryFormProps {
  query: string;
  setQuery: (query: string) => void;
  isExecuting: boolean;
  onExecuteClick: () => void;
  placeholder?: string;
  buttonText?: string;
  showClearButton?: boolean;
  showUploadButton?: boolean;
  onUploadClick?: () => void;
  className?: string;
  disabled?: boolean;
}

export function QueryForm({
  query,
  setQuery,
  isExecuting,
  onExecuteClick,
  placeholder = "Ask a question about your uploaded files...",
  buttonText = "Ask",
  showClearButton = true,
  showUploadButton = false,
  onUploadClick,
  className = "",
  disabled = false,
}: QueryFormProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [isMultiLine, setIsMultiLine] = useState(false);

  // Auto-resize textarea based on content
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      // Reset height to auto to get the correct scrollHeight
      textarea.style.height = 'auto';
      
      // Calculate the new height
      const scrollHeight = textarea.scrollHeight;
      const minHeight = 64; // 16 * 4 = 64px (h-16)
      const maxHeight = 200; // Maximum height before scrollbar appears
      
      // Set height with constraints
      const newHeight = Math.min(Math.max(scrollHeight, minHeight), maxHeight);
      textarea.style.height = `${newHeight}px`;
      
      // Determine if textarea is multi-line (more than minimum height)
      setIsMultiLine(newHeight > minHeight);
    }
  }, [query]);

  const handleClear = () => {
    setQuery("");
    setIsMultiLine(false);
    if (textareaRef.current) {
      textareaRef.current.style.height = '64px'; // Reset to minimum height
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Allow Enter to create new lines, but prevent form submission
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onExecuteClick();
    }
  };

  return (
    <div className={`relative -mt-16 px-0.5 z-10 ${className}`}>
      <div className="relative">
        <textarea
          ref={textareaRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="w-full min-h-16 px-4 pr-40 py-4 bg-slate-800/50 text-white placeholder-slate-400 focus:outline-none border-0 resize-none overflow-y-auto"
          style={{
            background:
              "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
            borderRadius: "16px",
            outline: "none",
            border: "none",
            minHeight: "64px",
            maxHeight: "200px",
          }}
          disabled={disabled || isExecuting}
          rows={1}
        />
        
        <div className={`absolute right-2 flex gap-2 ${isMultiLine ? 'bottom-2' : 'top-1/2 transform -translate-y-1/2'}`}>
          {showUploadButton && onUploadClick && (
            <Button
              variant="outline"
              onClick={onUploadClick}
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
              Upload
            </Button>
          )}
          <Button
            onClick={onExecuteClick}
            disabled={isExecuting || !query.trim() || disabled}
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
            {isExecuting ? "Executing..." : buttonText}
          </Button>
        </div>
      </div>
    </div>
  );
}
