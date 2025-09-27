"use client";

import React, { useRef, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { CustomTypewriter, TYPEWRITER_TEXTS, TypewriterTextType } from "@/components/ui/custom-typewriter";
import { Spinner } from "@/components/ui/loading/Spinner";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface QueryFormProps {
  query: string;
  setQuery: (query: string) => void;
  isExecuting: boolean;
  onExecuteClick: () => void;
  placeholder?: string;
  placeholderType?: TypewriterTextType;
  buttonText?: string;
  showClearButton?: boolean;
  showUploadButton?: boolean;
  onUploadClick?: () => void;
  className?: string;
  disabled?: boolean;
  enableTypewriter?: boolean;
  // Model selection props
  model?: string;
  onModelChange?: (model: string) => void;
  showModelSelector?: boolean;
}

export function QueryForm({
  query,
  setQuery,
  isExecuting,
  onExecuteClick,
  placeholder = "Ask a question about your uploaded files...",
  placeholderType = 'file',
  buttonText = "Ask",
  showClearButton = true,
  showUploadButton = false,
  onUploadClick,
  className = "",
  disabled = false,
  enableTypewriter = true,
  // Model selection props
  model = "gemini",
  onModelChange,
  showModelSelector = false,
}: QueryFormProps) {
  const textareaRef = useRef<HTMLDivElement>(null);
  const [isMultiLine, setIsMultiLine] = useState(false);
  const [hasUserInteracted, setHasUserInteracted] = useState(false);

  // Available AI models
  const modelOptions = [
    { value: "gemini", label: "Gemini" },
    { value: "llama-3.3-70b-versatile", label: "Llama 3.3 70B" },
    { value: "openai/gpt-oss-120b", label: "GPT OSS 120B" },
  ];

  // Auto-resize contentEditable div based on content
  useEffect(() => {
    const div = textareaRef.current;
    if (div) {
      // Only update content if it's different to avoid conflicts
      if (div.textContent !== query) {
        div.textContent = query;
      }
      
      // Reset height to auto to get the correct scrollHeight
      div.style.height = 'auto';
      
      // Calculate the new height
      const scrollHeight = div.scrollHeight;
      const minHeight = 24; // Single line height
      const maxHeight = 200; // Maximum height before scrollbar appears
      
      // Set height with constraints
      const newHeight = Math.min(Math.max(scrollHeight, minHeight), maxHeight);
      div.style.height = `${newHeight}px`;
      
      // Determine if div is multi-line (more than minimum height)
      setIsMultiLine(newHeight > minHeight);
    }
  }, [query]);

  // Set initial height on mount
  useEffect(() => {
    const div = textareaRef.current;
    if (div) {
      div.style.height = '24px'; // Set initial single line height
    }
  }, []);

  const handleClear = () => {
    setQuery("");
    setIsMultiLine(false);
    if (textareaRef.current) {
      textareaRef.current.style.height = '24px'; // Reset to single line height
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Stop typewriter effect when user starts typing
    setHasUserInteracted(true);
    
    // Allow Enter to create new lines, but prevent form submission
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onExecuteClick();
    }
  };

  const handleFocus = () => {
    // Stop typewriter effect when user focuses on the input
    setHasUserInteracted(true);
  };

  const handleInput = (e: React.FormEvent) => {
    // Stop typewriter effect when user types
    setHasUserInteracted(true);
    const text = e.currentTarget.textContent || '';
    setQuery(text);
  };

  return (
    <div 
      className={`relative -mt-16 px-0 z-10 ${className}`}
      style={{
        background: "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
        borderRadius: "25px",
        padding: "16px",
      }}
    >
      {/* ContentEditable div */}
      <div className="text-white text-sm leading-relaxed relative">
        <div
          ref={textareaRef}
          contentEditable
          suppressContentEditableWarning
          onInput={handleInput}
          onKeyDown={handleKeyDown}
          onFocus={handleFocus}
          className="w-full text-white resize-none overflow-y-auto"
          style={{
            outline: "none",
            border: "none",
            minHeight: "24px", // Single line height
            maxHeight: "200px",
            width: "100%",
            boxSizing: "border-box",
            overflow: "auto",
            background: "transparent",
            padding: "0",
            margin: "0",
            lineHeight: "24px", // Single line height
            whiteSpace: "pre-wrap", // Allow wrapping
            paddingRight: isMultiLine ? "0" : showModelSelector && onModelChange ? "180px" : "140px", // Space for buttons and model selector when single line
            direction: "ltr", // Ensure left-to-right text direction
            textAlign: "left", // Ensure left alignment
          }}
          data-placeholder={placeholder}
          suppressContentEditableWarning
        />
        
        {/* Placeholder overlay */}
        {!query && !hasUserInteracted && (
          <div 
            className="absolute top-0 left-0 pointer-events-none"
            style={{
              lineHeight: "24px",
              paddingRight: isMultiLine ? "0" : showModelSelector && onModelChange ? "180px" : "140px",
            }}
          >
            {enableTypewriter ? (
              <CustomTypewriter
                texts={TYPEWRITER_TEXTS[placeholderType]}
                className="text-sm"
                typeSpeed={8}
                deleteSpeed={5}
                deleteChunkSize={3}
                pauseTime={600}
                loop={true}
                startDelay={200}
              />
            ) : (
              <span className="text-slate-400">{placeholder}</span>
            )}
          </div>
        )}

        {/* Model Selector - inline when single line and enabled */}
        {!isMultiLine && showModelSelector && onModelChange && (
          <div className="absolute right-16 top-0 flex items-center" style={{ height: "24px" }}>
            <Select value={model} onValueChange={onModelChange}>
              <SelectTrigger 
                className="w-28 h-6 text-xs border-slate-600 bg-slate-800/50 text-white"
                style={{
                  background: "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
                  border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
                  borderRadius: "99px",
                }}
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-slate-800 border-slate-600">
                {modelOptions.map((option) => (
                  <SelectItem key={option.value} value={option.value} className="text-white hover:bg-slate-700">
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}

        {/* Buttons - inline when single line, separate section when multiline */}
        {!isMultiLine && (
          <div className={`absolute top-0 flex gap-2 items-center ${showModelSelector && onModelChange ? '-right-1' : '-right-1'}`} style={{ height: "24px" }}>
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
              {isExecuting ? (
                <Spinner size="sm" variant="accent-green" />
              ) : (
                buttonText
              )}
            </Button>
          </div>
        )}
      </div>

      {/* Separate button and model selector section when multiline */}
      {isMultiLine && (
        <div className="flex items-center justify-end gap-2 mt-2">
          {/* Model Selector for multiline */}
          {showModelSelector && onModelChange && (
            <Select value={model} onValueChange={onModelChange}>
              <SelectTrigger 
                className="w-36 h-10 text-xs border-slate-600 bg-slate-800/50 text-white"
                style={{
                  background: "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
                  border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
                  borderRadius: "99px",
                }}
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-slate-800 border-slate-600">
                {modelOptions.map((option) => (
                  <SelectItem key={option.value} value={option.value} className="text-white hover:bg-slate-700">
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
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
            {isExecuting ? (
              <Spinner size="sm" variant="accent-green" />
            ) : (
              buttonText
            )}
          </Button>
        </div>
      )}
    </div>
  );
}
