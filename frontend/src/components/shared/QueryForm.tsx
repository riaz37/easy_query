"use client";

import React, { useRef, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useAIModels } from "@/lib/hooks/use-ai-models";
import { Brain } from "lucide-react";

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
  // Model selection props
  showModelSelector?: boolean;
  selectedModel?: string;
  onModelChange?: (model: string) => void;
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
  // Model selection props
  showModelSelector = false,
  selectedModel,
  onModelChange,
}: QueryFormProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [isMultiLine, setIsMultiLine] = useState(false);
  
  // Use AI models hook
  const { getModelDisplayInfo, getModelsWithDisplayInfo } = useAIModels();

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

  // Set initial height on mount
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = '64px'; // Set initial height
    }
  }, []);

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
    <div className={`relative -mt-16 px-0 z-10 ${className}`}>
      <div className="relative">
        <textarea
          ref={textareaRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className={`w-full min-h-16 px-4 py-4 bg-slate-800/50 text-white placeholder-slate-400 focus:outline-none border-0 resize-none overflow-y-auto ${showModelSelector ? 'pr-80' : 'pr-40'}`}
          style={{
            background:
              "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
            borderRadius: "30px",
            outline: "none",
            border: "none",
            minHeight: "64px",
            maxHeight: "200px",
          }}
          disabled={disabled || isExecuting}
          rows={1}
        />
        
        <div className={`absolute right-2 flex gap-2 items-center ${isMultiLine ? 'bottom-3' : 'top-3'}`}>
          {/* Model Selector */}
          {showModelSelector && (
            <Select
              value={selectedModel}
              onValueChange={(value) => onModelChange?.(value)}
              disabled={disabled || isExecuting}
            >
              <SelectTrigger className="w-32 h-10 bg-slate-800/50 border-slate-700 text-white text-xs">
                <SelectValue>
                  {selectedModel && (() => {
                    const { name, icon: Icon, color } = getModelDisplayInfo(selectedModel);
                    return (
                      <div className="flex items-center gap-1">
                        <Icon className={`w-3 h-3 ${color}`} />
                        <span className="truncate">{name.split(' ')[0]}</span>
                      </div>
                    );
                  })()}
                </SelectValue>
              </SelectTrigger>
              <SelectContent className="bg-slate-800 border-slate-700 w-48">
                {getModelsWithDisplayInfo().map(({ value, name, icon: Icon, description, color }) => (
                  <SelectItem 
                    key={value} 
                    value={value} 
                    className="text-white hover:bg-slate-700 focus:bg-slate-700 cursor-pointer"
                  >
                    <div className="flex items-center gap-2">
                      <Icon className={`w-4 h-4 ${color}`} />
                      <div className="flex flex-col items-start">
                        <span className="text-sm font-medium">{name}</span>
                        {description && (
                          <span className="text-xs text-slate-400">{description}</span>
                        )}
                      </div>
                    </div>
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
            {isExecuting ? "Executing..." : buttonText}
          </Button>
        </div>
      </div>
    </div>
  );
}
