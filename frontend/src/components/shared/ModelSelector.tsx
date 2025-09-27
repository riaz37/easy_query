"use client";

import React, { useState, useEffect, useRef } from "react";
import { ChevronDown } from "lucide-react";

interface ModelSelectorProps {
  value: string;
  onValueChange: (value: string) => void;
  className?: string;
  size?: "sm" | "md" | "lg";
  disabled?: boolean;
}

const modelOptions = [
  { value: "gemini", label: "Gemini" },
  { value: "llama-3.3-70b-versatile", label: "Llama" },
  { value: "openai/gpt-oss-120b", label: "GPT" },
];

export function ModelSelector({
  value,
  onValueChange,
  className = "",
  size = "md",
  disabled = false,
}: ModelSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  
  const sizeClasses = {
    sm: "w-32 h-10 text-xs",
    md: "w-40 h-10 text-sm",
    lg: "w-44 h-12 text-base",
  };

  const selectedOption = modelOptions.find(option => option.value === value);

  const handleOptionClick = (optionValue: string) => {
    onValueChange(optionValue);
    setIsOpen(false);
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  return (
    <div ref={dropdownRef} className={`relative ${className}`}>
      <div
        className={`${sizeClasses[size]} flex items-center justify-between px-3 py-2 cursor-pointer transition-colors ${
          disabled ? 'opacity-50 cursor-not-allowed' : ''
        }`}
        style={{
          background: "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
          border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
          borderRadius: "99px",
          color: "white",
        }}
        onClick={() => !disabled && setIsOpen(!isOpen)}
      >
        <span className="text-white text-sm">
          {selectedOption?.label || "Select Model"}
        </span>
        <ChevronDown 
          className={`w-4 h-4 text-slate-400 transition-transform ${
            isOpen ? 'rotate-180' : ''
          }`} 
        />
      </div>
      
      {isOpen && !disabled && (
        <div 
          className="absolute top-full left-0 right-0 mt-1 rounded-lg shadow-lg"
          style={{
            background: "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
            border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
            backdropFilter: "blur(10px)",
            zIndex: 9999,
            minWidth: "max-content",
          }}
        >
          {modelOptions.map((option) => (
            <div
              key={option.value}
              className={`px-3 py-2 text-sm cursor-pointer transition-colors whitespace-nowrap ${
                value === option.value ? 'text-green-400' : 'text-white hover:bg-white/10'
              }`}
              onClick={() => handleOptionClick(option.value)}
            >
              {option.label}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}