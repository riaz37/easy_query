"use client";

import React from "react";
import { cn } from "@/lib/utils";

interface ContentWrapperProps {
  children: React.ReactNode;
  className?: string;
  spacing?: "sm" | "md" | "lg";
}

/**
 * Content wrapper that provides consistent horizontal spacing
 * for all content sections in the file query page.
 */
export function ContentWrapper({ 
  children, 
  className,
  spacing = "md" 
}: ContentWrapperProps) {
  const spacingClasses = {
    sm: "px-4",
    md: "px-16", // 64px = 16 * 4
    lg: "px-24", // 96px = 24 * 4
  };

  return (
    <div className={cn(spacingClasses[spacing], className)}>
      {children}
    </div>
  );
}
