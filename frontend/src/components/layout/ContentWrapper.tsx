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
    sm: "px-16",
    md: "px-32", // 128px = 32 * 4
    lg: "px-48", // 192px = 48 * 4
  };

  return (
    <div className={cn(spacingClasses[spacing], className)}>
      {children}
    </div>
  );
}
