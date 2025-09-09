"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Plus } from "lucide-react";
import { useTheme } from "@/store/theme-store";
import { cn } from "@/lib/utils";
import { EmptyStateProps } from "../types";

export function EmptyState({
  icon,
  title,
  description,
  actionLabel,
  onAction,
  isDark
}: EmptyStateProps) {
  const theme = useTheme();
  
  return (
    <div className="text-center py-12">
      <div className={cn(
        "mx-auto mb-4 flex justify-center items-center",
        theme === "dark" ? "text-emerald-400" : "text-emerald-500"
      )}>
        {icon}
      </div>
      <h3 className={cn(
        "text-lg font-semibold mb-2",
        theme === "dark" ? "text-white" : "text-gray-800"
      )}>
        {title}
      </h3>
      <p className={cn(
        "mb-4 font-medium",
        theme === "dark" ? "text-gray-300" : "text-gray-700"
      )}>
        {description}
      </p>
      <Button 
        onClick={onAction} 
        className="card-button-enhanced"
      >
        <Plus className="w-4 h-4 mr-2" />
        {actionLabel}
      </Button>
    </div>
  );
}
