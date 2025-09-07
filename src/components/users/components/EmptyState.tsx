"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Plus } from "lucide-react";
import { EmptyStateProps } from "../types";

export function EmptyState({
  icon,
  title,
  description,
  actionLabel,
  onAction,
  isDark
}: EmptyStateProps) {
  return (
    <div className="text-center py-12">
      <div className={`${isDark ? 'text-gray-400' : 'text-gray-500'} mx-auto mb-4`}>
        {icon}
      </div>
      <h3 className={`text-lg font-medium ${isDark ? 'text-white' : 'text-gray-900'} mb-2`}>
        {title}
      </h3>
      <p className={`${isDark ? 'text-gray-300' : 'text-gray-600'} mb-4`}>
        {description}
      </p>
      <Button 
        onClick={onAction} 
        className="bg-blue-500 hover:bg-blue-600 shadow-lg hover:shadow-blue-500/25 transition-all duration-200"
      >
        <Plus className="w-4 h-4 mr-2" />
        {actionLabel}
      </Button>
    </div>
  );
}
