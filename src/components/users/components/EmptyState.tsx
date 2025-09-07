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
      <div className="text-emerald-400 mx-auto mb-4">
        {icon}
      </div>
      <h3 className="text-lg font-medium text-white mb-2">
        {title}
      </h3>
      <p className="text-gray-300 mb-4">
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
