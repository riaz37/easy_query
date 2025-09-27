"use client";

import React from "react";
import { Plus, Building2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { EmptyStateProps } from "../types";

export function EmptyState({ onAddParentCompany }: EmptyStateProps) {
  return (
    <div className="flex items-center justify-center min-h-[400px]">
      <Card className="w-96 border-dashed border-2 border-emerald-400/30 bg-transparent hover:border-emerald-400/50 transition-all duration-300">
        <CardContent className="flex flex-col items-center justify-center p-12 text-center">
          {/* Icon */}
          <div className="relative mb-6">
            <div className="w-20 h-20 rounded-full bg-gradient-to-br from-emerald-500/20 to-emerald-600/20 border-2 border-emerald-400/30 flex items-center justify-center">
              <Building2 className="w-10 h-10 text-emerald-400" />
            </div>
            
            {/* Plus indicator */}
            <div className="absolute -bottom-2 -right-2 w-8 h-8 bg-emerald-500 rounded-full flex items-center justify-center shadow-lg">
              <Plus className="w-4 h-4 text-white" />
            </div>
          </div>

          {/* Text */}
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            No Companies Yet
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-6 max-w-sm">
            Get started by creating your first parent company. You can then add sub-companies and build your organizational structure.
          </p>

          {/* Action Button */}
          <Button
            onClick={onAddParentCompany}
            className="bg-emerald-600 hover:bg-emerald-700 text-white shadow-lg shadow-emerald-500/20 hover:shadow-emerald-500/30 transition-all duration-200"
          >
            <Plus className="w-4 h-4 mr-2" />
            Create Parent Company
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}