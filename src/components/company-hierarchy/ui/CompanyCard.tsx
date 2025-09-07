"use client";

import React from "react";
import { Building2, Plus, Upload } from "lucide-react";
import { Handle, Position } from "reactflow";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { CompanyCardProps } from "../types";

export function CompanyCard({
  company,
  onAddSubCompany,
  onUpload,
  isSelected = false,
  onSelect,
  level = 0,
}: CompanyCardProps) {
  const isMainCompany = level === 0;
  const companyType = company.id.startsWith("parent-") ? "parent" : "sub";

  return (
    <div className="relative group">
      {/* ReactFlow Handles */}
      {!isMainCompany && (
        <Handle
          type="target"
          position={Position.Top}
          id="top"
          className="!w-3 !h-3 !bg-emerald-500 !border-2 !border-emerald-500 !-top-1.5"
        />
      )}

      {isMainCompany && (
        <Handle
          type="source"
          position={Position.Bottom}
          id="bottom"
          className="!w-3 !h-3 !bg-emerald-500 !border-2 !border-emerald-500 !-bottom-1.5"
        />
      )}

      {/* Main Card */}
      <Card
        className={cn(
          "relative cursor-pointer transition-all duration-300",
          "border-emerald-400/20 bg-white/5 backdrop-blur-sm",
          "hover:bg-white/10 hover:border-emerald-400/40",
          "hover:shadow-lg hover:shadow-emerald-500/20 hover:scale-105",
          isSelected && "ring-2 ring-emerald-400/50 shadow-lg shadow-emerald-500/30 scale-105",
          isMainCompany ? "w-96 h-48" : "w-80 h-44"
        )}
        onClick={onSelect}
      >
        <CardContent className="p-6 h-full flex items-center gap-4">
          {/* Company Icon */}
          <div className="flex-shrink-0 relative">
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-emerald-500/20 to-emerald-600/20 border border-emerald-400/30 flex items-center justify-center">
              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-emerald-500 to-emerald-600 flex items-center justify-center shadow-lg">
                <Building2 className="w-6 h-6 text-white" />
              </div>
            </div>
            {/* Status Indicator */}
            <div className="absolute -top-1 -right-1 w-4 h-4 bg-emerald-400 rounded-full border-2 border-white dark:border-gray-900 animate-pulse" />
          </div>

          {/* Company Info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white truncate">
                {company.name}
              </h3>
            </div>
            
            <p className="text-sm text-gray-600 dark:text-gray-300 line-clamp-2 mb-2">
              {company.description || "No description available"}
            </p>
            
            {company.address && (
              <p className="text-xs text-gray-500 dark:text-gray-400 truncate flex items-center gap-1">
                <span>📍</span>
                {company.address}
              </p>
            )}
          </div>
        </CardContent>

        {/* Selected State Indicator */}
        {isSelected && (
          <Badge 
            variant="default" 
            className="absolute top-3 right-3 bg-emerald-500/20 text-emerald-400 border-emerald-400/30"
          >
            ACTIVE
          </Badge>
        )}

        {/* Action Buttons */}
        <div className="absolute bottom-3 right-3 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
          {isMainCompany && onAddSubCompany && (
            <Button
              size="sm"
              variant="outline"
              onClick={(e) => {
                e.stopPropagation();
                onAddSubCompany(company.id);
              }}
              className="border-emerald-400/50 text-emerald-400 hover:bg-emerald-400/10 hover:border-emerald-400 cursor-pointer transition-colors"
            >
              <Plus className="w-3 h-3 mr-1" />
              Add Sub
            </Button>
          )}

          {onUpload && (
            <Button
              size="sm"
              variant="outline"
              onClick={(e) => {
                e.stopPropagation();
                onUpload(company.id, company.name, companyType);
              }}
              className="border-blue-400/50 text-blue-400 hover:bg-blue-400/10 hover:border-blue-400 cursor-pointer transition-colors"
            >
              <Upload className="w-3 h-3 mr-1" />
              Upload
            </Button>
          )}
        </div>
      </Card>
    </div>
  );
}