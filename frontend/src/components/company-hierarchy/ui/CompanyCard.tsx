"use client";

import React from "react";
import { Building2, Plus, Upload } from "lucide-react";
import { Handle, Position } from "reactflow";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { useTheme } from "@/store/theme-store";
import { CompanyCardProps } from "../types";

export function CompanyCard({
  company,
  onAddSubCompany,
  onUpload,
  isSelected = false,
  onSelect,
  level = 0,
}: CompanyCardProps) {
  const theme = useTheme();
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
          "relative cursor-pointer query-content-gradient modal-enhanced",
          "hover:shadow-lg hover:shadow-emerald-500/20",
          isSelected &&
            "ring-2 ring-emerald-400/50 shadow-lg shadow-emerald-500/30",
          isMainCompany ? "w-[500px] h-[280px]" : "w-[480px] h-[260px]"
        )}
        onClick={onSelect}
      >
        <CardContent className="p-0 h-full relative">
          {/* Meta Logo - Background */}
          <div className="absolute left-0 top-0 w-80 h-full">
            <img
              src="/meta.svg"
              alt="Meta Logo"
              className="w-full h-full object-cover rounded-l-[30px]"
              style={{
                objectPosition: "left center",
              }}
            />
          </div>

          {/* Company Info - Overlapping text */}
          <div className="absolute left-48 top-0 right-0 h-full px-6 py-6 flex flex-col justify-center overflow-hidden z-10">
            <div className="flex items-center gap-2 mb-3">
              <h3 className="modal-title-enhanced text-xl truncate">
                {company.name}
              </h3>
            </div>

            <p className="modal-description-enhanced line-clamp-3 mb-4 leading-relaxed">
              {company.description ||
                "Automate refund processes with configurable policy enforcement."}
            </p>

            {company.address && (
              <p className="text-xs text-gray-300 truncate flex items-center gap-1">
                {company.address}
              </p>
            )}
          </div>
        </CardContent>

        {/* Action Buttons */}
        <div className="absolute bottom-4 right-4 flex gap-2 pointer-events-auto">
          {onUpload && (
            <Button
              size="sm"
              variant="outline"
              onClick={(e) => {
                e.stopPropagation();
                onUpload(company.id, company.name, companyType);
              }}
              className="border-emerald-400/50 text-white hover:bg-emerald-400/10 hover:border-emerald-400 cursor-pointer transition-colors bg-gray-800/50 px-4 py-2"
            >
              <Upload className="w-4 h-4 mr-2 text-emerald-400" />
              Upload
            </Button>
          )}
        </div>
      </Card>
    </div>
  );
}
