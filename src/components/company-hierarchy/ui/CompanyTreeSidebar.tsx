"use client";

import React from "react";
import Image from "next/image";
import {
  Building2,
  ChevronRight,
  ChevronDown,
  Plus,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { Company } from "../types";

interface CompanyTreeSidebarProps {
  companies: Company[];
  selectedParentForFlow: string | null;
  selectedCompany: string | null;
  onSelectParentForFlow: (parentId: string | null) => void;
  onSelectCompany: (companyId: string) => void;
  onAddSubCompany?: (parentId: string) => void;
}

export function CompanyTreeSidebar({
  companies,
  selectedParentForFlow,
  selectedCompany,
  onSelectParentForFlow,
  onSelectCompany,
  onAddSubCompany,
}: CompanyTreeSidebarProps) {
  if (companies.length === 0) {
    return null;
  }

  return (
    <Card className="w-80 bg-white/5 backdrop-blur-md border-emerald-400/20 shadow-xl">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-3 text-emerald-400">
          <div className="w-8 h-8 rounded-lg bg-emerald-500/20 border border-emerald-400/40 flex items-center justify-center">
            <Building2 className="w-4 h-4" />
          </div>
          Company Structure
        </CardTitle>

        {selectedParentForFlow && (
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-600 dark:text-gray-400">
              <span className="font-medium text-emerald-400">Viewing: </span>
              {companies.find((c) => c.id === selectedParentForFlow)?.name}
            </div>
            <Button
              size="sm"
              variant="outline"
              onClick={() => onSelectParentForFlow(null)}
              className="text-xs border-emerald-400/50 text-emerald-400 hover:bg-emerald-400/10 cursor-pointer transition-colors"
            >
              Show All
            </Button>
          </div>
        )}
      </CardHeader>

      <CardContent className="space-y-2 max-h-96 overflow-y-auto">
        {companies.map((company) => (
          <CompanyTreeItem
            key={company.id}
            company={company}
            isSelectedForFlow={selectedParentForFlow === company.id}
            selectedCompany={selectedCompany}
            onSelectForFlow={onSelectParentForFlow}
            onSelectCompany={onSelectCompany}
            onAddSubCompany={onAddSubCompany}
          />
        ))}
      </CardContent>
    </Card>
  );
}

interface CompanyTreeItemProps {
  company: Company;
  isSelectedForFlow: boolean;
  selectedCompany: string | null;
  onSelectForFlow: (parentId: string | null) => void;
  onSelectCompany: (companyId: string) => void;
  onAddSubCompany?: (parentId: string) => void;
}

function CompanyTreeItem({
  company,
  isSelectedForFlow,
  selectedCompany,
  onSelectForFlow,
  onSelectCompany,
  onAddSubCompany,
}: CompanyTreeItemProps) {
  const [isExpanded, setIsExpanded] = React.useState(true);
  const hasChildren = company.children && company.children.length > 0;

  return (
    <div className="space-y-1">
      {/* Parent Company */}
      <div
        className={cn(
          "flex items-center gap-2 p-2 rounded-lg cursor-pointer transition-all duration-200",
          "hover:bg-emerald-500/10 hover:border-emerald-400/30 border border-transparent active:scale-95",
          isSelectedForFlow && "bg-emerald-500/20 border-emerald-400/50"
        )}
        onClick={() => {
          if (isSelectedForFlow) {
            onSelectForFlow(null);
          } else {
            onSelectForFlow(company.id);
          }
        }}
      >
        {/* Expand/Collapse Button */}
        {hasChildren && (
          <Button
            variant="ghost"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              setIsExpanded(!isExpanded);
            }}
            className="p-0.5 h-auto w-auto hover:bg-emerald-400/20 cursor-pointer transition-colors"
          >
            {isExpanded ? (
              <ChevronDown className="w-3 h-3 text-emerald-400" />
            ) : (
              <ChevronRight className="w-3 h-3 text-emerald-400" />
            )}
          </Button>
        )}

        {/* Company Icon */}
        <div className="w-4 h-4 text-emerald-400">
          <Image 
            src="/filelogo.svg" 
            alt="Company" 
            width={16} 
            height={16} 
            className="w-4 h-4"
          />
        </div>

        {/* Status Dot */}
        <div className="w-2 h-2 rounded-full bg-emerald-400" />

        {/* Company Name */}
        <div className="flex-1 min-w-0 flex items-center gap-2">
          <span className="text-sm font-medium text-gray-900 dark:text-white truncate">
            {company.name}
          </span>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          {onAddSubCompany && (
            <Button
              variant="ghost"
              size="sm"
              onClick={(e) => {
                e.stopPropagation();
                onAddSubCompany(company.id);
              }}
              className="p-1 h-auto w-auto hover:bg-emerald-400/20 cursor-pointer transition-colors"
              title="Add Sub-Company"
            >
              <Plus className="w-3 h-3 text-emerald-400" />
            </Button>
          )}
        </div>
      </div>

      {/* Sub-Companies */}
      {hasChildren && isExpanded && (
        <div className="ml-4 space-y-1">
          {company.children!.map((child, index) => {
            const isLast = index === company.children!.length - 1;
            
            return (
              <div key={child.id} className="relative">
                {/* Tree connector lines */}
                <div className="absolute left-2 top-0 flex items-center h-8">
                  {!isLast && (
                    <div className="absolute left-0 top-4 w-px h-4 bg-emerald-400/30" />
                  )}
                  <div className="flex">
                    <div className="w-px h-4 bg-emerald-400/30" />
                    <div className="w-3 h-px bg-emerald-400/30 mt-4" />
                  </div>
                </div>

                <div
                  className={cn(
                    "flex items-center gap-2 p-2 ml-6 rounded-lg cursor-pointer transition-all duration-200",
                    "hover:bg-emerald-500/10 active:scale-95",
                    selectedCompany === child.id && "bg-emerald-500/15 border border-emerald-400/30"
                  )}
                  onClick={() => onSelectCompany(child.id)}
                >
                  {/* File Icon */}
                  <div className="w-4 h-4 text-emerald-400/70">
                    <Image 
                      src="/filelogo.svg" 
                      alt="Sub-Company" 
                      width={16} 
                      height={16} 
                      className="w-4 h-4"
                    />
                  </div>

                  {/* Status Dot */}
                  <div className="w-2 h-2 rounded-full bg-emerald-400/70" />

                  {/* Company Name */}
                  <div className="flex-1 min-w-0 flex items-center gap-2">
                    <span className="text-sm text-gray-700 dark:text-gray-300 truncate">
                      {child.name}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}