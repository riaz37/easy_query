"use client";

import React from "react";
import Image from "next/image";
import { ChevronRight, ChevronDown } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { Company } from "../types";

interface CompanyTreeSidebarProps {
  companies: Company[];
  selectedParentForFlow: string | null;
  selectedCompany: string | null;
  onSelectParentForFlow: (parentId: string | null) => void;
  onSelectCompany: (companyId: string) => void;
}

export function CompanyTreeSidebar({
  companies,
  selectedParentForFlow,
  selectedCompany,
  onSelectParentForFlow,
  onSelectCompany,
}: CompanyTreeSidebarProps) {
  if (companies.length === 0) {
    return null;
  }

  return (
    <Card className="w-80 query-content-gradient border-emerald-400/20 shadow-xl" style={{ borderRadius: '45px' }}>
      <CardContent className="space-y-2 max-h-96 overflow-y-auto pt-4">
        {companies.map((company) => (
          <CompanyTreeItem
            key={company.id}
            company={company}
            isSelectedForFlow={selectedParentForFlow === company.id}
            selectedCompany={selectedCompany}
            onSelectForFlow={onSelectParentForFlow}
            onSelectCompany={onSelectCompany}
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
}

function CompanyTreeItem({
  company,
  isSelectedForFlow,
  selectedCompany,
  onSelectForFlow,
  onSelectCompany,
}: CompanyTreeItemProps) {
  const [isExpanded, setIsExpanded] = React.useState(true);
  const hasChildren = company.children && company.children.length > 0;

  return (
    <div className="space-y-0">
      {/* Parent Company */}
      <div
        className={cn(
          "flex items-center gap-2 p-2 cursor-pointer transition-all duration-200",
          "hover:bg-emerald-500/10 active:scale-95"
        )}
        style={{
          borderRadius: "24px"
        }}
        onClick={() => {
          if (hasChildren) {
            setIsExpanded(!isExpanded);
          }
        }}
      >
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

        {/* Company Name */}
        <div className="flex-1 min-w-0 flex items-center gap-2">
          <span className="text-sm font-medium text-gray-900 dark:text-white truncate">
            {company.name}
          </span>
        </div>

        {/* Expand/Collapse Icon */}
        {hasChildren && (
          <div className="flex items-center">
            {isExpanded ? (
              <ChevronDown className="w-3 h-3 text-emerald-400" />
            ) : (
              <ChevronRight className="w-3 h-3 text-emerald-400" />
            )}
          </div>
        )}
      </div>

      {/* Sub-Companies Container */}
      {hasChildren && isExpanded && (
        <div className="relative">
          {/* Main vertical line - spans all children */}
          <div 
            className="absolute left-2 w-px bg-emerald-400"
            style={{
              top: '0px',
              height: `${company.children!.length * 36}px`
            }}
          />
          
          {/* Children items */}
          <div className="space-y-0">
            {company.children!.map((child, index) => {
              const isLast = index === company.children!.length - 1;
              
              return (
                <div key={child.id} className="relative flex items-center">
                  {/* Horizontal connector line */}
                  <div 
                    className="absolute left-2 w-3 h-px bg-emerald-400"
                    style={{
                      top: '18px' // Center of the item
                    }}
                  />
                  
                  {/* Child Company Item */}
                  <div
                    className={cn(
                      "flex items-center gap-2 p-2 ml-6 rounded-lg cursor-pointer transition-all duration-200",
                      "hover:bg-emerald-500/10 active:scale-95",
                      selectedCompany === child.id &&
                        "bg-emerald-500/15 border border-emerald-400/30"
                    )}
                    onClick={() => onSelectCompany(child.id)}
                    style={{
                      height: '36px',
                      minHeight: '36px'
                    }}
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
        </div>
      )}
    </div>
  );
}
