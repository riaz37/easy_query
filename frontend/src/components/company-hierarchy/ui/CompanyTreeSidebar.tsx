"use client";

import React from "react";
import Image from "next/image";
import { ChevronRight, ChevronDown, XIcon } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { Company } from "../types";

interface CompanyTreeSidebarProps {
  companies: Company[];
  selectedParentForFlow: string | null;
  selectedCompany: string | null;
  onSelectParentForFlow: (parentId: string | null) => void;
  onSelectCompany: (companyId: string) => void;
  onClose: () => void;
}

export function CompanyTreeSidebar({
  companies,
  selectedParentForFlow,
  selectedCompany,
  onSelectParentForFlow,
  onSelectCompany,
  onClose,
}: CompanyTreeSidebarProps) {

  if (companies.length === 0) {
    return null;
  }

  return (
    <div className="fixed bottom-32 left-1/2 transform -translate-x-1/2 z-60 flex-shrink-0">
      {/* Tree View Dropdown - Always visible */}
      <div className="transform transition-all duration-300 ease-in-out">
        <Card className="w-80 query-content-gradient border-emerald-400/20 shadow-xl" style={{ borderRadius: '45px' }}>
          {/* Fixed Header */}
          <div className="flex items-center justify-between p-4 pb-2 border-b border-emerald-400/20">
            <h3 className="text-lg font-semibold text-white">Company Tree</h3>
            <button
              onClick={onClose}
              className="modal-close-button cursor-pointer"
            >
              <XIcon className="h-5 w-5" />
            </button>
          </div>
          
          {/* Scrollable Company Tree Items */}
          <div className="max-h-80 overflow-y-auto">
            <div className="p-4 pt-2 space-y-2">
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
            </div>
          </div>
        </Card>
      </div>
    </div>
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
        <div className="relative ml-2">
          {/* SVG for curved tree lines */}
          <svg 
            className="absolute top-0 left-0 pointer-events-none"
            width="32"
            height={company.children!.length * 36}
            style={{ overflow: 'visible' }}
          >
            <defs>
              <style>{`
                .tree-line {
                  fill: none;
                  stroke: rgb(52, 211, 153);
                  stroke-width: 1;
                  opacity: 0.8;
                }
              `}</style>
            </defs>
            
            {/* Vertical line from parent */}
            <line 
              x1="8" 
              y1="0" 
              x2="8" 
              y2={company.children!.length > 1 ? (company.children!.length - 1) * 36 + 18 : 18}
              className="tree-line"
            />
            
            {/* Curved connectors for each child */}
            {company.children!.map((child, index) => {
              const y = index * 36 + 18; // Center of each child item
              
              return (
                <g key={child.id}>
                  {/* Longer curved horizontal connector */}
                  <path
                    d={`M 8 ${y} Q 14 ${y} 24 ${y}`}
                    className="tree-line"
                  />
                </g>
              );
            })}
          </svg>
          
          {/* Children items */}
          <div className="space-y-0">
            {company.children!.map((child, index) => (
              <div key={child.id} className="relative flex items-center">
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
            ))}
          </div>
        </div>
      )}
    </div>
  );
}