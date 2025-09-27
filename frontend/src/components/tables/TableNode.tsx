"use client";

import React, { memo } from "react";
import { Handle, Position, NodeProps } from "reactflow";
import { Database, Users, FileText, DollarSign, Calendar, Settings, Folder, Link } from "lucide-react";

interface TableNodeData {
  table: {
    name: string;
    full_name: string;
    columns?: any[];
    relationships?: any[];
  };
  label: string;
}

// Get appropriate icon based on table name
const getTableIcon = (tableName: string) => {
  const name = tableName.toLowerCase();
  if (name.includes('user') || name.includes('employee') || name.includes('person')) {
    return Users;
  }
  if (name.includes('transaction') || name.includes('payment') || name.includes('salary') || name.includes('expense')) {
    return DollarSign;
  }
  if (name.includes('report') || name.includes('document') || name.includes('file')) {
    return FileText;
  }
  if (name.includes('attendance') || name.includes('time') || name.includes('date')) {
    return Calendar;
  }
  if (name.includes('config') || name.includes('setting') || name.includes('permission')) {
    return Settings;
  }
  if (name.includes('project') || name.includes('department') || name.includes('company')) {
    return Folder;
  }
  return Database;
};

// Unified color scheme for all tables with query-content-gradient
const getTableColors = () => {
  return {
    icon: 'text-emerald-400',
    title: 'text-white',
    subtitle: 'text-gray-300',
    relationships: 'text-gray-300',
    dots: 'bg-emerald-400'
  };
};

export const TableNode = memo(({ data }: NodeProps<TableNodeData>) => {
  const { table, label } = data;
  const Icon = getTableIcon(table.name);
  const colors = getTableColors();

  return (
    <div className="relative group">
      {/* Connection handles */}
      <Handle
        type="target"
        position={Position.Left}
        className="w-3 h-3 bg-emerald-400 border-2 border-emerald-400 opacity-0 group-hover:opacity-100 transition-opacity"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="w-3 h-3 bg-emerald-400 border-2 border-emerald-400 opacity-0 group-hover:opacity-100 transition-opacity"
      />

      {/* Enhanced table card */}
      <div className="query-content-gradient p-6 min-w-[280px] max-w-[320px] shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 relative">
        {/* Corner border effect - closer to main border */}
        <div className="absolute top-0 left-0 right-0 bottom-0 rounded-[30px] pointer-events-none" 
             style={{
               background: `linear-gradient(132.56deg, rgba(19, 245, 132, 1) 0.71%, rgba(255, 255, 255, 0) 16.71%), linear-gradient(132.98deg, rgba(255, 255, 255, 0) 84.76%, rgba(19, 245, 132, 1) 99.35%)`,
               padding: '1px',
               WebkitMask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
               WebkitMaskComposite: 'xor',
               mask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
               maskComposite: 'exclude',
               zIndex: 1
             }}>
        </div>
        {/* Header with icon and title */}
        <div className="flex items-start gap-4 mb-4">
          <div className={`${colors.icon} p-3 rounded-lg bg-black/20`}>
            <Icon className="h-6 w-6" />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className={`${colors.title} font-semibold text-base leading-tight`}>
              {label}
            </h3>
            <p className={`${colors.subtitle} text-sm font-mono mt-1 truncate`}>
              {table.full_name}
            </p>
          </div>
        </div>

        {/* Column indicators */}
        <div className="flex items-center gap-3 mb-4">
          <div className="flex flex-wrap gap-1.5">
            {Array.from({ length: Math.min(table.columns?.length || 0, 10) }).map((_, i) => (
              <div
                key={i}
                className={`w-2.5 h-2.5 ${colors.dots} rounded-full`}
              />
            ))}
          </div>
          <span className={`${colors.subtitle} text-sm`}>
            {table.columns?.length || 0} columns
          </span>
        </div>

        {/* Relationship indicator */}
        {table.relationships && table.relationships.length > 0 && (
          <div className="flex items-center gap-3">
            <Link className={`${colors.icon} h-4 w-4`} />
            <span className={`${colors.relationships} text-sm`}>
              {table.relationships.length} relationship{table.relationships.length > 1 ? 's' : ''}
            </span>
          </div>
        )}

        {/* No relationships indicator */}
        {(!table.relationships || table.relationships.length === 0) && (
          <div className="flex items-center gap-3">
            <div className="w-4 h-4 bg-gray-500/60 rounded-full" />
            <span className="text-gray-400 text-sm">
              Isolated table
            </span>
          </div>
        )}
      </div>
    </div>
  );
});

TableNode.displayName = "TableNode";