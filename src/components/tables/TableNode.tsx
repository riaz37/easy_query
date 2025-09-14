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

// Unified green color scheme for all tables
const getTableColors = () => {
  return {
    bg: 'bg-emerald-900/20',
    border: 'border-emerald-500/40',
    icon: 'text-emerald-400',
    title: 'text-emerald-100',
    subtitle: 'text-emerald-300/60',
    dots: 'bg-emerald-400/60',
    relationships: 'text-emerald-300/60'
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
      <div className={`${colors.bg} ${colors.border} border-2 rounded-xl p-6 min-w-[280px] max-w-[320px] backdrop-blur-sm shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105`}>
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
            <div className="w-4 h-4 bg-gray-600/40 rounded-full" />
            <span className="text-gray-400/60 text-sm">
              Isolated table
            </span>
          </div>
        )}
      </div>
    </div>
  );
});

TableNode.displayName = "TableNode";