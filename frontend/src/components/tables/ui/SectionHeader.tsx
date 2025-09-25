import React from "react";

interface SectionHeaderProps {
  title: string;
  description?: string;
  actionButton?: React.ReactNode;
  icon?: React.ReactNode;
}

export function SectionHeader({ 
  title, 
  description, 
  actionButton, 
  icon 
}: SectionHeaderProps) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-4">
        {icon && (
          <div className="w-10 h-10 bg-gradient-to-br from-slate-500/20 to-slate-600/20 rounded-xl flex items-center justify-center border border-slate-500/30">
            {icon}
          </div>
        )}
        <div>
          <h2 className="text-3xl font-bold text-white">{title}</h2>
          {description && (
            <p className="text-slate-400 mt-2 text-lg">{description}</p>
          )}
        </div>
      </div>
      {actionButton && (
        <div className="flex gap-3">
          {actionButton}
        </div>
      )}
    </div>
  );
} 