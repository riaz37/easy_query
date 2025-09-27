"use client";

import React from "react";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

interface UseTableToggleProps {
  useTable: boolean;
  onToggle: (useTable: boolean) => void;
  className?: string;
}

export function UseTableToggle({ 
  useTable, 
  onToggle, 
  className = "" 
}: UseTableToggleProps) {
  return (
    <div className={`flex items-center gap-3 mb-6 ${className}`}>
      <Label htmlFor="use-table" className="text-white font-medium">
        Use Table
      </Label>
      <Switch
        id="use-table"
        checked={useTable}
        onCheckedChange={onToggle}
        style={{
          backgroundColor: "var(--white-12, rgba(255, 255, 255, 0.12))",
          backdropFilter: "blur(29.09090805053711px)"
        }}
        className="[&>span]:!bg-[var(--primary-light,rgba(158,251,205,1))]"
      />
    </div>
  );
}
