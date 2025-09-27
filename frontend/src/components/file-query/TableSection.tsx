"use client";

import React from "react";
import { TableSelector } from "@/components/data-query/TableSelector";
import { CheckCircle } from "lucide-react";

interface TableSectionProps {
  selectedTable: string | null;
  onTableSelect: (tableName: string) => void;
  currentDatabaseId: number | null;
  className?: string;
}

export function TableSection({
  selectedTable,
  onTableSelect,
  currentDatabaseId,
  className = "",
}: TableSectionProps) {
  return (
    <div
      className={`p-6 flex flex-col h-full ${className}`}
      style={{
        background:
          "linear-gradient(0deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.03)), linear-gradient(246.02deg, rgba(19, 245, 132, 0) 91.9%, rgba(19, 245, 132, 0.2) 114.38%), linear-gradient(59.16deg, rgba(19, 245, 132, 0) 71.78%, rgba(19, 245, 132, 0.2) 124.92%)",
        border: "1.5px solid",
        borderImageSource:
          "linear-gradient(158.39deg, rgba(255, 255, 255, 0.06) 14.19%, rgba(255, 255, 255, 1.5e-05) 50.59%, rgba(255, 255, 255, 1.5e-05) 68.79%, rgba(255, 255, 255, 0.015) 105.18%)",
        borderRadius: "30px",
        backdropFilter: "blur(30px)",
      }}
    >
      <div className="flex items-center gap-2 mb-4">
        <h3 className="text-white font-semibold text-xl">Table</h3>
      </div>
      <div className="space-y-4">
        <TableSelector
          databaseId={currentDatabaseId}
          onTableSelect={onTableSelect}
        />

        {/* Selected Table Indicator */}
        {selectedTable && (
          <div className="flex items-center gap-2 text-green-400 p-2 bg-green-500/10">
            <CheckCircle className="h-4 w-4" />
            <span className="text-sm">Selected: {selectedTable}</span>
          </div>
        )}
      </div>
    </div>
  );
}
