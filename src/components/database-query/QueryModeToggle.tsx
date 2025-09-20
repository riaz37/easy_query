"use client";

import React from "react";
import { Label } from "@/components/ui/label";

interface QueryModeToggleProps {
  mode: 'query' | 'reports';
  onModeChange: (mode: 'query' | 'reports') => void;
  hasDatabase: boolean;
  loading?: boolean;
}

export function QueryModeToggle({ mode, onModeChange, hasDatabase, loading = false }: QueryModeToggleProps) {
  return (
    <div className="flex items-center gap-3">
      <Label htmlFor="query-mode" className="text-white font-medium">
        Quick Query
      </Label>
      <button
        onClick={() => onModeChange(mode === 'query' ? 'reports' : 'query')}
        disabled={!hasDatabase || loading}
        className="relative inline-flex h-6 w-11 items-center transition-colors rounded-full disabled:opacity-50 disabled:cursor-not-allowed"
        style={{
          backgroundColor: "var(--white-12, rgba(255, 255, 255, 0.12))",
          backdropFilter: "blur(29.09090805053711px)"
        }}
      >
        <span
          className={`inline-block h-4 w-4 transform transition-transform rounded-full ${
            mode === 'query' ? "translate-x-6" : "translate-x-1"
          }`}
          style={{
            backgroundColor: "var(--primary-light, rgba(158, 251, 205, 1))"
          }}
        />
      </button>
      <Label htmlFor="query-mode" className="text-white font-medium">
        AI Reports
      </Label>
    </div>
  );
} 