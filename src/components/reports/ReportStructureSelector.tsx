import React, { useMemo, useCallback } from "react";
import { BarChart3 } from "lucide-react";
import { Spinner } from "@/components/ui/loading";

interface ReportStructureSelectorProps {
  reportStructure: any;
  selectedStructure: string;
  setSelectedStructure: (structure: string) => void;
  isGenerating: boolean;
  loading?: boolean;
  error?: string | null;
}

export function ReportStructureSelector({
  reportStructure,
  selectedStructure,
  setSelectedStructure,
  isGenerating,
  loading = false,
  error = null,
}: ReportStructureSelectorProps) {
  // Memoize the structure keys to prevent recreation
  const structureKeys = useMemo(() => {
    if (!reportStructure) return [];
    return Object.keys(reportStructure);
  }, [reportStructure]);

  // Memoize the selected structure content
  const selectedStructureContent = useMemo(() => {
    if (!reportStructure || !selectedStructure) return '';
    return reportStructure[selectedStructure] || '';
  }, [reportStructure, selectedStructure]);

  // Memoize the structure change handler
  const handleStructureChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedStructure(e.target.value);
  }, [setSelectedStructure]);

  // Memoize the loading state
  const loadingState = useMemo(() => {
    if (loading) {
      return (
        <div className="card-enhanced">
          <div className="card-content-enhanced">
            <div className="pt-12 pb-12 text-center">
              <Spinner size="md" variant="accent-green" className="mx-auto mb-4" />
              <p className="text-gray-400">Loading report templates...</p>
            </div>
          </div>
        </div>
      );
    }
    return null;
  }, [loading]);

  // Memoize the main content
  const mainContent = useMemo(() => {
    if (!reportStructure || structureKeys.length === 0) return null;

    return (
      <div className={`card-enhanced transition-all duration-300 ${
        isGenerating ? 'opacity-60 scale-95' : ''
      }`}>
        <div className="card-content-enhanced">
          <div className="card-header-enhanced">
            <div className="card-title-enhanced flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-emerald-400" />
              Report Template
            </div>
          </div>
          <div className="mt-4">
          <div className="space-y-3">
            <label className="text-sm font-medium text-white">
              Select Report Type
            </label>
            <select
              value={selectedStructure}
              onChange={handleStructureChange}
              disabled={isGenerating}
              className="w-full p-3 border rounded-lg bg-gray-800/50 border-emerald-400/30 text-white disabled:opacity-50"
            >
              {structureKeys.map((key) => (
                <option key={key} value={key}>
                  {key
                    .replace(/_/g, " ")
                    .replace(/\b\w/g, (l) => l.toUpperCase())}
                </option>
              ))}
            </select>
            <div className="text-xs text-gray-400 bg-gray-800/30 p-3 rounded border border-gray-700">
              <div className="font-medium mb-2">Template Preview:</div>
              <div className="whitespace-pre-wrap text-gray-300">
                {selectedStructureContent}
              </div>
            </div>
          </div>
          </div>
        </div>
      </div>
    );
  }, [reportStructure, structureKeys, selectedStructure, selectedStructureContent, isGenerating, handleStructureChange]);

  // Return the appropriate content based on state
  if (loadingState) return loadingState;
  if (error) {
    return (
      <div className="card-enhanced">
        <div className="card-content-enhanced">
          <div className="pt-12 pb-12 text-center">
            <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
              <BarChart3 className="w-8 h-8 text-red-400" />
            </div>
            <h3 className="text-red-400 text-lg font-medium mb-2">
              Error Loading Report Structure
            </h3>
            <p className="text-red-300 text-sm">{error}</p>
          </div>
        </div>
      </div>
    );
  }
  if (mainContent) return mainContent;
  
  // Return null if no structure is available
  return null;
} 