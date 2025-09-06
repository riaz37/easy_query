import React, { useMemo, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart3 } from "lucide-react";
import { Spinner } from "@/components/ui/loading";

interface ReportStructureSelectorProps {
  reportStructure: any;
  selectedStructure: string;
  setSelectedStructure: (structure: string) => void;
  isGenerating: boolean;
}

export function ReportStructureSelector({
  reportStructure,
  selectedStructure,
  setSelectedStructure,
  isGenerating,
}: ReportStructureSelectorProps) {
  // Memoize the structure keys to prevent recreation
  const structureKeys = useMemo(() => {
    if (!reportStructure.structure) return [];
    return Object.keys(reportStructure.structure);
  }, [reportStructure.structure]);

  // Memoize the selected structure content
  const selectedStructureContent = useMemo(() => {
    if (!reportStructure.structure || !selectedStructure) return '';
    return reportStructure.structure[selectedStructure] || '';
  }, [reportStructure.structure, selectedStructure]);

  // Memoize the structure change handler
  const handleStructureChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedStructure(e.target.value);
  }, [setSelectedStructure]);

  // Memoize the loading state
  const loadingState = useMemo(() => {
    if (reportStructure.isLoading) {
      return (
        <Card className="bg-gray-900/50 border-purple-400/30">
          <CardContent className="pt-12 pb-12 text-center">
            <Spinner size="md" variant="accent-purple" className="mx-auto mb-4" />
            <p className="text-gray-400">Loading report templates...</p>
          </CardContent>
        </Card>
      );
    }
    return null;
  }, [reportStructure.isLoading]);

  // Memoize the main content
  const mainContent = useMemo(() => {
    if (!reportStructure.structure || structureKeys.length === 0) return null;

    return (
      <Card className={`bg-gray-900/50 border-purple-400/30 transition-all duration-300 ${
        isGenerating ? 'opacity-60 scale-95' : ''
      }`}>
        <CardHeader>
          <CardTitle className="text-purple-400 flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            Report Template
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <label className="text-sm font-medium text-white">
              Select Report Type
            </label>
            <select
              value={selectedStructure}
              onChange={handleStructureChange}
              disabled={isGenerating}
              className="w-full p-3 border rounded-lg bg-gray-800/50 border-purple-400/30 text-white disabled:opacity-50"
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
        </CardContent>
      </Card>
    );
  }, [reportStructure.structure, structureKeys, selectedStructure, selectedStructureContent, isGenerating, handleStructureChange]);

  // Return the appropriate content based on state
  if (loadingState) return loadingState;
  if (mainContent) return mainContent;
  
  // Return null if no structure is available
  return null;
} 