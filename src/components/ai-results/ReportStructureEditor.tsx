"use client";

import React, { useState, useCallback, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { 
  Edit3, 
  Save, 
  X, 
  Plus, 
  Trash2, 
  Eye, 
  EyeOff,
  AlertCircle,
  CheckCircle,
  FileText,
  BarChart3
} from "lucide-react";
import { ReportResults, ReportSection } from "@/types/reports";
import { useReportStructure } from "@/lib/hooks/use-report-structure";
import { useUserContext } from "@/lib/hooks/use-user-context";

interface ReportStructureEditorProps {
  reportResults: ReportResults;
  onStructureUpdate?: (updatedResults: ReportResults) => void;
  onClose?: () => void;
}

interface EditableSection extends ReportSection {
  isEditing?: boolean;
  originalData?: ReportSection;
}

export function ReportStructureEditor({
  reportResults,
  onStructureUpdate,
  onClose
}: ReportStructureEditorProps) {
  const { user } = useUserContext();
  const reportStructure = useReportStructure();
  
  const [isEditing, setIsEditing] = useState(false);
  const [editableSections, setEditableSections] = useState<EditableSection[]>([]);
  const [expandedSections, setExpandedSections] = useState<Set<number>>(new Set());
  const [showPreview, setShowPreview] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Initialize editable sections from report results
  React.useEffect(() => {
    if (reportResults.results) {
      setEditableSections(reportResults.results.map(section => ({
        ...section,
        isEditing: false,
        originalData: { ...section }
      })));
    }
  }, [reportResults.results]);

  const toggleSectionExpansion = useCallback((index: number) => {
    setExpandedSections(prev => {
      const newSet = new Set(prev);
      if (newSet.has(index)) {
        newSet.delete(index);
      } else {
        newSet.add(index);
      }
      return newSet;
    });
  }, []);

  const startEditing = useCallback((index: number) => {
    setEditableSections(prev => 
      prev.map((section, i) => 
        i === index 
          ? { ...section, isEditing: true, originalData: { ...section } }
          : section
      )
    );
  }, []);

  const cancelEditing = useCallback((index: number) => {
    setEditableSections(prev => 
      prev.map((section, i) => 
        i === index && section.originalData
          ? { ...section.originalData, isEditing: false, originalData: undefined }
          : section
      )
    );
  }, []);

  const updateSection = useCallback((index: number, field: keyof ReportSection, value: any) => {
    setEditableSections(prev => 
      prev.map((section, i) => 
        i === index 
          ? { ...section, [field]: value }
          : section
      )
    );
  }, []);

  const saveSection = useCallback((index: number) => {
    setEditableSections(prev => 
      prev.map((section, i) => 
        i === index 
          ? { ...section, isEditing: false, originalData: undefined }
          : section
      )
    );
  }, []);

  const deleteSection = useCallback((index: number) => {
    setEditableSections(prev => prev.filter((_, i) => i !== index));
  }, []);

  const addNewSection = useCallback(() => {
    const newSection: EditableSection = {
      section_number: editableSections.length + 1,
      section_name: "New Section",
      query_number: editableSections.length + 1,
      query: "",
      success: true,
      isEditing: true,
      originalData: undefined
    };
    setEditableSections(prev => [...prev, newSection]);
  }, [editableSections.length]);

  const saveAllChanges = useCallback(async () => {
    if (!user?.id) {
      setError("User not authenticated");
      return;
    }

    setSaving(true);
    setError(null);

    try {
      // Create updated report results
      const updatedResults: ReportResults = {
        ...reportResults,
        results: editableSections.map(section => {
          const { isEditing, originalData, ...cleanSection } = section;
          return cleanSection;
        })
      };

      // Update the report structure in the backend
      const structureString = JSON.stringify(updatedResults, null, 2);
      await reportStructure.updateStructure(user.id, { report_structure: structureString });

      // Update session storage
      sessionStorage.setItem("reportResults", JSON.stringify(updatedResults));

      // Notify parent component
      onStructureUpdate?.(updatedResults);

      setIsEditing(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save changes");
    } finally {
      setSaving(false);
    }
  }, [user?.id, reportResults, editableSections, reportStructure, onStructureUpdate]);

  const hasChanges = useMemo(() => {
    return editableSections.some(section => section.isEditing);
  }, [editableSections]);

  const hasUnsavedChanges = useMemo(() => {
    return editableSections.some(section => 
      section.isEditing || 
      (section.originalData && JSON.stringify(section) !== JSON.stringify(section.originalData))
    );
  }, [editableSections]);

  if (!reportResults.results || reportResults.results.length === 0) {
    return (
      <Card className="bg-gray-900/50 border-gray-400/30">
        <CardContent className="pt-12 pb-12 text-center">
          <div className="w-16 h-16 bg-gray-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <FileText className="w-8 h-8 text-gray-400" />
          </div>
          <h3 className="text-white text-lg font-medium mb-2">
            No Report Sections Available
          </h3>
          <p className="text-gray-400">
            This report doesn't contain any sections to edit.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header Controls */}
      <Card className="bg-gray-900/50 border-gray-400/30">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-white">
              <BarChart3 className="w-5 h-5 text-emerald-400" />
              Report Structure Editor
            </CardTitle>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowPreview(!showPreview)}
                className="border-gray-600 text-gray-300 hover:bg-gray-700 cursor-pointer"
              >
                {showPreview ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                {showPreview ? "Hide Preview" : "Show Preview"}
              </Button>
              {!isEditing ? (
                <Button
                  onClick={() => setIsEditing(true)}
                  className="bg-emerald-600 hover:bg-emerald-700 cursor-pointer"
                >
                  <Edit3 className="w-4 h-4 mr-2" />
                  Edit Structure
                </Button>
              ) : (
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    onClick={() => setIsEditing(false)}
                    className="border-gray-600 text-gray-300 hover:bg-gray-700 cursor-pointer"
                  >
                    <X className="w-4 h-4 mr-2" />
                    Cancel
                  </Button>
                  <Button
                    onClick={saveAllChanges}
                    disabled={saving || !hasUnsavedChanges}
                    className="bg-emerald-600 hover:bg-emerald-700 cursor-pointer"
                  >
                    <Save className="w-4 h-4 mr-2" />
                    {saving ? "Saving..." : "Save Changes"}
                  </Button>
                </div>
              )}
              {onClose && (
                <Button
                  variant="ghost"
                  onClick={onClose}
                  className="text-gray-400 hover:text-white"
                >
                  <X className="w-4 h-4" />
                </Button>
              )}
            </div>
          </div>
        </CardHeader>
        {error && (
          <CardContent className="pt-0">
            <div className="flex items-center gap-2 p-3 bg-red-900/20 border border-red-500/30 rounded-lg">
              <AlertCircle className="w-4 h-4 text-red-400" />
              <span className="text-red-400 text-sm">{error}</span>
            </div>
          </CardContent>
        )}
        {hasUnsavedChanges && (
          <CardContent className="pt-0">
            <div className="flex items-center gap-2 p-3 bg-yellow-900/20 border border-yellow-500/30 rounded-lg">
              <AlertCircle className="w-4 h-4 text-yellow-400" />
              <span className="text-yellow-400 text-sm">You have unsaved changes</span>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Report Sections */}
      <div className="space-y-4">
        {editableSections.map((section, index) => (
          <Card key={index} className="bg-gray-900/50 border-gray-400/30">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Badge variant={section.success ? "default" : "destructive"}>
                    Section {section.section_number}
                  </Badge>
                  <span className="text-white font-medium">
                    {section.isEditing ? (
                      <Input
                        value={section.section_name}
                        onChange={(e) => updateSection(index, 'section_name', e.target.value)}
                        className="bg-gray-800 border-gray-600 text-white"
                        placeholder="Section name"
                      />
                    ) : (
                      section.section_name
                    )}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {isEditing && (
                    <>
                      {section.isEditing ? (
                        <>
                          <Button
                            size="sm"
                            onClick={() => saveSection(index)}
                            className="bg-emerald-600 hover:bg-emerald-700 cursor-pointer"
                          >
                            <CheckCircle className="w-4 h-4" />
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => cancelEditing(index)}
                            className="border-gray-600 text-gray-300 hover:bg-gray-700 cursor-pointer"
                          >
                            <X className="w-4 h-4" />
                          </Button>
                        </>
                      ) : (
                        <>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => startEditing(index)}
                            className="border-gray-600 text-gray-300 hover:bg-gray-700 cursor-pointer"
                          >
                            <Edit3 className="w-4 h-4" />
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => deleteSection(index)}
                            className="border-red-600 text-red-300 hover:bg-red-900/20 cursor-pointer"
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </>
                      )}
                    </>
                  )}
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => toggleSectionExpansion(index)}
                    className="text-gray-400 hover:text-white cursor-pointer"
                  >
                    {expandedSections.has(index) ? "Collapse" : "Expand"}
                  </Button>
                </div>
              </div>
            </CardHeader>
            
            {expandedSections.has(index) && (
              <CardContent className="space-y-4">
                {/* Query Section */}
                <div>
                  <label className="text-sm font-medium text-gray-300 mb-2 block">
                    Query {section.query_number}
                  </label>
                  {section.isEditing ? (
                    <Textarea
                      value={section.query}
                      onChange={(e) => updateSection(index, 'query', e.target.value)}
                      className="bg-gray-800 border-gray-600 text-white min-h-[100px]"
                      placeholder="Enter SQL query"
                    />
                  ) : (
                    <div className="bg-gray-800/50 p-3 rounded-lg border border-gray-700">
                      <code className="text-gray-300 text-sm">{section.query}</code>
                    </div>
                  )}
                </div>

                {/* Success Status */}
                {section.isEditing && (
                  <div className="flex items-center gap-2">
                    <label className="text-sm font-medium text-gray-300">
                      Success Status:
                    </label>
                    <select
                      value={section.success ? "true" : "false"}
                      onChange={(e) => updateSection(index, 'success', e.target.value === "true")}
                      className="bg-gray-800 border-gray-600 text-white px-3 py-1 rounded"
                    >
                      <option value="true">Success</option>
                      <option value="false">Failed</option>
                    </select>
                  </div>
                )}

                {/* Data Preview */}
                {showPreview && section.table && section.table.data && section.table.data.length > 0 && (
                  <div>
                    <label className="text-sm font-medium text-gray-300 mb-2 block">
                      Data Preview ({section.table.total_rows} rows, {section.table.columns.length} columns)
                    </label>
                    <div className="bg-gray-800/50 p-3 rounded-lg border border-gray-700 max-h-40 overflow-auto">
                      <div className="text-xs text-gray-400">
                        Columns: {section.table.columns.join(", ")}
                      </div>
                      <div className="text-xs text-gray-300 mt-2">
                        Sample data: {JSON.stringify(section.table.data.slice(0, 2), null, 2)}
                      </div>
                    </div>
                  </div>
                )}

                {/* Analysis Preview */}
                {showPreview && section.graph_and_analysis?.llm_analysis && (
                  <div>
                    <label className="text-sm font-medium text-gray-300 mb-2 block">
                      AI Analysis Preview
                    </label>
                    <div className="bg-gray-800/50 p-3 rounded-lg border border-gray-700 max-h-40 overflow-auto">
                      <div className="text-xs text-gray-300">
                        {section.graph_and_analysis.llm_analysis.analysis.substring(0, 200)}...
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            )}
          </Card>
        ))}

        {/* Add New Section Button */}
        {isEditing && (
          <Card className="bg-gray-900/30 border-dashed border-gray-600">
            <CardContent className="pt-6 pb-6 text-center">
              <Button
                onClick={addNewSection}
                variant="outline"
                className="border-gray-600 text-gray-300 hover:bg-gray-700 cursor-pointer"
              >
                <Plus className="w-4 h-4 mr-2" />
                Add New Section
              </Button>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
