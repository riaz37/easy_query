"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
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
  BarChart3,
  RefreshCw
} from "lucide-react";
import { useUserContext } from "@/lib/hooks/use-user-context";
import { ReportStructure } from "@/types/reports";
import { Spinner } from "@/components/ui/loading";
import { toast } from "sonner";

interface ReportStructureTabProps {
  className?: string;
  reportStructure?: string;
  reportStructureLoading?: boolean;
  reportStructureError?: string | null;
  onRefresh?: () => void;
}

interface EditableStructure {
  key: string;
  value: string;
  isEditing?: boolean;
  originalValue?: string;
}

export const ReportStructureTab = React.memo<ReportStructureTabProps>(({ 
  className,
  reportStructure: reportStructureString = '',
  reportStructureLoading = false,
  reportStructureError = null,
  onRefresh
}) => {
  const { user } = useUserContext();
  
  const [editableStructures, setEditableStructures] = useState<EditableStructure[]>([]);
  const [isEditing, setIsEditing] = useState(false);
  const [showPreview, setShowPreview] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [newStructureKey, setNewStructureKey] = useState("");
  const [newStructureValue, setNewStructureValue] = useState("");

  // Initialize editable structures from loaded structure
  useEffect(() => {
    if (reportStructureString) {
      try {
        const parsedStructure = JSON.parse(reportStructureString);
        const structures = Object.entries(parsedStructure).map(([key, value]) => ({
          key,
          value: String(value),
          isEditing: false,
          originalValue: String(value)
        }));
        setEditableStructures(structures);
      } catch (error) {
        console.error('Failed to parse report structure:', error);
        setEditableStructures([]);
      }
    } else {
      setEditableStructures([]);
    }
  }, [reportStructureString]);

  const startEditing = useCallback((index: number) => {
    setEditableStructures(prev => 
      prev.map((structure, i) => 
        i === index 
          ? { ...structure, isEditing: true, originalValue: structure.value }
          : structure
      )
    );
  }, []);

  const cancelEditing = useCallback((index: number) => {
    setEditableStructures(prev => 
      prev.map((structure, i) => 
        i === index && structure.originalValue !== undefined
          ? { ...structure, value: structure.originalValue, isEditing: false, originalValue: undefined }
          : structure
      )
    );
  }, []);

  const updateStructure = useCallback((index: number, field: 'key' | 'value', value: string) => {
    setEditableStructures(prev => 
      prev.map((structure, i) => 
        i === index 
          ? { ...structure, [field]: value }
          : structure
      )
    );
  }, []);

  const saveStructure = useCallback((index: number) => {
    setEditableStructures(prev => 
      prev.map((structure, i) => 
        i === index 
          ? { ...structure, isEditing: false, originalValue: undefined }
          : structure
      )
    );
  }, []);

  const deleteStructure = useCallback((index: number) => {
    setEditableStructures(prev => prev.filter((_, i) => i !== index));
  }, []);

  const addNewStructure = useCallback(() => {
    if (!newStructureKey.trim() || !newStructureValue.trim()) {
      toast.error("Please enter both key and value for the new structure");
      return;
    }

    // Check if key already exists
    if (editableStructures.some(s => s.key === newStructureKey.trim())) {
      toast.error("A structure with this key already exists");
      return;
    }

    const newStructure: EditableStructure = {
      key: newStructureKey.trim(),
      value: newStructureValue.trim(),
      isEditing: false,
      originalValue: undefined
    };

    setEditableStructures(prev => [...prev, newStructure]);
    setNewStructureKey("");
    setNewStructureValue("");
    toast.success("New structure added");
  }, [newStructureKey, newStructureValue, editableStructures]);

  const saveAllChanges = useCallback(async () => {
    if (!user?.user_id) {
      setError("User not authenticated");
      return;
    }

    setSaving(true);
    setError(null);

    try {
      // Convert editable structures back to ReportStructure format
      const structureObject: ReportStructure = {};
      editableStructures.forEach(structure => {
        structureObject[structure.key] = structure.value;
      });

      // Update the report structure in the backend using user current DB API
      const structureString = JSON.stringify(structureObject, null, 2);
      const { ServiceRegistry } = await import("@/lib/api");
      await ServiceRegistry.userCurrentDB.setUserCurrentDB(
        { report_structure: structureString },
        user.user_id
      );

      toast.success("Report structure updated successfully");
      setIsEditing(false);
      
      // Refresh the data
      if (onRefresh) {
        onRefresh();
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to save changes";
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setSaving(false);
    }
  }, [user?.user_id, editableStructures, onRefresh]);

  const hasUnsavedChanges = useCallback(() => {
    return editableStructures.some(structure => 
      structure.isEditing || 
      (structure.originalValue && structure.value !== structure.originalValue)
    );
  }, [editableStructures]);

  const resetToOriginal = useCallback(() => {
    if (reportStructureString) {
      try {
        const parsedStructure = JSON.parse(reportStructureString);
        const structures = Object.entries(parsedStructure).map(([key, value]) => ({
          key,
          value: String(value),
          isEditing: false,
          originalValue: String(value)
        }));
        setEditableStructures(structures);
        setIsEditing(false);
        toast.info("Changes reset to original");
      } catch (error) {
        console.error('Failed to parse report structure for reset:', error);
        setEditableStructures([]);
      }
    }
  }, [reportStructureString]);

  if (reportStructureLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Spinner size="lg" variant="primary" />
        <span className="ml-4 text-gray-400">Loading report structure...</span>
      </div>
    );
  }

  if (reportStructureError) {
    return (
      <Card className="bg-red-900/20 border-red-500/30">
        <CardContent className="pt-12 pb-12 text-center">
          <AlertCircle className="w-12 h-12 text-red-400 mx-auto mb-4" />
          <h3 className="text-red-400 text-lg font-medium mb-2">
            Error Loading Report Structure
          </h3>
          <p className="text-red-300 mb-4">{reportStructureError}</p>
          <Button
            onClick={onRefresh}
            className="bg-red-600 hover:bg-red-700"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className={className}>
      <div className="space-y-6">
        {/* Header Controls */}
        <Card className="bg-gray-900/50 border-gray-400/30">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2 text-white">
                <BarChart3 className="w-5 h-5 text-emerald-400" />
                Report Structure Management
              </CardTitle>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowPreview(!showPreview)}
                  className="border-gray-600 text-gray-300 hover:bg-gray-700"
                >
                  {showPreview ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  {showPreview ? "Hide Preview" : "Show Preview"}
                </Button>
                {!isEditing ? (
                  <Button
                    onClick={() => setIsEditing(true)}
                    className="bg-emerald-600 hover:bg-emerald-700"
                  >
                    <Edit3 className="w-4 h-4 mr-2" />
                    Edit Structure
                  </Button>
                ) : (
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      onClick={() => {
                        setIsEditing(false);
                        resetToOriginal();
                      }}
                      className="border-gray-600 text-gray-300 hover:bg-gray-700"
                    >
                      <X className="w-4 h-4 mr-2" />
                      Cancel
                    </Button>
                    <Button
                      onClick={saveAllChanges}
                      disabled={saving || !hasUnsavedChanges()}
                      className="bg-emerald-600 hover:bg-emerald-700"
                    >
                      <Save className="w-4 h-4 mr-2" />
                      {saving ? "Saving..." : "Save Changes"}
                    </Button>
                  </div>
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
          {hasUnsavedChanges() && (
            <CardContent className="pt-0">
              <div className="flex items-center gap-2 p-3 bg-yellow-900/20 border border-yellow-500/30 rounded-lg">
                <AlertCircle className="w-4 h-4 text-yellow-400" />
                <span className="text-yellow-400 text-sm">You have unsaved changes</span>
              </div>
            </CardContent>
          )}
        </Card>

        {/* Add New Structure */}
        {isEditing && (
          <Card className="bg-gray-900/30 border-dashed border-gray-600">
            <CardHeader>
              <CardTitle className="text-white text-lg">Add New Report Structure</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium text-gray-300 mb-2 block">
                    Structure Key
                  </label>
                  <Input
                    value={newStructureKey}
                    onChange={(e) => setNewStructureKey(e.target.value)}
                    placeholder="e.g., financial_report, sales_analysis"
                    className="bg-gray-800 border-gray-600 text-white"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-300 mb-2 block">
                    Structure Value
                  </label>
                  <Input
                    value={newStructureValue}
                    onChange={(e) => setNewStructureValue(e.target.value)}
                    placeholder="e.g., Financial Report Template"
                    className="bg-gray-800 border-gray-600 text-white"
                  />
                </div>
              </div>
              <Button
                onClick={addNewStructure}
                className="bg-emerald-600 hover:bg-emerald-700"
              >
                <Plus className="w-4 h-4 mr-2" />
                Add Structure
              </Button>
            </CardContent>
          </Card>
        )}

        {/* Report Structures List */}
        <div className="space-y-4">
          {editableStructures.length === 0 ? (
            <Card className="bg-gray-900/50 border-gray-400/30">
              <CardContent className="pt-12 pb-12 text-center">
                <div className="w-16 h-16 bg-gray-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
                  <FileText className="w-8 h-8 text-gray-400" />
                </div>
                <h3 className="text-white text-lg font-medium mb-2">
                  No Report Structures Found
                </h3>
                <p className="text-gray-400">
                  {isEditing 
                    ? "Add your first report structure using the form above."
                    : "Report structures will appear here once they are configured."
                  }
                </p>
              </CardContent>
            </Card>
          ) : (
            editableStructures.map((structure, index) => (
              <Card key={index} className="bg-gray-900/50 border-gray-400/30">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Badge variant="default" className="bg-emerald-600">
                        {structure.key}
                      </Badge>
                      {structure.isEditing ? (
                        <Input
                          value={structure.key}
                          onChange={(e) => updateStructure(index, 'key', e.target.value)}
                          className="bg-gray-800 border-gray-600 text-white"
                          placeholder="Structure key"
                        />
                      ) : (
                        <span className="text-white font-medium">
                          {structure.key.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase())}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      {isEditing && (
                        <>
                          {structure.isEditing ? (
                            <>
                              <Button
                                size="sm"
                                onClick={() => saveStructure(index)}
                                className="bg-emerald-600 hover:bg-emerald-700"
                              >
                                <CheckCircle className="w-4 h-4" />
                              </Button>
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => cancelEditing(index)}
                                className="border-gray-600 text-gray-300 hover:bg-gray-700"
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
                                className="border-gray-600 text-gray-300 hover:bg-gray-700"
                              >
                                <Edit3 className="w-4 h-4" />
                              </Button>
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => deleteStructure(index)}
                                className="border-red-600 text-red-300 hover:bg-red-900/20"
                              >
                                <Trash2 className="w-4 h-4" />
                              </Button>
                            </>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                </CardHeader>
                
                <CardContent>
                  <div className="space-y-3">
                    <label className="text-sm font-medium text-gray-300">
                      Structure Content:
                    </label>
                    {structure.isEditing ? (
                      <Textarea
                        value={structure.value}
                        onChange={(e) => updateStructure(index, 'value', e.target.value)}
                        className="bg-gray-800 border-gray-600 text-white min-h-[120px]"
                        placeholder="Enter structure content"
                      />
                    ) : (
                      <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700">
                        {showPreview ? (
                          <pre className="text-gray-300 text-sm whitespace-pre-wrap">
                            {structure.value}
                          </pre>
                        ) : (
                          <div className="text-gray-400 text-sm">
                            {structure.value.length > 100 
                              ? `${structure.value.substring(0, 100)}...` 
                              : structure.value
                            }
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </div>
      </div>
    </div>
  );
});

ReportStructureTab.displayName = 'ReportStructureTab';
