"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { 
  Plus, 
  RefreshCw,
  AlertCircle
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
      <div className="query-content-gradient rounded-[32px] p-6">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold text-white flex items-center gap-2">
                Report Structure
              </h2>
              <p className="text-gray-400 text-sm">
                {isEditing
                  ? "Edit report structures for the current database"
                  : "View and edit report structures"}
              </p>
            </div>
            <div className="flex gap-2">
              {!isEditing ? (
                <Button
                  onClick={onRefresh}
                  variant="outline"
                  size="icon"
                  className="border-0 text-white hover:bg-white/10 cursor-pointer"
                  style={{
                    background: "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
                    borderRadius: "118.8px",
                    width: "48px",
                    height: "48px",
                  }}
                >
                  <RefreshCw className="w-6 h-6" />
                </Button>
              ) : (
                <div className="flex gap-2">
                  <Button
                    onClick={saveAllChanges}
                    disabled={saving || !hasUnsavedChanges()}
                    className="bg-emerald-600 hover:bg-emerald-700"
                  >
                    {saving ? "Saving..." : "Save Changes"}
                  </Button>
                  <Button
                    onClick={() => {
                      setIsEditing(false);
                      resetToOriginal();
                    }}
                    variant="outline"
                    className="border-gray-600 text-gray-300 hover:bg-gray-700"
                  >
                    Cancel
                  </Button>
                </div>
              )}
              <Button
                onClick={() => setIsEditing(true)}
                variant="outline"
                size="icon"
                className="border-0 text-white hover:bg-white/10 cursor-pointer"
                style={{
                  background: "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
                  borderRadius: "118.8px",
                  width: "48px",
                  height: "48px",
                }}
              >
                <img src="/user-configuration/edit.svg" alt="Edit" className="w-6 h-6" />
              </Button>
            </div>
          </div>

          {/* Add New Structure Form */}
          {isEditing && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-white">Add New Report Structure</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label className="text-gray-400">Structure Key</Label>
                  <Input
                    value={newStructureKey}
                    onChange={(e) => setNewStructureKey(e.target.value)}
                    placeholder="e.g., financial_report, sales_analysis"
                    className="modal-input-enhanced mt-2"
                  />
                </div>
                <div>
                  <Label className="text-gray-400">Structure Value</Label>
                  <Input
                    value={newStructureValue}
                    onChange={(e) => setNewStructureValue(e.target.value)}
                    placeholder="e.g., Financial Report Template"
                    className="modal-input-enhanced mt-2"
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
            </div>
          )}

          {/* Report Structures List */}
          <div className="space-y-6">
            {editableStructures.length === 0 ? (
              <div className="text-center py-12">
                <div className="w-16 h-16 bg-gray-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
                  <AlertCircle className="w-8 h-8 text-gray-400" />
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
              </div>
            ) : (
              editableStructures.map((structure, index) => (
                <div key={index} className="query-content-gradient rounded-[16px] p-6">
                  <div className="space-y-4">
                    {/* Report Header */}
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="text-lg font-semibold text-white">
                          {structure.isEditing ? (
                            <Input
                              value={structure.key}
                              onChange={(e) => updateStructure(index, 'key', e.target.value)}
                              className="modal-input-enhanced text-lg font-semibold"
                              placeholder="Report name"
                            />
                          ) : (
                            structure.key.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase())
                          )}
                        </h3>
                        <p className="text-gray-400 text-sm">
                          View and edit business rules
                        </p>
                      </div>
                      <div className="flex items-center gap-2">
                        {structure.isEditing ? (
                          <>
                            <Button
                              size="sm"
                              onClick={() => saveStructure(index)}
                              className="bg-emerald-600 hover:bg-emerald-700"
                            >
                              Save
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => cancelEditing(index)}
                              className="border-gray-600 text-gray-300 hover:bg-gray-700"
                            >
                              Cancel
                            </Button>
                          </>
                        ) : (
                          <>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => startEditing(index)}
                              className="border-0 text-white hover:bg-white/10 cursor-pointer"
                              style={{
                                background: "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
                                borderRadius: "118.8px",
                                width: "40px",
                                height: "40px",
                              }}
                            >
                              <img
                                src="/user-configuration/reportedit.svg"
                                alt="Edit"
                                className="w-5 h-5"
                              />
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => deleteStructure(index)}
                              className="border-0 text-white hover:bg-white/10 cursor-pointer"
                              style={{
                                background: "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
                                borderRadius: "118.8px",
                                width: "40px",
                                height: "40px",
                              }}
                            >
                              <img
                                src="/user-configuration/reportdelete.svg"
                                alt="Delete"
                                className="w-5 h-5"
                              />
                            </Button>
                          </>
                        )}
                      </div>
                    </div>
                    
                    {/* Report Content - Same styling as business rules */}
                    <div className="space-y-3">
                      <Label className="text-gray-400">SQL Business Rules</Label>
                      <div className="query-content-gradient rounded-[16px] overflow-hidden">
                        <div className="p-4 max-h-[600px] overflow-y-auto">
                          {structure.isEditing ? (
                            <Textarea
                              value={structure.value}
                              onChange={(e) => updateStructure(index, 'value', e.target.value)}
                              className="modal-input-enhanced min-h-[200px] border-0 bg-transparent focus:ring-0 focus:ring-offset-0 rounded-[16px]"
                              placeholder="Enter SQL business rules for this report structure"
                            />
                          ) : (
                            structure.value ? (
                              <pre className="text-white whitespace-pre-wrap text-sm font-mono">
                                {structure.value}
                              </pre>
                            ) : (
                              <p className="text-gray-400 italic">
                                No business rules configured for this report structure.
                              </p>
                            )
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
});

ReportStructureTab.displayName = 'ReportStructureTab';
