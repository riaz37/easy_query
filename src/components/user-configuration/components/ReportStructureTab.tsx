"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Plus, RefreshCw, AlertCircle, X } from "lucide-react";
import { useUserContext } from "@/lib/hooks/use-user-context";
import { ReportStructure } from "@/types/reports";
import { Spinner } from "@/components/ui/loading";
import { toast } from "sonner";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Card, CardContent } from "@/components/ui/card";

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

export const ReportStructureTab = React.memo<ReportStructureTabProps>(
  ({
    className,
    reportStructure: reportStructureString = "",
    reportStructureLoading = false,
    reportStructureError = null,
    onRefresh,
  }) => {
    const { user } = useUserContext();

    const [editableStructures, setEditableStructures] = useState<
      EditableStructure[]
    >([]);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [newStructureKey, setNewStructureKey] = useState("");
    const [newStructureValue, setNewStructureValue] = useState("");
    const [isAddModalOpen, setIsAddModalOpen] = useState(false);

    // Initialize editable structures from loaded structure
    useEffect(() => {
      if (reportStructureString) {
        try {
          const parsedStructure = JSON.parse(reportStructureString);
          const structures = Object.entries(parsedStructure).map(
            ([key, value]) => ({
              key,
              value: String(value),
              isEditing: false,
              originalValue: String(value),
            })
          );
          setEditableStructures(structures);
        } catch (error) {
          console.error("Failed to parse report structure:", error);
          setEditableStructures([]);
        }
      } else {
        setEditableStructures([]);
      }
    }, [reportStructureString]);

    const startEditing = useCallback((index: number) => {
      setEditableStructures((prev) =>
        prev.map((structure, i) =>
          i === index
            ? { ...structure, isEditing: true, originalValue: structure.value }
            : structure
        )
      );
    }, []);

    const cancelEditing = useCallback((index: number) => {
      setEditableStructures((prev) =>
        prev.map((structure, i) =>
          i === index && structure.originalValue !== undefined
            ? {
                ...structure,
                value: structure.originalValue,
                isEditing: false,
                originalValue: undefined,
              }
            : structure
        )
      );
    }, []);

    const updateStructure = useCallback(
      (index: number, field: "key" | "value", value: string) => {
        setEditableStructures((prev) =>
          prev.map((structure, i) =>
            i === index ? { ...structure, [field]: value } : structure
          )
        );
      },
      []
    );

    const saveStructure = useCallback((index: number) => {
      setEditableStructures((prev) =>
        prev.map((structure, i) =>
          i === index
            ? { ...structure, isEditing: false, originalValue: undefined }
            : structure
        )
      );
    }, []);

    const deleteStructure = useCallback((index: number) => {
      setEditableStructures((prev) => prev.filter((_, i) => i !== index));
    }, []);

    const addNewStructure = useCallback(() => {
      if (!newStructureKey.trim() || !newStructureValue.trim()) {
        toast.error("Please enter both key and value for the new structure");
        return;
      }

      // Check if key already exists
      if (editableStructures.some((s) => s.key === newStructureKey.trim())) {
        toast.error("A structure with this key already exists");
        return;
      }

      const newStructure: EditableStructure = {
        key: newStructureKey.trim(),
        value: newStructureValue.trim(),
        isEditing: false,
        originalValue: undefined,
      };

      setEditableStructures((prev) => [...prev, newStructure]);
      setNewStructureKey("");
      setNewStructureValue("");
      setIsAddModalOpen(false);
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
        editableStructures.forEach((structure) => {
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

        // Refresh the data
        if (onRefresh) {
          onRefresh();
        }
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "Failed to save changes";
        setError(errorMessage);
        toast.error(errorMessage);
      } finally {
        setSaving(false);
      }
    }, [user?.user_id, editableStructures, onRefresh]);

    const hasUnsavedChanges = useCallback(() => {
      return editableStructures.some(
        (structure) =>
          structure.isEditing ||
          (structure.originalValue &&
            structure.value !== structure.originalValue)
      );
    }, [editableStructures]);

    const resetToOriginal = useCallback(() => {
      if (reportStructureString) {
        try {
          const parsedStructure = JSON.parse(reportStructureString);
          const structures = Object.entries(parsedStructure).map(
            ([key, value]) => ({
              key,
              value: String(value),
              isEditing: false,
              originalValue: String(value),
            })
          );
          setEditableStructures(structures);
          toast.info("Changes reset to original");
        } catch (error) {
          console.error("Failed to parse report structure for reset:", error);
          setEditableStructures([]);
        }
      }
    }, [reportStructureString]);

    if (reportStructureLoading) {
      return (
        <div className="flex items-center justify-center py-12">
          <Spinner size="lg" variant="primary" />
          <span className="ml-4 text-gray-400">
            Loading report structure...
          </span>
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
            <Button onClick={onRefresh} className="bg-red-600 hover:bg-red-700">
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
                  View and edit report structures
                </p>
              </div>
              <div className="flex gap-2">
                <Button
                  onClick={onRefresh}
                  variant="outline"
                  size="icon"
                  className="border-0 text-white hover:bg-white/10 cursor-pointer"
                  style={{
                    background:
                      "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
                    borderRadius: "118.8px",
                    width: "48px",
                    height: "48px",
                  }}
                >
                  <RefreshCw className="w-6 h-6" />
                </Button>

                <Button
                  variant="outline"
                  size="icon"
                  className="border-0 text-white hover:bg-white/10 cursor-pointer"
                  style={{
                    background:
                      "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
                    borderRadius: "118.8px",
                    width: "48px",
                    height: "48px",
                  }}
                  onClick={() => setIsAddModalOpen(true)}
                >
                  <img
                    src="/user-configuration/edit.svg"
                    alt="Add Structure"
                    className="w-6 h-6"
                  />
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Report Structures List */}
        <div className="space-y-6 mt-6">
          {editableStructures.length === 0 ? (
            <div className="query-content-gradient rounded-[32px] p-6">
              <div className="text-center py-12">
                <div className="w-16 h-16 bg-gray-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
                  <AlertCircle className="w-8 h-8 text-gray-400" />
                </div>
                <h3 className="text-white text-lg font-medium mb-2">
                  No Report Structures Found
                </h3>
                <p className="text-gray-400">
                  Report structures will appear here once they are configured.
                  Click the add button to create your first structure.
                </p>
              </div>
            </div>
          ) : (
            editableStructures.map((structure, index) => (
              <div
                key={index}
                className="query-content-gradient rounded-[32px] p-6"
              >
                <div className="space-y-4">
                  {/* Report Header */}
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-lg font-semibold text-white">
                        {structure.isEditing ? (
                          <Input
                            value={structure.key}
                            onChange={(e) =>
                              updateStructure(index, "key", e.target.value)
                            }
                            className="modal-input-enhanced text-lg font-semibold"
                            placeholder="Report name"
                          />
                        ) : (
                          structure.key
                            .replace(/_/g, " ")
                            .replace(/\b\w/g, (l) => l.toUpperCase())
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
                            onClick={() => saveStructure(index)}
                            className="modal-button-primary"
                          >
                            Save
                          </Button>
                          <Button
                            onClick={() => cancelEditing(index)}
                            className="modal-button-secondary"
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
                              background:
                                "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
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
                              background:
                                "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
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
                    <div className="query-content-gradient rounded-[16px] overflow-hidden">
                      {structure.isEditing ? (
                        <Textarea
                          value={structure.value}
                          onChange={(e) =>
                            updateStructure(
                              index,
                              "value",
                              e.target.value
                            )
                          }
                          className="modal-input-enhanced min-h-[200px] max-h-[600px] resize-y border-0 bg-transparent focus:ring-0 focus:ring-offset-0 rounded-[16px]"
                          placeholder="Enter SQL business rules for this report structure"
                        />
                      ) : structure.value ? (
                        <div className="p-4 max-h-[600px] overflow-y-auto">
                          <pre className="text-white whitespace-pre-wrap text-sm font-mono">
                            {structure.value}
                          </pre>
                        </div>
                      ) : (
                        <div className="p-4 max-h-[600px] overflow-y-auto">
                          <p className="text-gray-400 italic">
                            No business rules configured for this report
                            structure.
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Add New Structure Modal */}
        <Dialog open={isAddModalOpen} onOpenChange={setIsAddModalOpen}>
          <DialogContent
            className="p-0 border-0 bg-transparent"
            showCloseButton={false}
          >
            <div className="modal-enhanced">
              <div className="modal-content-enhanced">
                <DialogHeader className="modal-header-enhanced px-8 pt-6 pb-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <DialogTitle className="modal-title-enhanced flex items-center gap-3 text-xl">
                        Add New Structure
                      </DialogTitle>
                      <DialogDescription className="modal-description-enhanced text-sm">
                        Add User Add New Structure
                      </DialogDescription>
                    </div>
                    <button
                      onClick={() => setIsAddModalOpen(false)}
                      className="modal-close-button cursor-pointer"
                    >
                      <X className="h-5 w-5" />
                    </button>
                  </div>
                </DialogHeader>

                <div className="px-8 pb-6 space-y-4">
                  <div>
                    <Label className="modal-label-enhanced">
                      Structure Key
                    </Label>
                    <Input
                      value={newStructureKey}
                      onChange={(e) =>
                        setNewStructureKey(e.target.value)
                      }
                      placeholder="--"
                      className="modal-input-enhanced mt-2"
                    />
                  </div>
                  <div>
                    <Label className="modal-label-enhanced">
                      Structure Value
                    </Label>
                    <Input
                      value={newStructureValue}
                      onChange={(e) =>
                        setNewStructureValue(e.target.value)
                      }
                      placeholder="Easy Query"
                      className="modal-input-enhanced mt-2"
                    />
                  </div>

                  <div className="flex gap-3 pt-4">
                    <Button
                      onClick={() => setIsAddModalOpen(false)}
                      className="modal-button-secondary flex-1"
                    >
                      Cancel
                    </Button>
                    <Button
                      onClick={addNewStructure}
                      className="modal-button-primary flex-1"
                    >
                      Add
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>
    );
  }
);

ReportStructureTab.displayName = "ReportStructureTab";