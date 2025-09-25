import React from "react";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { FileText } from "lucide-react";
import type { BusinessRulesEditorProps } from "../types";

export const BusinessRulesEditor = React.memo<BusinessRulesEditorProps>(
  ({
    currentDatabaseId,
    businessRules,
    editorState,
    onContentChange,
    onEdit,
    onSave,
    onCancel,
    onReset,
  }) => {
    if (!currentDatabaseId) {
      return null;
    }

    return (
      <div className="query-content-gradient rounded-[32px] p-6">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold text-white flex items-center gap-2">
                Business Rules Editor
              </h2>
              <p className="text-gray-400 text-sm">
                {editorState.isEditing
                  ? "Edit business rules for the current database"
                  : "View and edit business rules"}
              </p>
            </div>
            <div className="flex gap-2">
              {!editorState.isEditing ? (
                <Button
                  onClick={onEdit}
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
                  <img
                    src="/user-configuration/reportedit.svg"
                    alt="Edit"
                    className="w-5 h-5"
                  />
                </Button>
              ) : (
                <div className="flex gap-2">
                  <Button
                    onClick={onSave}
                    disabled={!editorState.hasUnsavedChanges}
                    className="modal-button-primary"
                  >
                    Save
                  </Button>
                  <Button
                    onClick={onCancel}
                    className="modal-button-secondary"
                  >
                    Cancel
                  </Button>
                </div>
              )}
            </div>
          </div>
          {editorState.isEditing ? (
            <div className="space-y-4">
              <div>
                <Label htmlFor="businessRules" className="text-gray-400">
                  Business Rules Content
                </Label>
                <div className="mt-2 query-content-gradient rounded-[16px] overflow-hidden">
                  <Textarea
                    id="businessRules"
                    value={editorState.editedContent}
                    onChange={(e) => onContentChange(e.target.value)}
                    placeholder="Enter your business rules here..."
                    className="modal-input-enhanced min-h-[400px] max-h-[600px] resize-y border-0 bg-transparent focus:ring-0 focus:ring-offset-0 rounded-[16px]"
                  />
                </div>
                {editorState.contentError && (
                  <p className="text-red-400 text-sm mt-2">
                    {editorState.contentError}
                  </p>
                )}
                <div className="flex justify-between text-sm text-gray-400 mt-2">
                  <span>{editorState.editedContent.length} characters</span>
                  <span>
                    {
                      editorState.editedContent
                        .split("\n")
                        .filter((line) => line.trim().length > 0).length
                    }{" "}
                    rules
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div>
                <Label className="text-gray-400">Current Business Rules</Label>
                 <div className="mt-2 query-content-gradient rounded-[16px] overflow-hidden">
                   <div className="p-4 max-h-[600px] overflow-y-auto">
                     {businessRules.content ? (
                       <pre className="text-white whitespace-pre-wrap text-sm font-mono">
                         {businessRules.content}
                       </pre>
                     ) : (
                       <p className="text-gray-400 italic">
                         No business rules configured for this database.
                       </p>
                     )}
                   </div>
                 </div>
                <div className="flex justify-between text-sm text-gray-400 mt-2">
                  <span>{businessRules.content?.length || 0} characters</span>
                  <span>
                    {businessRules.content
                      ? businessRules.content
                          .split("\n")
                          .filter((line) => line.trim().length > 0).length
                      : 0}{" "}
                    rules
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }
);

BusinessRulesEditor.displayName = "BusinessRulesEditor";
