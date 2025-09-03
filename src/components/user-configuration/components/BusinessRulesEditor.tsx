import React from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { FileText } from 'lucide-react';
import type { BusinessRulesEditorProps } from '../types';

export const BusinessRulesEditor = React.memo<BusinessRulesEditorProps>(({
  currentDatabaseId,
  businessRules,
  editorState,
  onContentChange,
}) => {
  if (!currentDatabaseId) {
    return null;
  }

  return (
    <Card className="bg-slate-800/50 border-slate-700">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <FileText className="h-5 w-5 text-blue-400" />
          Business Rules Editor
        </CardTitle>
        <CardDescription className="text-gray-400">
          {editorState.isEditing
            ? 'Edit business rules for the current database'
            : 'View and edit business rules'}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {editorState.isEditing ? (
          <div className="space-y-4">
            <div>
              <Label
                htmlFor="businessRules"
                className="text-gray-400"
              >
                Business Rules Content
              </Label>
              <Textarea
                id="businessRules"
                value={editorState.editedContent}
                onChange={(e) => onContentChange(e.target.value)}
                placeholder="Enter your business rules here..."
                className="mt-2 bg-slate-700/50 border-slate-600 text-white min-h-[200px] resize-y"
              />
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
                      .split('\n')
                      .filter((line) => line.trim().length > 0).length
                  }{' '}
                  rules
                </span>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div>
              <Label className="text-gray-400">
                Current Business Rules
              </Label>
              <div className="mt-2 p-4 bg-slate-700/50 border border-slate-600 rounded-md min-h-[200px] max-h-[400px] overflow-y-auto">
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
              <div className="flex justify-between text-sm text-gray-400 mt-2">
                <span>
                  {businessRules.content?.length || 0} characters
                </span>
                <span>
                  {businessRules.content
                    ? businessRules.content
                        .split('\n')
                        .filter((line) => line.trim().length > 0).length
                    : 0}{' '}
                  rules
                </span>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
});

BusinessRulesEditor.displayName = 'BusinessRulesEditor';
