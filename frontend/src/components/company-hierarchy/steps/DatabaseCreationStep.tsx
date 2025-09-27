"use client";

import { useEffect, useState } from "react";
import { TaskProgress } from "@/components/ui/task-progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";

interface DatabaseCreationStepProps {
  currentTaskId: string | null;
  onTaskComplete: (success: boolean, result?: any) => void;
}

export function DatabaseCreationStep({
  currentTaskId,
  onTaskComplete,
}: DatabaseCreationStepProps) {
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!currentTaskId) {
      setError("No task ID provided for database creation");
    } else {
      setError(null);
    }
  }, [currentTaskId]);

  if (!currentTaskId) {
    return (
      <div className="space-y-4">
        <h3 className="text-lg font-medium text-red-400">Database Creation Error</h3>
        <Alert className="border-red-400/30 bg-red-900/20">
          <AlertCircle className="h-4 w-4 text-red-400" />
          <AlertDescription className="text-red-300">
            {error || "Failed to start database creation. Please go back and try again."}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-medium text-green-400">Creating Database</h3>
      <TaskProgress
        taskId={currentTaskId}
        onTaskComplete={onTaskComplete}
        title="Database Configuration"
        description="Setting up your database configuration and processing any uploaded files..."
        showCancelButton={false}
      />
    </div>
  );
}