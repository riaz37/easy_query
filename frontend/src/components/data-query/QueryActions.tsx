import { Button } from "@/components/ui/button";
import { Download, Copy, Save, Share2, Bookmark, Play, Trash2 } from "lucide-react";
import { toast } from "sonner";

interface QueryAction {
  label: string;
  icon: React.ReactNode;
  onClick: () => void;
  variant?: "default" | "outline" | "secondary" | "ghost";
  size?: "default" | "sm" | "lg";
  disabled?: boolean;
}

interface QueryActionsProps {
  // Legacy support for actions array
  actions?: QueryAction[];
  
  // New props for database query actions
  onExecute?: () => void;
  onClear?: () => void;
  onSave?: () => void;
  isExecuting?: boolean;
  hasQuery?: boolean;
  hasResults?: boolean;
  isDisabled?: boolean;
  
  className?: string;
  layout?: "horizontal" | "vertical";
}

export function QueryActions({ 
  actions,
  onExecute,
  onClear,
  onSave,
  isExecuting = false,
  hasQuery = false,
  hasResults = false,
  isDisabled = false,
  className = "",
  layout = "horizontal"
}: QueryActionsProps) {
  // Create actions array from props if not provided
  const queryActions: QueryAction[] = actions || [
    {
      label: isExecuting ? "Executing..." : "Execute Query",
      icon: <Play className="h-4 w-4" />,
      onClick: onExecute || (() => {}),
      variant: "default",
      size: "default",
      disabled: isDisabled || !hasQuery || isExecuting
    },
    {
      label: "Clear Results",
      icon: <Trash2 className="h-4 w-4" />,
      onClick: onClear || (() => {}),
      variant: "outline",
      size: "sm",
      disabled: !hasResults || isExecuting
    },
    {
      label: "Save Query",
      icon: <Save className="h-4 w-4" />,
      onClick: onSave || (() => {}),
      variant: "outline",
      size: "sm",
      disabled: !hasQuery || isExecuting
    }
  ];

  const containerClass = layout === "vertical" 
    ? "flex flex-col gap-2" 
    : "flex gap-2";

  return (
    <div className={`${containerClass} ${className}`}>
      {queryActions.map((action, index) => (
        <Button
          key={index}
          variant={action.variant || "outline"}
          size={action.size || "sm"}
          onClick={action.onClick}
          disabled={action.disabled}
          className="flex items-center gap-2"
        >
          {action.icon}
          {action.label}
        </Button>
      ))}
    </div>
  );
}

// Predefined action creators for common use cases
export const createDownloadAction = (data: any, filename: string) => ({
  label: "Download",
  icon: <Download className="h-4 w-4" />,
  onClick: () => {
    const dataStr = JSON.stringify(data, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
    toast.success("Results downloaded successfully");
  }
});

export const createCopyAction = (data: any) => ({
  label: "Copy",
  icon: <Copy className="h-4 w-4" />,
  onClick: () => {
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    toast.success("Results copied to clipboard");
  }
});

export const createSaveAction = (onSave: () => void) => ({
  label: "Save",
  icon: <Save className="h-4 w-4" />,
  onClick: onSave
});

export const createShareAction = (onShare: () => void) => ({
  label: "Share",
  icon: <Share2 className="h-4 w-4" />,
  onClick: onShare
});

export const createBookmarkAction = (onBookmark: () => void) => ({
  label: "Bookmark",
  icon: <Bookmark className="h-4 w-4" />,
  onClick: onBookmark
}); 