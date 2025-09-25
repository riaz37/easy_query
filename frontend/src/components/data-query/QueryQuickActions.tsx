import { Button } from "@/components/ui/button";
import { LucideIcon } from "lucide-react";

interface QuickAction {
  title: string;
  description: string;
  icon: LucideIcon;
  href: string;
  color: string;
  onClick?: () => void;
}

interface QueryQuickActionsProps {
  actions: QuickAction[];
  columns?: 2 | 3 | 4;
  variant?: 'grid' | 'list';
}

export function QueryQuickActions({ 
  actions, 
  columns = 4, 
  variant = 'grid' 
}: QueryQuickActionsProps) {
  const getGridCols = () => {
    switch (columns) {
      case 2: return "grid-cols-1 md:grid-cols-2";
      case 3: return "grid-cols-1 md:grid-cols-2 lg:grid-cols-3";
      case 4: return "grid-cols-1 md:grid-cols-2 lg:grid-cols-4";
      default: return "grid-cols-1 md:grid-cols-2 lg:grid-cols-4";
    }
  };

  if (variant === 'list') {
    return (
      <div className="space-y-4">
        {actions.map((action, index) => (
          <Button
            key={index}
            variant="outline"
            className="w-full h-auto p-4 flex items-center gap-3 hover:shadow-md transition-shadow"
            onClick={action.onClick}
          >
            <div className={`p-2 rounded-full ${action.color} text-white`}>
              <action.icon className="h-5 w-5" />
            </div>
            <div className="text-left flex-1">
              <div className="font-semibold">{action.title}</div>
              <div className="text-sm text-muted-foreground">
                {action.description}
              </div>
            </div>
          </Button>
        ))}
      </div>
    );
  }

  return (
    <div className={`grid ${getGridCols()} gap-4`}>
      {actions.map((action, index) => (
        <Button
          key={index}
          variant="outline"
          className="h-auto p-4 flex flex-col items-center gap-2 hover:shadow-md transition-shadow"
          onClick={action.onClick}
        >
          <div className={`p-2 rounded-full ${action.color} text-white`}>
            <action.icon className="h-5 w-5" />
          </div>
          <div className="text-center">
            <div className="font-semibold">{action.title}</div>
            <div className="text-xs text-muted-foreground">
              {action.description}
            </div>
          </div>
        </Button>
      ))}
    </div>
  );
} 