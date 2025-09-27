import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LucideIcon, BarChart3, Clock, CheckCircle, AlertTriangle, FileText } from "lucide-react";

interface StatCard {
  title: string;
  value: string | number;
  description: string;
  icon: LucideIcon;
  trend?: {
    value: number;
    isPositive: boolean;
  };
}

interface QueryStatsCardsProps {
  // New interface for stats array
  stats?: StatCard[];
  columns?: 2 | 3 | 4 | 5;
  
  // Legacy props for backward compatibility
  totalQueries?: number;
  isExecuting?: boolean;
  hasResults?: boolean;
  resultCount?: number;
  executionTime?: number;
  hasWarnings?: boolean;
}

export function QueryStatsCards({ 
  stats, 
  columns = 5,
  // Legacy props
  totalQueries,
  isExecuting,
  hasResults,
  resultCount,
  executionTime,
  hasWarnings
}: QueryStatsCardsProps) {
  
  // Generate stats from legacy props if stats array is not provided
  const getStatsFromLegacyProps = (): StatCard[] => {
    const legacyStats: StatCard[] = [];
    
    if (totalQueries !== undefined) {
      legacyStats.push({
        title: "Total Queries",
        value: totalQueries,
        description: "Total queries executed",
        icon: BarChart3
      });
    }
    
    if (resultCount !== undefined) {
      legacyStats.push({
        title: "Results",
        value: resultCount,
        description: "Query results found",
        icon: FileText
      });
    }
    
    if (executionTime !== undefined) {
      legacyStats.push({
        title: "Execution Time",
        value: `${executionTime}ms`,
        description: "Query execution time",
        icon: Clock
      });
    }
    
    if (hasResults !== undefined) {
      legacyStats.push({
        title: "Status",
        value: hasResults ? "Success" : "No Results",
        description: "Query execution status",
        icon: hasResults ? CheckCircle : AlertTriangle
      });
    }
    
    if (isExecuting !== undefined) {
      legacyStats.push({
        title: "Status",
        value: isExecuting ? "Executing..." : "Ready",
        description: "Current execution status",
        icon: isExecuting ? Clock : CheckCircle
      });
    }
    
    if (hasWarnings !== undefined) {
      legacyStats.push({
        title: "Warnings",
        value: hasWarnings ? "Yes" : "No",
        description: "Query warnings detected",
        icon: hasWarnings ? AlertTriangle : CheckCircle
      });
    }
    
    return legacyStats;
  };
  
  // Use provided stats or generate from legacy props
  const finalStats = stats || getStatsFromLegacyProps();
  
  // If no stats available, return null
  if (finalStats.length === 0) {
    return null;
  }
  
  const getGridCols = () => {
    switch (columns) {
      case 2: return "grid-cols-1 md:grid-cols-2";
      case 3: return "grid-cols-1 md:grid-cols-2 lg:grid-cols-3";
      case 4: return "grid-cols-1 md:grid-cols-2 lg:grid-cols-4";
      case 5: return "grid-cols-1 md:grid-cols-2 lg:grid-cols-5";
      default: return "grid-cols-1 md:grid-cols-2 lg:grid-cols-5";
    }
  };

  return (
    <div className={`grid ${getGridCols()} gap-6`}>
      {finalStats.map((stat, index) => (
        <Card key={index}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">{stat.title}</CardTitle>
            <stat.icon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stat.value}</div>
            <p className="text-xs text-muted-foreground">
              {stat.description}
            </p>
            {stat.trend && (
              <div className="flex items-center mt-2">
                <span className={`text-xs ${
                  stat.trend.isPositive ? 'text-green-600' : 'text-red-600'
                }`}>
                  {stat.trend.isPositive ? '+' : ''}{stat.trend.value}%
                </span>
                <span className="text-xs text-muted-foreground ml-1">from last month</span>
              </div>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  );
} 