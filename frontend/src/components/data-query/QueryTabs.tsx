import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ReactNode } from "react";

interface TabItem {
  value: string;
  label: string;
  content: ReactNode;
  icon?: ReactNode;
}

interface QueryTabsProps {
  tabs: TabItem[];
  defaultValue?: string;
  onValueChange?: (value: string) => void;
  className?: string;
}

export function QueryTabs({ 
  tabs, 
  defaultValue, 
  onValueChange,
  className = ""
}: QueryTabsProps) {
  const defaultTab = defaultValue || tabs[0]?.value;

  return (
    <Tabs 
      value={defaultTab} 
      onValueChange={onValueChange} 
      className={`space-y-6 ${className}`}
    >
      <TabsList className="grid w-full grid-cols-1 md:grid-cols-2 lg:grid-cols-4">
        {tabs.map((tab) => (
          <TabsTrigger key={tab.value} value={tab.value} className="flex items-center gap-2">
            {tab.icon}
            {tab.label}
          </TabsTrigger>
        ))}
      </TabsList>
      
      {tabs.map((tab) => (
        <TabsContent key={tab.value} value={tab.value}>
          {tab.content}
        </TabsContent>
      ))}
    </Tabs>
  );
} 