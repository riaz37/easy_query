import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Search, Play, Save, Loader2 } from "lucide-react";

interface QueryInputFormProps {
  title: string;
  icon: React.ReactNode;
  placeholder: string;
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onClear: () => void;
  onSave?: () => void;
  isLoading?: boolean;
  isDisabled?: boolean;
  rows?: number;
  showSaveButton?: boolean;
  submitButtonText?: string;
  clearButtonText?: string;
  hasErrors?: boolean;
  hasWarnings?: boolean;
}

export function QueryInputForm({
  title,
  icon,
  placeholder,
  value,
  onChange,
  onSubmit,
  onClear,
  onSave,
  isLoading = false,
  isDisabled = false,
  rows = 4,
  showSaveButton = false,
  submitButtonText = "Execute Query",
  clearButtonText = "Clear",
  hasErrors = false,
  hasWarnings = false,
}: QueryInputFormProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          {icon}
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="query-input">Enter your query</Label>
          <Textarea
            id="query-input"
            placeholder={placeholder}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            rows={rows}
            className={`resize-none font-mono text-sm ${
              hasErrors 
                ? 'border-red-500 focus:border-red-500 focus:ring-red-500' 
                : hasWarnings 
                ? 'border-yellow-500 focus:border-yellow-500 focus:ring-yellow-500'
                : ''
            }`}
            disabled={isDisabled}
          />
        </div>
        
        <div className="flex gap-2">
          <Button
            onClick={onSubmit}
            disabled={isDisabled || !value?.trim() || isLoading}
            className="flex-1"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Executing...
              </>
            ) : (
              <>
                {icon}
                {submitButtonText}
              </>
            )}
          </Button>
          <Button
            variant="outline"
            onClick={onClear}
            disabled={!value?.trim()}
          >
            {clearButtonText}
          </Button>
          {showSaveButton && onSave && (
            <Button
              variant="outline"
              onClick={onSave}
              disabled={!value?.trim()}
            >
              <Save className="h-4 w-4 mr-2" />
              Save
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
} 