import React, { useState } from 'react';
// Card components removed - now handled by parent component
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { FileText, Play, Save, Sparkles } from 'lucide-react';
import { ButtonLoader } from '@/components/ui/loading';
import { toast } from 'sonner';

interface FileQueryFormProps {
  onSubmit: (query: string, options: QueryOptions) => Promise<void>;
  onSave?: (query: string) => void;
  onClear?: () => void;
  isLoading?: boolean;
  disabled?: boolean;
  className?: string;
}

export interface QueryOptions {
  answerStyle: 'concise' | 'detailed';
  tableSpecific: boolean;
}

const defaultOptions: QueryOptions = {
  answerStyle: 'detailed',
  tableSpecific: false,
};

const querySuggestions = [
  "What is the main topic of this document?",
  "Summarize the key points",
  "Extract all dates mentioned",
  "Find all monetary amounts",
  "What are the main conclusions?",
  "List all people mentioned",
  "What are the key recommendations?",
  "Extract contact information",
];

export function FileQueryForm({
  onSubmit,
  onSave,
  onClear,
  isLoading = false,
  disabled = false,
  className = ""
}: FileQueryFormProps) {
  const [query, setQuery] = useState('');
  const [options, setOptions] = useState<QueryOptions>(defaultOptions);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!query.trim()) {
      toast.error('Please enter a query');
      return;
    }

    try {
      await onSubmit(query.trim(), options);
    } catch (error) {
      console.error('Query submission error:', error);
    }
  };

  const handleSave = () => {
    if (onSave && query.trim()) {
      onSave(query.trim());
    }
  };

  const handleClear = () => {
    setQuery('');
    if (onClear) {
      onClear();
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion);
  };

  return (
    <div className={className}>
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Query Input */}
        <div className="space-y-2">
          <Label htmlFor="file-query" className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Query your files
          </Label>
          <Textarea
            id="file-query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question about your uploaded files..."
            className="min-h-[100px] bg-gray-800/50 border-gray-600/30 text-white placeholder:text-gray-400 resize-none"
            disabled={disabled || isLoading}
            data-element="file-query-input"
          />
        </div>

        {/* Quick Suggestions */}
        <div className="space-y-2">
          <Label className="text-white">Quick suggestions:</Label>
          <div className="flex flex-wrap gap-2">
            {querySuggestions.map((suggestion, index) => (
              <Button
                key={index}
                type="button"
                variant="outline"
                size="sm"
                onClick={() => setQuery(suggestion)}
                disabled={disabled || isLoading}
                className="text-xs border-gray-600/30 text-gray-300 hover:bg-gray-700/50"
              >
                <Sparkles className="w-3 h-3 mr-1" />
                {suggestion}
              </Button>
            ))}
          </div>
        </div>

        


        {/* Action Buttons */}
        <div className="flex gap-2">
          <ButtonLoader
            type="submit"
            disabled={disabled || isLoading || !query.trim()}
            loading={isLoading}
            text="Processing..."
            size="md"
            variant="success"
            className="flex-1"
            data-element="file-query-submit"
          >
            <Play className="w-4 h-4 mr-2" />
            Execute Query
          </ButtonLoader>
          
          {onSave && (
            <Button
              type="button"
              variant="outline"
              onClick={handleSave}
              disabled={!query.trim()}
              className="border-emerald-400/30 text-emerald-400 hover:bg-emerald-400/10"
            >
              <Save className="w-4 h-4 mr-2" />
              Save
            </Button>
          )}
          
          <Button
            type="button"
            variant="outline"
            onClick={handleClear}
            disabled={!query.trim()}
            className="border-red-400/30 text-red-400 hover:bg-red-400/10"
          >
            Clear
          </Button>
        </div>
      </form>
    </div>
  );
} 