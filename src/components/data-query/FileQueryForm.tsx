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
  useIntentReranker: boolean;
  useChunkReranker: boolean;
  useDualEmbeddings: boolean;
  intentTopK: number;
  chunkTopK: number;
  maxChunksForAnswer: number;
  answerStyle: 'concise' | 'detailed';
  tableSpecific: boolean;
}

const defaultOptions: QueryOptions = {
  useIntentReranker: false,
  useChunkReranker: false,
  useDualEmbeddings: true,
  intentTopK: 20,
  chunkTopK: 40,
  maxChunksForAnswer: 40,
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
  const [showAdvanced, setShowAdvanced] = useState(false);

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

        {/* Advanced Options */}
        {showAdvanced && (
          <div className="space-y-4 p-4 bg-gray-800/50 rounded-lg border border-gray-600/30">
            <Label className="text-white font-medium">Advanced Query Options</Label>
            
            <div className="grid grid-cols-2 gap-4">
              {/* Reranker Options */}
              <div className="space-y-3">
                <Label className="text-sm text-gray-300">Reranker Settings</Label>
                <div className="space-y-2">
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={options.useIntentReranker}
                      onChange={(e) => setOptions(prev => ({ ...prev, useIntentReranker: e.target.checked }))}
                      className="rounded bg-gray-800 border-gray-600"
                    />
                    <span className="text-sm text-gray-300">Use Intent Reranker</span>
                  </label>
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={options.useChunkReranker}
                      onChange={(e) => setOptions(prev => ({ ...prev, useChunkReranker: e.target.checked }))}
                      className="rounded bg-gray-800 border-gray-600"
                    />
                    <span className="text-sm text-gray-300">Use Chunk Reranker</span>
                  </label>
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={options.useDualEmbeddings}
                      onChange={(e) => setOptions(prev => ({ ...prev, useDualEmbeddings: e.target.checked }))}
                      className="rounded bg-gray-800 border-gray-600"
                    />
                    <span className="text-sm text-gray-300">Use Dual Embeddings</span>
                  </label>
                </div>
              </div>

              {/* Numeric Parameters */}
              <div className="space-y-3">
                <Label className="text-sm text-gray-300">Retrieval Parameters</Label>
                <div className="grid grid-cols-3 gap-2">
                  <div>
                    <Label className="text-xs text-gray-400">Intent Top K</Label>
                    <input
                      type="number"
                      value={options.intentTopK}
                      onChange={(e) => setOptions(prev => ({ ...prev, intentTopK: Number(e.target.value) }))}
                      className="w-full px-2 py-1 text-sm border rounded bg-gray-800 border-gray-600 text-white"
                      min="1"
                      max="100"
                    />
                  </div>
                  <div>
                    <Label className="text-xs text-gray-400">Chunk Top K</Label>
                    <input
                      type="number"
                      value={options.chunkTopK}
                      onChange={(e) => setOptions(prev => ({ ...prev, chunkTopK: Number(e.target.value) }))}
                      className="w-full px-2 py-1 text-sm border rounded bg-gray-800 border-gray-600 text-white"
                      min="1"
                      max="100"
                    />
                  </div>
                  <div>
                    <Label className="text-xs text-gray-400">Max Chunks</Label>
                    <input
                      type="number"
                      value={options.maxChunksForAnswer}
                      onChange={(e) => setOptions(prev => ({ ...prev, maxChunksForAnswer: Number(e.target.value) }))}
                      className="w-full px-2 py-1 text-sm border rounded bg-gray-800 border-gray-600 text-white"
                      min="1"
                      max="100"
                    />
                  </div>
                </div>
              </div>

              <div className="col-span-2 space-y-2">
                <Label className="text-sm font-medium text-gray-300">Answer Style</Label>
                <div className="flex gap-2">
                  <label className="flex items-center space-x-2">
                    <input
                      type="radio"
                      name="answerStyle"
                      value="concise"
                      checked={options.answerStyle === 'concise'}
                      onChange={(e) => setOptions(prev => ({ ...prev, answerStyle: e.target.value as 'concise' | 'detailed' }))}
                      className="rounded bg-gray-800 border-gray-600"
                    />
                    <span className="text-sm text-gray-300">Concise</span>
                  </label>
                  <label className="flex items-center space-x-2">
                    <input
                      type="radio"
                      name="answerStyle"
                      value="detailed"
                      checked={options.answerStyle === 'detailed'}
                      onChange={(e) => setOptions(prev => ({ ...prev, answerStyle: e.target.value as 'concise' | 'detailed' }))}
                      className="rounded bg-gray-800 border-gray-600"
                    />
                    <span className="text-sm text-gray-300">Detailed</span>
                  </label>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Show/Hide Advanced Options Toggle */}
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="border-purple-400/30 text-purple-400 hover:bg-purple-400/10"
        >
          {showAdvanced ? 'Hide Advanced Options' : 'Show Advanced Options'}
        </Button>

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
              className="border-blue-400/30 text-blue-400 hover:bg-blue-400/10"
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