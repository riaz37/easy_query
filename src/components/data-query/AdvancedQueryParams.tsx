import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Settings, ChevronDown, ChevronUp, RotateCcw } from "lucide-react";

// Simple version without props for standalone use
export function AdvancedQueryParams() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [params, setParams] = useState({
    useIntentReranker: false,
    useChunkReranker: false,
    useDualEmbeddings: true,
    intentTopK: 20,
    chunkTopK: 40,
    chunkSource: "files",
    maxChunksForAnswer: 40,
    answerStyle: "detailed",
  });

  const handleReset = () => {
    setParams({
      useIntentReranker: false,
      useChunkReranker: false,
      useDualEmbeddings: true,
      intentTopK: 20,
      chunkTopK: 40,
      chunkSource: "files",
      maxChunksForAnswer: 40,
      answerStyle: "detailed",
    });
  };

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-emerald-400 hover:bg-emerald-400/10 p-0 h-auto"
        >
          <Settings className="w-4 h-4 mr-2" />
          Advanced Parameters
          {isExpanded ? (
            <ChevronUp className="w-4 h-4 ml-2" />
          ) : (
            <ChevronDown className="w-4 h-4 ml-2" />
          )}
        </Button>
        {isExpanded && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleReset}
            className="text-gray-400 hover:text-gray-300 hover:bg-gray-700/50"
          >
            <RotateCcw className="w-3 h-3 mr-1" />
            Reset
          </Button>
        )}
      </div>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="space-y-4 p-4 bg-gray-800/30 rounded-lg border border-gray-600/30">
          {/* Reranking Options */}
          <div className="space-y-3">
            <Label className="text-sm font-medium text-emerald-400">Reranking Options</Label>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="intent-reranker" className="text-sm text-gray-300">
                  Use Intent Reranker
                </Label>
                <Switch
                  id="intent-reranker"
                  checked={params.useIntentReranker}
                  onCheckedChange={(checked) =>
                    setParams(prev => ({ ...prev, useIntentReranker: checked }))
                  }
                />
              </div>
              <div className="flex items-center justify-between">
                <Label htmlFor="chunk-reranker" className="text-sm text-gray-300">
                  Use Chunk Reranker
                </Label>
                <Switch
                  id="chunk-reranker"
                  checked={params.useChunkReranker}
                  onCheckedChange={(checked) =>
                    setParams(prev => ({ ...prev, useChunkReranker: checked }))
                  }
                />
              </div>
              <div className="flex items-center justify-between">
                <Label htmlFor="dual-embeddings" className="text-sm text-gray-300">
                  Use Dual Embeddings
                </Label>
                <Switch
                  id="dual-embeddings"
                  checked={params.useDualEmbeddings}
                  onCheckedChange={(checked) =>
                    setParams(prev => ({ ...prev, useDualEmbeddings: checked }))
                  }
                />
              </div>
            </div>
          </div>

          {/* Retrieval Parameters */}
          <div className="space-y-3">
            <Label className="text-sm font-medium text-emerald-400">Retrieval Parameters</Label>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <Label htmlFor="intent-topk" className="text-xs text-gray-400">
                  Intent Top K
                </Label>
                <Input
                  id="intent-topk"
                  type="number"
                  min="1"
                  max="100"
                  value={params.intentTopK}
                  onChange={(e) =>
                    setParams(prev => ({ ...prev, intentTopK: parseInt(e.target.value) || 20 }))
                  }
                  className="bg-gray-800/50 border-gray-600/30 text-white"
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="chunk-topk" className="text-xs text-gray-400">
                  Chunk Top K
                </Label>
                <Input
                  id="chunk-topk"
                  type="number"
                  min="1"
                  max="100"
                  value={params.chunkTopK}
                  onChange={(e) =>
                    setParams(prev => ({ ...prev, chunkTopK: parseInt(e.target.value) || 40 }))
                  }
                  className="bg-gray-800/50 border-gray-600/30 text-white"
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="max-chunks" className="text-xs text-gray-400">
                  Max Chunks for Answer
                </Label>
                <Input
                  id="max-chunks"
                  type="number"
                  min="1"
                  max="100"
                  value={params.maxChunksForAnswer}
                  onChange={(e) =>
                    setParams(prev => ({ ...prev, maxChunksForAnswer: parseInt(e.target.value) || 40 }))
                  }
                  className="bg-gray-800/50 border-gray-600/30 text-white"
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="answer-style" className="text-xs text-gray-400">
                  Answer Style
                </Label>
                <Select
                  value={params.answerStyle}
                  onValueChange={(value) =>
                    setParams(prev => ({ ...prev, answerStyle: value }))
                  }
                >
                  <SelectTrigger className="bg-gray-800/50 border-gray-600/30 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-gray-800 border-gray-600">
                    <SelectItem value="concise" className="text-white">Concise</SelectItem>
                    <SelectItem value="detailed" className="text-white">Detailed</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          {/* Info Text */}
          <div className="text-xs text-gray-400 space-y-1">
            <p>• Intent reranker improves query understanding</p>
            <p>• Chunk reranker refines content selection</p>
            <p>• Higher Top K values return more candidates</p>
            <p>• Max chunks controls answer comprehensiveness</p>
          </div>
        </div>
      )}
    </div>
  );
} 