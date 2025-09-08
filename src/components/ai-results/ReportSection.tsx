"use client";

import React, { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { FileText, Brain, ChevronDown, ChevronUp } from "lucide-react";
import { EnhancedResultsTable } from "./EnhancedResultsTable";
import { DynamicGraph } from "./DynamicGraph";

interface ReportSectionProps {
  section: any;
  index: number;
  expandedAnalysis: Set<number>;
  toggleAnalysis: (index: number) => void;
}

export function ReportSection({
  section,
  index,
  expandedAnalysis,
  toggleAnalysis,
}: ReportSectionProps) {
  const [showGraph, setShowGraph] = useState(true);

  return (
    <div className="card-enhanced">
      <div className="card-content-enhanced">
        <div className="card-header-enhanced">
          <div className="card-title-enhanced flex items-center gap-2">
            <FileText className="w-5 h-5 text-emerald-400" />
            Section {section.section_number}: {section.section_name}
          </div>
        </div>
        <div className="mt-4 space-y-4">
        <div className="flex items-center gap-2">
          <Badge variant={section.success ? "default" : "destructive"}>
            {section.success ? "Success" : "Failed"}
          </Badge>
          <span className="text-gray-400 text-sm">
            Query {section.query_number}
          </span>
        </div>

        {/* Query Text */}
        <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700">
          <div className="text-sm text-gray-300 mb-2 font-medium">Query:</div>
          <div className="text-white text-sm">{section.query}</div>
        </div>

        {/* Graph and Analysis Info */}
        {section.graph_and_analysis && (
          <div className="bg-emerald-900/20 p-4 rounded-lg border border-emerald-400/30">
            <div className="text-sm text-emerald-300 mb-2 font-medium">
              Generated Graph:
            </div>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-400">Type:</span>
                <span className="text-white ml-2">
                  {section.graph_and_analysis.graph_type}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Theme:</span>
                <span className="text-white ml-2">
                  {section.graph_and_analysis.theme}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Analysis Info */}
        {section.analysis && (
          <div className="bg-emerald-900/20 p-4 rounded-lg border border-emerald-400/30">
            <div className="text-sm text-emerald-300 mb-2 font-medium">
              Analysis:
            </div>
            <div className="text-white text-sm">
              {typeof section.analysis === "string"
                ? section.analysis
                : JSON.stringify(section.analysis, null, 2)}
            </div>
          </div>
        )}

        {/* LLM Analysis */}
        {section.graph_and_analysis?.llm_analysis && (
          <div className="bg-gradient-to-r from-emerald-900/30 to-emerald-800/30 p-4 rounded-lg border border-emerald-400/30">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-emerald-500/20 rounded-full flex items-center justify-center">
                  <span className="text-emerald-400 text-sm">🤖</span>
                </div>
                <div className="text-lg text-emerald-300 font-semibold">
                  AI-Powered Analysis
                </div>
              </div>

              <Button
                onClick={() => toggleAnalysis(index)}
                variant="ghost"
                size="sm"
                className="text-emerald-400 hover:text-emerald-300 hover:bg-emerald-400/10"
              >
                {expandedAnalysis.has(index) ? "Collapse" : "Expand"} Analysis
              </Button>
            </div>

            {expandedAnalysis.has(index) ? (
              <div className="space-y-4">
                {/* Executive Summary */}
                <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700">
                  <div className="text-sm text-emerald-200 mb-2 font-medium">
                    Executive Summary:
                  </div>
                  <div className="text-white text-sm leading-relaxed whitespace-pre-line">
                    {
                      section.graph_and_analysis?.llm_analysis?.analysis.split(
                        "\n\n"
                      )[0]
                    }
                  </div>
                </div>

                {/* Key Insights */}
                {section.graph_and_analysis?.llm_analysis?.analysis.includes(
                  "KEY INSIGHTS:"
                ) && (
                  <div className="bg-emerald-900/30 p-4 rounded-lg border border-emerald-400/30">
                    <div className="text-sm text-emerald-200 mb-2 font-medium">
                      Key Insights:
                    </div>
                    <div className="text-white text-sm leading-relaxed">
                      {(() => {
                        const insightsMatch =
                          section.graph_and_analysis?.llm_analysis?.analysis.match(
                            /KEY INSIGHTS:(.*?)(?=TRENDS AND PATTERNS:|ANOMALIES:|BUSINESS IMPLICATIONS:|RECOMMENDATIONS:|$)/s
                          );
                        if (insightsMatch) {
                          return insightsMatch[1]
                            .trim()
                            .split("\n")
                            .map((insight, i) => (
                              <div
                                key={i}
                                className="flex items-start gap-2 mb-2"
                              >
                                <span className="text-emerald-400 text-xs mt-1">
                                  •
                                </span>
                                <span>
                                  {insight.replace(/^•\s*/, "").trim()}
                                </span>
                              </div>
                            ));
                        }
                        return null;
                      })()}
                    </div>
                  </div>
                )}

                {/* Trends and Patterns */}
                {section.graph_and_analysis?.llm_analysis?.analysis.includes(
                  "TRENDS AND PATTERNS:"
                ) && (
                  <div className="bg-emerald-900/30 p-4 rounded-lg border border-emerald-400/30">
                    <div className="text-sm text-emerald-200 mb-2 font-medium">
                      Trends & Patterns:
                    </div>
                    <div className="text-white text-sm leading-relaxed">
                      {(() => {
                        const trendsMatch =
                          section.graph_and_analysis?.llm_analysis?.analysis.match(
                            /TRENDS AND PATTERNS:(.*?)(?=ANOMALIES:|BUSINESS IMPLICATIONS:|RECOMMENDATIONS:|$)/s
                          );
                        if (trendsMatch) {
                          return trendsMatch[1]
                            .trim()
                            .split("\n")
                            .map((trend, i) => (
                              <div
                                key={i}
                                className="flex items-start gap-2 mb-2"
                              >
                                <span className="text-emerald-400 text-xs mt-1">
                                  •
                                </span>
                                <span>{trend.replace(/^•\s*/, "").trim()}</span>
                              </div>
                            ));
                        }
                        return null;
                      })()}
                    </div>
                  </div>
                )}

                {/* Business Implications */}
                {section.graph_and_analysis?.llm_analysis?.analysis.includes(
                  "BUSINESS IMPLICATIONS:"
                ) && (
                  <div className="bg-emerald-900/30 p-4 rounded-lg border border-emerald-400/30">
                    <div className="text-sm text-emerald-200 mb-2 font-medium">
                      Business Implications:
                    </div>
                    <div className="text-white text-sm leading-relaxed">
                      {(() => {
                        const implicationsMatch =
                          section.graph_and_analysis?.llm_analysis?.analysis.match(
                            /BUSINESS IMPLICATIONS:(.*?)(?=RECOMMENDATIONS:|$)/s
                          );
                        if (implicationsMatch) {
                          return implicationsMatch[1]
                            .trim()
                            .split("\n")
                            .map((implication, i) => (
                              <div
                                key={i}
                                className="flex items-start gap-2 mb-2"
                              >
                                <span className="text-emerald-400 text-xs mt-1">
                                  •
                                </span>
                                <span>
                                  {implication.replace(/^•\s*/, "").trim()}
                                </span>
                              </div>
                            ));
                        }
                        return null;
                      })()}
                    </div>
                  </div>
                )}

                {/* Recommendations */}
                {section.graph_and_analysis?.llm_analysis?.analysis.includes(
                  "RECOMMENDATIONS:"
                ) && (
                  <div className="bg-green-900/30 p-4 rounded-lg border border-green-400/30">
                    <div className="text-sm text-green-200 mb-2 font-medium">
                      Strategic Recommendations:
                    </div>
                    <div className="text-white text-sm leading-relaxed">
                      {(() => {
                        const recommendationsMatch =
                          section.graph_and_analysis?.llm_analysis?.analysis.match(
                            /RECOMMENDATIONS:(.*?)$/s
                          );
                        if (recommendationsMatch) {
                          return recommendationsMatch[1]
                            .trim()
                            .split("\n")
                            .map((recommendation, i) => (
                              <div
                                key={i}
                                className="flex items-start gap-2 mb-2"
                              >
                                <span className="text-green-400 text-xs mt-1">
                                  •
                                </span>
                                <span>
                                  {recommendation.replace(/^•\s*/, "").trim()}
                                </span>
                              </div>
                            ));
                        }
                        return null;
                      })()}
                    </div>
                  </div>
                )}

                {/* Analysis Metadata */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4 border-t border-gray-700/50">
                  {section.graph_and_analysis?.llm_analysis
                    ?.analysis_subject && (
                    <div className="bg-gray-800/50 p-3 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">
                        Analysis Subject
                      </div>
                      <div className="text-white text-sm font-medium">
                        {
                          section.graph_and_analysis.llm_analysis
                            .analysis_subject
                        }
                      </div>
                    </div>
                  )}
                  {section.graph_and_analysis?.llm_analysis?.data_coverage && (
                    <div className="bg-gray-800/50 p-3 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">
                        Data Coverage
                      </div>
                      <div className="text-white text-sm font-medium">
                        {section.graph_and_analysis.llm_analysis.data_coverage}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-3 p-3 bg-emerald-900/10 rounded-lg border border-emerald-400/20">
                <span className="text-emerald-400 text-sm">
                  Click to expand AI analysis...
                </span>
              </div>
            )}
          </div>
        )}

        {/* Enhanced Data Table with Pagination */}
        {section.table &&
          section.table.data &&
          section.table.data.length > 0 && (
            <div className="space-y-4">
              <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700">
                <div className="text-sm text-gray-300 mb-3 font-medium">
                  Data Preview ({section.table.total_rows} rows,{" "}
                  {section.table.columns.length} columns):
                </div>
              </div>

              {/* Use Enhanced Table Component */}
              <EnhancedResultsTable
                data={section.table.data}
                columns={section.table.columns}
              />
            </div>
          )}

        {/* Graph Visualization */}
        {section.graph_and_analysis && section.table && section.table.data && (
          <div className="mt-6">
            <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700">
              <div className="flex items-center justify-between">
                <div className="text-sm text-gray-300 font-medium">
                  Data Visualization:
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowGraph(!showGraph)}
                  className="text-gray-400 hover:text-white"
                >
                  {showGraph ? <ChevronUp /> : <ChevronDown />}
                </Button>
              </div>
            </div>

            {showGraph && (
              <DynamicGraph
                graphData={section.graph_and_analysis}
                tableData={section.table.data}
                columns={section.table.columns}
              />
            )}
          </div>
        )}
        </div>
      </div>
    </div>
  );
}
