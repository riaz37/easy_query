"use client";

import React, { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { FileText, Brain } from "lucide-react";
import { QueryResultsTable } from "@/components/database-query/QueryResultsTable";
import { DynamicGraph } from "./DynamicGraph";

interface ReportSectionProps {
  section: any;
  index: number;
  expandedSections: Set<number>;
  toggleSection: (index: number) => void;
}

export function ReportSection({
  section,
  index,
  expandedSections,
  toggleSection,
}: ReportSectionProps) {
  const isExpanded = expandedSections.has(index);

  return (
    <div className="query-content-gradient p-6 rounded-lg shadow-lg hover:shadow-xl transition-all duration-200">
      <div className="mb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button 
              onClick={() => toggleSection(index)}
              className="w-8 h-8 bg-gray-600/30 rounded-full flex items-center justify-center hover:bg-gray-500/40 transition-colors"
            >
              <img src="/ai-results/plus.svg" alt="Plus" className="w-4 h-4" />
            </button>
            <h2 className="text-xl text-white flex items-center gap-3 mb-2">
              <FileText className="h-5 w-5 text-emerald-400" />
              Section {section.section_number}: {section.section_name}
            </h2>
          </div>
        </div>
      </div>
      
      {isExpanded && (
        <div className="space-y-4">
          <div>
            <div className="text-sm text-gray-300 mb-2 font-medium">Query:</div>
            <div className="text-white text-sm">{section.query}</div>
          </div>


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
            <div className="space-y-4">
              {/* Executive Summary */}
              <div 
                className="p-4"
                style={{
                  background: "var(--primary-8, rgba(19, 245, 132, 0.08))",
                  border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
                  borderRadius: "16px"
                }}
              >
                <div 
                  className="text-sm mb-2 font-medium"
                  style={{ color: "var(--text-secondary, rgba(223, 227, 232, 1))" }}
                >
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
                <div 
                  className="p-4"
                  style={{
                    background: "var(--primary-8, rgba(19, 245, 132, 0.08))",
                    border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
                    borderRadius: "16px"
                  }}
                >
                  <div 
                    className="text-sm mb-2 font-medium"
                    style={{ color: "var(--text-secondary, rgba(223, 227, 232, 1))" }}
                  >
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
                              <span>{insight.trim()}</span>
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
                <div 
                  className="p-4"
                  style={{
                    background: "var(--primary-8, rgba(19, 245, 132, 0.08))",
                    border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
                    borderRadius: "16px"
                  }}
                >
                  <div 
                    className="text-sm mb-2 font-medium"
                    style={{ color: "var(--text-secondary, rgba(223, 227, 232, 1))" }}
                  >
                    Trends and Patterns:
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
                              <span className="text-blue-400 text-xs mt-1">
                                •
                              </span>
                              <span>{trend.trim()}</span>
                            </div>
                          ));
                      }
                      return null;
                    })()}
                  </div>
                </div>
              )}

              {/* Anomalies */}
              {section.graph_and_analysis?.llm_analysis?.analysis.includes(
                "ANOMALIES:"
              ) && (
                <div 
                  className="p-4"
                  style={{
                    background: "var(--primary-8, rgba(19, 245, 132, 0.08))",
                    border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
                    borderRadius: "16px"
                  }}
                >
                  <div 
                    className="text-sm mb-2 font-medium"
                    style={{ color: "var(--text-secondary, rgba(223, 227, 232, 1))" }}
                  >
                    Anomalies:
                  </div>
                  <div className="text-white text-sm leading-relaxed">
                    {(() => {
                      const anomaliesMatch =
                        section.graph_and_analysis?.llm_analysis?.analysis.match(
                          /ANOMALIES:(.*?)(?=BUSINESS IMPLICATIONS:|RECOMMENDATIONS:|$)/s
                        );
                      if (anomaliesMatch) {
                        return anomaliesMatch[1]
                          .trim()
                          .split("\n")
                          .map((anomaly, i) => (
                            <div
                              key={i}
                              className="flex items-start gap-2 mb-2"
                            >
                              <span className="text-orange-400 text-xs mt-1">
                                •
                              </span>
                              <span>{anomaly.trim()}</span>
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
                <div 
                  className="p-4"
                  style={{
                    background: "var(--primary-8, rgba(19, 245, 132, 0.08))",
                    border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
                    borderRadius: "16px"
                  }}
                >
                  <div 
                    className="text-sm mb-2 font-medium"
                    style={{ color: "var(--text-secondary, rgba(223, 227, 232, 1))" }}
                  >
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
                              <span className="text-purple-400 text-xs mt-1">
                                •
                              </span>
                              <span>{implication.trim()}</span>
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
                <div 
                  className="p-4"
                  style={{
                    background: "var(--primary-8, rgba(19, 245, 132, 0.08))",
                    border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
                    borderRadius: "16px"
                  }}
                >
                  <div 
                    className="text-sm mb-2 font-medium"
                    style={{ color: "var(--text-secondary, rgba(223, 227, 232, 1))" }}
                  >
                    Recommendations:
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
                              <span>{recommendation.trim()}</span>
                            </div>
                          ));
                      }
                      return null;
                    })()}
                  </div>
                </div>
              )}

            </div>
          )}

          {/* Data Visualization */}
          {section.graph_and_analysis && section.table && section.table.data && (
            <div className="mb-6">
              <DynamicGraph
                graphData={section.graph_and_analysis}
                tableData={section.table.data}
                columns={section.table.columns}
              />
            </div>
          )}

          {/* Data Table - Matching Database Query Results */}
          {section.table &&
            section.table.data &&
            section.table.data.length > 0 && (
              <div>
                <h3 className="modal-title-enhanced text-lg font-semibold mb-4">
                  Data Table
                </h3>
                <QueryResultsTable
                  data={section.table.data}
                  columns={section.table.columns}
                />
              </div>
            )}
        </div>
      )}
    </div>
  );
}