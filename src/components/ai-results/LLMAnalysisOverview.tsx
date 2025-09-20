"use client";

import React from "react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain, TrendingUp, Lightbulb, Target, CheckCircle } from "lucide-react";

interface LLMAnalysisOverviewProps {
  reportResults: any;
}

export function LLMAnalysisOverview({
  reportResults,
}: LLMAnalysisOverviewProps) {
  if (!reportResults.results || !reportResults.results.some((section: any) => section.graph_and_analysis?.llm_analysis)) {
    return (
      <div className="query-content-gradient p-6 rounded-lg shadow-lg hover:shadow-xl transition-all duration-200">
        <div className="mb-4">
          <h2 className="text-xl text-white flex items-center gap-3 mb-2">
            <Brain className="h-5 w-5 text-emerald-400" />
            AI Analysis Overview
          </h2>
          <p className="text-emerald-200 text-sm">
            High-level insights and analysis from your report
          </p>
        </div>
        <div className="pt-12 pb-12 text-center">
          <div className="w-16 h-16 bg-gray-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <Brain className="w-8 h-8 text-gray-400" />
          </div>
          <h3 className="text-white text-lg font-medium mb-2">
            No AI Analysis Available
          </h3>
          <p className="text-gray-400">
            This report doesn't contain any AI-generated analysis yet.
            Generate a new report with AI analysis enabled to see insights here.
          </p>
        </div>
      </div>
    );
  }

  const analysisSections = reportResults.results.filter((section: any) => section.graph_and_analysis?.llm_analysis);
  
  // Extract high-level insights from all analyses
  const totalInsights = analysisSections.reduce((total: number, section: any) => {
    const analysis = section.graph_and_analysis?.llm_analysis?.analysis || "";
    const insightsCount = (analysis.match(/KEY INSIGHTS:/g) || []).length;
    const trendsCount = (analysis.match(/TRENDS AND PATTERNS:/g) || []).length;
    const implicationsCount = (analysis.match(/BUSINESS IMPLICATIONS:/g) || []).length;
    const recommendationsCount = (analysis.match(/RECOMMENDATIONS:/g) || []).length;
    return total + insightsCount + trendsCount + implicationsCount + recommendationsCount;
  }, 0);

  // Get unique analysis subjects
  const analysisSubjects = [...new Set(
    analysisSections
      .map((section: any) => section.graph_and_analysis?.llm_analysis?.analysis_subject)
      .filter(Boolean)
  )];

  return (
    <div className="query-content-gradient p-6 rounded-lg shadow-lg hover:shadow-xl transition-all duration-200">
      <div className="mb-4">
        <h2 className="text-xl text-white flex items-center gap-3 mb-2">
          <Brain className="h-5 w-5 text-emerald-400" />
          AI Analysis Overview
        </h2>
        <p className="text-emerald-200 text-sm">
          High-level insights and analysis from your report
        </p>
      </div>
      <div className="space-y-6">
        {/* Analysis Summary Card */}
        <div className="bg-gradient-to-r from-emerald-900/30 to-emerald-800/30 p-6 rounded-lg border border-emerald-400/30">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-xl font-semibold text-white mb-2">
                AI-Powered Insights Summary
              </h3>
              <p className="text-sm text-gray-300">
                {analysisSections.length} comprehensive AI analyses completed
              </p>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold text-emerald-400">
                {analysisSections.length}
              </div>
              <div className="text-xs text-gray-400">Analyses</div>
            </div>
          </div>
        </div>

        {/* Quick Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-gradient-to-r from-emerald-900/30 to-emerald-800/30 p-4 rounded-lg border border-emerald-400/30">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-emerald-500/20 rounded-full flex items-center justify-center">
                <Lightbulb className="w-5 h-5 text-emerald-400" />
              </div>
              <div>
                <div className="text-2xl font-bold text-emerald-400">
                  {analysisSections.filter((section: any) => 
                    section.graph_and_analysis?.llm_analysis?.analysis.includes("KEY INSIGHTS:")
                  ).length}
                </div>
                <div className="text-xs text-gray-400">Key Insights</div>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-emerald-900/30 to-emerald-800/30 p-4 rounded-lg border border-emerald-400/30">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-emerald-500/20 rounded-full flex items-center justify-center">
                <TrendingUp className="w-5 h-5 text-emerald-400" />
              </div>
              <div>
                <div className="text-2xl font-bold text-emerald-400">
                  {analysisSections.filter((section: any) => 
                    section.graph_and_analysis?.llm_analysis?.analysis.includes("TRENDS AND PATTERNS:")
                  ).length}
                </div>
                <div className="text-xs text-gray-400">Trends Found</div>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-emerald-900/30 to-emerald-800/30 p-4 rounded-lg border border-emerald-400/30">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-emerald-500/20 rounded-full flex items-center justify-center">
                <Target className="w-5 h-5 text-emerald-400" />
              </div>
              <div>
                <div className="text-2xl font-bold text-emerald-400">
                  {analysisSections.filter((section: any) => 
                    section.graph_and_analysis?.llm_analysis?.analysis.includes("BUSINESS IMPLICATIONS:")
                  ).length}
                </div>
                <div className="text-xs text-gray-400">Business Implications</div>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-green-900/30 to-green-800/30 p-4 rounded-lg border border-green-400/30">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-green-500/20 rounded-full flex items-center justify-center">
                <CheckCircle className="w-5 h-5 text-green-400" />
              </div>
              <div>
                <div className="text-2xl font-bold text-green-400">
                  {analysisSections.filter((section: any) => 
                    section.graph_and_analysis?.llm_analysis?.analysis.includes("RECOMMENDATIONS:")
                  ).length}
                </div>
                <div className="text-xs text-gray-400">Recommendations</div>
              </div>
            </div>
          </div>
        </div>

        {/* Analysis Subjects */}
        {analysisSubjects.length > 0 && (
          <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700">
            <h4 className="text-white font-medium mb-3">Analysis Subjects</h4>
            <div className="flex flex-wrap gap-2">
              {analysisSubjects.map((subject: string, index: number) => (
                <Badge key={index} variant="secondary" className="bg-emerald-500/20 text-emerald-300 border-emerald-400/30">
                  {subject}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* Navigation Hint */}
        <div className="bg-gradient-to-r from-emerald-900/20 to-green-900/20 p-4 rounded-lg border border-emerald-400/30">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-emerald-500/20 rounded-full flex items-center justify-center">
              <span className="text-emerald-400 text-sm">💡</span>
            </div>
            <div className="text-sm text-emerald-300">
              <span className="font-medium">Tip:</span> Expand individual report sections below to view detailed AI analysis, key insights, trends, and strategic recommendations for each query.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}