"use client";

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  ArrowLeft, 
  FileText, 
  BarChart3, 
  CheckCircle, 
  Database,
  Download,
  Clock,
  Eye,
  FileDown
} from 'lucide-react';
import { ReportResults } from '@/types/reports';
import { generateAndDownloadPDF, generatePDFBlob } from '@/lib/utils/pdf-report-generator';

export default function AIResultsPage() {
  const router = useRouter();
  const [reportResults, setReportResults] = useState<ReportResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [pdfGenerating, setPdfGenerating] = useState(false);

  useEffect(() => {
    // Get results from sessionStorage
    const storedResults = sessionStorage.getItem('reportResults');
    if (storedResults) {
      try {
        const results = JSON.parse(storedResults);
        setReportResults(results);
      } catch (error) {
        console.error('Failed to parse report results:', error);
      }
    }
    setLoading(false);
  }, []);

  const handleBackToQuery = () => {
    router.push('/database-query');
  };

  const handleDownloadPDF = async () => {
    if (!reportResults) return;
    
    setPdfGenerating(true);
    try {
      // Generate and download PDF
      generateAndDownloadPDF(reportResults, `AI_Report_${new Date().toISOString().split('T')[0]}.pdf`);
    } catch (error) {
      console.error('Failed to generate PDF:', error);
      alert('Failed to generate PDF. Please try again.');
    } finally {
      setPdfGenerating(false);
    }
  };

  const handlePreviewPDF = async () => {
    if (!reportResults) return;
    
    setPdfGenerating(true);
    try {
      // Generate PDF blob and open in new tab
      const blob = generatePDFBlob(reportResults);
      const url = URL.createObjectURL(blob);
      window.open(url, '_blank');
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to preview PDF:', error);
      alert('Failed to preview PDF. Please try again.');
    } finally {
      setPdfGenerating(false);
    }
  };

  const handleDownloadText = () => {
    if (!reportResults) return;
    
    // Create a downloadable text report
    const reportText = generateReportText(reportResults);
    const blob = new Blob([reportText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `AI_Report_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const generateReportText = (results: ReportResults): string => {
    let report = 'AI-GENERATED REPORT\n';
    report += '='.repeat(50) + '\n\n';
    report += `Database ID: ${results.database_id}\n`;
    report += `Total Queries: ${results.total_queries}\n`;
    report += `Successful Queries: ${results.successful_queries}\n`;
    report += `Failed Queries: ${results.failed_queries}\n\n`;

    if (results.results) {
      results.results.forEach((section, index) => {
        report += `Section ${section.section_number}: ${section.section_name}\n`;
        report += '-'.repeat(30) + '\n';
        report += `Query ${section.query_number}: ${section.query}\n`;
        report += `Status: ${section.success ? 'Success' : 'Failed'}\n\n`;
      });
    }

    return report;
  };

  if (loading) {
    return (
      <div className="w-full min-h-screen relative bg-gradient-to-br from-slate-900 via-slate-800 to-emerald-900">
        <div className="pt-24 pb-8">
          <div className="container mx-auto px-4">
            <div className="text-center">
              <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-400 mx-auto"></div>
              <p className="text-white mt-4">Loading report results...</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!reportResults) {
    return (
      <div className="w-full min-h-screen relative bg-gradient-to-br from-slate-900 via-slate-800 to-emerald-900">
        <div className="pt-24 pb-8">
          <div className="container mx-auto px-4">
            <div className="max-w-4xl mx-auto">
              <Card className="bg-gray-900/50 border-red-400/30">
                <CardContent className="pt-12 pb-12 text-center">
                  <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
                    <FileText className="w-8 h-8 text-red-400" />
                  </div>
                  <h3 className="text-white text-lg font-medium mb-2">
                    No Report Results Found
                  </h3>
                  <p className="text-gray-400 mb-4">
                    Please generate a report first from the Database Query page
                  </p>
                  <Button onClick={handleBackToQuery} className="bg-blue-600 hover:bg-blue-700">
                    <ArrowLeft className="w-4 h-4 mr-2" />
                    Back to Database Query
                  </Button>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full min-h-screen relative bg-gradient-to-br from-slate-900 via-slate-800 to-emerald-900">
      <div className="pt-24 pb-8">
        <div className="container mx-auto px-4">
          <div className="max-w-6xl mx-auto">
            {/* Header */}
            <div className="mb-8">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-gradient-to-br from-purple-500/30 to-purple-600/20 rounded-xl flex items-center justify-center border border-purple-500/40">
                    <FileText className="w-6 h-6 text-purple-400" />
                  </div>
                  <div>
                    <h1 className="text-3xl font-bold text-white">
                      AI Report Results
                    </h1>
                    <p className="text-gray-400">
                      Comprehensive analysis and insights generated by AI
                    </p>
                  </div>
                </div>
              </div>

              {/* Action Buttons - Prominent PDF Actions */}
              <div className="bg-gradient-to-r from-purple-900/20 to-blue-900/20 p-6 rounded-xl border border-purple-400/30">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-2">Export Report</h3>
                    <p className="text-gray-400 text-sm">
                      Download your AI-generated report in multiple formats
                    </p>
                  </div>
                  <div className="flex gap-3">
                    <Button
                      onClick={handleDownloadPDF}
                      disabled={pdfGenerating}
                      className="bg-green-600 hover:bg-green-700 text-white px-6 py-3"
                    >
                      <Download className="w-4 h-4 mr-2" />
                      {pdfGenerating ? 'Generating PDF...' : 'Download PDF'}
                    </Button>
                    <Button
                      onClick={handlePreviewPDF}
                      disabled={pdfGenerating}
                      variant="outline"
                      className="border-blue-400/30 text-blue-400 hover:bg-blue-400/10 px-6 py-3"
                    >
                      <Eye className="w-4 h-4 mr-2" />
                      Preview PDF
                    </Button>
                    <Button
                      onClick={handleDownloadText}
                      variant="outline"
                      className="border-purple-400/30 text-purple-400 hover:bg-purple-400/10 px-6 py-3"
                    >
                      <FileDown className="w-4 h-4 mr-2" />
                      Download Text
                    </Button>
                  </div>
                </div>
                
                {/* PDF Generation Progress */}
                {pdfGenerating && (
                  <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700">
                    <div className="flex items-center gap-3">
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-green-400"></div>
                      <span className="text-green-400 text-sm">Generating professional PDF report...</span>
                    </div>
                    <p className="text-gray-400 text-xs mt-2">
                      This may take a few moments for complex reports
                    </p>
                  </div>
                )}
              </div>

              {/* Back Button */}
              <div className="flex justify-end mt-4">
                <Button
                  onClick={handleBackToQuery}
                  variant="outline"
                  className="border-blue-400/30 text-blue-400 hover:bg-blue-400/10"
                >
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Back to Database Query
                </Button>
              </div>
                      </div>

            {/* Report Summary */}
            <Card className="bg-gray-900/50 border-purple-400/30 mb-6">
              <CardHeader>
                <CardTitle className="text-purple-400 flex items-center gap-2">
                  <CheckCircle className="w-5 h-5" />
                  Report Summary
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-white">{reportResults.total_queries}</div>
                    <div className="text-sm text-gray-400">Total Queries</div>
                      </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-400">
                      {reportResults.successful_queries}
                    </div>
                    <div className="text-sm text-gray-400">Successful</div>
                      </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-red-400">
                      {reportResults.failed_queries}
                    </div>
                    <div className="text-sm text-gray-400">Failed</div>
                      </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-400">
                      {reportResults.database_id}
                    </div>
                    <div className="text-sm text-gray-400">Database ID</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

            {/* Report Sections */}
            {reportResults.results && reportResults.results.length > 0 && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-white">Report Sections</h2>
                
                {reportResults.results.map((section, index) => (
                  <Card key={index} className="bg-gray-900/50 border-blue-400/30">
                  <CardHeader>
                      <CardTitle className="text-blue-400 flex items-center gap-2">
                        <BarChart3 className="w-5 h-5" />
                        Section {section.section_number}: {section.section_name}
                    </CardTitle>
                  </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="flex items-center gap-2">
                        <Badge variant={section.success ? "default" : "destructive"}>
                          {section.success ? "Success" : "Failed"}
                        </Badge>
                        <span className="text-sm text-gray-400">
                          Query {section.query_number}
                          </span>
                      </div>

                      <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700">
                        <div className="text-sm text-gray-300 mb-2">Query:</div>
                        <div className="text-white">{section.query}</div>
                      </div>

                      {/* Graph Analysis */}
                      {section.graph_and_analysis && (
                        <div className="bg-purple-900/20 p-4 rounded-lg border border-purple-400/30">
                          <div className="text-sm text-purple-300 mb-2 font-medium">Generated Graph:</div>
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                              <span className="text-gray-400">Type:</span>
                              <span className="text-white ml-2">{section.graph_and_analysis.graph_type}</span>
                            </div>
                            <div>
                              <span className="text-gray-400">Theme:</span>
                              <span className="text-white ml-2">{section.graph_and_analysis.theme}</span>
                            </div>
                          </div>
                          <div className="mt-2">
                            <span className="text-gray-400">Path:</span>
                            <div className="text-white text-xs break-all mt-1">
                              {section.graph_and_analysis.image_url}
                                      </div>
                                      </div>
                                    </div>
                            )}

                      {/* Additional Analysis */}
                      {section.analysis && (
                        <div className="bg-green-900/20 p-4 rounded-lg border border-green-400/30">
                          <div className="text-sm text-green-300 mb-2 font-medium">Analysis:</div>
                          <div className="text-white text-sm">
                            {typeof section.analysis === 'string' 
                              ? section.analysis 
                              : JSON.stringify(section.analysis, null, 2)
                            }
                          </div>
                                      </div>
                      )}

                      {/* LLM Analysis */}
                      {section.llm_analysis && (
                        <div className="bg-blue-900/20 p-4 rounded-lg border border-blue-400/30">
                          <div className="text-sm text-blue-300 mb-2 font-medium">AI Insights:</div>
                          
                          {/* Display analysis in a structured way */}
                          {section.llm_analysis.analysis && (
                            <div className="mb-4">
                              <div className="text-xs text-blue-200 mb-2 font-medium">Executive Summary:</div>
                              <div className="text-white text-sm leading-relaxed whitespace-pre-line">
                                {section.llm_analysis.analysis}
                                      </div>
                                    </div>
                          )}
                          
                          {/* Display metadata */}
                          <div className="grid grid-cols-2 gap-4 text-xs">
                            {section.llm_analysis.analysis_subject && (
                              <div>
                                <span className="text-blue-200">Subject:</span>
                                <span className="text-white ml-2">{section.llm_analysis.analysis_subject}</span>
                                </div>
                            )}
                            {section.llm_analysis.data_coverage && (
                              <div>
                                <span className="text-blue-200">Coverage:</span>
                                <span className="text-white ml-2">{section.llm_analysis.data_coverage}</span>
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {/* Data Table Preview */}
                      {section.table && section.table.data && section.table.data.length > 0 && (
                        <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700">
                          <div className="text-sm text-gray-300 mb-3 font-medium">
                            Data Preview ({section.table.total_rows} rows, {section.table.columns.length} columns):
                          </div>
                          
                          {/* Show first few rows */}
                            <div className="overflow-x-auto">
                            <table className="w-full text-xs">
                                <thead>
                                <tr className="border-b border-gray-600">
                                  {section.table.columns.map((col, colIndex) => (
                                    <th key={colIndex} className="text-left p-2 text-gray-300 font-medium">
                                      {col}
                                      </th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                {section.table.data.slice(0, 5).map((row, rowIndex) => (
                                  <tr key={rowIndex} className="border-b border-gray-700">
                                    {section.table.columns.map((col, colIndex) => (
                                      <td key={colIndex} className="p-2 text-white">
                                        {String(row[col] || 'N/A').substring(0, 50)}
                                        {String(row[col] || 'N/A').length > 50 ? '...' : ''}
                                        </td>
                                      ))}
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          
                          {section.table.data.length > 5 && (
                            <div className="text-xs text-gray-400 mt-2 text-center">
                              Showing first 5 rows of {section.table.data.length} total rows
                            </div>
                          )}
                        </div>
                      )}
                  </CardContent>
                </Card>
                ))}
              </div>
            )}

            {/* No Sections Message */}
            {(!reportResults.results || reportResults.results.length === 0) && (
              <Card className="bg-gray-900/50 border-gray-400/30">
                  <CardContent className="pt-12 pb-12 text-center">
                  <div className="w-16 h-16 bg-gray-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
                    <FileText className="w-8 h-8 text-gray-400" />
                    </div>
                    <h3 className="text-white text-lg font-medium mb-2">
                    Report Generated Successfully
                    </h3>
                  <p className="text-gray-400">
                    The report has been generated with {reportResults.total_queries} queries.
                    {reportResults.successful_queries > 0 && (
                      <span className="text-green-400"> {reportResults.successful_queries} queries were successful.</span>
                    )}
                  </p>
                  </CardContent>
                </Card>
            )}

            {/* Processing Details */}
            {reportResults.summary && (
              <Card className="bg-gray-900/50 border-green-400/30 mt-6">
                <CardHeader>
                  <CardTitle className="text-green-400 flex items-center gap-2">
                    <Clock className="w-5 h-5" />
                    Processing Details
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-white">
                        {reportResults.summary.total_sections}
                      </div>
                      <div className="text-sm text-gray-400">Total Sections</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-400">
                        {reportResults.summary.success_rate}%
                      </div>
                      <div className="text-sm text-gray-400">Success Rate</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-400">
                        {reportResults.summary.total_processing_time?.toFixed(2) || 'N/A'}s
                      </div>
                      <div className="text-sm text-gray-400">Total Time</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-400">
                        {reportResults.summary.average_processing_time?.toFixed(2) || 'N/A'}s
                      </div>
                      <div className="text-sm text-gray-400">Avg per Query</div>
                    </div>
                  </div>
                  
                  <Separator className="my-4 bg-gray-700" />
                  
                  <div className="text-center">
                    <div className="text-sm text-gray-400 mb-2">Processing Method:</div>
                    <Badge variant="outline" className="border-green-400/30 text-green-400">
                      {reportResults.summary.processing_method}
                    </Badge>
              </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
