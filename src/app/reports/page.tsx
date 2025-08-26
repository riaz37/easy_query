import React from 'react';
import { ReportGenerator } from '@/components/reports';

export default function ReportsPage() {
  return (
    <div className="container mx-auto py-8 px-4">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Report Generation</h1>
        <p className="text-gray-600">
          Generate comprehensive reports using AI-powered query analysis and data processing.
        </p>
      </div>
      
      <ReportGenerator 
        configId={1}
        onReportComplete={(results) => {
          console.log('Report completed:', results);
        }}
      />
    </div>
  );
} 