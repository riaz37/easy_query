import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import html2canvas from 'html2canvas';
import { ReportResults, ReportSection } from '@/types/reports';

export interface PDFGenerationOptions {
  format?: 'A4' | 'Letter' | 'Legal';
  landscape?: boolean;
  margin?: number;
  fontSize?: number;
  maxRowsPerPage?: number;
  maxColumnsPerPage?: number;
  enableSmartSplitting?: boolean;
}

export class SmartPDFGenerator {
  private doc: jsPDF;
  private currentY: number = 20;
  private pageWidth: number;
  private pageHeight: number;
  private margin: number = 20;
  private useLandscape: boolean = false;
  private fontSize: number = 10;
  private maxRowsPerPage: number = 30;
  private maxColumnsPerPage: number = 8;

  constructor(options?: PDFGenerationOptions) {
    this.useLandscape = options?.landscape || false;
    this.margin = options?.margin || 20;
    this.fontSize = options?.fontSize || 10;
    this.maxRowsPerPage = options?.maxRowsPerPage || 30;
    this.maxColumnsPerPage = options?.maxColumnsPerPage || 8;
    
    this.doc = new jsPDF(this.useLandscape ? 'l' : 'p', 'mm', options?.format || 'A4');
    this.pageWidth = this.doc.internal.pageSize.getWidth();
    this.pageHeight = this.doc.internal.pageSize.getHeight();
  }

  /**
   * Smart PDF generation that automatically handles large datasets
   */
  generateReport(results: ReportResults): jsPDF {
    // Auto-detect if we need landscape mode
    this.detectOptimalOrientation(results);
    
    // Add title page
    this.addTitlePage(results);
    
    // Add executive summary
    this.addExecutiveSummary(results);
    
    // Add detailed results with smart table handling
    if (results.results) {
      results.results.forEach((section, index) => {
        this.addSection(section, index + 1);
      });
    }
    
    // Add summary and metadata
    this.addSummary(results);
    
    return this.doc;
  }

  /**
   * Auto-detect optimal orientation based on data complexity
   */
  private detectOptimalOrientation(results: ReportResults): void {
    const hasWideTables = results.results?.some(section => 
      section.table && section.table.columns && section.table.columns.length > this.maxColumnsPerPage
    ) || false;

    const hasLongTables = results.results?.some(section => 
      section.table && section.table.data && section.table.data.length > this.maxRowsPerPage * 2
    ) || false;

    // Switch to landscape for wide tables or very long tables
    if (hasWideTables || hasLongTables) {
      this.useLandscape = true;
      this.doc = new jsPDF('l', 'mm', 'A4');
      this.pageWidth = this.doc.internal.pageSize.getWidth();
      this.pageHeight = this.doc.internal.pageSize.getHeight();
    }
  }

  /**
   * Add title page
   */
  private addTitlePage(results: ReportResults): void {
    this.doc.setFillColor(41, 128, 185);
    this.doc.rect(0, 0, this.pageWidth, 60, 'F');
    
    // Title
    this.doc.setTextColor(255, 255, 255);
    this.doc.setFontSize(28);
    this.doc.setFont('helvetica', 'bold');
    this.doc.text('AI-GENERATED REPORT', this.pageWidth / 2, 35, { align: 'center' });
    
    // Subtitle
    this.doc.setFontSize(16);
    this.doc.setFont('helvetica', 'normal');
    this.doc.text('Comprehensive Data Analysis', this.pageWidth / 2, 50, { align: 'center' });
    
    // Reset text color
    this.doc.setTextColor(0, 0, 0);
    
    // Report metadata
    this.currentY = 80;
    this.addMetadataRow('Database ID', results.database_id.toString());
    this.addMetadataRow('Total Queries', results.total_queries.toString());
    this.addMetadataRow('Successful Queries', results.successful_queries.toString());
    this.addMetadataRow('Failed Queries', results.failed_queries.toString());
    this.addMetadataRow('Generated On', new Date().toLocaleDateString());
    
    this.currentY += 20;
  }

  /**
   * Add executive summary
   */
  private addExecutiveSummary(results: ReportResults): void {
    this.addPageBreak();
    
    this.doc.setFontSize(20);
    this.doc.setFont('helvetica', 'bold');
    this.doc.text('Executive Summary', this.margin, this.currentY);
    this.currentY += 15;
    
    this.doc.setFontSize(12);
    this.doc.setFont('helvetica', 'normal');
    
    // Add orientation info if landscape
    if (this.useLandscape) {
      this.doc.setFontSize(10);
      this.doc.setFont('helvetica', 'italic');
      this.doc.setTextColor(255, 140, 0);
      this.doc.text('Note: This report uses landscape orientation to accommodate large data tables.', this.margin, this.currentY);
      this.currentY += 8;
      this.doc.setTextColor(0, 0, 0);
    }
    
    const summary = results.summary;
    if (summary) {
      const summaryText = [
        'This AI-generated report provides comprehensive insights into the knowledge base system.',
        '',
        'Key Metrics:',
        `• Total Queries: ${results.total_queries}`,
        `• Success Rate: ${((results.successful_queries / results.total_queries) * 100).toFixed(1)}%`,
        `• Failed Queries: ${results.failed_queries}`,
        '',
        'The analysis covers multiple sections with detailed results for each query,',
        'including data tables, success/failure status, and comprehensive insights.'
      ];
      
      summaryText.forEach(line => {
        if (line.trim() === '') {
          this.currentY += 5;
        } else {
          const wrappedLines = this.wrapText(line, this.pageWidth - (this.margin * 2), 12);
          wrappedLines.forEach(wrappedLine => {
            this.doc.text(wrappedLine, this.margin, this.currentY);
            this.currentY += 6;
          });
        }
      });
    }
    
    this.currentY += 20;
  }

  /**
   * Add detailed section with smart table handling
   */
  private addSection(section: ReportSection, sectionNumber: number): void {
    this.addPageBreak();
    
    // Section header
    this.doc.setFontSize(18);
    this.doc.setFont('helvetica', 'bold');
    this.doc.text(`Section ${sectionNumber}: ${section.section_name}`, this.margin, this.currentY);
    this.currentY += 15;
    
    // Query information
    this.doc.setFontSize(14);
    this.doc.setFont('helvetica', 'bold');
    this.doc.text(`Query ${section.query_number}:`, this.margin, this.currentY);
    this.currentY += 8;
    
    // Query text
    this.doc.setFontSize(12);
    this.doc.setFont('helvetica', 'normal');
    const queryLines = this.wrapText(section.query, this.pageWidth - (this.margin * 2), 12);
    queryLines.forEach(line => {
      this.doc.text(line, this.margin, this.currentY);
      this.currentY += 6;
    });
    
    this.currentY += 10;
    
    // Status
    this.doc.setFontSize(12);
    this.doc.setFont('helvetica', 'bold');
    const statusText = section.success ? 'Status: SUCCESS' : 'Status: FAILED';
    const statusColor = section.success ? [0, 128, 0] : [255, 0, 0];
    this.doc.setTextColor(statusColor[0], statusColor[1], statusColor[2]);
    this.doc.text(statusText, this.margin, this.currentY);
    this.doc.setTextColor(0, 0, 0);
    this.currentY += 10;
    
    // Add table with smart handling
    if (section.table) {
      this.addSmartTable(section.table);
    }
    
    // Add graph visualization if available
    if (section.graph_and_analysis && section.table) {
      this.addGraphVisualization(section.graph_and_analysis, section.table);
    }
    
    this.currentY += 20;
  }

  /**
   * Smart table handling for large datasets
   */
  private addSmartTable(table: any): void {
    this.addPageBreak();
    
    // Table title
    this.doc.setFontSize(14);
    this.doc.setFont('helvetica', 'bold');
    this.doc.text('Data Results:', this.margin, this.currentY);
    this.currentY += 10;
    
    const tableData = table.data || [];
    const tableHeaders = table.columns || [];
    
    // Debug logging
    console.log('PDF Generator - Table Data Structure:', {
      headers: tableHeaders,
      dataLength: tableData.length,
      firstRow: tableData[0],
      dataType: Array.isArray(tableData) ? 'array' : typeof tableData,
      firstRowType: tableData[0] ? (Array.isArray(tableData[0]) ? 'array' : typeof tableData[0]) : 'undefined'
    });
    
    if (tableData.length > 0 && tableHeaders.length > 0) {
      // Check if table needs special handling
      const isWideTable = tableHeaders.length > this.maxColumnsPerPage;
      const isLongTable = tableData.length > this.maxRowsPerPage;
      
      if (isWideTable || isLongTable) {
        this.addComplexTable(tableHeaders, tableData);
      } else {
        this.addSimpleTable(tableHeaders, tableData);
      }
    } else {
      console.warn('PDF Generator - No table data or headers found:', { tableData, tableHeaders });
      // Add a note about no data
      this.doc.setFontSize(10);
      this.doc.setFont('helvetica', 'italic');
      this.doc.setTextColor(255, 140, 0);
      this.doc.text('Note: No data available for this table.', this.margin, this.currentY);
      this.currentY += 8;
      this.doc.setTextColor(0, 0, 0);
    }
  }

  /**
   * Add graph visualization to PDF
   */
  private addGraphVisualization(graphData: any, tableData: any): void {
    this.addPageBreak();
    
    // Graph title
    this.doc.setFontSize(14);
    this.doc.setFont('helvetica', 'bold');
    this.doc.text('Data Visualization:', this.margin, this.currentY);
    this.currentY += 10;
    
    // Graph type and configuration
    this.doc.setFontSize(12);
    this.doc.setFont('helvetica', 'normal');
    this.doc.text(`Graph Type: ${graphData.graph_type}`, this.margin, this.currentY);
    this.currentY += 8;
    
    // Column mapping information
    this.doc.text(`X-Axis: ${graphData.column_mapping.x}`, this.margin, this.currentY);
    this.currentY += 6;
    this.doc.text(`Y-Axis: ${graphData.column_mapping.y}`, this.margin, this.currentY);
    this.currentY += 6;
    
    if (graphData.column_mapping.color) {
      this.doc.text(`Color: ${graphData.column_mapping.color}`, this.margin, this.currentY);
      this.currentY += 6;
    }
    
    this.currentY += 10;
    
    // Add a note about graph data
    this.doc.setFontSize(10);
    this.doc.setFont('helvetica', 'italic');
    this.doc.setTextColor(255, 140, 0);
    this.doc.text('Note: Interactive graph visualization is available in the web interface.', this.margin, this.currentY);
    this.currentY += 8;
    this.doc.setTextColor(0, 0, 0);
    
    // Add sample data summary for the graph
    if (tableData.data && tableData.data.length > 0) {
      this.doc.setFontSize(10);
      this.doc.setFont('helvetica', 'normal');
      this.doc.text('Graph Data Summary:', this.margin, this.currentY);
      this.currentY += 6;
      
      // Show first few data points
      const sampleData = tableData.data.slice(0, 5);
      sampleData.forEach((row: any, index: number) => {
        const xValue = row[graphData.column_mapping.x] || 'N/A';
        const yValue = row[graphData.column_mapping.y] || 'N/A';
        this.doc.text(`${index + 1}. ${graphData.column_mapping.x}: ${xValue}, ${graphData.column_mapping.y}: ${yValue}`, this.margin + 10, this.currentY);
        this.currentY += 5;
      });
      
      if (tableData.data.length > 5) {
        this.doc.text(`... and ${tableData.data.length - 5} more data points`, this.margin + 10, this.currentY);
        this.currentY += 5;
      }
    }
    
    this.currentY += 10;
  }

  /**
   * Handle complex tables (wide or long)
   */
  private addComplexTable(headers: string[], data: any[]): void {
    const isWide = headers.length > this.maxColumnsPerPage;
    
    if (isWide) {
      // Split wide tables into column groups
      this.addWideTable(headers, data);
    } else {
      // Split long tables into page groups
      this.addLongTable(headers, data);
    }
  }

  /**
   * Handle wide tables by splitting columns
   */
  private addWideTable(headers: string[], data: any[]): void {
    this.doc.setFontSize(10);
    this.doc.setFont('helvetica', 'italic');
    this.doc.setTextColor(255, 140, 0);
    this.doc.text(`Note: Table has ${headers.length} columns. Showing data in optimized format.`, this.margin, this.currentY);
    this.currentY += 8;
    this.doc.setTextColor(0, 0, 0);
    
    // Split columns into groups
    const columnGroups = [];
    for (let i = 0; i < headers.length; i += this.maxColumnsPerPage) {
      columnGroups.push(headers.slice(i, i + this.maxColumnsPerPage));
    }
    
    columnGroups.forEach((columns, groupIndex) => {
      if (groupIndex > 0) {
        this.addPageBreak();
      }
      
      this.doc.setFontSize(12);
      this.doc.setFont('helvetica', 'bold');
      this.doc.text(`Data Group ${groupIndex + 1} (Columns ${groupIndex * this.maxColumnsPerPage + 1}-${Math.min((groupIndex + 1) * this.maxColumnsPerPage, headers.length)})`, this.margin, this.currentY);
      this.currentY += 8;
      
      const groupData = data.map((row: any) => 
        columns.map((col: string) => {
          const value = row[col];
          if (value === null || value === undefined) return 'N/A';
          return String(value).length > 25 ? String(value).substring(0, 25) + '...' : String(value);
        })
      );
      
      this.addSimpleTable(columns, groupData);
    });
  }

  /**
   * Handle long tables by splitting pages
   */
  private addLongTable(headers: string[], data: any[]): void {
    this.doc.setFontSize(10);
    this.doc.setFont('helvetica', 'italic');
    this.doc.setTextColor(255, 140, 0);
    this.doc.text(`Note: Table has ${data.length} rows. Showing data across multiple pages.`, this.margin, this.currentY);
    this.currentY += 8;
    this.doc.setTextColor(0, 0, 0);
    
    // Split data into page groups
    for (let i = 0; i < data.length; i += this.maxRowsPerPage) {
      if (i > 0) {
        this.addPageBreak();
      }
      
      const pageData = data.slice(i, i + this.maxRowsPerPage);
      const pageNumber = Math.floor(i / this.maxRowsPerPage) + 1;
      const totalPages = Math.ceil(data.length / this.maxRowsPerPage);
      
      this.doc.setFontSize(10);
      this.doc.setFont('helvetica', 'italic');
      this.doc.text(`Page ${pageNumber} of ${totalPages}`, this.margin, this.currentY);
      this.currentY += 6;
      
      this.addSimpleTable(headers, pageData);
    }
  }

  /**
   * Add simple table using jsPDF autoTable
   */
  private addSimpleTable(headers: string[], data: any[]): void {
    try {
      // Convert object data to array format for autoTable
      const formattedData = data.map(row => {
        if (Array.isArray(row)) {
          // If already an array, use as is
          return row;
        } else if (typeof row === 'object' && row !== null) {
          // Convert object to array of values
          return headers.map(header => {
            const value = row[header];
            if (value === null || value === undefined) return '';
            return String(value);
          });
        } else {
          // Fallback for other data types
          return [String(row)];
        }
      });

      autoTable(this.doc, {
        head: [headers],
        body: formattedData,
        startY: this.currentY,
        margin: { left: this.margin, right: this.margin },
        styles: {
          fontSize: this.fontSize,
          cellPadding: 2,
          overflow: 'linebreak',
          halign: 'left',
          valign: 'middle',
        },
        headStyles: {
          fillColor: [41, 128, 185],
          textColor: 255,
          fontStyle: 'bold',
          fontSize: this.fontSize + 1,
        },
        alternateRowStyles: {
          fillColor: [245, 245, 245],
        },
        didDrawPage: (data) => {
          // Update current Y position after table
          this.currentY = data.cursor.y + 10;
        }
      });
    } catch (error) {
      console.error('Error adding table to PDF:', error);
      // Fallback: add table data as text
      this.addTableAsText(headers, data);
    }
  }

  /**
   * Fallback method to add table as text
   */
  private addTableAsText(headers: string[], data: any[]): void {
    this.doc.setFontSize(this.fontSize);
    this.doc.setFont('helvetica', 'bold');
    
    // Headers
    headers.forEach((header, index) => {
      const x = this.margin + (index * 30);
      if (x < this.pageWidth - this.margin) {
        this.doc.text(header, x, this.currentY);
      }
    });
    this.currentY += 8;
    
    // Convert object data to array format if needed
    const formattedData = data.map(row => {
      if (Array.isArray(row)) {
        return row;
      } else if (typeof row === 'object' && row !== null) {
        return headers.map(header => {
          const value = row[header];
          if (value === null || value === undefined) return '';
          return String(value);
        });
      } else {
        return [String(row)];
      }
    });
    
    // Data rows
    this.doc.setFont('helvetica', 'normal');
    formattedData.slice(0, this.maxRowsPerPage).forEach(row => {
      row.forEach((cell: any, index: number) => {
        const x = this.margin + (index * 30);
        if (x < this.pageWidth - this.margin) {
          const cellText = String(cell || '').substring(0, 20);
          this.doc.text(cellText, x, this.currentY);
        }
      });
      this.currentY += 6;
      
      if (this.currentY > this.pageHeight - 50) {
        this.addPageBreak();
      }
    });
    
    if (formattedData.length > this.maxRowsPerPage) {
      this.currentY += 5;
      this.doc.setFontSize(10);
      this.doc.setFont('helvetica', 'italic');
      this.doc.text(`... and ${formattedData.length - this.maxRowsPerPage} more rows`, this.margin, this.currentY);
      this.currentY += 8;
    }
  }

  /**
   * Add summary section
   */
  private addSummary(results: ReportResults): void {
    this.addPageBreak();
    
    this.doc.setFontSize(18);
    this.doc.setFont('helvetica', 'bold');
    this.doc.text('Summary & Recommendations', this.margin, this.currentY);
    this.currentY += 15;
    
    this.doc.setFontSize(12);
    this.doc.setFont('helvetica', 'normal');
    
    const summaryText = [
      'This AI-generated report provides comprehensive insights into the knowledge base system.',
      '',
      'Key Metrics:',
      `• Total Queries: ${results.total_queries}`,
      `• Success Rate: ${((results.successful_queries / results.total_queries) * 100).toFixed(1)}%`,
      `• Failed Queries: ${results.failed_queries}`,
      '',
      'Recommendations:',
      '• Review failed queries to identify patterns and improve system performance',
      '• Monitor processing times and optimize for better efficiency',
      '• Consider expanding the knowledge base based on successful query patterns',
      '• Implement automated monitoring for system health and performance',
      '',
      'Generated on: ' + new Date().toLocaleDateString() + ' at ' + new Date().toLocaleTimeString()
    ];
    
    summaryText.forEach(line => {
      if (line.trim() === '') {
        this.currentY += 5;
      } else {
        const wrappedLines = this.wrapText(line, this.pageWidth - (this.margin * 2), 12);
        wrappedLines.forEach(wrappedLine => {
          this.doc.text(wrappedLine, this.margin, this.currentY);
          this.currentY += 6;
        });
      }
    });
    
    this.currentY += 20;
    
    // Footer
    this.doc.setFontSize(10);
    this.doc.setFont('helvetica', 'italic');
    this.doc.text('Generated by Easy Query Knowledge Base AI System', this.pageWidth / 2, this.currentY, { align: 'center' });
  }

  /**
   * Add metadata row
   */
  private addMetadataRow(label: string, value: string): void {
    this.doc.setFontSize(12);
    this.doc.setFont('helvetica', 'bold');
    this.doc.text(`${label}:`, this.margin, this.currentY);
    
    this.doc.setFont('helvetica', 'normal');
    this.doc.text(value, this.margin + 40, this.currentY);
    
    this.currentY += 8;
  }

  /**
   * Add page break if needed
   */
  private addPageBreak(): void {
    if (this.currentY > this.pageHeight - 50) {
      this.doc.addPage();
      this.currentY = 20;
    }
  }

  /**
   * Wrap text to fit within specified width
   */
  private wrapText(text: string, maxWidth: number, fontSize: number): string[] {
    const words = text.split(' ');
    const lines: string[] = [];
    let currentLine = '';
    
    words.forEach(word => {
      const testLine = currentLine + (currentLine ? ' ' : '') + word;
      const testWidth = this.doc.getTextWidth(testLine);
      
      if (testWidth > maxWidth && currentLine) {
        lines.push(currentLine);
        currentLine = word;
      } else {
        currentLine = testLine;
      }
    });
    
    if (currentLine) {
      lines.push(currentLine);
    }
    
    return lines;
  }

  /**
   * Download the generated PDF
   */
  download(filename?: string): void {
    const defaultFilename = `AI_Report_${new Date().toISOString().split('T')[0]}.pdf`;
    this.doc.save(filename || defaultFilename);
  }

  /**
   * Get the PDF as a blob for preview or other uses
   */
  getBlob(): Blob {
    return this.doc.output('blob');
  }
}

/**
 * Smart PDF generation with automatic optimization
 */
export function generateAndDownloadPDF(results: ReportResults, filename?: string): void {
  // Auto-detect optimal settings
  const hasWideTables = results.results?.some(section => 
    section.table && section.table.columns && section.table.columns.length > 8
  ) || false;
  
  const hasLongTables = results.results?.some(section => 
    section.table && section.table.data && section.table.data.length > 100
  ) || false;
  
  const generator = new SmartPDFGenerator({
    landscape: hasWideTables || hasLongTables,
    maxRowsPerPage: hasLongTables ? 20 : 30,
    maxColumnsPerPage: hasWideTables ? 6 : 8,
    enableSmartSplitting: true
  });
  
  const pdf = generator.generateReport(results);
  generator.download(filename);
}

/**
 * Generate PDF and return as blob for preview
 */
export function generatePDFBlob(results: ReportResults): Blob {
  const hasWideTables = results.results?.some(section => 
    section.table && section.table.columns && section.table.columns.length > 8
  ) || false;
  
  const hasLongTables = results.results?.some(section => 
    section.table && section.table.data && section.table.data.length > 100
  ) || false;
  
  const generator = new SmartPDFGenerator({
    landscape: hasWideTables || hasLongTables,
    maxRowsPerPage: hasLongTables ? 20 : 30,
    maxColumnsPerPage: hasWideTables ? 6 : 8,
    enableSmartSplitting: true
  });
  
  generator.generateReport(results);
  return generator.getBlob();
} 