import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import { ReportResults, ReportSection } from '@/types/reports';

export class PDFReportGenerator {
  private doc: jsPDF;
  private currentY: number = 20;
  private pageWidth: number;
  private pageHeight: number;
  private margin: number = 20;
  private useLandscape: boolean = false;

  constructor(useLandscape: boolean = false) {
    this.useLandscape = useLandscape;
    this.doc = new jsPDF(useLandscape ? 'l' : 'p', 'mm', 'a4');
    this.pageWidth = this.doc.internal.pageSize.getWidth();
    this.pageHeight = this.doc.internal.pageSize.getHeight();
  }

  /**
   * Generate a complete PDF report from AI results
   */
  generateReport(results: ReportResults): jsPDF {
    // Add title page
    this.addTitlePage(results);
    
    // Add executive summary
    this.addExecutiveSummary(results);
    
    // Add detailed results for each section
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
    this.addMetadataRow('Processing Time', `${results.total_processing_time?.toFixed(2) || 'N/A'} seconds`);
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
    
    // Add landscape mode warning if applicable
    if (this.useLandscape) {
      this.doc.setFontSize(10);
      this.doc.setFont('helvetica', 'italic');
      this.doc.setTextColor(255, 140, 0); // Orange color for warning
      this.doc.text('Note: This report uses landscape orientation to accommodate wide data tables.', this.margin, this.currentY);
      this.currentY += 8;
      this.doc.setTextColor(0, 0, 0); // Reset to black
    }
    
    const summary = results.summary;
    if (summary) {
      const summaryText = [
        `This AI-generated report contains ${summary.total_sections} sections with ${summary.total_queries} queries.`,
        `All queries were processed successfully with a ${summary.success_rate}% success rate.`,
        `The total processing time was ${summary.total_processing_time?.toFixed(2) || 'N/A'} seconds.`,
        `The report covers comprehensive data analysis including employee attendance, salary information, and other business metrics.`
      ];
      
      summaryText.forEach(text => {
        this.doc.text(text, this.margin, this.currentY);
        this.currentY += 8;
      });
    }
    
    this.currentY += 10;
  }

  /**
   * Add a section with its data and analysis
   */
  private addSection(section: ReportSection, sectionIndex: number): void {
    this.addPageBreak();
    
    // Section header
    this.doc.setFontSize(18);
    this.doc.setFont('helvetica', 'bold');
    this.doc.text(`Section ${section.section_number}: ${section.section_name}`, this.margin, this.currentY);
    this.currentY += 15;
    
    // Query details
    this.doc.setFontSize(14);
    this.doc.setFont('helvetica', 'bold');
    this.doc.text(`Query ${section.query_number}:`, this.margin, this.currentY);
    this.currentY += 8;
    
    this.doc.setFontSize(12);
    this.doc.setFont('helvetica', 'normal');
    
    // Wrap long queries
    const queryLines = this.wrapText(section.query, this.pageWidth - 2 * this.margin, 12);
    queryLines.forEach(line => {
      this.doc.text(line, this.margin, this.currentY);
      this.currentY += 6;
    });
    
    this.currentY += 10;
    
    // Status
    this.doc.setFontSize(12);
    this.doc.setFont('helvetica', 'bold');
    this.doc.text(`Status: ${section.success ? '✅ Success' : '❌ Failed'}`, this.margin, this.currentY);
    this.currentY += 10;
    
    // Add table data if available
    if (section.table && section.table.data && section.table.data.length > 0) {
      this.addDataTable(section.table);
    }
    
    // Add LLM analysis if available
    if (section.llm_analysis) {
      this.addLLMAnalysis(section.llm_analysis);
    }
    
    this.currentY += 15;
  }

  /**
   * Add data table
   */
  private addDataTable(table: any): void {
    this.addPageBreak();
    
    this.doc.setFontSize(14);
    this.doc.setFont('helvetica', 'bold');
    this.doc.text('Data Summary', this.margin, this.currentY);
    this.currentY += 10;
    
    // Table info
    this.doc.setFontSize(12);
    this.doc.setFont('helvetica', 'normal');
    this.doc.text(`Total Rows: ${table.total_rows} | Columns: ${table.columns.join(', ')}`, this.margin, this.currentY);
    this.currentY += 10;
    
    // Create table with autoTable - handle wide tables
    const tableData = table.data.slice(0, 50); // Limit to first 50 rows for PDF
    
    // Calculate optimal column widths for wide tables
    const pageWidth = this.pageWidth - (2 * this.margin);
    const columnCount = table.columns.length;
    
    // For tables with many columns, use landscape orientation or adjust column widths
    let tableConfig: any = {
      head: [table.columns],
      body: tableData.map((row: any) => 
        table.columns.map((col: string) => {
          const value = row[col];
          if (value === null || value === undefined) return 'N/A';
          // Truncate long values to prevent table overflow
          return String(value).length > 30 ? String(value).substring(0, 30) + '...' : String(value);
        })
      ),
      startY: this.currentY,
      margin: { left: this.margin, right: this.margin },
      styles: {
        fontSize: 7, // Smaller font for wide tables
        cellPadding: 1, // Reduce padding
        overflow: 'linebreak', // Handle text overflow
        halign: 'left',
        valign: 'middle',
      },
      headStyles: {
        fillColor: [41, 128, 185],
        textColor: 255,
        fontStyle: 'bold',
        fontSize: 8,
      },
      alternateRowStyles: {
        fillColor: [245, 245, 245],
      },
      // Handle wide tables
      didParseCell: function(data: any) {
        // Set maximum column width to prevent overflow
        const maxWidth = pageWidth / Math.min(columnCount, 8); // Max 8 columns per row
        data.cell.width = Math.min(data.cell.width, maxWidth);
      },
      // Auto-adjust column widths
      columnStyles: {},
    };
    
    // For very wide tables (more than 8 columns), create multiple tables
    if (columnCount > 8) {
      this.doc.setFontSize(10);
      this.doc.setFont('helvetica', 'italic');
      this.doc.setTextColor(255, 140, 0); // Orange color for optimization note
      this.doc.text(`Note: Table has ${columnCount} columns. Showing data in optimized format.`, this.margin, this.currentY);
      this.currentY += 8;
      this.doc.setTextColor(0, 0, 0); // Reset to black
      
      // Split columns into groups
      const columnGroups = [];
      for (let i = 0; i < columnCount; i += 6) {
        columnGroups.push(table.columns.slice(i, i + 6));
      }
      
      columnGroups.forEach((columns, groupIndex) => {
        if (groupIndex > 0) {
          this.addPageBreak();
        }
        
        this.doc.setFontSize(12);
        this.doc.setFont('helvetica', 'bold');
        this.doc.text(`Data Group ${groupIndex + 1} (Columns ${groupIndex * 6 + 1}-${Math.min((groupIndex + 1) * 6, columnCount)})`, this.margin, this.currentY);
        this.currentY += 8;
        
        const groupData = tableData.map((row: any) => 
          columns.map((col: string) => {
            const value = row[col];
            if (value === null || value === undefined) return 'N/A';
            return String(value).length > 25 ? String(value).substring(0, 25) + '...' : String(value);
          })
        );
        
        autoTable(this.doc, {
          ...tableConfig,
          head: [columns],
          body: groupData,
          startY: this.currentY,
          styles: {
            ...tableConfig.styles,
            fontSize: 8, // Slightly larger for grouped tables
          },
        });
        
        const finalY = (this.doc as any).lastAutoTable.finalY;
        this.currentY = finalY + 10;
      });
    } else {
      // Standard table for normal width
      autoTable(this.doc, tableConfig);
      
      // Update current Y position
      const finalY = (this.doc as any).lastAutoTable.finalY;
      this.currentY = finalY + 10;
    }
    
    if (table.data.length > 50) {
      this.doc.setFontSize(10);
      this.doc.setFont('helvetica', 'italic');
      this.doc.text(`Note: Showing first 50 rows out of ${table.data.length} total rows`, this.margin, this.currentY);
      this.currentY += 8;
    }
  }

  /**
   * Add LLM analysis
   */
  private addLLMAnalysis(analysis: any): void {
    this.addPageBreak();
    
    this.doc.setFontSize(14);
    this.doc.setFont('helvetica', 'bold');
    this.doc.text('AI Analysis', this.margin, this.currentY);
    this.currentY += 10;
    
    this.doc.setFontSize(12);
    this.doc.setFont('helvetica', 'normal');
    
    if (analysis.analysis) {
      const analysisLines = this.wrapText(analysis.analysis, this.pageWidth - 2 * this.margin, 12);
      analysisLines.forEach(line => {
        this.doc.text(line, this.margin, this.currentY);
        this.currentY += 6;
      });
    }
    
    this.currentY += 10;
  }

  /**
   * Add summary section
   */
  private addSummary(results: ReportResults): void {
    this.addPageBreak();
    
    this.doc.setFontSize(16);
    this.doc.setFont('helvetica', 'bold');
    this.doc.text('Report Summary', this.margin, this.currentY);
    this.currentY += 15;
    
    this.doc.setFontSize(12);
    this.doc.setFont('helvetica', 'normal');
    
    const summary = results.summary;
    if (summary) {
      this.addMetadataRow('Total Sections', summary.total_sections.toString());
      this.addMetadataRow('Total Queries', summary.total_queries.toString());
      this.addMetadataRow('Success Rate', `${summary.success_rate}%`);
      this.addMetadataRow('Processing Method', summary.processing_method);
      this.addMetadataRow('Total Processing Time', `${summary.total_processing_time?.toFixed(2) || 'N/A'} seconds`);
      this.addMetadataRow('Average Time per Query', `${summary.average_processing_time?.toFixed(2) || 'N/A'} seconds`);
    }
    
    this.currentY += 20;
    
    // Footer
    this.doc.setFontSize(10);
    this.doc.setFont('helvetica', 'italic');
    this.doc.text('Generated by ESAP Knowledge Base AI System', this.pageWidth / 2, this.currentY, { align: 'center' });
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
 * Convenience function to generate and download a PDF report
 */
export function generateAndDownloadPDF(results: ReportResults, filename?: string): void {
  // Check if we have wide tables that need landscape mode
  const hasWideTables = results.results?.some(section => 
    section.table && section.table.columns && section.table.columns.length > 8
  ) || false;
  
  const generator = new PDFReportGenerator(hasWideTables);
  const pdf = generator.generateReport(results);
  generator.download(filename);
}

/**
 * Generate PDF and return as blob for preview
 */
export function generatePDFBlob(results: ReportResults): Blob {
  // Check if we have wide tables that need landscape mode
  const hasWideTables = results.results?.some(section => 
    section.table && section.table.columns && section.table.columns.length > 8
  ) || false;
  
  const generator = new PDFReportGenerator(hasWideTables);
  generator.generateReport(results);
  return generator.getBlob();
} 