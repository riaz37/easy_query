export interface ReportStructure {
  [key: string]: string;
}

export interface GenerateReportRequest {
  user_id: string;
  user_query: string;
}

export interface GenerateReportResponse {
  task_id: string;
  status: 'accepted' | 'processing' | 'completed' | 'failed';
  message: string;
  user_id: string;
  timestamp: string;
}

export interface ReportTaskStatus {
  task_id: string;
  user_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: string;
  current_step: string;
  total_queries: number;
  processed_queries: number;
  successful_queries: number;
  failed_queries: number;
  created_at: string;
  started_at: string;
  completed_at?: string;
  processing_time_seconds?: number;
  progress_percentage: number;
  error?: string | null;
  results?: ReportResults;
}

export interface ReportResults {
  success: boolean;
  database_id: number;
  total_queries: number;
  successful_queries: number;
  failed_queries: number;
  results?: ReportSection[]; // This is the actual data array
  total_processing_time?: number;
  summary?: {
    total_sections: number;
    total_queries: number;
    successful_queries: number;
    failed_queries: number;
    success_rate: number;
    total_processing_time: number;
    average_processing_time: number;
    sections_processed: number;
    processing_method: string;
    database_id: number;
    errors_summary: Record<string, any>;
  };
}

export interface ReportSection {
  section_number: number;
  section_name: string;
  query_number: number;
  query: string;
  success: boolean;
  table?: {
    total_rows: number;
    columns: string[];
    data: any[];
  };
  graph_and_analysis?: GraphAnalysis;
  analysis?: any;
  llm_analysis?: {
    analysis: string;
    analysis_subject: string;
    data_coverage: string;
    metadata: any;
  };
}

export interface GraphAnalysis {
  graph_type: string;
  theme: string;
  image_url: string;
}

export interface UpdateReportStructureRequest {
  report_structure: string;
}

export interface ReportGenerationOptions {
  onProgress?: (status: ReportTaskStatus) => void;
  onComplete?: (results: ReportResults) => void;
  onError?: (error: Error) => void;
  pollInterval?: number;
  timeout?: number;
}

export interface ReportHistoryItem {
  id: string;
  user_id: string;
  user_query: string;
  status: string;
  created_at: string;
  completed_at?: string;
  processing_time_seconds?: number;
}

export interface ReportFilterOptions {
  status?: string;
  date_from?: string;
  date_to?: string;
  user_id?: string;
} 