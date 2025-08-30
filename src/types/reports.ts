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
}

export interface GraphAnalysis {
  graph_type: string;
  theme: string;
  image_url: string;
  column_mapping?: {
    x: string;
    y: string;
    color?: string;
    size?: string;
    group?: string;
  };
  analysis?: {
    total_records: number;
    columns_count: number;
    sql_query: string;
    generation_time: string;
    data_summary: string;
    query: string;
  };
  llm_analysis?: {
    analysis: string;
    analysis_subject: string;
    data_coverage: string;
    metadata: {
      analysis_timestamp: string;
      data_summary: {
        total_rows: number;
        total_columns: number;
        columns: string[];
        first_100_rows: number;
        last_100_rows: number;
      };
      query: string;
      subject: string;
    };
  };
}

// Available graph types for reference
export type GraphType = 
  // Basic Charts
  | "bar" | "column" | "line" | "area" | "scatter" | "bubble"
  // Distribution Charts
  | "histogram" | "box" | "violin" | "density" | "qq_plot"
  // Composition Charts
  | "pie" | "donut" | "treemap" | "sunburst" | "funnel"
  // Comparison Charts
  | "grouped_bar" | "stacked_bar" | "waterfall"
  // Time Series Charts
  | "multi_line" | "step" | "candlestick"
  // Correlation Charts
  | "heatmap" | "correlation_matrix"
  // Geographic Charts
  | "choropleth" | "scatter_map" | "bubble_map"
  // Advanced Charts
  | "radar" | "polar" | "3d_scatter" | "surface" | "contour";

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

/**
 * User Tasks Types
 */
export interface UserTask {
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

export interface UserTasksResponse {
  user_id: string;
  total_tasks: number;
  tasks: UserTask[];
}

export interface GetUserTasksRequest {
  userId: string;
  limit?: number;
  offset?: number;
  status?: UserTask['status'];
} 