// Store-related types and interfaces

// File upload store types
export interface FileMeta {
    id: string;
    name: string;
    size: number;
    type: string;
    status: string;
    progress: number;
  }
  
  export interface FileUploadState {
    uploadedFiles: File[];
    fileMetas: FileMeta[];
    processing: boolean;
    status: "pending" | "running" | "completed" | null;
    bundleId: string | null;
    bundleStatus: any;
    initialResponse: any;
    error: string | null;
    polling: boolean;
    pollingRef?: NodeJS.Timeout | null;
    setFiles: (files: File[], metas: FileMeta[]) => void;
    setProcessing: (processing: boolean) => void;
    setStatus: (status: "pending" | "running" | "completed" | null) => void;
    setBundleId: (id: string | null) => void;
    setBundleStatus: (status: any) => void;
    setInitialResponse: (resp: any) => void;
    setError: (err: string | null) => void;
    reset: () => void;
    startPolling: (bundleId: string) => void;
    stopPolling: () => void;
  }
  
  // Query history types
  export interface FileQueryHistory {
    id: string;
    question: string;
    answer: string;
    timestamp: Date;
    files?: string[];
  }
  
  // UI state types
  export interface UIState {
    theme: 'light' | 'dark';
    sidebarCollapsed: boolean;
    activeModal: string | null;
    loading: boolean;
    notifications: Notification[];
  }
  
  export interface Notification {
    id: string;
    type: 'success' | 'error' | 'warning' | 'info';
    title?: string;
    message: string;
    duration?: number;
    timestamp: Date;
  }