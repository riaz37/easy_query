import { useState, useCallback, useRef, useEffect } from 'react';
import { toast } from 'sonner';
import { fileService } from '@/lib/api/services/file-service';
import type { UploadedFile, FileQueryResult, QueryOptions } from '@/components/data-query';

export interface FileQueryManagerState {
  uploadedFiles: UploadedFile[];
  queryResults: FileQueryResult[];
  isExecuting: boolean;
  queryError: string | null;
  executionTime: number;
}

export interface FileQueryManagerActions {
  uploadFiles: (files: File[]) => Promise<void>;
  removeFile: (fileId: string) => void;
  clearAllFiles: () => void;
  executeQuery: (query: string, options: QueryOptions, userId: string) => Promise<FileQueryResult[]>;
  clearResults: () => void;
  clearError: () => void;
}

export function useFileQueryManager(): [FileQueryManagerState, FileQueryManagerActions] {
  // State
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [queryResults, setQueryResults] = useState<FileQueryResult[]>([]);
  const [isExecuting, setIsExecuting] = useState(false);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [executionTime, setExecutionTime] = useState<number>(0);
  
  // Refs for cleanup
  const pollingIntervals = useRef<Record<string, NodeJS.Timeout>>({});
  const abortController = useRef<AbortController | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Clear all polling intervals
      Object.values(pollingIntervals.current).forEach(interval => clearInterval(interval));
      
      // Abort any ongoing requests
      if (abortController.current) {
        abortController.current.abort();
      }
    };
  }, []);

  // Upload files
  const uploadFiles = useCallback(async (files: File[]) => {
    if (files.length === 0) return;

    // Create initial file entries
    const newFiles: UploadedFile[] = files.map(file => ({
      id: `${Date.now()}-${Math.random()}`,
      file,
      status: 'pending' as const,
      progress: 0,
    }));

    setUploadedFiles(prev => [...prev, ...newFiles]);

    try {
      // Generate descriptions and table names
      const fileDescriptions = files.map(file => `Uploaded file: ${file.name}`);
      const tableNames = files.map(file => `file_${file.name.replace(/[^a-zA-Z0-9]/g, '_')}`);

      // Upload files using the file service
      const uploadResponse = await fileService.uploadToSmartFileSystem({
        files,
        file_descriptions: fileDescriptions,
        table_names: tableNames,
      });

      if (uploadResponse.success && uploadResponse.data) {
        const bundleId = uploadResponse.data.bundle_id;
        
        // Update files with bundle ID and start polling
        setUploadedFiles(prev => prev.map(uploadFile => {
          const matchingFile = newFiles.find(f => f.id === uploadFile.id);
          if (matchingFile) {
            return {
              ...uploadFile,
              bundleId,
              status: 'processing' as const,
              progress: 0,
            };
          }
          return uploadFile;
        }));

        // Start progress polling
        startProgressPolling(bundleId, newFiles);
        
        toast.success(`Files uploaded successfully! Bundle ID: ${bundleId}`);
      } else {
        throw new Error(uploadResponse.error || 'File upload failed');
      }
    } catch (error) {
      console.error('File upload error:', error);
      const errorMessage = error instanceof Error ? error.message : 'File upload failed';
      toast.error(`Upload failed: ${errorMessage}`);
      
      // Mark files as failed
      setUploadedFiles(prev => prev.map(uploadFile => {
        const matchingFile = newFiles.find(f => f.id === uploadFile.id);
        if (matchingFile) {
          return {
            ...uploadFile,
            status: 'failed' as const,
            error: errorMessage,
          };
        }
        return uploadFile;
      }));
    }
  }, []);

  // Start progress polling for a bundle
  const startProgressPolling = useCallback((bundleId: string, files: UploadedFile[]) => {
    console.log('Starting progress polling for bundle:', bundleId, 'with files:', files.map(f => f.file.name));
    
    const pollInterval = setInterval(async () => {
      try {
        const statusResponse = await fileService.getBundleTaskStatus(bundleId);
        
        if (statusResponse.success && statusResponse.data) {
          const status = statusResponse.data;
          console.log('Progress update for bundle:', bundleId, 'status:', status);
          
          // Update progress for each file
          setUploadedFiles(prev => prev.map(uploadFile => {
            if (uploadFile.bundleId !== bundleId) return uploadFile;
            
            const task = status.individual_tasks.find(task => 
              task.filename.toLowerCase() === uploadFile.file.name.toLowerCase() ||
              task.filename.toLowerCase().includes(uploadFile.file.name.toLowerCase()) ||
              uploadFile.file.name.toLowerCase().includes(task.filename.toLowerCase())
            );
            
            if (task) {
              let newStatus: UploadedFile['status'] = 'processing';
              let progress = 0;
              let error: string | undefined;
              
              if (task.status === 'completed') {
                newStatus = 'completed';
                progress = 100;
              } else if (task.status === 'failed') {
                newStatus = 'failed';
                error = task.error_message || 'Processing failed';
              } else if (task.status === 'processing') {
                progress = parseInt(task.progress) || 0;
              }
              
              return {
                ...uploadFile,
                status: newStatus,
                progress,
                error,
              };
            }
            
            return uploadFile;
          }));
          
          // Check if all files are completed
          if (status.status === 'completed' || status.status === 'failed') {
            clearInterval(pollInterval);
            
            // Clean up interval
            delete pollingIntervals.current[bundleId];
            
            if (status.status === 'completed') {
              toast.success(`All files processed successfully! Total: ${status.completed_files}`);
            } else {
              toast.error(`Some files failed to process. Completed: ${status.completed_files}, Failed: ${status.failed_files}`);
            }
          }
        }
      } catch (error) {
        console.error('Progress polling error:', error);
      }
    }, 2000); // Poll every 2 seconds
    
    // Store interval for cleanup
    pollingIntervals.current[bundleId] = pollInterval;
  }, []);

  // Remove file
  const removeFile = useCallback((fileId: string) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId));
    
    // Clean up any polling intervals for this file
    const file = uploadedFiles.find(f => f.id === fileId);
    if (file?.bundleId && pollingIntervals.current[file.bundleId]) {
      clearInterval(pollingIntervals.current[file.bundleId]);
      delete pollingIntervals.current[file.bundleId];
    }
  }, [uploadedFiles]);

  // Clear all files
  const clearAllFiles = useCallback(() => {
    // Clear all polling intervals
    Object.values(pollingIntervals.current).forEach(interval => clearInterval(interval));
    pollingIntervals.current = {};
    setUploadedFiles([]);
  }, []);

  // Execute query
  const executeQuery = useCallback(async (query: string, options: QueryOptions, userId: string): Promise<FileQueryResult[]> => {
    if (!query.trim()) {
      throw new Error('Query cannot be empty');
    }

    if (!userId) {
      throw new Error('User ID is required');
    }

    // Abort any ongoing request
    if (abortController.current) {
      abortController.current.abort();
    }
    
    // Create new abort controller
    abortController.current = new AbortController();

    setIsExecuting(true);
    setQueryError(null);
    setQueryResults([]);
    setExecutionTime(0);

    const startTime = Date.now();

    try {
      // Get file IDs from completed uploads
      const completedFileIds = uploadedFiles
        .filter(file => file.status === 'completed')
        .map(file => file.id);

      // Execute file search
      const response = await fileService.searchFiles({
        query: query.trim(),
        user_id: userId,
        use_intent_reranker: options.useIntentReranker,
        use_chunk_reranker: options.useChunkReranker,
        use_dual_embeddings: options.useDualEmbeddings,
        intent_top_k: options.intentTopK,
        chunk_top_k: options.chunkTopK,
        max_chunks_for_answer: options.maxChunksForAnswer,
        answer_style: options.answerStyle,
        table_specific: options.tableSpecific,
        file_ids: completedFileIds.length > 0 ? completedFileIds : undefined,
      }, abortController.current.signal);

      if (response.success && response.data) {
        const searchResponse = response.data;
        console.log('File search response:', searchResponse);
        
        // Extract results from the answer sources or create structured result
        let results: FileQueryResult[] = [];
        if (searchResponse.answer && searchResponse.answer.sources && Array.isArray(searchResponse.answer.sources)) {
          results = searchResponse.answer.sources.map((source, index) => ({
            id: `result-${index}`,
            answer: source.content || source.text || JSON.stringify(source),
            confidence: searchResponse.answer.confidence,
            sources_used: searchResponse.answer.sources_used,
            query: searchResponse.query,
            ...source,
          }));
        } else if (searchResponse.answer) {
          // If no sources, create a result from the answer
          results = [{
            id: 'result-0',
            answer: searchResponse.answer.answer,
            confidence: searchResponse.answer.confidence,
            sources_used: searchResponse.answer.sources_used,
            query: searchResponse.query,
          }];
        }
        
        setQueryResults(results);
        setExecutionTime(Date.now() - startTime);
        
        return results;
      } else {
        throw new Error(response.error || 'Query execution failed');
      }
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('Query was aborted');
        return [];
      }
      
      console.error('File query execution error:', error);
      const errorMessage = error instanceof Error ? error.message : 'An unexpected error occurred';
      setQueryError(errorMessage);
      throw error;
    } finally {
      setIsExecuting(false);
    }
  }, [uploadedFiles]);

  // Clear results
  const clearResults = useCallback(() => {
    setQueryResults([]);
    setExecutionTime(0);
  }, []);

  // Clear error
  const clearError = useCallback(() => {
    setQueryError(null);
  }, []);

  // State
  const state: FileQueryManagerState = {
    uploadedFiles,
    queryResults,
    isExecuting,
    queryError,
    executionTime,
  };

  // Actions
  const actions: FileQueryManagerActions = {
    uploadFiles,
    removeFile,
    clearAllFiles,
    executeQuery,
    clearResults,
    clearError,
  };

  return [state, actions];
} 