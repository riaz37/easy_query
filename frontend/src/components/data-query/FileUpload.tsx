import React, { useState, useCallback, useRef } from 'react';
// Card components removed - now handled by parent component
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Upload, File, X, CheckCircle, AlertCircle, Database } from 'lucide-react';
import { Spinner } from '@/components/ui/loading';
import { toast } from 'sonner';
import { ServiceRegistry } from '@/lib/api/services/service-registry';
import { useAuthContext } from '@/components/providers/AuthContextProvider';

export interface UploadedFile {
  id: string;
  file: File;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'failed';
  progress: number;
  bundleId?: string;
  error?: string;
}

interface FileUploadProps {
  onFilesUploaded: (fileIds: string[]) => void;
  onUploadStatusChange: (files: UploadedFile[]) => void;
  onTableUsageChange?: (useTable: boolean) => void;
  disabled?: boolean;
  className?: string;
}

export function FileUpload({ 
  onFilesUploaded, 
  onUploadStatusChange, 
  onTableUsageChange,
  disabled = false,
  className = "" 
}: FileUploadProps) {
  const { user } = useAuthContext();
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [pollingIntervals, setPollingIntervals] = useState<Record<string, NodeJS.Timeout>>({});
  const [useTable, setUseTable] = useState(true); // Default to using tables
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handle file selection
  const handleFileSelect = useCallback((files: FileList | null) => {
    if (!files || files.length === 0) return;

    const newFiles: UploadedFile[] = Array.from(files).map(file => ({
      id: `${Date.now()}-${Math.random()}`,
      file,
      status: 'pending',
      progress: 0,
    }));

    console.log('Adding new files:', newFiles.map(f => ({ name: f.file.name, id: f.id, status: f.status })));
    
    setUploadedFiles(prev => {
      const updated = [...prev, ...newFiles];
      console.log('Updated uploadedFiles state after selection:', updated.map(f => ({ name: f.file.name, id: f.id, status: f.status })));
      return updated;
    });
    
    // Don't call onUploadStatusChange here - it will be called when files are actually uploaded
  }, []);

  // Upload files to the API
  const uploadFiles = useCallback(async (files: UploadedFile[]) => {
    if (files.length === 0) return;

    setIsUploading(true);
    
    try {
      // Generate descriptions and table names
      const fileDescriptions = files.map(uploadFile => `Uploaded file: ${uploadFile.file.name}`);
      const tableNames = useTable ? files.map(uploadFile => `file_${uploadFile.file.name.replace(/[^a-zA-Z0-9]/g, '_')}`) : [];

      // Check if user is authenticated
      if (!user?.user_id) {
        throw new Error('User not authenticated. Please log in again.');
      }

      // Upload files using the file service
      const uploadResponse = await ServiceRegistry.file.uploadToSmartFileSystem({
        files: files.map(uploadFile => uploadFile.file),
        file_descriptions: fileDescriptions,
        table_names: tableNames,
        user_ids: user.user_id,
        use_table: useTable,
      });

      if (uploadResponse.success && uploadResponse.data) {
        const bundleId = uploadResponse.data.bundle_id;
        
        // Update files with bundle ID and start polling
        setUploadedFiles(prev => {
          const updated = prev.map(uploadFile => {
            const matchingFile = files.find(f => f.id === uploadFile.id);
            if (matchingFile) {
              console.log(`Updating file ${uploadFile.file.name} with bundle ID ${bundleId} and status processing`);
              return {
                ...uploadFile,
                bundleId,
                status: 'processing' as const,
                progress: 0,
              };
            }
            return uploadFile;
          });
          console.log('Updated uploadedFiles state:', updated);
          return updated;
        });

        // Start progress polling
        startProgressPolling(bundleId);
        
        console.log(`Started progress polling for bundle: ${bundleId}`);
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
        const matchingFile = files.find(f => f.id === uploadFile.id);
        if (matchingFile) {
          return {
            ...uploadFile,
            status: 'failed' as const,
            error: errorMessage,
          };
        }
        return uploadFile;
      }));
    } finally {
      setIsUploading(false);
    }
  }, []);

  // Start progress polling for a bundle
  const startProgressPolling = useCallback((bundleId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const statusResponse = await ServiceRegistry.file.getBundleTaskStatus(bundleId);
        
        if (statusResponse.success && statusResponse.data) {
          const status = statusResponse.data;
          console.log('Progress update for bundle:', bundleId, 'status:', status);
          
          // Update progress for each file using current state
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
              
              // Handle different status values from API (case-insensitive)
              const taskStatus = task.status?.toLowerCase();
              console.log(`Task ${task.filename} status: ${taskStatus}, progress: ${task.progress}`);
              
              if (taskStatus === 'completed' || taskStatus === 'success') {
                newStatus = 'completed';
                progress = 100;
              } else if (taskStatus === 'failed' || taskStatus === 'error') {
                newStatus = 'failed';
                error = task.error_message || 'Processing failed';
              } else if (taskStatus === 'processing' || taskStatus === 'running' || taskStatus === 'pending') {
                newStatus = 'processing';
                // Parse progress - handle both string and number formats
                if (typeof task.progress === 'number') {
                  progress = task.progress;
                } else if (typeof task.progress === 'string') {
                  // Try to extract number from progress string
                  const progressMatch = task.progress.match(/(\d+)/);
                  progress = progressMatch ? parseInt(progressMatch[1]) : 0;
                } else {
                  progress = 0;
                }
              }
              
              console.log(`Updating file ${uploadFile.file.name} to status: ${newStatus}, progress: ${progress}`);
              
              return {
                ...uploadFile,
                status: newStatus,
                progress,
                error,
              };
            }
            
            return uploadFile;
          }));
          
          // Check if all files are completed (handle both uppercase and lowercase status)
          const bundleStatus = status.status?.toLowerCase();
          if (bundleStatus === 'completed' || bundleStatus === 'success' || bundleStatus === 'failed' || bundleStatus === 'error') {
            console.log(`Bundle ${bundleId} completed with status: ${bundleStatus}`);
            clearInterval(pollInterval);
            
            // Clean up interval
            setPollingIntervals(prev => {
              const newIntervals = { ...prev };
              delete newIntervals[bundleId];
              return newIntervals;
            });
            
            // Update parent component with current state
            setUploadedFiles(currentFiles => {
              const updatedFiles = currentFiles.map(uploadFile => {
                if (uploadFile.bundleId === bundleId) {
                  const isCompleted = bundleStatus === 'completed' || bundleStatus === 'success';
                  return {
                    ...uploadFile,
                    status: isCompleted ? 'completed' : 'failed',
                    progress: isCompleted ? 100 : 0,
                  };
                }
                return uploadFile;
              });
              
              return updatedFiles;
            });
            
            if (bundleStatus === 'completed' || bundleStatus === 'success') {
              toast.success(`All files processed successfully! Total: ${status.completed_files}`);
            } else {
              toast.error(`Some files failed to process. Completed: ${status.completed_files}, Failed: ${status.failed_files}`);
            }
          }
        } else {
          console.error('Failed to get bundle status:', statusResponse.error);
        }
      } catch (error) {
        console.error('Progress polling error:', error);
        // Don't stop polling on error, just log it
      }
    }, 2000); // Poll every 2 seconds
    
    // Store interval for cleanup
    setPollingIntervals(prev => ({
      ...prev,
      [bundleId]: pollInterval
    }));
  }, [onUploadStatusChange]);

  // Remove file
  const removeFile = useCallback((fileId: string) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId));
    
    // Clean up any polling intervals for this file
    const file = uploadedFiles.find(f => f.id === fileId);
    if (file?.bundleId && pollingIntervals[file.bundleId]) {
      clearInterval(pollingIntervals[file.bundleId]);
      setPollingIntervals(prev => {
        const newIntervals = { ...prev };
        delete newIntervals[file.bundleId!];
        return newIntervals;
      });
    }
  }, [uploadedFiles, pollingIntervals]);

  // Clear all files
  const clearAllFiles = useCallback(() => {
    // Clear all polling intervals
    Object.values(pollingIntervals).forEach(interval => clearInterval(interval));
    setPollingIntervals({});
    setUploadedFiles([]);
    onUploadStatusChange([]);
  }, [pollingIntervals, onUploadStatusChange]);

  // Cleanup effect to clear intervals on unmount
  React.useEffect(() => {
    return () => {
      // Clear all polling intervals when component unmounts
      Object.values(pollingIntervals).forEach(interval => clearInterval(interval));
    };
  }, [pollingIntervals]);

  // Sync with parent component when uploadedFiles changes
  React.useEffect(() => {
    if (uploadedFiles.length > 0) {
      onUploadStatusChange(uploadedFiles);
    }
  }, [uploadedFiles, onUploadStatusChange]);

  // Notify parent when table usage changes
  React.useEffect(() => {
    console.log('FileUpload: useTable changed to:', useTable, 'onTableUsageChange exists:', !!onTableUsageChange);
    if (onTableUsageChange && typeof onTableUsageChange === 'function') {
      try {
        onTableUsageChange(useTable);
      } catch (error) {
        console.error('Error calling onTableUsageChange:', error);
      }
    }
  }, [useTable, onTableUsageChange]);

  // Get status icon
  const getStatusIcon = (status: UploadedFile['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'failed':
        return <AlertCircle className="w-4 h-4 text-red-600" />;
      case 'processing':
        return <Spinner size="sm" variant="accent-green" />;
      case 'uploading':
        return <Spinner size="sm" variant="accent-green" />;
      default:
        return <File className="w-4 h-4 text-gray-600" />;
    }
  };

  // Get status color
  const getStatusColor = (status: UploadedFile['status']) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-200';
      case 'failed':
        return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-200';
      case 'processing':
        return 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/20 dark:text-emerald-200';
      case 'uploading':
        return 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/20 dark:text-emerald-200';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-200';
    }
  };

  // Handle drag and drop
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.currentTarget.classList.add('border-emerald-500', 'bg-emerald-50');
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.currentTarget.classList.remove('border-emerald-500', 'bg-emerald-50');
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.currentTarget.classList.remove('border-emerald-500', 'bg-emerald-50');
    handleFileSelect(e.dataTransfer.files);
  }, [handleFileSelect]);

  return (
    <div className={className}>
      {/* Header with status */}
      {uploadedFiles.length > 0 && (
        <div className="flex items-center justify-between mb-4">
          <Badge variant="outline" className="border-emerald-400/30 text-emerald-400">
            {uploadedFiles.filter(f => f.status === 'completed').length} / {uploadedFiles.length} Complete
          </Badge>
          <Button
            variant="outline"
            size="sm"
            onClick={clearAllFiles}
            disabled={isUploading}
            className="border-red-400/30 text-red-400 hover:bg-red-400/10"
          >
            Clear All
          </Button>
        </div>
      )}

      <div className="space-y-4">
        {/* Table Usage Toggle */}
        <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg border border-gray-600/30">
          <div className="flex items-center gap-3">
            <Database className="w-5 h-5 text-emerald-400" />
            <div>
              <Label htmlFor="use-table" className="text-sm font-medium text-white">
                Use Table Names
              </Label>
              <p className="text-xs text-gray-400">
                {useTable ? 'Files will be associated with table names' : 'Files will be processed without table names'}
              </p>
            </div>
          </div>
          <Switch
            id="use-table"
            checked={useTable}
            onCheckedChange={setUseTable}
            disabled={isUploading}
          />
        </div>

        {/* File Drop Zone */}
        <div
          className={`border-2 border-dashed border-gray-600/50 rounded-lg p-6 text-center transition-colors ${
            disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:border-emerald-400 hover:bg-emerald-900/20'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !disabled && fileInputRef.current?.click()}
          data-element="upload-area"
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={(e) => handleFileSelect(e.target.files)}
            className="hidden"
            accept=".txt,.pdf,.doc,.docx,.csv,.xlsx,.xls"
            disabled={disabled || isUploading}
          />
          
          {isUploading ? (
            <>
              <Spinner size="lg" variant="accent-green" className="mx-auto mb-4" />
              <p className="text-lg font-medium text-white">
                Uploading files...
              </p>
              <p className="text-sm text-gray-400">
                Please wait while your files are being processed
              </p>
            </>
          ) : (
            <>
              <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-lg font-medium text-white">
                Click to upload files or drag and drop
              </p>
              <p className="text-sm text-gray-400">
                Supports TXT, PDF, DOC, DOCX, CSV, XLSX, XLS
              </p>
            </>
          )}
        </div>

        {/* Upload Button */}
        {uploadedFiles.length > 0 && !isUploading && (
          <Button
            onClick={() => {
              const pendingFiles = uploadedFiles.filter(f => f.status === 'pending');
              console.log('Upload button clicked, pending files:', pendingFiles.map(f => ({ name: f.file.name, id: f.id, status: f.status })));
              uploadFiles(pendingFiles);
            }}
            disabled={disabled || uploadedFiles.filter(f => f.status === 'pending').length === 0 || !user?.user_id}
            className="w-full bg-emerald-600 hover:bg-emerald-700"
            data-element="upload-button"
          >
            <Upload className="w-4 h-4 mr-2" />
            Upload {uploadedFiles.filter(f => f.status === 'pending').length} Files
            {useTable && (
              <Badge variant="outline" className="ml-2 text-xs">
                With Tables
              </Badge>
            )}
          </Button>
        )}

        {/* Authentication Warning */}
        {!user?.user_id && (
          <div className="p-3 bg-yellow-500/20 border border-yellow-400/30 rounded-lg">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-yellow-400" />
              <span className="text-yellow-400 text-sm font-medium">Authentication Required</span>
            </div>
            <p className="text-yellow-300 text-xs mt-1">Please log in to upload files</p>
          </div>
        )}

        {/* Uploaded Files List */}
        {uploadedFiles.length > 0 && (
          <div className="space-y-3">
            <h4 className="text-sm font-medium text-white">Uploaded Files</h4>
            <div className="space-y-2 max-h-60 overflow-y-auto">
              {uploadedFiles.map((uploadedFile) => (
                <div key={uploadedFile.id} className="flex items-center justify-between p-3 bg-gray-800/30 rounded-lg border border-gray-600/30">
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    {getStatusIcon(uploadedFile.status)}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-white truncate">
                        {uploadedFile.file.name}
                      </p>
                      <div className="flex items-center gap-2 mt-1">
                        <Badge className={`text-xs ${getStatusColor(uploadedFile.status)}`}>
                          {uploadedFile.status}
                        </Badge>
                        <span className="text-xs text-gray-400">
                          {(uploadedFile.file.size / 1024 / 1024).toFixed(2)} MB
                        </span>
                      </div>
                      {uploadedFile.error && (
                        <p className="text-xs text-red-400 mt-1">{uploadedFile.error}</p>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    {(uploadedFile.status === 'uploading' || uploadedFile.status === 'processing') && (
                      <div className="w-16">
                        <Progress value={uploadedFile.progress} className="h-2" />
                      </div>
                    )}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeFile(uploadedFile.id)}
                      disabled={isUploading}
                      className="text-red-400 hover:text-red-300 hover:bg-red-900/20"
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
} 