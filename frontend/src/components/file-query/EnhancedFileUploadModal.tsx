"use client";

import React, { useState, useCallback, useRef } from "react";
import { useDropzone } from "react-dropzone";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { X, Upload, CheckCircle, AlertCircle, File } from "lucide-react";
import { toast } from "sonner";
import { Spinner } from "@/components/ui/loading";
import { ServiceRegistry } from "@/lib/api/services/service-registry";
import { useAuthContext } from "@/components/providers/AuthContextProvider";

interface UploadedFile {
  id: string;
  file: File;
  status: "pending" | "uploading" | "processing" | "completed" | "failed";
  progress: number;
  bundleId?: string;
  error?: string;
}

interface EnhancedFileUploadModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onFilesUploaded: (fileIds: string[]) => void;
  onUploadStatusChange: (files: UploadedFile[]) => void;
  disabled?: boolean;
}

export function EnhancedFileUploadModal({
  open,
  onOpenChange,
  onFilesUploaded,
  onUploadStatusChange,
  disabled = false,
}: EnhancedFileUploadModalProps) {
  const { user } = useAuthContext();
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [pollingIntervals, setPollingIntervals] = useState<Record<string, NodeJS.Timeout>>({});

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (disabled || isUploading) return;

      const newFiles: UploadedFile[] = acceptedFiles.map((file) => ({
        id: Math.random().toString(36).substr(2, 9),
        file,
        status: "pending",
        progress: 0,
      }));

      setUploadedFiles((prev) => [...prev, ...newFiles]);
      onUploadStatusChange([...uploadedFiles, ...newFiles]);
    },
    [disabled, isUploading, uploadedFiles, onUploadStatusChange]
  );

  // Upload files to the API
  const uploadFiles = useCallback(async (files: UploadedFile[]) => {
    if (files.length === 0) return;

    setIsUploading(true);
    
    try {
      // Generate descriptions and table names
      const fileDescriptions = files.map(uploadFile => `Uploaded file: ${uploadFile.file.name}`);
      const tableNames = files.map(uploadFile => `file_${uploadFile.file.name.replace(/[^a-zA-Z0-9]/g, '_')}`);

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
        use_table: true,
      });

      if (uploadResponse.success && uploadResponse.data) {
        const bundleId = uploadResponse.data.bundle_id;
        
        // Update files with bundle ID and start polling
        setUploadedFiles((prev) => {
          const updated = prev.map(uploadFile => {
            const matchingFile = files.find(f => f.id === uploadFile.id);
            if (matchingFile) {
              return {
                ...uploadFile,
                bundleId,
                status: 'processing' as const,
                progress: 0,
              };
            }
            return uploadFile;
          });
          return updated;
        });

        // Start progress polling
        startProgressPolling(bundleId);
        
        toast.success(`Files uploaded successfully! Bundle ID: ${bundleId}`);
      } else {
        throw new Error(uploadResponse.error || 'File upload failed');
      }
    } catch (error) {
      console.error('File upload error:', error);
      const errorMessage = error instanceof Error ? error.message : 'File upload failed';
      toast.error(`Upload failed: ${errorMessage}`);
      
      // Mark files as failed
      setUploadedFiles((prev) => prev.map(uploadFile => {
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
  }, [user?.user_id]);

  // Start progress polling for a bundle
  const startProgressPolling = useCallback((bundleId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const statusResponse = await ServiceRegistry.file.getBundleTaskStatus(bundleId);
        
        if (statusResponse.success && statusResponse.data) {
          const status = statusResponse.data;
          
          // Update progress for each file using current state
          setUploadedFiles((prev) => prev.map(uploadFile => {
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
          const bundleStatus = status.status?.toLowerCase();
          if (bundleStatus === 'completed' || bundleStatus === 'success' || bundleStatus === 'failed' || bundleStatus === 'error') {
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
              
              // Extract completed file IDs for callback
              const completedFiles = updatedFiles
        .filter((file) => file.status === "completed")
        .map((file) => file.id);

      if (completedFiles.length > 0) {
        onFilesUploaded(completedFiles);
      }
              
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
  }, [onFilesUploaded]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/plain": [".txt"],
      "application/pdf": [".pdf"],
      "application/msword": [".doc"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        [".docx"],
      "text/csv": [".csv"],
      "application/vnd.ms-excel": [".xls"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [
        ".xlsx",
      ],
    },
    maxFiles: 10,
    maxSize: 50 * 1024 * 1024, // 50MB
    disabled: disabled || isUploading,
  });

  const handleRemoveFile = (fileId: string) => {
    setUploadedFiles((prev) => prev.filter((file) => file.id !== fileId));
    
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
  };

  const handleClearAll = () => {
    // Clear all polling intervals
    Object.values(pollingIntervals).forEach(interval => clearInterval(interval));
    setPollingIntervals({});
    setUploadedFiles([]);
    onUploadStatusChange([]);
    toast.info("All files cleared");
  };

  const handleClose = () => {
    onOpenChange(false);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

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

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="p-0 border-0 bg-transparent"
        showCloseButton={false}
        style={{
          width: "800px",
          maxWidth: "800px",
          maxHeight: "80vh",
        }}
      >
        <div className="modal-enhanced">
          <div className="modal-content-enhanced flex flex-col overflow-hidden">
            <DialogHeader className="modal-header-enhanced">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <DialogTitle className="modal-title-enhanced">
                    File Upload
                  </DialogTitle>
                  <p className="modal-description-enhanced">
                    Upload files to query and analyze their content
                  </p>
                </div>
                <button
                  onClick={handleClose}
                  className="modal-close-button cursor-pointer"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
            </DialogHeader>

            <div className="modal-form-content flex-1 overflow-y-auto px-6 pb-6 min-h-0">
              {/* File Drop Zone */}
              <div
                {...getRootProps()}
                className={`border rounded-2xl cursor-pointer transition-all duration-300 relative overflow-hidden flex-1 w-full ${
                  isDragActive
                    ? "border-green-400 bg-green-400/10 scale-105"
                    : ""
                }`}
              >
                <input {...getInputProps()} />

                <div className="flex h-64">
                  {/* Upload Icon - Left Side - Full width like system card */}
                  <div className="flex-shrink-0 w-1/2 h-full relative overflow-hidden">
                    <img
                      src="/tables/uploadXL.svg"
                      alt="Upload Files"
                      className={`w-full h-full object-cover transition-all duration-300 ${
                        isDragActive ? "scale-110" : ""
                      }`}
                      style={{
                        objectPosition: "center center",
                      }}
                    />
                  </div>

                  {/* Text Content - Right Side */}
                  <div className="flex-1 flex flex-col justify-center">
                    {isDragActive ? (
                      <div className="space-y-1">
                        <p className="text-lg font-semibold text-green-400">
                          Drop the files here...
                        </p>
                        <p className="text-xs text-green-300">Release to upload</p>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <p className="text-lg font-semibold text-white">
                          Drop or select files
                        </p>
                        <p className="text-xs text-slate-400">
                          Drop files here or{" "}
                          <span className="text-green-400 font-medium">
                            click
                          </span>{" "}
                          to browse through your machine
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Authentication Warning */}
              {!user?.user_id && (
                <div className="p-3 bg-yellow-500/20 border border-yellow-400/30 rounded-lg mb-4">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 text-yellow-400" />
                    <span className="text-yellow-400 text-sm font-medium">Authentication Required</span>
                  </div>
                  <p className="text-yellow-300 text-xs mt-1">Please log in to upload files</p>
                </div>
              )}

              {/* Selected Files List */}
              {uploadedFiles.length > 0 && (
                <div className="w-full mt-6">
                  <div className="mb-4">
                    <h3 className="text-white font-medium">Selected Files</h3>
                  </div>

                  <div className="space-y-2 max-h-40 overflow-y-auto">
                    {uploadedFiles.map((file) => (
                      <div
                        key={file.id}
                        className="flex items-center justify-between w-full p-4"
                      >
                        {/* Left Side - File Icon and File Name */}
                        <div className="flex items-center flex-1 min-w-0">
                          <div className="flex-shrink-0">
                            <img
                              src="/tables/excelfile.svg"
                              alt="File"
                              className="h-8 w-8"
                            />
                          </div>
                          <div className="min-w-0 flex-1">
                            <p className="text-white font-medium truncate text-base">
                              {file.file.name}
                            </p>
                            <p className="text-slate-400 text-sm mt-1">
                              {formatFileSize(file.file.size)}
                            </p>
                          </div>
                        </div>

                        {/* Right Side - Status and Actions */}
                        <div className="flex items-center  flex-shrink-0 ml-4">
                            <div className="flex items-center gap-2">
                            {getStatusIcon(file.status)}
                            </div>

                          {/* Only show remove button if file is not completed */}
                          {file.status !== "completed" && (
                            <button
                              onClick={() => handleRemoveFile(file.id)}
                              className="p-2 hover:bg-red-500/20 rounded-lg transition-all duration-200 cursor-pointer group"
                              title="Remove file"
                              disabled={isUploading}
                            >
                              <img
                                src="/tables/cross.svg"
                                alt="Remove"
                                className="h-5 w-5 opacity-70 group-hover:opacity-100 transition-opacity"
                              />
                            </button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="modal-footer-enhanced">
                <div className="flex justify-between items-center w-full">
                  {/* Left side - Cancel and Upload buttons */}
                  <div className="flex gap-3">
                    <Button
                      variant="outline"
                      onClick={handleClose}
                      className="modal-button-secondary"
                    >
                      Cancel
                    </Button>
                    <Button
                      onClick={() => {
                        const pendingFiles = uploadedFiles.filter(f => f.status === 'pending');
                        if (pendingFiles.length > 0) {
                          uploadFiles(pendingFiles);
                        } else {
                        handleClose();
                        }
                      }}
                      disabled={uploadedFiles.length === 0 || isUploading || !user?.user_id}
                      className="modal-button-primary min-w-[140px]"
                    >
                      {isUploading ? "Uploading..." : "Upload Files"}
                    </Button>
                  </div>

                  {/* Right side - Clear All button */}
                  <div style={{ marginRight: "-1.5rem" }}>
                    {uploadedFiles.length > 0 && (
                      <Button
                        onClick={handleClearAll}
                        variant="destructive"
                        className="h-10"
                        style={{
                          background: "var(--error-8, rgba(255, 86, 48, 0.08))",
                          color: "var(--error-main, rgba(255, 86, 48, 1))",
                          border:
                            "1px solid var(--error-16, rgba(255, 86, 48, 0.16))",
                          borderRadius: "99px",
                        }}
                      >
                        Clear All
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}