"use client";

import React, { useState, useCallback, useRef } from "react";
import { useDropzone } from "react-dropzone";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { X, Upload } from "lucide-react";
import { toast } from "sonner";
import { Spinner } from "@/components/ui/loading";

interface UploadedFile {
  id: string;
  file: File;
  status: "uploading" | "completed" | "error";
  progress: number;
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
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (disabled || isUploading) return;

      const newFiles: UploadedFile[] = acceptedFiles.map((file) => ({
        id: Math.random().toString(36).substr(2, 9),
        file,
        status: "uploading",
        progress: 0,
      }));

      setUploadedFiles((prev) => [...prev, ...newFiles]);
      onUploadStatusChange([...uploadedFiles, ...newFiles]);

      // Simulate upload process
      newFiles.forEach((uploadedFile) => {
        simulateUpload(uploadedFile);
      });
    },
    [disabled, isUploading, uploadedFiles, onUploadStatusChange]
  );

  const simulateUpload = (uploadedFile: UploadedFile) => {
    const interval = setInterval(() => {
      setUploadedFiles((prev) =>
        prev.map((file) =>
          file.id === uploadedFile.id
            ? {
                ...file,
                progress: Math.min(file.progress + Math.random() * 30, 100),
              }
            : file
        )
      );
    }, 200);

    setTimeout(() => {
      clearInterval(interval);
      setUploadedFiles((prev) =>
        prev.map((file) =>
          file.id === uploadedFile.id
            ? { ...file, status: "completed", progress: 100 }
            : file
        )
      );

      // Extract file IDs for callback
      const completedFiles = uploadedFiles
        .filter((file) => file.status === "completed")
        .map((file) => file.id);
      
      if (completedFiles.length > 0) {
        onFilesUploaded(completedFiles);
      }
    }, 2000);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/plain": [".txt"],
      "application/pdf": [".pdf"],
      "application/msword": [".doc"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
      "text/csv": [".csv"],
      "application/vnd.ms-excel": [".xls"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
    },
    maxFiles: 10,
    maxSize: 50 * 1024 * 1024, // 50MB
    disabled: disabled || isUploading,
  });

  const handleRemoveFile = (fileId: string) => {
    setUploadedFiles((prev) => prev.filter((file) => file.id !== fileId));
    onUploadStatusChange(uploadedFiles.filter((file) => file.id !== fileId));
  };

  const handleClearAll = () => {
    setUploadedFiles([]);
    onUploadStatusChange([]);
    toast.info("All files cleared");
  };


  const handleClose = () => {
    onOpenChange(false);
  };


  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
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
                <button onClick={handleClose} className="modal-close-button cursor-pointer">
                  <X className="h-5 w-5" />
                </button>
              </div>
            </DialogHeader>

            <div className="modal-form-content flex-1 overflow-y-auto px-6 pb-6 min-h-0">
              {/* File Drop Zone */}
              <div
                {...getRootProps()}
                className={`modal-input-enhanced border-2 border-dashed rounded-2xl cursor-pointer transition-all duration-300 relative overflow-hidden flex-1 max-w-2xl mx-auto ${
                  isDragActive
                    ? "border-green-400 bg-green-400/10 scale-105"
                    : "hover:border-slate-500 hover:bg-slate-700/20"
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
                  <div className="flex-1 px-6 flex flex-col justify-center">
                    {isDragActive ? (
                      <div className="space-y-2">
                        <p className="text-lg font-semibold text-green-400">
                          Drop the files here...
                        </p>
                        <p className="text-green-300">Release to upload</p>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <p className="text-lg font-semibold text-white">
                          Drop or select files
                        </p>
                        <p className="text-slate-400">
                          Drop files here or{" "}
                          <span className="text-green-400 font-medium">click</span> to
                          browse through your machine
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Selected Files List */}
              {uploadedFiles.length > 0 && (
                <div className="max-w-2xl mx-auto w-full mt-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-white font-medium">Selected Files</h3>
                    <button
                      onClick={handleClearAll}
                      className="text-red-400 hover:text-red-300 text-sm"
                    >
                      Clear All
                    </button>
                  </div>
                  
                  <div className="space-y-2 max-h-40 overflow-y-auto">
                    {uploadedFiles.map((file) => (
                      <div
                        key={file.id}
                        className="flex items-center justify-between w-full p-4 bg-slate-800/30 rounded-lg border border-slate-600/30"
                      >
                        {/* Left Side - File Name */}
                        <div className="flex items-center gap-4 flex-1 min-w-0">
                          <div className="flex-1 min-w-0">
                            <p className="text-white text-sm font-medium truncate">
                              {file.file.name}
                            </p>
                            <p className="text-slate-400 text-xs">
                              {formatFileSize(file.file.size)}
                            </p>
                          </div>
                        </div>
                        
                        {/* Right Side - Status and Actions */}
                        <div className="flex items-center gap-2">
                          {file.status === "uploading" && (
                            <div className="flex items-center gap-2">
                              <Spinner size="sm" variant="accent-green" />
                              <span className="text-green-400 text-xs">
                                {Math.round(file.progress)}%
                              </span>
                            </div>
                          )}
                          
                          {file.status === "completed" && (
                            <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                          )}
                          
                          <button
                            onClick={() => handleRemoveFile(file.id)}
                            className="text-red-400 hover:text-red-300 p-1"
                          >
                            <X className="h-4 w-4" />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="modal-footer-enhanced">
                <Button
                  variant="outline"
                  onClick={handleClose}
                  className="modal-button-secondary w-full sm:w-auto"
                >
                  Cancel
                </Button>
                <Button
                  onClick={() => {
                    toast.success("Files uploaded successfully!");
                    handleClose();
                  }}
                  disabled={uploadedFiles.length === 0 || isUploading}
                  className="modal-button-primary min-w-[140px] w-full sm:w-auto"
                >
                  {isUploading ? "Uploading..." : "Upload Files"}
                </Button>
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
