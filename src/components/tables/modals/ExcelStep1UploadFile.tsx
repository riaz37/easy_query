"use client";

import React, { useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Label } from "@/components/ui/label";
import {
  Upload,
  FileSpreadsheet,
  FileText,
  Trash2,
  ArrowRight,
  AlertCircle,
} from "lucide-react";
import { useDropzone } from "react-dropzone";

interface ExcelStep1UploadFileProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  onNext: () => void;
}

export function ExcelStep1UploadFile({
  onFileSelect,
  selectedFile,
  onNext,
}: ExcelStep1UploadFileProps) {
  // File drop zone configuration
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      onFileSelect(file);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/vnd.ms-excel": [".xls"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
    },
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024, // 50MB
  });

  const handleRemoveFile = () => {
    onFileSelect(null as any);
  };

  return (
    <div className="space-y-6">
      {/* File Drop Zone */}
      <div
        {...getRootProps()}
        className={`modal-input-enhanced border-2 border-dashed rounded-2xl p-16 text-center cursor-pointer transition-all duration-300 relative overflow-hidden ${
          isDragActive
            ? "border-green-400 bg-green-400/10 scale-105"
            : "hover:border-slate-500 hover:bg-slate-700/20"
        }`}
      >
        <input {...getInputProps()} />
        
        {/* Background Circuit Pattern */}
        <div className="absolute inset-0 opacity-20">
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
            <div className="w-32 h-32 border-2 border-dashed border-green-400 rounded-full flex items-center justify-center">
              <div className="w-24 h-24 border border-green-400 rounded-full flex items-center justify-center">
                <div className="w-16 h-16 border border-green-400 rounded-full"></div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="relative z-10 space-y-6">
          {/* Main Upload Icon */}
          <div className="flex justify-center">
            <div className={`p-6 rounded-full transition-all duration-300 ${
              isDragActive 
                ? "bg-green-500/20 text-green-400 scale-110" 
                : "bg-green-500/10 text-green-400"
            }`}>
              <img 
                src="/tables/uploadexcel.svg" 
                alt="Upload Excel" 
                className="h-20 w-20"
              />
            </div>
          </div>
          
          {isDragActive ? (
            <div className="space-y-2">
              <p className="text-lg font-semibold text-green-400">Drop the Excel file here...</p>
              <p className="text-green-300">Release to upload</p>
            </div>
          ) : (
            <div className="space-y-4">
              <p className="text-lg font-semibold text-white">
                Drop or select file
              </p>
              <p className="text-slate-400">
                Drop files here or <span className="text-green-400 font-medium">click</span> to browse through your machine
              </p>
              <div className="flex items-center justify-center gap-2 text-sm text-slate-500">
                <FileText className="h-4 w-4" />
                <span>Supports .xlsx and .xls files up to 50MB</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Selected File Info */}
      {selectedFile && (
        <div className="space-y-4">
          <div className="flex items-center gap-4 p-4 bg-gradient-to-r from-green-500/10 to-blue-500/10 rounded-xl border border-green-500/30">
            <div className="p-2 bg-green-500/20 rounded-lg">
              <img 
                src="/tables/uploadexcel.svg" 
                alt="Upload Excel" 
                className="h-6 w-6"
              />
            </div>
            <div className="flex-1">
              <p className="text-white font-semibold">{selectedFile.name}</p>
              <p className="text-slate-400">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
            <Button
              onClick={handleRemoveFile}
              variant="ghost"
              size="sm"
              className="text-red-400 hover:text-red-300 hover:bg-red-500/10"
            >
              <Trash2 className="h-5 w-5" />
            </Button>
          </div>

          {/* File List */}
          <div className="space-y-2">
            <Label className="text-sm font-medium text-slate-400">Uploaded Files</Label>
            <div className="flex items-center gap-2 p-3 bg-slate-700/30 rounded-lg">
              <img 
                src="/tables/uploadexcel.svg" 
                alt="Upload Excel" 
                className="h-4 w-4"
              />
              <span className="text-white text-sm">{selectedFile.name}</span>
              <Button
                onClick={handleRemoveFile}
                variant="ghost"
                size="sm"
                className="ml-auto text-red-400 hover:text-red-300 hover:bg-red-500/10 p-1"
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Next Button */}
      <div className="flex justify-end">
        <Button
          onClick={onNext}
          disabled={!selectedFile}
          className="modal-button-primary"
        >
          Continue
          <ArrowRight className="h-5 w-5 ml-2" />
        </Button>
      </div>
    </div>
  );
}
