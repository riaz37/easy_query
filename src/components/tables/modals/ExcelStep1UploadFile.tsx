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
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (file) {
        onFileSelect(file);
      }
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/vnd.ms-excel": [".xls"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [
        ".xlsx",
      ],
    },
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024, // 50MB
  });

  const handleRemoveFile = () => {
    onFileSelect(null as any);
  };

  return (
    <div className="space-y-6 h-full flex flex-col">
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
              alt="Upload Excel"
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
                  Drop the Excel file here...
                </p>
                <p className="text-green-300">Release to upload</p>
              </div>
            ) : (
              <div className="space-y-2">
                <p className="text-lg font-semibold text-white">
                  Drop or select file
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

      {/* Selected File List */}
      {selectedFile && (
        <div className="max-w-2xl mx-auto w-full">
          <div className="flex items-center justify-between w-full p-4">
            {/* Left Side - Excel Icon and File Name */}
            <div className="flex items-center gap-4 flex-1 min-w-0">
              <div className="flex-shrink-0">
                <img
                  src="/tables/excelfile.svg"
                  alt="Excel File"
                  className="h-8 w-8"
                />
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-white font-medium truncate text-base">
                  {selectedFile.name}
                </p>
                <p className="text-slate-400 text-sm mt-1">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>

            {/* Right Side - Remove Button */}
            <div className="flex-shrink-0 ml-4">
              <button
                onClick={handleRemoveFile}
                className="p-2 hover:bg-red-500/20 rounded-lg transition-all duration-200 cursor-pointer group"
                title="Remove file"
              >
                <img 
                  src="/tables/cross.svg" 
                  alt="Remove" 
                  className="h-5 w-5 opacity-70 group-hover:opacity-100 transition-opacity" 
                />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Next Button */}
      <div className="modal-footer-enhanced">
        <Button
          onClick={onNext}
          disabled={!selectedFile}
          className="modal-button-primary w-full sm:w-auto"
        >
          Continue
        </Button>
      </div>
    </div>
  );
}