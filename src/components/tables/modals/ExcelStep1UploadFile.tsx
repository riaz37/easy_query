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
    <div className="space-y-6">
      {/* File Drop Zone */}
      <div
        {...getRootProps()}
        className={`modal-input-enhanced border-2 border-dashed rounded-2xl cursor-pointer transition-all duration-300 relative overflow-hidden ${
          isDragActive
            ? "border-green-400 bg-green-400/10 scale-105"
            : "hover:border-slate-500 hover:bg-slate-700/20"
        }`}
      >
        <input {...getInputProps()} />

        <div className="flex h-50">
          {/* Upload Icon - Left Side - Full width like system card */}
          <div className="flex-shrink-0 w-1/2 h-full relative overflow-hidden">
            <img
              src="/tables/uploadexcel.svg"
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
        <div className="space-y-3">
          <div className="flex items-center gap-4 p-4 bg-slate-700/30 rounded-xl border border-slate-600/50">
            {/* Excel Icon */}
            <div className="flex-shrink-0">
              <img
                src="/tables/excelfile.svg"
                alt="Excel File"
                className="h-6 w-6"
              />
            </div>

            {/* File Name */}
            <div className="flex-1 min-w-0">
              <p className="text-white font-medium truncate">
                {selectedFile.name}
              </p>
              <p className="text-slate-400 text-sm">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>

            {/* Remove Button */}
            <button
              onClick={handleRemoveFile}
              className="flex-shrink-0 p-2 hover:bg-red-500/20 rounded-lg transition-colors cursor-pointer"
            >
              <img src="/tables/cross.svg" alt="Remove" className="h-4 w-4" />
            </button>
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
