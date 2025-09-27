"use client";

import { cn } from "@/lib/utils";
import { LoadingProps } from "./types";

interface ReportListSkeletonProps extends LoadingProps {
  reportCount?: number;
  showActions?: boolean;
  showPagination?: boolean;
}

export function ReportListSkeleton({ 
  size = "md", 
  className,
  reportCount = 5,
  showActions = true,
  showPagination = true
}: ReportListSkeletonProps) {
  const cellSizes = {
    xs: "h-3",
    sm: "h-4", 
    md: "h-5",
    lg: "h-6",
    xl: "h-7"
  };

  const cellHeight = cellSizes[size];

  return (
    <div className={cn("w-full", className)}>
      {/* Modal Container Skeleton */}
      <div className="modal-enhanced">
        <div 
          className="modal-content-enhanced overflow-hidden"
          style={{
            background: `linear-gradient(0deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.03)),
linear-gradient(230.27deg, rgba(19, 245, 132, 0) 71.59%, rgba(19, 245, 132, 0.2) 98.91%),
linear-gradient(67.9deg, rgba(19, 245, 132, 0) 66.65%, rgba(19, 245, 132, 0.2) 100%)`,
            backdropFilter: "blur(30px)"
          }}
        >
          {/* Reports List Skeleton */}
          <div className="p-6">
            <div className="rounded-t-xl overflow-hidden">
              <table className="w-full">
                {/* Table Header Skeleton */}
                <thead>
                  <tr 
                    style={{
                      background: "var(--components-Table-Head-filled, rgba(145, 158, 171, 0.08))",
                      borderRadius: "12px 12px 0 0"
                    }}
                  >
                    <th className="px-6 py-4 text-left rounded-tl-xl">
                      <div 
                        className={cn("rounded animate-pulse", cellHeight, "w-32")}
                        style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                      />
                    </th>
                    <th className="px-6 py-4 text-left">
                      <div 
                        className={cn("rounded animate-pulse", cellHeight, "w-20")}
                        style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                      />
                    </th>
                    <th className="px-6 py-4 text-left">
                      <div 
                        className={cn("rounded animate-pulse", cellHeight, "w-16")}
                        style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                      />
                    </th>
                    <th className="px-6 py-4 text-right rounded-tr-xl">
                      <div 
                        className={cn("rounded animate-pulse", cellHeight, "w-20")}
                        style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                      />
                    </th>
                  </tr>
                </thead>
                
                {/* Table Rows Skeleton */}
                <tbody>
                  {Array.from({ length: reportCount }).map((_, index) => (
                    <tr 
                      key={`skeleton-row-${index}`}
                      className="border-b border-white/10"
                    >
                      {/* Report Title Column */}
                      <td className="px-6 py-4">
                        <div className="space-y-2">
                          <div 
                            className={cn("rounded animate-pulse", cellHeight, "w-48")}
                            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                          />
                          <div 
                            className={cn("rounded animate-pulse", cellHeight, "w-32")}
                            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                          />
                        </div>
                      </td>
                      
                      {/* Created Date Column */}
                      <td className="px-6 py-4">
                        <div className="space-y-1">
                          <div 
                            className={cn("rounded animate-pulse", cellHeight, "w-24")}
                            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                          />
                          <div 
                            className={cn("rounded animate-pulse", cellHeight, "w-16")}
                            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                          />
                        </div>
                      </td>
                      
                      {/* Queries Count Column */}
                      <td className="px-6 py-4">
                        <div 
                          className={cn("rounded animate-pulse", cellHeight, "w-8")}
                          style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                        />
                      </td>
                      
                      {/* Actions Column */}
                      <td className="px-6 py-4 text-right">
                        {showActions && (
                          <div className="flex items-center gap-2 justify-end">
                            {/* View Button Skeleton */}
                            <div 
                              className="rounded-full animate-pulse h-8 w-20"
                              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                            />
                            {/* Download Button Skeleton */}
                            <div 
                              className="rounded-full animate-pulse h-8 w-24"
                              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                            />
                          </div>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Pagination Controls Skeleton */}
          {showPagination && (
            <div className="px-6 py-4 flex items-center justify-between">
              {/* Pagination Info Skeleton */}
              <div className="space-y-1">
                <div 
                  className={cn("rounded animate-pulse", cellHeight, "w-48")}
                  style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                />
              </div>
              
              {/* Pagination Buttons Skeleton */}
              <div className="flex items-center gap-2">
                <div 
                  className="rounded-full animate-pulse h-8 w-20"
                  style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                />
                <div 
                  className="rounded animate-pulse h-6 w-16"
                  style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                />
                <div 
                  className="rounded-full animate-pulse h-8 w-16"
                  style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
