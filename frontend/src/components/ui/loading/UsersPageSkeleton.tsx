"use client";

import { cn } from "@/lib/utils";
import { LoadingProps } from "./types";

interface UsersPageSkeletonProps extends LoadingProps {
  activeTab?: "mssql" | "vector";
  showTabs?: boolean;
  showSearch?: boolean;
  showActions?: boolean;
  showPagination?: boolean;
  rowCount?: number;
}

export function UsersPageSkeleton({ 
  size = "md", 
  className,
  activeTab = "mssql",
  showTabs = true,
  showSearch = true,
  showActions = true,
  showPagination = true,
  rowCount = 5
}: UsersPageSkeletonProps) {
  const textSizes = {
    xs: "h-3",
    sm: "h-4", 
    md: "h-5",
    lg: "h-6",
    xl: "h-7"
  };

  const cellHeight = textSizes[size];
  const isMSSQL = activeTab === "mssql";

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
          {/* Header Section */}
          <div className="p-6">
            {/* Tabs Skeleton */}
            {showTabs && (
              <div className="flex gap-8 mb-6">
                <div 
                  className="text-sm font-medium pb-2 border-b-2 rounded animate-pulse"
                  style={{ 
                    background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                    width: "200px",
                    height: "24px"
                  }}
                />
                <div 
                  className="text-sm font-medium pb-2 border-b-2 rounded animate-pulse"
                  style={{ 
                    background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                    width: "200px",
                    height: "24px"
                  }}
                />
              </div>
            )}

            {/* Action Bar Skeleton */}
            <div className="flex items-center justify-between">
              {/* Search Input Skeleton */}
              {showSearch && (
                <div className="flex items-center gap-4 flex-1">
                  <div 
                    className="relative flex-1 max-w-md rounded-full animate-pulse"
                    style={{
                      background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                      height: "50px"
                    }}
                  />
                </div>
              )}

              {/* Action Buttons Skeleton */}
              {showActions && (
                <div className="flex items-center gap-3">
                  <div 
                    className="rounded-full animate-pulse"
                    style={{
                      background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                      width: "48px",
                      height: "48px"
                    }}
                  />
                  <div 
                    className="rounded-full animate-pulse"
                    style={{
                      background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                      width: "48px",
                      height: "48px"
                    }}
                  />
                  <div 
                    className="rounded-full animate-pulse"
                    style={{
                      background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                      width: "48px",
                      height: "48px"
                    }}
                  />
                </div>
              )}
            </div>
          </div>

          {/* Table Section Skeleton */}
          <div className="overflow-x-auto px-6 pb-6">
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
                        className="rounded animate-pulse"
                        style={{
                          background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                          width: "20px",
                          height: "20px"
                        }}
                      />
                    </th>
                    <th className="px-6 py-4 text-left">
                      <div 
                        className={cn("rounded animate-pulse", cellHeight, "w-24")}
                        style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                      />
                    </th>
                    {isMSSQL ? (
                      <>
                        <th className="px-6 py-4 text-left">
                          <div 
                            className={cn("rounded animate-pulse", cellHeight, "w-32")}
                            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                          />
                        </th>
                        <th className="px-6 py-4 text-left">
                          <div 
                            className={cn("rounded animate-pulse", cellHeight, "w-28")}
                            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                          />
                        </th>
                        <th className="px-6 py-4 text-left">
                          <div 
                            className={cn("rounded animate-pulse", cellHeight, "w-20")}
                            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                          />
                        </th>
                      </>
                    ) : (
                      <>
                        <th className="px-6 py-4 text-left">
                          <div 
                            className={cn("rounded animate-pulse", cellHeight, "w-24")}
                            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                          />
                        </th>
                        <th className="px-6 py-4 text-left">
                          <div 
                            className={cn("rounded animate-pulse", cellHeight, "w-28")}
                            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                          />
                        </th>
                        <th className="px-6 py-4 text-left">
                          <div 
                            className={cn("rounded animate-pulse", cellHeight, "w-20")}
                            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                          />
                        </th>
                      </>
                    )}
                    <th className="px-6 py-4 text-left">
                      <div 
                        className={cn("rounded animate-pulse", cellHeight, "w-16")}
                        style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                      />
                    </th>
                    <th className="px-6 py-4 text-right text-white font-medium text-sm rounded-tr-xl">
                      <div 
                        className={cn("rounded animate-pulse", cellHeight, "w-16")}
                        style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                      />
                    </th>
                  </tr>
                </thead>
                
                {/* Table Rows Skeleton */}
                <tbody>
                  {Array.from({ length: rowCount }).map((_, index) => (
                    <tr 
                      key={`skeleton-row-${index}`}
                      className="border-b border-white/10"
                    >
                      {/* Checkbox Column */}
                      <td className="px-6 py-4">
                        <div 
                          className="rounded animate-pulse"
                          style={{
                            background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                            width: "20px",
                            height: "20px"
                          }}
                        />
                      </td>
                      
                      {/* User ID Column */}
                      <td className="px-6 py-4">
                        <div 
                          className={cn("rounded animate-pulse", cellHeight, "w-48")}
                          style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                        />
                      </td>
                      
                      {/* Conditional Columns based on tab */}
                      {isMSSQL ? (
                        <>
                          {/* Parent Company Column */}
                          <td className="px-6 py-4">
                            <div 
                              className={cn("rounded animate-pulse", cellHeight, "w-32")}
                              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                            />
                          </td>
                          
                          {/* Sub Companies Column */}
                          <td className="px-6 py-4">
                            <div 
                              className={cn("rounded animate-pulse", cellHeight, "w-24")}
                              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                            />
                          </td>
                          
                          {/* Databases Column */}
                          <td className="px-6 py-4">
                            <div 
                              className={cn("rounded animate-pulse", cellHeight, "w-8")}
                              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                            />
                          </td>
                        </>
                      ) : (
                        <>
                          {/* Database Column */}
                          <td className="px-6 py-4">
                            <div 
                              className={cn("rounded animate-pulse", cellHeight, "w-24")}
                              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                            />
                          </td>
                          
                          {/* Access Level Column */}
                          <td className="px-6 py-4">
                            <div 
                              className={cn("rounded animate-pulse", cellHeight, "w-16")}
                              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                            />
                          </td>
                          
                          {/* Tables Column */}
                          <td className="px-6 py-4">
                            <div 
                              className={cn("rounded animate-pulse", cellHeight, "w-20")}
                              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                            />
                          </td>
                        </>
                      )}
                      
                      {/* Date Column */}
                      <td className="px-6 py-4">
                        <div 
                          className={cn("rounded animate-pulse", cellHeight, "w-20")}
                          style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                        />
                      </td>
                      
                      {/* Actions Column */}
                      <td className="px-6 py-4 text-right">
                        <div 
                          className="rounded-full animate-pulse"
                          style={{
                            background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                            width: "32px",
                            height: "32px"
                          }}
                        />
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
              {/* Left: Selection Status Skeleton */}
              <div className="flex items-center">
                <div 
                  className={cn("rounded animate-pulse", cellHeight, "w-48")}
                  style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                />
              </div>
              
              {/* Right: Pagination Controls Skeleton */}
              <div className="flex items-center gap-6">
                {/* Rows per page selector skeleton */}
                <div className="flex items-center gap-2">
                  <div 
                    className={cn("rounded animate-pulse", cellHeight, "w-24")}
                    style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                  />
                  <div 
                    className="rounded-full animate-pulse"
                    style={{
                      background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                      width: "60px",
                      height: "32px"
                    }}
                  />
                </div>
                
                {/* Page info and controls skeleton */}
                <div className="flex items-center gap-2">
                  <div 
                    className={cn("rounded animate-pulse", cellHeight, "w-20")}
                    style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                  />
                  <div className="flex gap-1">
                    <div 
                      className="rounded-full animate-pulse"
                      style={{
                        background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                        width: "32px",
                        height: "32px"
                      }}
                    />
                    <div 
                      className="rounded-full animate-pulse"
                      style={{
                        background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                        width: "32px",
                        height: "32px"
                      }}
                    />
                    <div 
                      className="rounded-full animate-pulse"
                      style={{
                        background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                        width: "32px",
                        height: "32px"
                      }}
                    />
                    <div 
                      className="rounded-full animate-pulse"
                      style={{
                        background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                        width: "32px",
                        height: "32px"
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
