import React, { useState, useMemo } from "react";
import {
  useReactTable,
  getCoreRowModel,
  flexRender,
  getSortedRowModel,
  getFilteredRowModel,
  ColumnDef,
} from "@tanstack/react-table";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { cn } from "@/lib/utils";
import FilterIcon from "@/icons/sidebar/filterIcon";
import { useResolvedTheme, useTheme } from "@/store/theme-store";
import { ChevronUp, ChevronDown, MoreVertical } from "lucide-react";

export type TableColumn = {
  key: string;
  label: string;
  sortable?: boolean;
};

export type DataTableProps = {
  columns?: TableColumn[];
  allKeys?: string[];
  data?: Record<string, any>[];
  pageSizeOptions?: number[];
  defaultPageSize?: number;
};

const STATUS_COLORS: Record<string, { variant: "default" | "secondary" | "destructive" | "outline" }> = {
  Pending: { variant: "secondary" },
  Active: { variant: "default" },
  Approved: { variant: "outline" },
  Reject: { variant: "destructive" },
};

function StatusBadge({ status }: { status: string }) {
  const { variant } = STATUS_COLORS[status] || STATUS_COLORS["Pending"];
  return (
    <Badge variant={variant} className="min-w-[72px] justify-center">
      {status}
    </Badge>
  );
}

export default function DataTable({
  columns = [],
  allKeys,
  data = [],
  pageSizeOptions = [7, 10, 20, 30],
  defaultPageSize = 7,
}: DataTableProps) {
  const theme = useTheme();
  const [showFilters, setShowFilters] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [pageSize, setPageSize] = useState(defaultPageSize);
  const [currentPage, setCurrentPage] = useState(1);
  const [sorting, setSorting] = useState<any>([]);
  const [selectedRows, setSelectedRows] = useState<string[]>([]);

  // Filtering logic
  const filteredData = useMemo(() => {
    let filtered = data;
    // Global search
    if (searchTerm) {
      filtered = filtered.filter((row) =>
        Object.values(row).some((val) =>
          String(val).toLowerCase().includes(searchTerm.toLowerCase())
        )
      );
    }
    // Per-column filters
    Object.entries(filters).forEach(([key, value]) => {
      if (value) {
        filtered = filtered.filter((row) =>
          String(row[key] ?? "")
            .toLowerCase()
            .includes(value.toLowerCase())
        );
      }
    });
    return filtered;
  }, [data, searchTerm, filters]);

  // Pagination logic
  const totalPages = Math.ceil(filteredData.length / pageSize);
  const startIndex = (currentPage - 1) * pageSize;
  const endIndex = startIndex + pageSize;
  const currentData = filteredData.slice(startIndex, endIndex);

  // Row selection logic
  const rowKey =
    columns.find((col) => col.key.toLowerCase().includes("id"))?.key ||
    columns[0]?.key ||
    "id";
  const allChecked =
    currentData.length > 0 &&
    currentData.every((row) => selectedRows.includes(String(row[rowKey])));
  const toggleAll = () => {
    if (allChecked) {
      setSelectedRows(
        selectedRows.filter(
          (id) => !currentData.some((row) => String(row[rowKey]) === id)
        )
      );
    } else {
      setSelectedRows([
        ...selectedRows,
        ...currentData
          .map((row) => String(row[rowKey]))
          .filter((id) => !selectedRows.includes(id)),
      ]);
    }
  };
  const toggleRow = (id: string) =>
    setSelectedRows((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );

  // Filter out columns that have no data
  const columnsWithData = useMemo(() => {
    if (!data || data.length === 0) return columns;

    return columns.filter((col) => {
      // Check if any row has non-empty data for this column
      return data.some((row) => {
        const value = row[col.key];
        return (
          value !== null &&
          value !== undefined &&
          value !== "" &&
          String(value).trim() !== ""
        );
      });
    });
  }, [columns, data]);

  // TanStack Table setup
  const tableColumns: ColumnDef<any>[] = useMemo(
    () => [
      {
        id: "select",
        header: ({ table }) => (
          <Checkbox
            checked={allChecked}
            onCheckedChange={toggleAll}
            aria-label="Select all"
          />
        ),
        cell: ({ row }) => (
          <Checkbox
            checked={selectedRows.includes(String(row.original[rowKey]))}
            onCheckedChange={() => toggleRow(String(row.original[rowKey]))}
            aria-label="Select row"
          />
        ),
        enableSorting: false,
        enableHiding: false,
      },
      ...columnsWithData.map((col) => ({
        accessorKey: col.key,
        header: ({ column }: any) => {
          return (
            <Button
              variant="ghost"
              onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
              className="h-auto p-0 font-semibold hover:bg-transparent"
            >
              {col.label}
              {col.sortable && (
                <div className="ml-2 flex flex-col">
                  <ChevronUp className={cn(
                    "h-3 w-3",
                    column.getIsSorted() === "asc" ? "text-foreground" : "text-muted-foreground"
                  )} />
                  <ChevronDown className={cn(
                    "h-3 w-3 -mt-1",
                    column.getIsSorted() === "desc" ? "text-foreground" : "text-muted-foreground"
                  )} />
                </div>
              )}
            </Button>
          );
        },
        cell: ({ getValue }: any) =>
          col.key.toLowerCase() === "status" ? (
            <StatusBadge status={getValue()} />
          ) : (
            <span className="text-sm">{String(getValue() ?? "")}</span>
          ),
        enableSorting: col.sortable,
      })),
      {
        id: "actions",
        header: "",
        cell: () => (
          <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
            <MoreVertical className="h-4 w-4" />
          </Button>
        ),
        enableSorting: false,
        enableHiding: false,
      },
    ],
    [columnsWithData, selectedRows, allChecked]
  );

  const table = useReactTable({
    data: currentData,
    columns: tableColumns,
    state: {
      sorting,
    },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    manualPagination: true,
    pageCount: totalPages,
  });

  return (
    <Card className={cn(
      "w-full p-6",
      theme === "dark" 
        ? "border-[3px] border-[rgba(0,191,111,0.27)] bg-gradient-to-b from-[rgba(0,191,111,0.025)] via-[rgba(0,191,111,0.09)] to-[rgba(0,191,111,0.02)] backdrop-blur-[32px] shadow-none"
        : "border-[1.5px] border-[#e1f4ea] bg-white shadow-[0_2px_12px_0_rgba(0,0,0,0.06)]"
    )}>
      <CardContent className="p-0 space-y-4">
        {/* Filter Toggle */}
        <div className="flex items-center">
          <Button
            variant="outline"
            onClick={() => setShowFilters((v) => !v)}
            className={cn(
              "flex items-center gap-2",
              theme === "dark" 
                ? "bg-white/10 border-none hover:bg-white/20" 
                : "bg-[#f0f9f5] border-[#e1f4ea] hover:bg-[#e1f4ea]"
            )}
          >
            <FilterIcon fill={theme === "dark" ? "#fff" : "#000"} />
            {showFilters ? "Hide Filter" : "Show Filter"}
          </Button>
        </div>

        {/* Filters */}
        {showFilters && (
          <Card className={cn(
            "p-4 space-y-4",
            theme === "dark" 
              ? "bg-white/10 border-none" 
              : "bg-[#f0f9f5] border-[#e1f4ea]"
          )}>
            {/* Global Search */}
            <div className="flex items-center gap-2">
              <Input
                placeholder="Search in all columns..."
                value={searchTerm}
                onChange={(e) => {
                  setSearchTerm(e.target.value);
                  setCurrentPage(1);
                }}
                className="flex-1 max-w-sm"
              />
              {searchTerm && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSearchTerm("")}
                >
                  Clear
                </Button>
              )}
            </div>

            {/* Column Filters */}
            <div className="flex flex-wrap gap-2">
              {columnsWithData.map((col) => (
                <div key={col.key} className="flex items-center gap-2">
                  <Input
                    placeholder={`Filter ${col.label}...`}
                    value={filters[col.key] || ""}
                    onChange={(e) => {
                      setFilters((prev) => ({
                        ...prev,
                        [col.key]: e.target.value,
                      }));
                      setCurrentPage(1);
                    }}
                    className="w-40"
                  />
                  {filters[col.key] && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        setFilters((prev) => {
                          const newFilters = { ...prev };
                          delete newFilters[col.key];
                          return newFilters;
                        });
                        setCurrentPage(1);
                      }}
                    >
                      ×
                    </Button>
                  )}
                </div>
              ))}
            </div>
          </Card>
        )}

        {/* Table */}
        <div className="rounded-lg border overflow-hidden">
          <Table>
            <TableHeader className={cn(
              theme === "dark" 
                ? "bg-white/7" 
                : "bg-[#f0f9f5]"
            )}>
              {table.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id} className="hover:bg-transparent">
                  {headerGroup.headers.map((header) => (
                    <TableHead key={header.id} className="font-semibold">
                      {header.isPlaceholder
                        ? null
                        : flexRender(
                            header.column.columnDef.header,
                            header.getContext()
                          )}
                    </TableHead>
                  ))}
                </TableRow>
              ))}
            </TableHeader>
            <TableBody>
              {table.getRowModel().rows?.length ? (
                table.getRowModel().rows.map((row) => (
                  <TableRow
                    key={row.id}
                    data-state={row.getIsSelected() && "selected"}
                  >
                    {row.getVisibleCells().map((cell) => (
                      <TableCell key={cell.id}>
                        {flexRender(
                          cell.column.columnDef.cell,
                          cell.getContext()
                        )}
                      </TableCell>
                    ))}
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell
                    colSpan={tableColumns.length}
                    className="h-24 text-center"
                  >
                    No results.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>

        {/* Pagination */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">
              Rows per page:
            </span>
            <select
              value={pageSize}
              onChange={(e) => {
                setPageSize(Number(e.target.value));
                setCurrentPage(1);
              }}
              className={cn(
                "border rounded-md px-3 py-1 text-sm bg-background",
                "focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent"
              )}
            >
              {pageSizeOptions.map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-4">
            <span className="text-sm text-muted-foreground">
              {filteredData.length === 0
                ? "0"
                : `${startIndex + 1}-${Math.min(
                    endIndex,
                    filteredData.length
                  )} of ${filteredData.length}`}
            </span>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
                disabled={currentPage === 1}
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() =>
                  setCurrentPage((prev) => Math.min(prev + 1, totalPages))
                }
                disabled={currentPage === totalPages || totalPages === 0}
              >
                Next
              </Button>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}