// SQL Parser utility to transform API response to table visualization format

export interface ParsedTable {
  name: string;
  full_name: string;
  columns: Array<{
    name: string;
    type: string;
    is_primary?: boolean;
    is_foreign?: boolean;
    is_required?: boolean;
  }>;
  relationships: Array<{
    type: string;
    related_table: string;
    via_column: string;
    via_related: string;
  }>;
}

export interface ParsedApiResponse {
  tables: ParsedTable[];
  metadata: {
    total_tables: number;
    matched_tables: number;
    unmatched_tables: number;
  };
}

/**
 * Parse SQL CREATE TABLE statement to extract table information
 */
function parseCreateTableStatement(sql: string): ParsedTable | null {
  try {
    // Extract table name
    const tableNameMatch = sql.match(/CREATE TABLE \[?(\w+)\]?\s*\(/i);
    if (!tableNameMatch) return null;

    const tableName = tableNameMatch[1];
    const fullName = `dbo.${tableName}`;

    // Extract columns
    const columns: ParsedTable["columns"] = [];
    const relationships: ParsedTable["relationships"] = [];

    // Split by lines and process each column/constraint
    const lines = sql.split("\n").map((line) => line.trim());

    for (const line of lines) {
      // Skip empty lines and comments
      if (!line || line.startsWith("/*") || line.startsWith("*/")) continue;

      // Parse column definitions
      const columnMatch = line.match(/^\[?(\w+)\]?\s+(\w+(?:\([^)]+\))?)/);
      if (columnMatch) {
        const [, columnName, columnType] = columnMatch;

        // Skip constraint lines
        if (
          columnName.toUpperCase().startsWith("CONSTRAINT") ||
          columnName.toUpperCase().startsWith("PRIMARY") ||
          columnName.toUpperCase().startsWith("FOREIGN")
        ) {
          continue;
        }

        columns.push({
          name: columnName,
          type: columnType,
          is_primary: line.includes("PRIMARY KEY"),
          is_foreign:
            line.includes("FOREIGN KEY") ||
            columnName.toLowerCase().startsWith("fk_"),
          is_required: line.includes("NOT NULL"),
        });
      }

      // Parse foreign key constraints
      const fkMatch = line.match(
        /FOREIGN KEY\s*\(\[?(\w+)\]?\)\s*REFERENCES\s*\[?(\w+)\]?\s*\(\[?(\w+)\]?\)/i
      );
      if (fkMatch) {
        const [, sourceColumn, targetTable, targetColumn] = fkMatch;
        relationships.push({
          type: "many_to_one",
          related_table: `dbo.${targetTable}`,
          via_column: sourceColumn,
          via_related: targetColumn,
        });
      }
    }

    return {
      name: formatTableName(tableName),
      full_name: fullName,
      columns,
      relationships,
    };
  } catch (error) {
    console.warn("Failed to parse CREATE TABLE statement:", error);
    return null;
  }
}

/**
 * Format table name for display (convert camelCase/snake_case to readable format)
 */
function formatTableName(tableName: string): string {
  // Handle common patterns
  const formatted = tableName
    // Convert camelCase to spaces
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    // Convert snake_case to spaces
    .replace(/_/g, " ")
    // Capitalize first letter of each word
    .replace(/\b\w/g, (l) => l.toUpperCase())
    // Handle common abbreviations
    .replace(/\bId\b/g, "ID")
    .replace(/\bDb\b/g, "DB")
    .replace(/\bApi\b/g, "API")
    .replace(/\bUrl\b/g, "URL");

  return formatted;
}

/**
 * Extract CREATE TABLE statements from API response text
 */
function extractCreateTableStatements(responseText: string): string[] {
  const createTableRegex = /CREATE TABLE[\s\S]*?(?=\n\n\/\*|CREATE TABLE|$)/gi;
  const matches = responseText.match(createTableRegex) || [];
  return matches.filter((match) => match.trim().length > 0);
}

/**
 * Parse the full API response and extract table information
 */
export function parseApiResponse(apiResponse: any): ParsedApiResponse {
  try {
    // Handle different response formats
    let responseText = "";

    // Check if it's the UserCurrentDBTableData format
    if (apiResponse && apiResponse.table_info) {
      // If table_info has tables array (already parsed)
      if (
        apiResponse.table_info.tables &&
        Array.isArray(apiResponse.table_info.tables)
      ) {
        // Check if the tables have proper structure (not just table names)
        const firstTable = apiResponse.table_info.tables[0];
        if (firstTable && (firstTable.table_name || firstTable.name) && firstTable.table_name !== "Unknown") {
          return {
            tables: apiResponse.table_info.tables.map((table: any) => ({
              name: formatTableName(table.table_name || table.name || "Unknown"),
              full_name:
                table.full_name ||
                `dbo.${table.table_name || table.name || "unknown"}`,
              columns: Array.isArray(table.columns) ? table.columns.map((col: any) => ({
                name: col.name || col.column_name || "unknown",
                type: col.type || col.data_type || col.Type || "unknown",
                is_primary: col.is_primary || false,
                is_foreign: col.is_foreign || false,
                is_required: col.is_required || !col.is_nullable || false,
              })) : [],
              relationships: Array.isArray(table.relationships) ? table.relationships : [],
            })),
            metadata: {
              total_tables:
                apiResponse.table_info.metadata?.total_tables ||
                apiResponse.table_info.tables.length,
              matched_tables:
                apiResponse.table_info.metadata?.processed_tables || apiResponse.table_info.tables.length,
              unmatched_tables:
                apiResponse.table_info.metadata?.failed_tables || 0,
            },
          };
        } else {
          // This is likely an array of table names, not full table objects
          const tables: ParsedTable[] = apiResponse.table_info.tables
            .filter((tableName: any) => typeof tableName === 'string' && tableName !== "Unknown")
            .map((tableName: string) => ({
              name: formatTableName(tableName),
              full_name: `dbo.${tableName}`,
              columns: [],
              relationships: [],
            }));

          return {
            tables,
            metadata: {
              total_tables: apiResponse.table_info.metadata?.total_tables || tables.length,
              matched_tables: tables.length,
              unmatched_tables: 0,
            },
          };
        }
      }

      // If table_info is a string (SQL statements)
      if (typeof apiResponse.table_info === "string") {
        responseText = apiResponse.table_info;
      }
      // If table_info has db_schema property
      else if (apiResponse.table_info.db_schema) {
        responseText = apiResponse.table_info.db_schema;
      }
    }
    // Handle direct string response (like from curl)
    else if (typeof apiResponse === "string") {
      responseText = apiResponse;
    }
    // Handle response with db_schema property
    else if (
      apiResponse.db_schema &&
      typeof apiResponse.db_schema === "string"
    ) {
      responseText = apiResponse.db_schema;
    }
    // Handle response with data property
    else if (apiResponse.data && typeof apiResponse.data === "string") {
      responseText = apiResponse.data;
    }

    if (!responseText) {
      // Try to extract from schema_tables list if available
      if (
        apiResponse.schema_tables &&
        Array.isArray(apiResponse.schema_tables)
      ) {
        const tables: ParsedTable[] = apiResponse.schema_tables.map(
          (tableName: string) => ({
            name: formatTableName(tableName),
            full_name: `dbo.${tableName}`,
            columns: [],
            relationships: [],
          })
        );

        return {
          tables,
          metadata: {
            total_tables: tables.length,
            matched_tables: apiResponse.metadata?.total_matches || 0,
            unmatched_tables:
              tables.length - (apiResponse.metadata?.total_matches || 0),
          },
        };
      }

      return {
        tables: [],
        metadata: {
          total_tables: 0,
          matched_tables: 0,
          unmatched_tables: 0,
        },
      };
    }

    // Extract CREATE TABLE statements
    const createTableStatements = extractCreateTableStatements(responseText);

    // Parse each CREATE TABLE statement
    const tables: ParsedTable[] = [];
    for (const statement of createTableStatements) {
      const parsedTable = parseCreateTableStatement(statement);
      if (parsedTable) {
        tables.push(parsedTable);
      }
    }

    // Extract metadata if available
    let metadata = {
      total_tables: tables.length,
      matched_tables: 0,
      unmatched_tables: 0,
    };

    if (typeof apiResponse === "object" && apiResponse.metadata) {
      metadata = {
        total_tables: apiResponse.metadata.total_schema_tables || tables.length,
        matched_tables: apiResponse.metadata.total_matches || 0,
        unmatched_tables:
          (apiResponse.metadata.total_schema_tables || tables.length) -
          (apiResponse.metadata.total_matches || 0),
      };
    }

    return {
      tables,
      metadata,
    };
  } catch (error) {
    console.error("Failed to parse API response:", error);
    return {
      tables: [],
      metadata: {
        total_tables: 0,
        matched_tables: 0,
        unmatched_tables: 0,
      },
    };
  }
}

/**
 * Filter tables to show only those with relationships or important tables
 */
export function filterImportantTables(
  tables: ParsedTable[],
  maxTables: number = 15
): ParsedTable[] {
  // Sort by importance: tables with relationships first, then by column count
  const sortedTables = [...tables].sort((a, b) => {
    const aScore = a.relationships.length * 10 + a.columns.length;
    const bScore = b.relationships.length * 10 + b.columns.length;
    return bScore - aScore;
  });

  return sortedTables.slice(0, maxTables);
}
