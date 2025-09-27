import { ApiResponse, PaginatedResponse } from "@/types/api";

/**
 * Transforms raw API response data to a standardized format
 */
export function transformResponse<T = any>(response: any): ApiResponse<T> {
  // If response is already in the expected format, return it
  if (response && "success" in response && "data" in response) {
    return response as ApiResponse<T>;
  }

  // Handle API response format with status_code and payload
  if (response && "status_code" in response && "payload" in response) {
    return {
      success: response.status_code >= 200 && response.status_code < 300,
      data: response.payload as T,
      timestamp: new Date().toISOString(),
    };
  }

  // Otherwise, transform to standard format
  return {
    success: true,
    data: response as T,
    timestamp: new Date().toISOString(),
  };
}

/**
 * Transforms paginated response data
 */
export function transformPaginatedResponse<T = any>(
  response: any,
  page: number,
  limit: number
): PaginatedResponse<T> {
  // If response is already in the expected format, return it
  if (
    response &&
    "success" in response &&
    "data" in response &&
    "pagination" in response
  ) {
    return response as PaginatedResponse<T>;
  }

  // Extract data array
  const data = Array.isArray(response) ? response : response?.data || [];

  // Calculate pagination info
  const total = response?.total || data.length;
  const totalPages = Math.ceil(total / limit);

  return {
    success: true,
    data: data as T[],
    timestamp: new Date().toISOString(),
    pagination: {
      page,
      limit,
      total,
      totalPages,
      hasNext: page < totalPages,
      hasPrev: page > 1,
    },
  };
}

/**
 * Transforms request data before sending to API
 */
export function transformRequest<T = any>(data: T): any {
  // Handle Date objects
  if (data instanceof Date) {
    return data.toISOString();
  }

  // Handle arrays
  if (Array.isArray(data)) {
    return data.map((item) => transformRequest(item));
  }

  // Handle objects
  if (data !== null && typeof data === "object") {
    return Object.entries(data).reduce((acc, [key, value]) => {
      // Skip undefined values
      if (value === undefined) {
        return acc;
      }

      // Transform value
      acc[key] = transformRequest(value);
      return acc;
    }, {} as Record<string, any>);
  }

  // Return primitive values as is
  return data;
}

/**
 * Transforms error responses
 */
export function transformErrorResponse(error: any): Error {
  // If error is already an Error instance, return it
  if (error instanceof Error) {
    return error;
  }

  // If error has a message property, use it
  if (error && typeof error === "object" && "message" in error) {
    return new Error(error.message as string);
  }

  // Otherwise, create a generic error
  return new Error("An unknown error occurred");
}

/**
 * Transforms file upload data for multipart/form-data requests
 */
export function transformFileUploadData(
  file: File,
  additionalData?: Record<string, any>
): FormData {
  const formData = new FormData();
  formData.append("file", file);

  // Add additional data if provided
  if (additionalData) {
    Object.entries(additionalData).forEach(([key, value]) => {
      if (value !== undefined) {
        formData.append(key, String(value));
      }
    });
  }

  return formData;
}

/**
 * Transforms query parameters for API requests
 */
export function transformQueryParams(
  params: Record<string, any>
): Record<string, string> {
  return Object.entries(params).reduce((acc, [key, value]) => {
    // Skip undefined or null values
    if (value === undefined || value === null) {
      return acc;
    }

    // Handle arrays
    if (Array.isArray(value)) {
      return {
        ...acc,
        [key]: value.join(","),
      };
    }

    // Handle dates
    if (value instanceof Date) {
      return {
        ...acc,
        [key]: value.toISOString(),
      };
    }

    // Handle objects
    if (typeof value === "object") {
      return {
        ...acc,
        [key]: JSON.stringify(value),
      };
    }

    // Handle primitives
    return {
      ...acc,
      [key]: String(value),
    };
  }, {} as Record<string, string>);
}
