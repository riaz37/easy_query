# Remote URL Setup for Image Access

## Overview

This document explains how to configure the system to generate remote URLs for image files instead of local URLs.

## Problem

By default, the system generates local URLs like:
```
http://localhost:8200/storage/graphs/images/graph_bar_1234567890.png
```

But you need remote URLs like:
```
https://your-server.com/storage/graphs/images/graph_bar_1234567890.png
```

## Solution

### Step 1: Set Environment Variables

Add the following environment variables to your `.env` file:

```bash
# Remote server URL for image access
SERVER_URL=https://your-server.com

# Local server URL (optional, for other files)
BASE_URL=http://localhost:8200
```

### Step 2: Verify Configuration

Run the test script to verify your configuration:

```bash
python Report_generator/utilites/test_remote_urls.py
```

### Step 3: Test via API

Use the test endpoints to verify URL generation:

```bash
# Test URL conversion
curl http://localhost:8200/test-url-conversion

# Test image generation with URLs
curl http://localhost:8200/test-image-urls
```

## How It Works

### URL Generation Flow

1. **Image Creation**: `export_graph_to_image()` saves image to local storage
2. **Path Conversion**: `convert_file_path_to_url()` converts local path to remote URL
3. **Remote URL**: Uses `SERVER_URL` environment variable for image URLs
4. **Response**: Returns remote URL in the `files.image_url` field

### Key Functions

- `get_image_server_url()`: Reads `SERVER_URL` from environment
- `convert_file_path_to_url()`: Converts file paths to accessible URLs
- `export_graph_to_image()`: Saves images and generates URLs

### URL Patterns

- **Images**: `{SERVER_URL}/storage/graphs/images/{filename}`
- **HTML**: `{BASE_URL}/storage/graphs/html/{filename}`

## Troubleshooting

### Issue: URLs still showing localhost

**Solution**: Check that `SERVER_URL` is set correctly in your environment:

```bash
echo $SERVER_URL
```

### Issue: Environment variable not being read

**Solution**: Ensure `.env` file is in the project root and contains:

```bash
SERVER_URL=https://your-server.com
```

### Issue: Images not accessible remotely

**Solution**: Ensure your server is configured to serve static files from the storage directory.

### Issue: SSL certificate errors

**Solution**: If using self-signed certificates, the system includes SSL verification bypass in HTTP calls.

## Testing

### Manual Test

1. Set environment variables
2. Generate a report
3. Check the `image_url` field in the response
4. Verify the URL starts with your `SERVER_URL`

### Automated Test

Run the test script:

```bash
python Report_generator/utilites/test_remote_urls.py
```

## Example Response

After configuration, your API response will include:

```json
{
  "success": true,
  "files": {
    "image_url": "https://your-server.com/storage/graphs/images/graph_bar_1234567890.png",
    "html_url": "http://localhost:8200/storage/graphs/html/graph_bar_1234567890.html"
  }
}
```

## Security Considerations

- Ensure your `SERVER_URL` is accessible from your target clients
- Consider using HTTPS for production environments
- Verify that your server properly serves static files from the storage directory
- Implement proper access controls if needed

## Support

If you encounter issues:

1. Check the test script output
2. Verify environment variables
3. Check server logs for errors
4. Use the test endpoints to debug URL generation
