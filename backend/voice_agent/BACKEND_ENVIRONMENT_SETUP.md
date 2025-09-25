# 🔧 Backend Environment Setup Guide

## Problem Solved ✅

The backend `voice_agent_router.py` had **hardcoded production URLs** that prevented proper environment switching. This has been fixed!

## 🔄 What Was Changed

### 1. **Created Backend Environment File** (`.env`)
```bash
# ===========================================
# BACKEND ENVIRONMENT CONFIGURATION
# ===========================================

# 🔄 ENVIRONMENT MODE
# Values: "development" or "production"
ENVIRONMENT=development

# 🌐 BACKEND URLS
# Development (localhost)
DEV_BACKEND_URL=https://localhost:8200

# Production (server)
PROD_BACKEND_URL=https://176.9.16.194:8200

# 🔑 API KEYS
# GOOGLE_API_KEY=your_google_api_key_here
```

### 2. **Updated `voice_agent_router.py`**
- ✅ Added `get_backend_url()` function
- ✅ Made `/connect` endpoint dynamic
- ✅ Enhanced `/health` endpoint with environment info
- ✅ Removed hardcoded production URLs

### 3. **Dynamic URL Generation**
The backend now automatically:
- Detects environment from `.env` file
- Selects appropriate backend URL
- Converts HTTPS to WSS for WebSockets
- Returns correct URLs in `/connect` endpoint

## 🚀 How to Deploy

### For Development (localhost):
1. **Set environment in `.env`:**
   ```bash
   ENVIRONMENT=development
   ```

2. **Start backend:**
   ```bash
   cd /Users/nilab/Desktop/projects/Esap-database_agent/Knowledge_base_backend
   python main.py --http  # or without --http for HTTPS
   ```

3. **Test:**
   ```bash
   curl -k https://localhost:8200/voice/health
   curl -k -X POST https://localhost:8200/voice/connect -H "Content-Type: application/json" -d '{"user_id": "test"}'
   ```

### For Production (server):
1. **Set environment in `.env`:**
   ```bash
   ENVIRONMENT=production
   ```

2. **Deploy to server:**
   ```bash
   # Copy updated files to production server
   # Restart the backend service
   ```

3. **Test:**
   ```bash
   curl -k https://176.9.16.194:8200/voice/health
   curl -k -X POST https://176.9.16.194:8200/voice/connect -H "Content-Type: application/json" -d '{"user_id": "test"}'
   ```

## 📋 Expected Results

### Health Endpoint Response:
```json
{
  "status": "healthy",
  "environment": {
    "mode": "development",
    "backend_url": "https://localhost:8200",
    "websocket_base": "wss://localhost:8200"
  },
  "websockets": {
    "conversation": "wss://localhost:8200/voice/ws?user_id=your_user_id",
    "tools": "wss://localhost:8200/voice/ws/tools?user_id=your_user_id"
  }
}
```

### Connect Endpoint Response:
```json
{
  "ws_url": "wss://localhost:8200/voice/ws?user_id=test_user",
  "user_id": "test_user",
  "note": "WebSocket connection will use the provided user_id",
  "environment": "development",
  "backend_url": "https://localhost:8200"
}
```

## 🔍 Verification Steps

1. **Check Environment Loading:**
   ```bash
   cd voice_agent
   python3 -c "
   import os
   from dotenv import load_dotenv
   load_dotenv(override=True)
   print('ENVIRONMENT:', os.getenv('ENVIRONMENT'))
   print('DEV_BACKEND_URL:', os.getenv('DEV_BACKEND_URL'))
   print('PROD_BACKEND_URL:', os.getenv('PROD_BACKEND_URL'))
   "
   ```

2. **Test Health Endpoint:**
   ```bash
   curl -k https://your-backend-url/voice/health | python3 -m json.tool
   ```

3. **Test Connect Endpoint:**
   ```bash
   curl -k -X POST https://your-backend-url/voice/connect \
     -H "Content-Type: application/json" \
     -d '{"user_id": "test_user"}' | python3 -m json.tool
   ```

## 🎯 Frontend Integration

The frontend will now work correctly with both environments:

- **Development**: `NEXT_PUBLIC_ENVIRONMENT=development` → connects to localhost
- **Production**: `NEXT_PUBLIC_ENVIRONMENT=production` → connects to server

Both frontend and backend will use the same environment URLs!

## 🚨 Important Notes

1. **Restart Required**: Changes to `.env` require backend restart
2. **File Location**: `.env` must be in `/voice_agent/` directory
3. **Environment Variables**: Must be prefixed correctly (no `NEXT_PUBLIC_` for backend)
4. **SSL Certificates**: Development may need `NODE_TLS_REJECT_UNAUTHORIZED=0`

## 🔧 Troubleshooting

### If tools still don't show:
1. Check backend logs for environment loading
2. Verify `.env` file is in correct location
3. Ensure backend was restarted after changes
4. Test both health and connect endpoints
5. Check frontend environment matches backend

### If WebSocket connections fail:
1. Verify environment URLs are correct
2. Check SSL certificate configuration
3. Test direct WebSocket connection
4. Review browser console for errors
