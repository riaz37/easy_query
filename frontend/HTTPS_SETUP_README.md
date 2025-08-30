# Voice Agent Frontend - HTTPS Setup Guide

## 🔒 HTTPS Configuration Complete

The frontend has been updated to work with the HTTPS backend configuration. Here's what has been changed and how to run it.

## 📋 What Was Fixed

### ✅ Backend Changes:
1. **SSL Certificates**: Created self-signed certificates (`cert.pem`, `key.pem`) for development
2. **Router Prefix**: Added `/voice` prefix to all voice agent endpoints
3. **WSS URLs**: Updated all WebSocket URLs to use `wss://` (secure WebSocket)
4. **Endpoint Updates**: All health check and connect endpoints now return correct HTTPS/WSS URLs

### ✅ Frontend Changes:
1. **Configuration System**: Created centralized config in `lib/config.ts`
2. **Environment Variables**: Added `env.local` for backend URL configuration
3. **WebSocket URLs**: Updated to use WSS with proper `/voice` prefix
4. **SSL Development**: Added Next.js configuration for self-signed certificates
5. **HTTPS Scripts**: Added `dev:https` and `start:https` npm scripts

## 🚀 How to Run

### 1. Start the Backend (HTTPS)
```bash
# From the root directory
cd /Users/nilab/Desktop/projects/Esap-database_agent/Knowledge_base_backend
python main.py
```
The backend will start on: `https://localhost:8200`

### 2. Start the Frontend
```bash
# From the frontend directory
cd voice_agent/frontend
npm run dev
```
The frontend will start on: `http://localhost:3000` (with SSL certificate handling for backend connections)

## 🔗 Updated Endpoints

### Backend Endpoints (with /voice prefix):
- Health Check: `https://localhost:8200/voice/health`
- Connect: `https://localhost:8200/voice/connect`
- Test Database: `https://localhost:8200/voice/test-database-search`

### WebSocket Endpoints:
- Voice Conversation: `wss://localhost:8200/voice/ws?user_id=your_user_id`
- Tool Commands: `wss://localhost:8200/voice/ws/tools?user_id=your_user_id`

## ⚙️ Configuration

### Environment Variables (env.local):
```bash
# Backend URL
NEXT_PUBLIC_BACKEND_URL=https://localhost:8200

# Development settings
NEXT_PUBLIC_DEV_MODE=true
NEXT_PUBLIC_IGNORE_SSL_ERRORS=true
```

### For Production:
Update the backend URL in `env.local`:
```bash
NEXT_PUBLIC_BACKEND_URL=https://your-production-domain.com
```

## 🔧 SSL Certificate Handling

### Development (Self-Signed):
- Certificates are automatically created in the backend root directory
- Browser will show security warnings - click "Proceed anyway"
- For Chrome: Navigate to `chrome://flags/#allow-insecure-localhost` and enable

### Production:
- Replace `cert.pem` and `key.pem` with valid SSL certificates
- Update the backend URL in environment variables

## 🎯 Features Working:
- ✅ Voice conversation over secure WebSocket (WSS)
- ✅ Tool commands via separate secure WebSocket
- ✅ Real-time message monitoring
- ✅ Database search testing
- ✅ CORS properly configured for HTTPS
- ✅ Self-signed certificate support

## 🐛 Troubleshooting

### Common Issues:

1. **"Certificate not trusted" error**:
   - Accept the self-signed certificate in your browser
   - Navigate to `https://localhost:8200` first to accept the backend certificate

2. **WebSocket connection failed**:
   - Ensure backend is running on HTTPS
   - Check that SSL certificates exist in the backend directory
   - Verify the browser accepts self-signed certificates

3. **CORS errors**:
   - The backend is configured to allow all origins for development
   - For production, update CORS settings in `main.py`

### Test the Setup:
1. Visit `https://localhost:8200/voice/health` - should return JSON health status
2. Visit `https://localhost:3000` - frontend should load
3. Click "Connect to Voice Agent" - should establish WebSocket connection
4. Use the "Test Tool" button to verify database search functionality

## 📚 Technical Details

### WebSocket Configuration:
- Uses Pipecat's WebSocket transport with protobuf serialization
- Automatic reconnection on connection loss
- User session management with unique user IDs
- Tool command routing via separate WebSocket channel

### Security:
- All connections use TLS/SSL encryption
- Self-signed certificates for development
- Environment-based configuration for different deployment environments
- CORS protection configured

## 🎉 Ready to Use!

Your voice agent frontend is now fully configured for HTTPS operation with the backend. The system supports:
- Secure voice conversations
- Real-time tool interactions
- Database search capabilities
- WebSocket monitoring and debugging
