# Voice Agent Frontend - Troubleshooting Guide

## 🔧 Common Issues and Solutions

### Issue 1: WebSocket Connection Failed
**Symptoms:**
- "WebSocket connection to 'wss://localhost:8200/voice/ws/tools?user_id=frontend_user' failed"
- "WebSocket is closed before the connection is established"

**Solutions:**
1. **Ensure Backend is Running:**
   ```bash
   cd /Users/nilab/Desktop/projects/Esap-database_agent/Knowledge_base_backend
   python main.py
   ```

2. **Check Backend Health:**
   ```bash
   curl -k https://localhost:8200/voice/health
   ```

3. **Accept SSL Certificate:**
   - Open `https://localhost:8200` in your browser
   - Click "Advanced" → "Proceed to localhost (unsafe)"
   - This tells your browser to accept the self-signed certificate

### Issue 2: SSL Certificate Errors
**Symptoms:**
- "ERR_CERT_AUTHORITY_INVALID"
- "Failed to connect / invalid auth bundle from base url"

**Solutions:**
1. **For Chrome/Edge:**
   - Navigate to `chrome://flags/#allow-insecure-localhost`
   - Enable "Allow invalid certificates for resources loaded from localhost"
   - Restart browser

2. **For Firefox:**
   - Go to `about:config`
   - Search for `security.enterprise_roots.enabled`
   - Set to `true`

3. **For Safari:**
   - Go to Safari → Preferences → Advanced
   - Check "Show Develop menu in menu bar"
   - Develop → Disable Cross-Origin Restrictions

### Issue 3: React Key Warnings
**Symptoms:**
- "Encountered two children with the same key"

**Status:** ✅ **Fixed** - Updated message ID generation to use unique keys

### Issue 4: Backend Not Accessible
**Symptoms:**
- "Cannot reach backend - please ensure it is running"

**Solutions:**
1. **Check if backend is running:**
   ```bash
   ps aux | grep python
   ```

2. **Check if port 8200 is in use:**
   ```bash
   lsof -i :8200
   ```

3. **Restart backend:**
   ```bash
   cd /Users/nilab/Desktop/projects/Esap-database_agent/Knowledge_base_backend
   python main.py
   ```

### Issue 5: Frontend Not Loading
**Symptoms:**
- Frontend shows "Disconnected" status
- No error messages in console

**Solutions:**
1. **Check frontend is running:**
   ```bash
   cd voice_agent/frontend
   npm run dev
   ```

2. **Clear browser cache:**
   - Hard refresh: `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (Mac)

3. **Check environment variables:**
   - Ensure `.env.local` exists with correct backend URL

## 🧪 Testing Your Setup

### Use the Connection Test Component
1. Open the frontend at `http://localhost:3000`
2. Look for the "Connection Test" component
3. Click "Run Tests" to check:
   - Health endpoint connectivity
   - Connect endpoint functionality
   - WebSocket connection

### Manual Testing
1. **Test Backend Health:**
   ```bash
   curl -k https://localhost:8200/voice/health
   ```

2. **Test Connect Endpoint:**
   ```bash
   curl -k -X POST https://localhost:8200/voice/connect \
     -H "Content-Type: application/json" \
     -d '{"user_id": "test_user"}'
   ```

3. **Test WebSocket (using wscat):**
   ```bash
   npm install -g wscat
   wscat -c "wss://localhost:8200/voice/ws/tools?user_id=test_user"
   ```

## 🔍 Debug Information

### Frontend Logs
Check browser console for:
- WebSocket connection attempts
- Error messages
- Connection status updates

### Backend Logs
Check terminal where backend is running for:
- WebSocket connection logs
- Error messages
- Request processing logs

### Network Tab
In browser DevTools → Network tab:
- Check for failed requests
- Verify WebSocket connections
- Look for SSL certificate errors

## 🚀 Quick Fix Checklist

- [ ] Backend running on `https://localhost:8200`
- [ ] Frontend running on `http://localhost:3000`
- [ ] SSL certificates exist (`cert.pem`, `key.pem`)
- [ ] Browser accepts self-signed certificates
- [ ] No firewall blocking port 8200
- [ ] Environment variables configured correctly

## 📞 Getting Help

If you're still having issues:

1. **Check the logs** in both frontend console and backend terminal
2. **Use the Connection Test** component to identify specific issues
3. **Verify all prerequisites** from the checklist above
4. **Check the main README** for setup instructions

## 🔄 Reset Everything

If all else fails, try a complete reset:

```bash
# Stop all processes
pkill -f "python main.py"
pkill -f "next dev"

# Restart backend
cd /Users/nilab/Desktop/projects/Esap-database_agent/Knowledge_base_backend
python main.py

# In another terminal, restart frontend
cd voice_agent/frontend
npm run dev
```
