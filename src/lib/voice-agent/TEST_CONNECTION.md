# WebSocket Connection & Navigation Test Guide

## Quick Test Steps

### 1. **Check WebSocket Connection**
Open browser console and look for:
```
✅ Voice WebSocket connected successfully
🎤 WebSocket URL: wss://176.9.16.194:8200/voice/ws/tools?user_id=frontend_user
🎤 WebSocket readyState: 1
```

### 2. **Test Backend Message Reception**
When backend sends navigation command, you should see:
```
🎤 WebSocket message received: {...}
🎤 Parsed WebSocket data: {...}
🧭 Navigation result message structure: {...}
🧭 ✅ Navigation result received from backend: {...}
🧭 📋 Message details: {...}
🧭 🚀 About to execute navigation command...
```

### 3. **Test Navigation Execution**
You should see:
```
🧭 NavigationService imported successfully
🧭 Current page from NavigationService: [current_page]
🧭 Target page from backend: [target_page]
🧭 Executing page navigation to: [target_page] from: [current_page]
🧭 ✅ Navigation command executed successfully
```

## Debug Commands

### **In Browser Console:**
```javascript
// Check if voice agent is connected
const voiceAgent = document.querySelector('[data-voice-agent]')?.__voiceAgent

// Check WebSocket connection
console.log('WebSocket connected:', voiceAgent?.webSocketService?.isConnected())

// Check current page state
console.log('Current page:', voiceAgent?.getCurrentPageState())

// Force refresh page state
voiceAgent?.refreshPageState()

// Test manual navigation
voiceAgent?.navigateToPage('file-query')
```

## Common Issues & Solutions

### **Issue 1: WebSocket not connecting**
**Check:**
- Backend is running on `176.9.16.194:8200`
- Endpoint `/voice/ws/tools` exists
- No firewall blocking WebSocket connection

**Solution:**
```typescript
// Update config if needed
VOICE_AGENT_CONFIG.BACKEND_URL = 'https://176.9.16.194:8200'
VOICE_AGENT_CONFIG.WEBSOCKET_ENDPOINTS.VOICE_WS = '/voice/ws/tools'
```

### **Issue 2: Messages received but not processed**
**Check:**
- Console shows "WebSocket message received"
- Message format matches expected structure
- No errors in message processing

**Solution:**
```typescript
// Add more logging
console.log('🎤 Raw message:', event.data)
console.log('🎤 Parsed data:', data)
console.log('🎤 Message type:', data.type)
```

### **Issue 3: Navigation not executing**
**Check:**
- NavigationService is imported successfully
- Current page vs target page
- No errors in executeNavigationCommand

**Solution:**
```typescript
// Force navigation execution
NavigationService.executeNavigation('dashboard')
```

## Test Message Format

### **Send this via WebSocket to test:**
```json
{
  "type": "navigation_result",
  "action": "navigate",
  "data": {
    "Action_type": "navigation",
    "interaction_type": "page_navigation",
    "page": "file-query",
    "previous_page": "dashboard"
  }
}
```

### **Expected Response:**
```
🧭 ✅ Navigation result received from backend: {...}
🧭 🚀 About to execute navigation command...
🧭 NavigationService imported successfully
🧭 Current page from NavigationService: dashboard
🧭 Target page from backend: file-query
🧭 Executing page navigation to: file-query from: dashboard
🧭 ✅ Navigation command executed successfully
```

## Connection Status Check

### **WebSocket States:**
- `0` (CONNECTING): Connection is being established
- `1` (OPEN): Connection is ready for communication
- `2` (CLOSING): Connection is closing
- `3` (CLOSED): Connection is closed

### **Health Check:**
```typescript
// Check connection health
const ws = voiceAgent?.webSocketService
console.log('Connected:', ws?.isConnected())
console.log('State:', ws?.getConnectionState())
console.log('URL:', ws?.ws?.url)
```

## Troubleshooting Steps

1. **Check WebSocket Connection**
   - Verify backend is running
   - Check WebSocket endpoint exists
   - Confirm no firewall issues

2. **Check Message Reception**
   - Verify messages are received
   - Check message format
   - Confirm parsing works

3. **Check Navigation Execution**
   - Verify NavigationService import
   - Check current page state
   - Confirm navigation logic

4. **Check State Updates**
   - Verify page state changes
   - Check UI updates
   - Confirm event emission 