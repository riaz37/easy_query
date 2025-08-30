# Voice Agent Navigation Debug Guide

## How to Test Navigation

### 1. **Check Current Page State**
```typescript
import { useVoiceClient } from '@/lib/hooks/use-voice-client'

function DebugComponent() {
  const { getCurrentPageState, refreshPageState } = useVoiceClient()
  
  const checkState = () => {
    const state = getCurrentPageState()
    console.log('🧭 Current page state:', state)
  }
  
  return (
    <div>
      <button onClick={checkState}>Check Page State</button>
      <button onClick={refreshPageState}>Refresh Page State</button>
    </div>
  )
}
```

### 2. **Monitor Console Logs**
Look for these log messages:
- `🧭 Detected current page from URL: [page]`
- `🧭 Page changed from [old] to [new] (source: [source])`
- `🧭 NavigationService state updated: [old] → [new]`
- `🧭 Navigation command received from backend: [data]`
- `🧭 Executing page navigation to: [page] from: [current]`

### 3. **Test Backend Navigation Command**
Send this via WebSocket to test:
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

### 4. **Expected Flow**
1. **Backend sends** navigation command
2. **WebSocketService receives** and logs the command
3. **NavigationService executes** the navigation
4. **VoiceAgentService updates** its state
5. **Hook notifies** components of state change
6. **UI updates** to reflect new page

### 5. **Common Issues & Solutions**

#### Issue: Navigation not executing
**Check:**
- Console logs for "Navigation command received from backend"
- WebSocket connection status
- Current page vs target page (should be different)

#### Issue: Page state out of sync
**Solution:**
- Call `refreshPageState()` to force refresh
- Check browser console for page change logs
- Verify URL matches expected page

#### Issue: Duplicate navigation
**Check:**
- Backend is not sending multiple commands
- Current page detection is working correctly
- NavigationService state is synchronized

### 6. **Debug Commands**

#### In Browser Console:
```javascript
// Get voice agent instance
const voiceAgent = window.voiceAgent // if exposed

// Check current page
console.log('Current page:', voiceAgent?.getCurrentPageState())

// Force refresh
voiceAgent?.refreshPageState()

// Test navigation
voiceAgent?.navigateToPage('file-query')
```

### 7. **Backend Integration Checklist**

- [ ] Backend sends `type: "navigation_result"`
- [ ] Navigation data in `data` object
- [ ] `Action_type: "navigation"` in data
- [ ] `interaction_type: "page_navigation"`
- [ ] `page` field contains target page
- [ ] `previous_page` field is set correctly

### 8. **Testing Scenarios**

#### Scenario 1: Navigate to different page
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

#### Scenario 2: Navigate to same page (should skip)
```json
{
  "type": "navigation_result",
  "action": "navigate",
  "result": {
    "Action_type": "navigation",
    "interaction_type": "page_navigation",
    "page": "dashboard",
    "previous_page": "dashboard"
  }
}
```

#### Scenario 3: Button click
```json
{
  "type": "navigation_result",
  "action": "navigate",
  "result": {
    "Action_type": "navigation",
    "interaction_type": "button_click",
    "element_name": "upload_button",
    "page": "file-query"
  }
}
```

## Success Indicators

✅ **Navigation working correctly when:**
- Console shows "Navigation command received from backend"
- Console shows "Executing page navigation to: [page] from: [current]"
- Console shows "Navigation command executed successfully"
- Page state updates correctly
- UI reflects the new page

❌ **Navigation failing when:**
- No console logs for navigation commands
- Page state doesn't update
- UI doesn't change
- Errors in console
- WebSocket connection issues 