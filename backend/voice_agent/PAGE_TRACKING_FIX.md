# Page Tracking Fix for Voice Agent

## Issue Description

The voice agent was experiencing a mismatch between the AI's understanding of the current page and the navigation tool's internal page tracking. This caused the following problems:

### Symptoms
1. **AI says**: "You are currently on the tables page"
2. **Navigation tool reports**: "Page 'dashboard' has no interactive elements"
3. **User confusion**: The AI thinks the user is on one page, but the navigation tool thinks they're on another

### Root Cause
The navigation tool was always initializing user sessions with "dashboard" as the default current page, regardless of the actual page the user was on. This happened because:

1. The `NavigationTool` class had a hardcoded default of "dashboard" in its `__init__` method
2. When creating new user sessions, it always started with "dashboard"
3. The actual current page from the WebSocket connection wasn't being passed to the navigation tool
4. **CRITICAL**: The agent manager was being registered with a global navigation tool instead of the session-specific tool

## Solution Implemented

### 1. Updated NavigationTool Constructor
Modified the `NavigationTool` class to accept an `initial_current_page` parameter:

```python
def __init__(self, rtvi_processor: RTVIProcessor, task=None, initial_current_page: str = "dashboard"):
    super().__init__(rtvi_processor, task)
    self.user_sessions: Dict[str, Dict[str, Any]] = {}
    self.initial_current_page = initial_current_page
```

### 2. Updated User Session Initialization
Modified the user session initialization to use the provided initial current page:

```python
if user_id not in self.user_sessions:
    self.user_sessions[user_id] = {
        "current_page": self.initial_current_page,  # Use initial current page
        "previous_page": None,
        "interaction_history": []
    }
```

### 3. Updated Factory Function
Modified the `create_navigation_tool` factory function to accept and pass the initial current page:

```python
def create_navigation_tool(rtvi_processor: RTVIProcessor, task=None, initial_current_page: str = "dashboard") -> NavigationTool:
    return NavigationTool(rtvi_processor, task, initial_current_page)
```

### 4. Updated Voice Agent Router
Updated all places where navigation tools are created to pass the current page:

```python
# In create_gemini_live_llm
navigation_tool_temp = create_navigation_tool(rtvi_temp, initial_current_page=current_page)

# In create_text_gemini_live_llm
navigation_tool_temp = create_navigation_tool(rtvi_temp, initial_current_page=current_page)

# In run_simplified_conversation_bot
session_navigation_tool = create_navigation_tool(rtvi, initial_current_page=current_page)

# In run_text_conversation_bot
navigation_tool_instance = create_navigation_tool(rtvi, initial_current_page=current_page)
```

### 5. Fixed Agent Manager Registration
**CRITICAL FIX**: Updated the agent manager to use the session-specific navigation tool instead of the global one:

```python
# Before (WRONG):
agent_manager.register_tools(
    [navigation_tool.get_tool_definition()],  # Global tool
    {"navigation": navigation_tool}           # Global tool
)

# After (CORRECT):
agent_manager.register_tools(
    [session_navigation_tool.get_tool_definition()],  # Session-specific tool
    {"navigation": session_navigation_tool}           # Session-specific tool
)
```

### 6. Removed Global Navigation Tool
Removed the global navigation tool creation that was causing the issue:

```python
# Removed:
# navigation_tool = create_navigation_tool(rtvi)  # Global tool without current page
```

### 7. Removed Manual Initialization
Removed the manual initialization code that was trying to set the current page after tool creation, since the tool now initializes with the correct page.

## Testing

### Test Scripts
Created comprehensive test suites to verify the fix:

```bash
cd voice_agent
python test_page_tracking.py      # Basic page tracking tests
python test_current_page_fix.py   # Agent manager integration tests
```

### Test Cases
1. **Initial Page Tracking**: Verify tool starts with correct initial page
2. **Page Navigation**: Verify navigation between pages works correctly
3. **Element Interaction**: Verify elements are available on correct pages
4. **Multiple Users**: Verify independent page tracking for different users
5. **Agent Manager Integration**: Verify agent manager uses correct navigation tool
6. **Session-Specific Tools**: Verify each session has its own tool instance

## Expected Behavior After Fix

### Before Fix
```
User: "Can you load the tables?"
AI: "You are currently on the tables page"
Navigation Tool: "Page 'dashboard' has no interactive elements"
AI: "The system indicated that the dashboard page has no interactive elements, which is unexpected since we are on the tables page"
```

### After Fix
```
User: "Can you load the tables?"
AI: "You are currently on the tables page"
Navigation Tool: "Successfully clicked 'load tables' on tables page"
AI: "I've loaded the tables for you. The tables are now displayed on your screen."
```

## Files Modified

1. **`voice_agent/tools/navigation_tool.py`**
   - Updated constructor to accept `initial_current_page`
   - Updated user session initialization
   - Updated factory function

2. **`voice_agent/voice_agent_router.py`**
   - Updated all navigation tool creation calls
   - **CRITICAL**: Fixed agent manager registration to use session-specific tools
   - Removed global navigation tool creation
   - Removed manual initialization code
   - Updated function signatures

3. **`voice_agent/test_page_tracking.py`** (new)
   - Basic page tracking test suite

4. **`voice_agent/test_current_page_fix.py`** (new)
   - Agent manager integration test suite

5. **`voice_agent/PAGE_TRACKING_FIX.md`** (this file)
   - Complete documentation

## Key Insights

### The Critical Issue
The main problem wasn't just the navigation tool initialization - it was that the agent manager was being registered with the **wrong navigation tool instance**. The system was:

1. Creating a session-specific navigation tool with the correct current page
2. But registering the agent manager with a global navigation tool that had "dashboard" as the default
3. When function calls were made, the agent manager used the global tool instead of the session-specific one

### The Solution
The fix involved ensuring that:
1. Each session creates its own navigation tool with the correct current page
2. The agent manager is registered with the session-specific tool, not the global one
3. All tool references use the session-specific instances

## Benefits

1. **Consistent Page Tracking**: AI and navigation tool now agree on current page
2. **Better User Experience**: No more confusing error messages
3. **Proper Functionality**: Page-specific elements work correctly
4. **Multi-User Support**: Each user has independent page tracking
5. **Session Isolation**: Each session has its own tool instance
6. **Maintainable Code**: Clear separation of concerns

## Future Considerations

1. **Page State Persistence**: Consider persisting page state across sessions
2. **Page Validation**: Add validation for page names and transitions
3. **Page History**: Track page navigation history for better context
4. **Page-Specific Tools**: Add page-specific tool availability
5. **Tool Instance Management**: Implement proper cleanup of session-specific tools
