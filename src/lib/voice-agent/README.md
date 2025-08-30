# Voice Agent Architecture

This directory contains a clean, modular architecture for the ESAP Voice Agent system. The architecture separates concerns into distinct services and provides a clean API for voice interactions.

## Architecture Overview

```
src/lib/voice-agent/
├── config/           # Configuration and constants
├── services/         # Individual service modules
├── types/           # TypeScript type definitions
├── VoiceAgentService.ts  # Main orchestrator service
├── index.ts         # Public exports
└── README.md        # This file
```

## Core Components

### 1. Types (`types/index.ts`)
- `VoiceMessage`: Represents different types of voice messages
- `NavigationData`: Contains navigation command data
- `VoiceClientState`: Current state of the voice client
- `VoiceClientActions`: Available actions for the voice client
- `VoiceClientHook`: Complete hook interface

### 2. Configuration (`config/index.ts`)
- `VOICE_AGENT_CONFIG`: Centralized configuration object
- Backend URLs and endpoints
- Navigation patterns for voice commands (legacy, now backend-driven)
- Timing configurations
- Default values

### 3. Services

#### MessageService (`services/MessageService.ts`)
- Creates and formats different types of messages
- Handles message ID generation
- Provides formatting utilities for navigation and system messages

#### NavigationService (`services/NavigationService.ts`)
- **Backend-Driven Navigation**: Executes navigation commands from backend
- Detects current page from URL
- Executes navigation actions
- Emits navigation events
- **No longer parses voice commands** - that's handled by backend

#### WebSocketService (`services/WebSocketService.ts`)
- Manages WebSocket connections
- **Handles navigation commands from backend**
- Processes backend messages and executes commands
- Provides reconnection logic
- **Primary navigation execution point**

#### RTVIService (`services/RTVIService.ts`)
- Manages audio connections via RTVI
- Handles voice recognition
- **Audio processing only - no navigation logic**
- Manages audio devices

### 4. Main Service (`VoiceAgentService.ts`)
- Orchestrates all individual services
- Manages overall state
- Provides unified API for voice operations
- Handles service coordination

## Navigation Architecture (Backend-Driven)

### How Navigation Works Now

```
User Voice → Backend AI Processing → Navigation Command → WebSocket → Frontend → Execute
```

**Key Points:**
1. **Backend controls all navigation decisions**
2. **Frontend only executes commands, doesn't decide**
3. **No voice command parsing on frontend**
4. **Clear separation of concerns**

### Backend Command Format

```typescript
// Backend sends this via WebSocket
{
  "Action_type": "navigation",
  "interaction_type": "page_navigation", 
  "page": "file-query",
  "timestamp": "2024-01-01T12:00:00Z",
  "user_id": "voice_user",
  "success": true
}
```

### Frontend Execution

```typescript
// WebSocketService receives and executes commands
private executeNavigationCommand(data: any): void {
  switch (data.interaction_type) {
    case 'page_navigation':
      NavigationService.executeNavigation(data.page)
      break
    case 'button_click':
      NavigationService.executeElementClick(data.element_name)
      break
    // ... other cases
  }
}
```

## Usage

### Basic Hook Usage

```typescript
import { useVoiceClient } from '@/lib/hooks/use-voice-client'

function MyComponent() {
  const {
    isConnected,
    isInConversation,
    connectionStatus,
    messages,
    connect,
    disconnect,
    startConversation,
    stopConversation,
    navigateToPage,
    // ... other methods
  } = useVoiceClient()

  // Use the voice client functionality
}
```

### Direct Service Usage

```typescript
import { VoiceAgentService } from '@/lib/voice-agent'

const voiceAgent = new VoiceAgentService(
  (state) => console.log('State changed:', state),
  (message) => console.log('New message:', message)
)

await voiceAgent.connect()
await voiceAgent.startConversation()
```

## Features

### Voice Navigation
- **Backend-controlled navigation** - no frontend parsing
- Automatic page detection from URL
- Support for all major application pages
- Event emission for UI components

### Audio Management
- RTVI client integration
- Audio device management
- Voice recognition and synthesis
- Connection health monitoring

### Tool Communication
- WebSocket-based tool communication
- Backend integration
- **Direct command execution**
- Error handling and reconnection

### State Management
- Centralized state management
- Reactive state updates
- Message history
- Connection status tracking

## Event System

The voice agent emits various events that components can listen to:

```typescript
// Navigation events (triggered by backend commands)
window.addEventListener('voice-navigation', (event) => {
  const { page, previousPage, type } = event.detail
  // Handle navigation
})

// Click events
window.addEventListener('voice-click', (event) => {
  const { elementName, page, type } = event.detail
  // Handle element clicks
})

// Search events
window.addEventListener('voice-search', (event) => {
  const { query, type, page, interactionType } = event.detail
  // Handle searches
})
```

## Why Backend-Driven Navigation is Better

1. **Single Source of Truth**: Backend controls all navigation decisions
2. **No Duplicate Logic**: Frontend only executes, doesn't decide
3. **Better User Experience**: Navigation happens when backend confirms intent
4. **Easier Debugging**: Clear command flow from backend to frontend
5. **More Reliable**: No false positives from bot confirmation messages
6. **Scalable**: Backend can implement complex navigation logic
7. **Consistent**: All voice interactions go through same backend pipeline

## Migration from Frontend Parsing

The system has been migrated from frontend voice parsing to backend command execution:

- ❌ **Removed**: Frontend voice command parsing
- ❌ **Removed**: Bot transcript navigation analysis
- ✅ **Added**: Backend command execution in WebSocketService
- ✅ **Added**: Clear separation between audio and navigation
- ✅ **Added**: Robust command handling with error handling

## Error Handling

The system includes comprehensive error handling:

- Connection failures with retry logic
- Audio device errors
- WebSocket disconnections
- Invalid navigation commands from backend
- Service initialization failures

## Testing

The architecture supports testing through:

- Mockable service interfaces
- Event-based communication
- Isolated service testing
- Backend command simulation

## Future Enhancements

- Plugin system for custom voice commands
- Advanced audio processing
- Multi-language support
- Voice command learning
- Performance optimizations
- Accessibility improvements
- Backend command validation
- Command queuing and batching 