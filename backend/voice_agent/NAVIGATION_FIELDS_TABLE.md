# 🧭 Voice Agent Navigation System - Complete Field Reference

## 📊 Complete Field Matrix by Action Type

| Field | navigate | click | search | file_search | file_upload | view_report | generate_report |
|-------|----------|-------|--------|-------------|-------------|-------------|-----------------|
| **Action_type** | `"navigation"` | `"clicked"` | `"clicked"` | `"clicked"` | `"clicked"` | `"clicked"` | `"clicked"` |
| **param** | `"name"` | `"clicked,name"` | `"search,question"` | `"query,table_specific,tables[]"` | `"file_descriptions[],table_names[]"` | `"clicked,name,report_request"` | `"clicked,name,report_query"` |
| **value** | `target_page` | `"true,button_name"` | `"true,search_query"` | `"query,table_specific,tables"` | `"file_descriptions,table_names"` | `"true,view report,report_request"` | `"true,report generation,report_query"` |
| **page** | `target_page` | `current_page` | `current_page` | `current_page` | `current_page` | `current_page` | `current_page` |
| **previous_page** | `current_page` | `previous_page` | `previous_page` | `previous_page` | `previous_page` | `previous_page` | `previous_page` |
| **interaction_type** | `"page_navigation"` | `"button_click"` | `"database_search"` | `"file_search"` | `"file_upload"` | `"view_report"` | `"generate_report"` |
| **clicked** | `false` | `true` | `true` | `true` | `true` | `true` | `true` |
| **element_name** | `null` | `button_name` | `"search"` | `"search"` | `"upload"` | `"view report"` | `"report generation"` |
| **search_query** | `null` | `null` | `search_query` | `search_query` | `null` | `null` | `null` |
| **report_request** | `null` | `null` | `null` | `null` | `null` | `report_request` | `null` |
| **report_query** | `null` | `null` | `null` | `null` | `null` | `null` | `report_query` |
| **upload_request** | `null` | `null` | `null` | `null` | `upload_request` | `null` | `null` |
| **db_id** | `null` | `db_id` (if set_db) | `null` | `null` | `null` | `null` | `null` |
| **table_specific** | `false` | `false` | `false` | `table_specific` | `false` | `false` | `false` |
| **tables** | `[]` | `[]` | `[]` | `tables[]` | `[]` | `[]` | `[]` |
| **file_descriptions** | `[]` | `[]` | `[]` | `[]` | `file_descriptions[]` | `[]` | `[]` |
| **table_names** | `[]` | `[]` | `[]` | `[]` | `table_names[]` | `[]` | `[]` |
| **context** | `context` | `context` | `context` | `context` | `context` | `context` | `context` |
| **timestamp** | `ISO_timestamp` | `ISO_timestamp` | `ISO_timestamp` | `ISO_timestamp` | `ISO_timestamp` | `ISO_timestamp` | `ISO_timestamp` |
| **user_id** | `user_id` | `user_id` | `user_id` | `user_id` | `user_id` | `user_id` | `user_id` |
| **success** | `true` | `true` | `true` | `true` | `true` | `true` | `true` |
| **error_message** | `null` | `null` | `null` | `null` | `null` | `null` | `null` |

---

## 🎯 Field Descriptions & Logic

### **Core Action Fields**
- **Action_type**: Distinguishes between `"navigation"` (page changes) and `"clicked"` (user interactions)
- **param**: Comma-separated parameter names for the action
- **value**: Comma-separated parameter values corresponding to param names
- **page**: Current page after action execution
- **previous_page**: Page before action execution
- **interaction_type**: Specific type of interaction performed

### **User Interaction Fields**
- **clicked**: `true` for user interactions, `false` for pure navigation
- **element_name**: Name of the element interacted with (button, form, etc.)
- **search_query**: User's search query for database/file searches
- **report_request**: User's report viewing request
- **report_query**: User's report generation query
- **upload_request**: User's file upload request

### **Database & File Fields**
- **db_id**: Database identifier (for set database actions)
- **table_specific**: Whether search is limited to specific tables
- **tables**: Array of table names for table-specific searches
- **file_descriptions**: Array of file descriptions for uploads
- **table_names**: Array of target table names for uploads

### **System Fields**
- **context**: Additional context from voice agent
- **timestamp**: ISO format timestamp of action
- **user_id**: Identifier of the user performing action
- **success**: Whether action completed successfully
- **error_message**: Error message if action failed

---

## 🖥️ Frontend Logic Implementation

### **1. WebSocket Message Handler**

```javascript
// WebSocket connection and message handling
class VoiceNavigationHandler {
    constructor() {
        this.websocket = null;
        this.currentPage = 'dashboard';
        this.previousPage = null;
        this.userId = 'frontend_user';
        this.initializeWebSocket();
    }

    initializeWebSocket() {
        this.websocket = new WebSocket('ws://localhost:8000/ws/voice');
        
        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleNavigationMessage(data);
        };

        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    handleNavigationMessage(data) {
        console.log('🎯 Received navigation message:', data);
        
        // Update page state
        this.previousPage = this.currentPage;
        this.currentPage = data.page;
        
        // Route to appropriate handler based on interaction_type
        switch(data.interaction_type) {
            case 'page_navigation':
                this.handlePageNavigation(data);
                break;
            case 'button_click':
                this.handleButtonClick(data);
                break;
            case 'database_search':
                this.handleDatabaseSearch(data);
                break;
            case 'file_search':
                this.handleFileSearch(data);
                break;
            case 'file_upload':
                this.handleFileUpload(data);
                break;
            case 'view_report':
                this.handleViewReport(data);
                break;
            case 'generate_report':
                this.handleGenerateReport(data);
                break;
            default:
                console.warn('Unknown interaction type:', data.interaction_type);
        }
    }
}
```

### **2. Action-Specific Handlers**

```javascript
// Page Navigation Handler
handlePageNavigation(data) {
    console.log(`🧭 Navigating from ${data.previous_page} to ${data.page}`);
    
    // Update UI to show new page
    this.showPage(data.page);
    
    // Update navigation state
    this.updateNavigationState(data);
    
    // Show success message
    this.showNotification(`Navigated to ${data.page} page`, 'success');
}

// Button Click Handler
handleButtonClick(data) {
    console.log(`🖱️ Button clicked: ${data.element_name}`);
    
    // Highlight the clicked element
    this.highlightElement(data.element_name);
    
    // Handle special button types
    if (data.element_name === 'set database') {
        this.handleSetDatabase(data);
    } else {
        this.handleGenericButton(data);
    }
    
    // Show feedback
    this.showNotification(`Clicked ${data.element_name}`, 'info');
}

// Database Search Handler
handleDatabaseSearch(data) {
    console.log(`🔍 Database search: ${data.search_query}`);
    
    // Highlight search box
    this.highlightElement('search');
    
    // Show search in progress
    this.showSearchProgress(data.search_query);
    
    // Execute search
    this.executeDatabaseSearch(data.search_query);
    
    // Show results
    this.showNotification(`Searching for: ${data.search_query}`, 'info');
}

// File Search Handler
handleFileSearch(data) {
    console.log(`📁 File search: ${data.search_query}`);
    console.log(`📊 Table specific: ${data.table_specific}`);
    console.log(`📋 Tables: ${data.tables.join(', ')}`);
    
    // Highlight search box
    this.highlightElement('search');
    
    // Show search parameters
    this.showFileSearchParams(data);
    
    // Execute file search
    this.executeFileSearch(data);
    
    // Show results
    const tableInfo = data.table_specific ? ` in tables: ${data.tables.join(', ')}` : '';
    this.showNotification(`File search: ${data.search_query}${tableInfo}`, 'info');
}

// File Upload Handler
handleFileUpload(data) {
    console.log(`📤 File upload: ${data.upload_request}`);
    console.log(`📄 File descriptions: ${data.file_descriptions.join(', ')}`);
    console.log(`📋 Target tables: ${data.table_names.join(', ')}`);
    
    // Highlight upload area
    this.highlightElement('upload');
    
    // Show upload parameters
    this.showUploadParams(data);
    
    // Execute upload
    this.executeFileUpload(data);
    
    // Show results
    const fileInfo = `${data.file_descriptions.length} file(s)`;
    const tableInfo = data.table_names.length > 0 ? ` to tables: ${data.table_names.join(', ')}` : '';
    this.showNotification(`Uploading ${fileInfo}${tableInfo}`, 'info');
}

// Report Handlers
handleViewReport(data) {
    console.log(`📊 View report: ${data.report_request}`);
    
    // Highlight report button
    this.highlightElement('view report');
    
    // Show report viewer
    this.showReportViewer(data.report_request);
    
    this.showNotification(`Viewing report: ${data.report_request}`, 'info');
}

handleGenerateReport(data) {
    console.log(`📈 Generate report: ${data.report_query}`);
    
    // Highlight report generation button
    this.highlightElement('report generation');
    
    // Show report generator
    this.showReportGenerator(data.report_query);
    
    this.showNotification(`Generating report: ${data.report_query}`, 'info');
}
```

### **3. UI Update Methods**

```javascript
// UI Update Methods
class VoiceNavigationHandler {
    // ... existing code ...

    showPage(pageName) {
        // Hide all pages
        document.querySelectorAll('.page').forEach(page => {
            page.style.display = 'none';
        });
        
        // Show target page
        const targetPage = document.getElementById(`${pageName}-page`);
        if (targetPage) {
            targetPage.style.display = 'block';
            targetPage.scrollIntoView({ behavior: 'smooth' });
        }
    }

    highlightElement(elementName) {
        // Remove previous highlights
        document.querySelectorAll('.highlighted').forEach(el => {
            el.classList.remove('highlighted');
        });
        
        // Find and highlight element
        const element = this.findElementByName(elementName);
        if (element) {
            element.classList.add('highlighted');
            
            // Auto-remove highlight after 3 seconds
            setTimeout(() => {
                element.classList.remove('highlighted');
            }, 3000);
        }
    }

    findElementByName(elementName) {
        // Search by various selectors
        const selectors = [
            `[data-element="${elementName}"]`,
            `[aria-label*="${elementName}"]`,
            `[title*="${elementName}"]`,
            `button:contains("${elementName}")`,
            `input[placeholder*="${elementName}"]`,
            `.${elementName.replace(/\s+/g, '-')}`,
            `#${elementName.replace(/\s+/g, '-')}`
        ];
        
        for (const selector of selectors) {
            const element = document.querySelector(selector);
            if (element) return element;
        }
        
        return null;
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }

    updateNavigationState(data) {
        // Update breadcrumb
        this.updateBreadcrumb(data.previous_page, data.page);
        
        // Update page title
        document.title = `${data.page} - Voice Agent`;
        
        // Update navigation menu
        this.updateNavigationMenu(data.page);
    }
}
```

### **4. CSS for Visual Feedback**

```css
/* Highlight styles for voice interactions */
.highlighted {
    animation: voice-highlight 0.5s ease-in-out;
    box-shadow: 0 0 20px rgba(0, 123, 255, 0.5);
    border: 2px solid #007bff;
    border-radius: 4px;
}

@keyframes voice-highlight {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

/* Notification styles */
.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 15px 20px;
    border-radius: 5px;
    color: white;
    font-weight: bold;
    z-index: 1000;
    animation: slide-in 0.3s ease-out;
}

.notification-info {
    background-color: #17a2b8;
}

.notification-success {
    background-color: #28a745;
}

.notification-warning {
    background-color: #ffc107;
    color: #212529;
}

.notification-error {
    background-color: #dc3545;
}

@keyframes slide-in {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

/* Page transition styles */
.page {
    transition: opacity 0.3s ease-in-out;
}

.page.fade-out {
    opacity: 0;
}

.page.fade-in {
    opacity: 1;
}
```

### **5. Error Handling**

```javascript
// Error handling for navigation
class VoiceNavigationHandler {
    // ... existing code ...

    handleError(data) {
        console.error('❌ Navigation error:', data.error_message);
        
        // Show error notification
        this.showNotification(data.error_message, 'error');
        
        // Highlight error state
        this.highlightError(data);
        
        // Log error for debugging
        this.logError(data);
    }

    highlightError(data) {
        // Find the element that caused the error
        const element = this.findElementByName(data.element_name);
        if (element) {
            element.classList.add('error-highlight');
            
            // Add error tooltip
            this.addErrorTooltip(element, data.error_message);
        }
    }

    addErrorTooltip(element, message) {
        const tooltip = document.createElement('div');
        tooltip.className = 'error-tooltip';
        tooltip.textContent = message;
        
        element.appendChild(tooltip);
        
        // Remove tooltip after 5 seconds
        setTimeout(() => {
            tooltip.remove();
            element.classList.remove('error-highlight');
        }, 5000);
    }
}
```

---

## 🎯 Usage Examples

### **Example 1: Database Search**
```javascript
// Voice command: "Search for financial reports"
const response = {
    "Action_type": "clicked",
    "interaction_type": "database_search",
    "element_name": "search",
    "search_query": "financial reports",
    "clicked": true,
    "page": "database-query"
};

// Frontend handles:
// 1. Highlights search box
// 2. Shows "Searching for: financial reports"
// 3. Executes database search
// 4. Displays results
```

### **Example 2: File Upload**
```javascript
// Voice command: "Upload sales data to finance table"
const response = {
    "Action_type": "clicked",
    "interaction_type": "file_upload",
    "element_name": "upload",
    "upload_request": "sales data",
    "file_descriptions": ["sales data"],
    "table_names": ["finance"],
    "clicked": true,
    "page": "file-query"
};

// Frontend handles:
// 1. Highlights upload area
// 2. Shows "Uploading 1 file(s) to tables: finance"
// 3. Opens file selector
// 4. Processes upload
```

### **Example 3: Page Navigation**
```javascript
// Voice command: "Go to dashboard"
const response = {
    "Action_type": "navigation",
    "interaction_type": "page_navigation",
    "element_name": null,
    "clicked": false,
    "page": "dashboard",
    "previous_page": "database-query"
};

// Frontend handles:
// 1. Smoothly transitions to dashboard
// 2. Updates navigation state
// 3. Shows "Navigated to dashboard page"
```

---

## 🚀 Implementation Checklist

### **Frontend Setup**
- [ ] Initialize WebSocket connection
- [ ] Create message handler
- [ ] Implement action-specific handlers
- [ ] Add visual feedback (highlighting, notifications)
- [ ] Set up error handling
- [ ] Add CSS animations and transitions

### **Element Identification**
- [ ] Add `data-element` attributes to interactive elements
- [ ] Ensure proper ARIA labels
- [ ] Test element finding logic
- [ ] Add fallback selectors

### **Testing**
- [ ] Test all action types
- [ ] Verify visual feedback
- [ ] Test error scenarios
- [ ] Validate WebSocket reconnection
- [ ] Test on different browsers

### **Performance**
- [ ] Optimize element finding
- [ ] Debounce rapid voice commands
- [ ] Cache frequently accessed elements
- [ ] Monitor WebSocket performance

---

## 📝 Notes

1. **Element Names**: Use consistent naming for `data-element` attributes
2. **Error Handling**: Always provide user feedback for errors
3. **Accessibility**: Ensure voice navigation works with screen readers
4. **Performance**: Optimize for real-time voice interaction
5. **Testing**: Test with various voice commands and scenarios

This comprehensive system provides a robust foundation for voice-controlled web navigation with clear visual feedback and error handling.
