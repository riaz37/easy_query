/**
 * 🧭 Voice Agent Navigation System - Frontend Implementation
 * Complete JavaScript implementation for voice-controlled web navigation
 * 
 * Features:
 * - WebSocket connection to voice agent backend
 * - Real-time navigation message handling
 * - Visual feedback (highlighting, notifications)
 * - Error handling and recovery
 * - Page transitions and state management
 */

class VoiceNavigationHandler {
    constructor(config = {}) {
        // Configuration
        this.websocketUrl = config.websocketUrl || 'ws://localhost:8000/ws/voice';
        this.userId = config.userId || 'frontend_user';
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000;
        
        // State management
        this.websocket = null;
        this.currentPage = 'dashboard';
        this.previousPage = null;
        this.isConnected = false;
        this.messageQueue = [];
        
        // UI elements cache
        this.elementCache = new Map();
        this.notificationContainer = null;
        
        // Initialize
        this.initialize();
    }

    /**
     * Initialize the voice navigation system
     */
    initialize() {
        console.log('🎯 Initializing Voice Navigation Handler...');
        
        // Create notification container
        this.createNotificationContainer();
        
        // Initialize WebSocket connection
        this.initializeWebSocket();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Show initial status
        this.showNotification('Voice navigation system initialized', 'success');
    }

    /**
     * Create notification container for displaying messages
     */
    createNotificationContainer() {
        this.notificationContainer = document.createElement('div');
        this.notificationContainer.id = 'voice-notification-container';
        this.notificationContainer.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            pointer-events: none;
        `;
        document.body.appendChild(this.notificationContainer);
    }

    /**
     * Initialize WebSocket connection
     */
    initializeWebSocket() {
        try {
            this.websocket = new WebSocket(this.websocketUrl);
            
            this.websocket.onopen = () => {
                console.log('✅ WebSocket connected to voice agent');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.showNotification('Voice agent connected', 'success');
                
                // Process any queued messages
                this.processMessageQueue();
            };
            
            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleNavigationMessage(data);
                } catch (error) {
                    console.error('❌ Error parsing WebSocket message:', error);
                    this.showNotification('Error processing voice command', 'error');
                }
            };
            
            this.websocket.onclose = () => {
                console.log('🔌 WebSocket connection closed');
                this.isConnected = false;
                this.showNotification('Voice agent disconnected', 'warning');
                this.attemptReconnect();
            };
            
            this.websocket.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
                this.showNotification('Voice agent connection error', 'error');
            };
            
        } catch (error) {
            console.error('❌ Failed to initialize WebSocket:', error);
            this.showNotification('Failed to connect to voice agent', 'error');
        }
    }

    /**
     * Attempt to reconnect to WebSocket
     */
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`🔄 Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            
            setTimeout(() => {
                this.initializeWebSocket();
            }, this.reconnectDelay * this.reconnectAttempts);
        } else {
            console.error('❌ Max reconnection attempts reached');
            this.showNotification('Voice agent connection lost. Please refresh the page.', 'error');
        }
    }

    /**
     * Set up event listeners for page interactions
     */
    setupEventListeners() {
        // Listen for page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible' && !this.isConnected) {
                this.attemptReconnect();
            }
        });
        
        // Listen for window focus to check connection
        window.addEventListener('focus', () => {
            if (!this.isConnected) {
                this.attemptReconnect();
            }
        });
    }

    /**
     * Handle incoming navigation messages from voice agent
     */
    handleNavigationMessage(data) {
        console.log('🎯 Received navigation message:', data);
        
        // Validate message structure
        if (!this.validateMessage(data)) {
            console.error('❌ Invalid message structure:', data);
            return;
        }
        
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
                console.warn('⚠️ Unknown interaction type:', data.interaction_type);
                this.showNotification(`Unknown action: ${data.interaction_type}`, 'warning');
        }
    }

    /**
     * Validate message structure
     */
    validateMessage(data) {
        const requiredFields = ['interaction_type', 'page', 'Action_type', 'clicked'];
        return requiredFields.every(field => data.hasOwnProperty(field));
    }

    /**
     * Handle page navigation
     */
    handlePageNavigation(data) {
        console.log(`🧭 Navigating from ${data.previous_page} to ${data.page}`);
        
        // Update UI to show new page
        this.showPage(data.page);
        
        // Update navigation state
        this.updateNavigationState(data);
        
        // Show success message
        this.showNotification(`Navigated to ${data.page} page`, 'success');
        
        // Trigger page-specific initialization
        this.initializePage(data.page);
    }

    /**
     * Handle button clicks
     */
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

    /**
     * Handle database search
     */
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

    /**
     * Handle file search
     */
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

    /**
     * Handle file upload
     */
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

    /**
     * Handle view report
     */
    handleViewReport(data) {
        console.log(`📊 View report: ${data.report_request}`);
        
        // Highlight report button
        this.highlightElement('view report');
        
        // Show report viewer
        this.showReportViewer(data.report_request);
        
        this.showNotification(`Viewing report: ${data.report_request}`, 'info');
    }

    /**
     * Handle generate report
     */
    handleGenerateReport(data) {
        console.log(`📈 Generate report: ${data.report_query}`);
        
        // Highlight report generation button
        this.highlightElement('report generation');
        
        // Show report generator
        this.showReportGenerator(data.report_query);
        
        this.showNotification(`Generating report: ${data.report_query}`, 'info');
    }

    /**
     * Show page with smooth transition
     */
    showPage(pageName) {
        // Hide all pages with fade out
        document.querySelectorAll('.page').forEach(page => {
            page.classList.add('fade-out');
        });
        
        // Wait for fade out, then show target page
        setTimeout(() => {
            // Hide all pages
            document.querySelectorAll('.page').forEach(page => {
                page.style.display = 'none';
                page.classList.remove('fade-out');
            });
            
            // Show target page
            const targetPage = document.getElementById(`${pageName}-page`);
            if (targetPage) {
                targetPage.style.display = 'block';
                targetPage.classList.add('fade-in');
                targetPage.scrollIntoView({ behavior: 'smooth' });
                
                // Remove fade-in class after animation
                setTimeout(() => {
                    targetPage.classList.remove('fade-in');
                }, 300);
            } else {
                console.warn(`⚠️ Page not found: ${pageName}-page`);
                this.showNotification(`Page ${pageName} not found`, 'warning');
            }
        }, 150);
    }

    /**
     * Highlight element with animation
     */
    highlightElement(elementName) {
        // Remove previous highlights
        document.querySelectorAll('.voice-highlighted').forEach(el => {
            el.classList.remove('voice-highlighted');
        });
        
        // Find and highlight element
        const element = this.findElementByName(elementName);
        if (element) {
            element.classList.add('voice-highlighted');
            
            // Scroll element into view if needed
            element.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            // Auto-remove highlight after 3 seconds
            setTimeout(() => {
                element.classList.remove('voice-highlighted');
            }, 3000);
        } else {
            console.warn(`⚠️ Element not found: ${elementName}`);
        }
    }

    /**
     * Find element by name using multiple strategies
     */
    findElementByName(elementName) {
        // Check cache first
        if (this.elementCache.has(elementName)) {
            const cachedElement = this.elementCache.get(elementName);
            if (document.contains(cachedElement)) {
                return cachedElement;
            } else {
                this.elementCache.delete(elementName);
            }
        }
        
        // Search by various selectors
        const selectors = [
            `[data-element="${elementName}"]`,
            `[aria-label*="${elementName}"]`,
            `[title*="${elementName}"]`,
            `button:contains("${elementName}")`,
            `input[placeholder*="${elementName}"]`,
            `.${elementName.replace(/\s+/g, '-')}`,
            `#${elementName.replace(/\s+/g, '-')}`,
            `[class*="${elementName.replace(/\s+/g, '-')}"]`,
            `[id*="${elementName.replace(/\s+/g, '-')}"]`
        ];
        
        for (const selector of selectors) {
            try {
                const element = document.querySelector(selector);
                if (element) {
                    this.elementCache.set(elementName, element);
                    return element;
                }
            } catch (error) {
                console.warn(`Invalid selector: ${selector}`);
            }
        }
        
        return null;
    }

    /**
     * Show notification message
     */
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `voice-notification voice-notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">${this.getNotificationIcon(type)}</span>
                <span class="notification-message">${message}</span>
            </div>
        `;
        
        // Add to container
        this.notificationContainer.appendChild(notification);
        
        // Trigger animation
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.remove();
                }
            }, 300);
        }, 5000);
    }

    /**
     * Get notification icon based on type
     */
    getNotificationIcon(type) {
        const icons = {
            'info': 'ℹ️',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌'
        };
        return icons[type] || icons.info;
    }

    /**
     * Update navigation state
     */
    updateNavigationState(data) {
        // Update breadcrumb
        this.updateBreadcrumb(data.previous_page, data.page);
        
        // Update page title
        document.title = `${data.page} - Voice Agent`;
        
        // Update navigation menu
        this.updateNavigationMenu(data.page);
        
        // Update URL if needed
        this.updateURL(data.page);
    }

    /**
     * Update breadcrumb navigation
     */
    updateBreadcrumb(previousPage, currentPage) {
        const breadcrumb = document.querySelector('.breadcrumb');
        if (breadcrumb) {
            breadcrumb.innerHTML = `
                <span class="breadcrumb-item">${previousPage || 'Home'}</span>
                <span class="breadcrumb-separator">→</span>
                <span class="breadcrumb-item active">${currentPage}</span>
            `;
        }
    }

    /**
     * Update navigation menu
     */
    updateNavigationMenu(currentPage) {
        // Remove active class from all menu items
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        
        // Add active class to current page
        const currentMenuItem = document.querySelector(`[data-page="${currentPage}"]`);
        if (currentMenuItem) {
            currentMenuItem.classList.add('active');
        }
    }

    /**
     * Update URL without page reload
     */
    updateURL(page) {
        const url = new URL(window.location);
        url.searchParams.set('page', page);
        window.history.pushState({ page }, '', url);
    }

    /**
     * Initialize page-specific functionality
     */
    initializePage(pageName) {
        switch(pageName) {
            case 'dashboard':
                this.initializeDashboard();
                break;
            case 'database-query':
                this.initializeDatabaseQuery();
                break;
            case 'file-query':
                this.initializeFileQuery();
                break;
            default:
                console.log(`📄 Initializing page: ${pageName}`);
        }
    }

    /**
     * Initialize dashboard page
     */
    initializeDashboard() {
        console.log('🏠 Initializing dashboard...');
        // Add dashboard-specific initialization here
    }

    /**
     * Initialize database query page
     */
    initializeDatabaseQuery() {
        console.log('🗄️ Initializing database query page...');
        // Add database query-specific initialization here
    }

    /**
     * Initialize file query page
     */
    initializeFileQuery() {
        console.log('📁 Initializing file query page...');
        // Add file query-specific initialization here
    }

    /**
     * Handle set database button
     */
    handleSetDatabase(data) {
        console.log(`🗄️ Setting database: ${data.db_id}`);
        // Add set database logic here
    }

    /**
     * Handle generic button clicks
     */
    handleGenericButton(data) {
        console.log(`🔘 Generic button: ${data.element_name}`);
        // Add generic button logic here
    }

    /**
     * Show search progress
     */
    showSearchProgress(query) {
        console.log(`🔍 Search progress: ${query}`);
        // Add search progress UI here
    }

    /**
     * Execute database search
     */
    executeDatabaseSearch(query) {
        console.log(`🔍 Executing database search: ${query}`);
        // Add database search logic here
    }

    /**
     * Show file search parameters
     */
    showFileSearchParams(data) {
        console.log(`📁 File search params:`, data);
        // Add file search params UI here
    }

    /**
     * Execute file search
     */
    executeFileSearch(data) {
        console.log(`📁 Executing file search:`, data);
        // Add file search logic here
    }

    /**
     * Show upload parameters
     */
    showUploadParams(data) {
        console.log(`📤 Upload params:`, data);
        // Add upload params UI here
    }

    /**
     * Execute file upload
     */
    executeFileUpload(data) {
        console.log(`📤 Executing file upload:`, data);
        // Add file upload logic here
    }

    /**
     * Show report viewer
     */
    showReportViewer(reportRequest) {
        console.log(`📊 Showing report viewer: ${reportRequest}`);
        // Add report viewer logic here
    }

    /**
     * Show report generator
     */
    showReportGenerator(reportQuery) {
        console.log(`📈 Showing report generator: ${reportQuery}`);
        // Add report generator logic here
    }

    /**
     * Process queued messages
     */
    processMessageQueue() {
        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();
            this.handleNavigationMessage(message);
        }
    }

    /**
     * Send message to voice agent
     */
    sendMessage(message) {
        if (this.isConnected && this.websocket) {
            this.websocket.send(JSON.stringify(message));
        } else {
            this.messageQueue.push(message);
            this.showNotification('Message queued - reconnecting...', 'warning');
        }
    }

    /**
     * Disconnect from voice agent
     */
    disconnect() {
        if (this.websocket) {
            this.websocket.close();
        }
        this.isConnected = false;
        console.log('🔌 Disconnected from voice agent');
    }

    /**
     * Get current status
     */
    getStatus() {
        return {
            connected: this.isConnected,
            currentPage: this.currentPage,
            previousPage: this.previousPage,
            reconnectAttempts: this.reconnectAttempts
        };
    }
}

// CSS styles for voice navigation
const voiceNavigationStyles = `
/* Voice Navigation Styles */
.voice-highlighted {
    animation: voice-highlight 0.5s ease-in-out;
    box-shadow: 0 0 20px rgba(0, 123, 255, 0.5);
    border: 2px solid #007bff;
    border-radius: 4px;
    position: relative;
    z-index: 1000;
}

@keyframes voice-highlight {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.voice-notification {
    background: white;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    margin-bottom: 10px;
    padding: 12px 16px;
    transform: translateX(100%);
    transition: transform 0.3s ease-out;
    pointer-events: auto;
    max-width: 300px;
}

.voice-notification.show {
    transform: translateX(0);
}

.voice-notification-content {
    display: flex;
    align-items: center;
    gap: 8px;
}

.voice-notification-icon {
    font-size: 16px;
}

.voice-notification-message {
    font-size: 14px;
    font-weight: 500;
    color: #333;
}

.voice-notification-info {
    border-left: 4px solid #17a2b8;
}

.voice-notification-success {
    border-left: 4px solid #28a745;
}

.voice-notification-warning {
    border-left: 4px solid #ffc107;
}

.voice-notification-error {
    border-left: 4px solid #dc3545;
}

.page {
    transition: opacity 0.3s ease-in-out;
}

.page.fade-out {
    opacity: 0;
}

.page.fade-in {
    opacity: 1;
}

/* Error highlight styles */
.error-highlight {
    animation: error-highlight 0.5s ease-in-out;
    box-shadow: 0 0 20px rgba(220, 53, 69, 0.5);
    border: 2px solid #dc3545;
    border-radius: 4px;
}

@keyframes error-highlight {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.error-tooltip {
    position: absolute;
    top: -40px;
    left: 50%;
    transform: translateX(-50%);
    background: #dc3545;
    color: white;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    white-space: nowrap;
    z-index: 1001;
}

.error-tooltip::after {
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 4px solid transparent;
    border-top-color: #dc3545;
}
`;

// Add styles to document
const styleSheet = document.createElement('style');
styleSheet.textContent = voiceNavigationStyles;
document.head.appendChild(styleSheet);

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VoiceNavigationHandler;
} else if (typeof window !== 'undefined') {
    window.VoiceNavigationHandler = VoiceNavigationHandler;
}

// Auto-initialize if script is loaded directly
if (typeof window !== 'undefined' && !window.voiceNavigationHandler) {
    window.voiceNavigationHandler = new VoiceNavigationHandler();
    console.log('🎯 Voice Navigation Handler auto-initialized');
}
