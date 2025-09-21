'use client'

import React, { useEffect, useRef } from 'react'
import { useRouter } from 'next/navigation'
import { useVoiceAgent } from '@/components/providers/VoiceAgentContextProvider'

interface VoiceNavigationHandlerProps {
  children: React.ReactNode
}

export function VoiceNavigationHandler({ children }: VoiceNavigationHandlerProps) {
  const router = useRouter()
  const { currentPage, previousPage, isReady, isLoading } = useVoiceAgent()
  const eventListenersRef = useRef<(() => void)[]>([])

  useEffect(() => {
    // Only set up event listeners if service is ready
    if (isLoading || !isReady) {
      return
    }

    // Handle page navigation
    const handleNavigation = (event: CustomEvent) => {
      const { page, previousPage, type } = event.detail
      console.log(`🧭 Voice navigation: ${type} to ${page}`)
      
      // Navigate to the page
      router.push(`/${page}`)
      
      // Show visual feedback
      showNavigationFeedback(page, type)
    }

    // Handle element clicks
    const handleClick = (event: CustomEvent) => {
      const { elementName, page, type } = event.detail
      console.log(`🖱️ Voice click: ${elementName} on ${page}`)
      
      // Find and click the element
      clickElementByName(elementName)
      
      // Show visual feedback
      showClickFeedback(elementName)
    }

    // Handle search commands
    const handleSearch = (event: CustomEvent) => {
      const { query, type, page, interactionType } = event.detail
      console.log(`🔍 Voice search: ${type} search for "${query}" on ${page}`)
      
      // Navigate to appropriate search page
      if (type === 'database') {
        router.push('/database-query')
        // Trigger search after navigation
        setTimeout(() => {
          triggerDatabaseSearch(query)
        }, 500)
      } else {
        router.push('/file-query')
        // Trigger search after navigation
        setTimeout(() => {
          triggerFileSearch(query)
        }, 500)
      }
      
      // Show visual feedback
      showSearchFeedback(query, type)
    }

    // Handle file upload commands
    const handleUpload = (event: CustomEvent) => {
      const { descriptions, tableNames, page, type } = event.detail
      console.log(`📤 Voice upload: ${descriptions.join(', ')} to ${tableNames.join(', ')}`)
      
      // Navigate to file upload page
      router.push('/file-query')
      
      // Show visual feedback
      showUploadFeedback(descriptions, tableNames)
    }

    // Handle report viewing
    const handleViewReport = (event: CustomEvent) => {
      const { request, page, type } = event.detail
      console.log(`📊 Voice view report: ${request}`)
      
      // Navigate to reports page
      router.push('/ai-reports')
      
      // Show visual feedback
      showReportFeedback(request, 'view')
    }

    // Handle report generation
    const handleGenerateReport = (event: CustomEvent) => {
      const { query, page, type } = event.detail
      console.log(`📈 Voice generate report: ${query}`)
      
      // Navigate to report generation page
      router.push('/ai-reports')
      
      // Show visual feedback
      showReportFeedback(query, 'generate')
    }

    // Add event listeners
    window.addEventListener('voice-navigation', handleNavigation as EventListener)
    window.addEventListener('voice-click', handleClick as EventListener)
    window.addEventListener('voice-search', handleSearch as EventListener)
    window.addEventListener('voice-upload', handleUpload as EventListener)
    window.addEventListener('voice-view-report', handleViewReport as EventListener)
    window.addEventListener('voice-generate-report', handleGenerateReport as EventListener)

    // Store cleanup functions
    eventListenersRef.current = [
      () => window.removeEventListener('voice-navigation', handleNavigation as EventListener),
      () => window.removeEventListener('voice-click', handleClick as EventListener),
      () => window.removeEventListener('voice-search', handleSearch as EventListener),
      () => window.removeEventListener('voice-upload', handleUpload as EventListener),
      () => window.removeEventListener('voice-view-report', handleViewReport as EventListener),
      () => window.removeEventListener('voice-generate-report', handleGenerateReport as EventListener),
    ]

    return () => {
      // Cleanup event listeners
      eventListenersRef.current.forEach(cleanup => cleanup())
    }
  }, [router, isReady, isLoading])

  // Don't initialize if service is not ready
  if (isLoading || !isReady) {
    return <>{children}</>
  }

  // Helper functions for actual UI interactions
  const clickElementByName = (elementName: string) => {
    // Try different selectors to find the element
    const selectors = [
      `[data-element="${elementName}"]`,
      `[aria-label*="${elementName}"]`,
      `[title*="${elementName}"]`,
      `button:contains("${elementName}")`,
      `input[placeholder*="${elementName}"]`,
      `.${elementName.replace(/\s+/g, '-')}`,
      `#${elementName.replace(/\s+/g, '-')}`
    ]

    for (const selector of selectors) {
      try {
        const element = document.querySelector(selector)
        if (element) {
          // Highlight the element with Tailwind classes
          element.classList.add('ring-4', 'ring-blue-500', 'ring-opacity-50', 'scale-105', 'transition-all', 'duration-300', 'ease-out')
          
          // Click the element
          if (element instanceof HTMLElement) {
            element.click()
          }
          
          // Remove highlight after 3 seconds
          setTimeout(() => {
            element.classList.remove('ring-4', 'ring-blue-500', 'ring-opacity-50', 'scale-105', 'transition-all', 'duration-300', 'ease-out')
          }, 3000)
          
          return
        }
      } catch (error) {
        console.warn(`Selector failed: ${selector}`, error)
      }
    }
    
    console.warn(`Could not find element: ${elementName}`)
  }

  const triggerDatabaseSearch = (query: string) => {
    // Find search input and trigger search
    const searchInput = document.querySelector('input[placeholder*="search"], input[placeholder*="query"]') as HTMLInputElement
    if (searchInput) {
      searchInput.value = query
      searchInput.dispatchEvent(new Event('input', { bubbles: true }))
      
      // Find and click search button
      const searchButton = document.querySelector('button:contains("Search"), button:contains("Query")') as HTMLButtonElement
      if (searchButton) {
        searchButton.click()
      }
    }
  }

  const triggerFileSearch = (query: string) => {
    // Find file search input and trigger search
    const searchInput = document.querySelector('input[placeholder*="file"], input[placeholder*="search"]') as HTMLInputElement
    if (searchInput) {
      searchInput.value = query
      searchInput.dispatchEvent(new Event('input', { bubbles: true }))
      
      // Find and click search button
      const searchButton = document.querySelector('button:contains("Search"), button:contains("Find")') as HTMLButtonElement
      if (searchButton) {
        searchButton.click()
      }
    }
  }

  // Visual feedback functions
  const showNavigationFeedback = (page: string, type: string) => {
    showNotification(`🧭 Navigated to ${page} page`, 'success')
  }

  const showClickFeedback = (elementName: string) => {
    showNotification(`🖱️ Clicked ${elementName}`, 'info')
  }

  const showSearchFeedback = (query: string, type: string) => {
    showNotification(`🔍 Searching for: ${query}`, 'info')
  }

  const showUploadFeedback = (descriptions: string[], tableNames: string[]) => {
    const fileInfo = `${descriptions.length} file(s)`
    const tableInfo = tableNames.length > 0 ? ` to tables: ${tableNames.join(', ')}` : ''
    showNotification(`📤 Uploading ${fileInfo}${tableInfo}`, 'info')
  }

  const showReportFeedback = (request: string, action: 'view' | 'generate') => {
    const actionText = action === 'view' ? 'Viewing' : 'Generating'
    showNotification(`📊 ${actionText} report: ${request}`, 'info')
  }

  const showNotification = (message: string, type: 'info' | 'success' | 'warning' | 'error' = 'info') => {
    // Create notification element with Tailwind classes
    const notification = document.createElement('div')
    
    // Base classes
    const baseClasses = 'fixed top-5 right-5 px-5 py-4 rounded-lg text-white font-bold z-50 max-w-xs break-words shadow-lg transform transition-all duration-300 ease-out'
    
    // Type-specific classes
    const typeClasses = {
      info: 'bg-blue-500',
      success: 'bg-green-500',
      warning: 'bg-yellow-500 text-gray-900',
      error: 'bg-red-500'
    }
    
    notification.className = `${baseClasses} ${typeClasses[type]} animate-slide-in-right`
    notification.textContent = message
    
    // Add to page
    document.body.appendChild(notification)
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
      if (notification.parentNode) {
        notification.classList.add('animate-slide-out-right')
        setTimeout(() => {
          if (notification.parentNode) {
            notification.remove()
          }
        }, 300)
      }
    }, 5000)
  }

  return <>{children}</>
}