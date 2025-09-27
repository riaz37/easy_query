import { ServiceRegistry } from '../api';
import { ReportTaskStatus } from '../../types/reports';

interface TaskMonitor {
  taskId: string;
  intervalId: NodeJS.Timeout;
  onProgress?: (status: ReportTaskStatus) => void;
  onComplete?: (results: any) => void;
  onError?: (error: Error) => void;
  pollInterval: number;
}

class MultiTaskMonitor {
  private monitors: Map<string, TaskMonitor> = new Map();
  private isDestroyed = false;

  /**
   * Start monitoring a task
   */
  startMonitoring(
    taskId: string,
    options: {
      onProgress?: (status: ReportTaskStatus) => void;
      onComplete?: (results: any) => void;
      onError?: (error: Error) => void;
      pollInterval?: number;
    } = {}
  ): void {
    // Stop existing monitoring for this task if it exists
    this.stopMonitoring(taskId);

    const pollInterval = options.pollInterval || 2000;
    
    const intervalId = setInterval(async () => {
      if (this.isDestroyed) {
        clearInterval(intervalId);
        return;
      }

      try {
        console.log(`[MultiTaskMonitor] Polling task status for: ${taskId}`);
        const status = await ServiceRegistry.reports.getTaskStatus(taskId);
        console.log(`[MultiTaskMonitor] Task ${taskId} status:`, status);
        
        // Call progress callback
        options.onProgress?.(status);

        // Check if completed
        if (status.status === 'completed' && status.results) {
          console.log(`[MultiTaskMonitor] Task ${taskId} completed successfully:`, status.results);
          options.onComplete?.(status.results);
          this.stopMonitoring(taskId);
        }

        // Check if failed
        if (status.status === 'failed') {
          console.error(`[MultiTaskMonitor] Task ${taskId} failed:`, status.error);
          const error = new Error(status.error || 'Task failed');
          options.onError?.(error);
          this.stopMonitoring(taskId);
        }
      } catch (error) {
        console.error(`[MultiTaskMonitor] Error polling task ${taskId}:`, error);
        const errorObj = error instanceof Error ? error : new Error('Polling failed');
        options.onError?.(errorObj);
        this.stopMonitoring(taskId);
      }
    }, pollInterval);

    // Store the monitor
    this.monitors.set(taskId, {
      taskId,
      intervalId,
      onProgress: options.onProgress,
      onComplete: options.onComplete,
      onError: options.onError,
      pollInterval,
    });

    console.log(`[MultiTaskMonitor] Started monitoring task: ${taskId}`);
  }

  /**
   * Stop monitoring a specific task
   */
  stopMonitoring(taskId: string): void {
    const monitor = this.monitors.get(taskId);
    if (monitor) {
      clearInterval(monitor.intervalId);
      this.monitors.delete(taskId);
      console.log(`[MultiTaskMonitor] Stopped monitoring task: ${taskId}`);
    }
  }

  /**
   * Stop monitoring all tasks
   */
  stopAllMonitoring(): void {
    console.log(`[MultiTaskMonitor] Stopping all monitoring (${this.monitors.size} tasks)`);
    for (const [taskId, monitor] of this.monitors) {
      clearInterval(monitor.intervalId);
    }
    this.monitors.clear();
  }

  /**
   * Check if a task is being monitored
   */
  isMonitoring(taskId: string): boolean {
    return this.monitors.has(taskId);
  }

  /**
   * Get all monitored task IDs
   */
  getMonitoredTasks(): string[] {
    return Array.from(this.monitors.keys());
  }

  /**
   * Get the number of active monitors
   */
  getActiveMonitorCount(): number {
    return this.monitors.size;
  }

  /**
   * Destroy the monitor and clean up all resources
   */
  destroy(): void {
    this.isDestroyed = true;
    this.stopAllMonitoring();
    console.log('[MultiTaskMonitor] Destroyed');
  }
}

// Create a singleton instance
export const multiTaskMonitor = new MultiTaskMonitor();

// Clean up on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    multiTaskMonitor.destroy();
  });
}

export default multiTaskMonitor;
