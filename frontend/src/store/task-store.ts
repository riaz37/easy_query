import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { multiTaskMonitor } from '@/lib/services/multi-task-monitor';

export interface Task {
  id: string;
  type: 'report_generation' | 'query_execution' | 'data_processing' | 'file_upload';
  title: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number; // 0-100
  createdAt: Date;
  startedAt?: Date;
  completedAt?: Date;
  error?: string;
  result?: any;
  metadata?: Record<string, any>;
}

export interface TaskStore {
  // State
  tasks: Task[];
  activeTasks: Task[];
  completedTasks: Task[];
  failedTasks: Task[];
  isTaskListOpen: boolean;
  
  // Actions
  addTask: (task: Omit<Task, 'id' | 'createdAt' | 'status' | 'progress'>) => string;
  updateTask: (id: string, updates: Partial<Task>) => void;
  removeTask: (id: string) => void;
  clearCompletedTasks: () => void;
  clearAllTasks: () => void;
  
  // UI Actions
  toggleTaskList: () => void;
  openTaskList: () => void;
  closeTaskList: () => void;
  
  // Computed
  getTaskById: (id: string) => Task | undefined;
  getTasksByType: (type: Task['type']) => Task[];
  getActiveTasksCount: () => number;
  getCompletedTasksCount: () => number;
  getFailedTasksCount: () => number;
}

export const useTaskStore = create<TaskStore>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    tasks: [],
    activeTasks: [],
    completedTasks: [],
    failedTasks: [],
    isTaskListOpen: false,

    // Add a new task
    addTask: (taskData) => {
      const id = `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const newTask: Task = {
        ...taskData,
        id,
        status: 'pending',
        progress: 0,
        createdAt: new Date(),
      };

      set((state) => ({
        tasks: [...state.tasks, newTask],
        activeTasks: [...state.activeTasks, newTask],
      }));

      return id;
    },

    // Update an existing task
    updateTask: (id, updates) => {
      set((state) => {
        const taskIndex = state.tasks.findIndex(task => task.id === id);
        if (taskIndex === -1) return state;

        const updatedTask = { ...state.tasks[taskIndex], ...updates };
        const newTasks = [...state.tasks];
        newTasks[taskIndex] = updatedTask;

        // Update task lists based on status
        let newActiveTasks = state.activeTasks.filter(task => task.id !== id);
        let newCompletedTasks = state.completedTasks.filter(task => task.id !== id);
        let newFailedTasks = state.failedTasks.filter(task => task.id !== id);

        if (updatedTask.status === 'running' || updatedTask.status === 'pending') {
          newActiveTasks.push(updatedTask);
        } else if (updatedTask.status === 'completed') {
          newCompletedTasks.push(updatedTask);
        } else if (updatedTask.status === 'failed') {
          newFailedTasks.push(updatedTask);
        }

        return {
          tasks: newTasks,
          activeTasks: newActiveTasks,
          completedTasks: newCompletedTasks,
          failedTasks: newFailedTasks,
        };
      });
    },

    // Remove a task
    removeTask: (id) => {
      set((state) => ({
        tasks: state.tasks.filter(task => task.id !== id),
        activeTasks: state.activeTasks.filter(task => task.id !== id),
        completedTasks: state.completedTasks.filter(task => task.id !== id),
        failedTasks: state.failedTasks.filter(task => task.id !== id),
      }));
    },

    // Clear completed tasks
    clearCompletedTasks: () => {
      set((state) => ({
        tasks: state.tasks.filter(task => task.status !== 'completed'),
        completedTasks: [],
      }));
    },

    // Clear all tasks
    clearAllTasks: () => {
      set({
        tasks: [],
        activeTasks: [],
        completedTasks: [],
        failedTasks: [],
      });
    },

    // Toggle task list visibility
    toggleTaskList: () => {
      set((state) => ({
        isTaskListOpen: !state.isTaskListOpen,
      }));
    },

    // Open task list
    openTaskList: () => {
      set({ isTaskListOpen: true });
    },

    // Close task list
    closeTaskList: () => {
      set({ isTaskListOpen: false });
    },

    // Get task by ID
    getTaskById: (id) => {
      return get().tasks.find(task => task.id === id);
    },

    // Get tasks by type
    getTasksByType: (type) => {
      return get().tasks.filter(task => task.type === type);
    },

    // Get active tasks count
    getActiveTasksCount: () => {
      return get().activeTasks.length;
    },

    // Get completed tasks count
    getCompletedTasksCount: () => {
      return get().completedTasks.length;
    },

    // Get failed tasks count
    getFailedTasksCount: () => {
      return get().failedTasks.length;
    },
  }))
);

// Task management utilities
export const TaskUtils = {
  // Start a task
  startTask: (id: string) => {
    useTaskStore.getState().updateTask(id, {
      status: 'running',
      startedAt: new Date(),
    });
  },

  // Complete a task
  completeTask: (id: string, result?: any) => {
    useTaskStore.getState().updateTask(id, {
      status: 'completed',
      progress: 100,
      completedAt: new Date(),
      result,
    });
  },

  // Fail a task
  failTask: (id: string, error: string) => {
    useTaskStore.getState().updateTask(id, {
      status: 'failed',
      completedAt: new Date(),
      error,
    });
  },

  // Update task progress
  updateProgress: (id: string, progress: number) => {
    useTaskStore.getState().updateTask(id, { progress });
  },

  // Cancel a task
  cancelTask: (id: string) => {
    useTaskStore.getState().updateTask(id, {
      status: 'cancelled',
      completedAt: new Date(),
    });
  },

  // Start monitoring a report task with multi-task support
  startReportMonitoring: (taskId: string, reportTaskId: string, options: {
    onProgress?: (status: any) => void;
    onComplete?: (results: any) => void;
    onError?: (error: Error) => void;
    pollInterval?: number;
  } = {}) => {
    multiTaskMonitor.startMonitoring(reportTaskId, {
      onProgress: (status) => {
        // Update the task progress in the store
        TaskUtils.updateProgress(taskId, status.progress_percentage || 0);
        options.onProgress?.(status);
      },
      onComplete: (results) => {
        // Complete the task in the store
        TaskUtils.completeTask(taskId, results);
        options.onComplete?.(results);
      },
      onError: (error) => {
        // Fail the task in the store
        TaskUtils.failTask(taskId, error.message);
        options.onError?.(error);
      },
      pollInterval: options.pollInterval || 2000,
    });
  },

  // Stop monitoring a specific task
  stopReportMonitoring: (reportTaskId: string) => {
    multiTaskMonitor.stopMonitoring(reportTaskId);
  },

  // Stop all monitoring
  stopAllMonitoring: () => {
    multiTaskMonitor.stopAllMonitoring();
  },

  // Get active monitor count
  getActiveMonitorCount: () => {
    return multiTaskMonitor.getActiveMonitorCount();
  },
};
