interface AIConfiguration {
  userId: string;
  databaseId: number;
  databaseName: string;
  timestamp: string;
}

export class AIConfigService {
  private static readonly CONFIG_KEY = 'aiConfiguration';

  /**
   * Save AI configuration
   */
  static async saveConfiguration(config: Omit<AIConfiguration, 'timestamp'>): Promise<void> {
    try {
      const configWithTimestamp: AIConfiguration = {
        ...config,
        timestamp: new Date().toISOString(),
      };

      localStorage.setItem(this.CONFIG_KEY, JSON.stringify(configWithTimestamp));
    } catch (error) {
      console.error('Failed to save AI configuration:', error);
      throw new Error('Failed to save AI configuration');
    }
  }

  /**
   * Load AI configuration
   */
  static async loadConfiguration(): Promise<AIConfiguration | null> {
    try {
      const savedConfig = localStorage.getItem(this.CONFIG_KEY);
      if (!savedConfig) {
        return null;
      }

      const config = JSON.parse(savedConfig) as AIConfiguration;
      
      // Validate the configuration structure
      if (!config.userId || !config.databaseId || !config.databaseName) {
        console.warn('Invalid AI configuration found, clearing...');
        this.clearConfiguration();
        return null;
      }

      return config;
    } catch (error) {
      console.error('Failed to load AI configuration:', error);
      this.clearConfiguration(); // Clear corrupted config
      return null;
    }
  }

  /**
   * Clear AI configuration
   */
  static clearConfiguration(): void {
    try {
      localStorage.removeItem(this.CONFIG_KEY);
    } catch (error) {
      console.error('Failed to clear AI configuration:', error);
    }
  }

  /**
   * Check if configuration exists and is valid
   */
  static async hasValidConfiguration(): Promise<boolean> {
    const config = await this.loadConfiguration();
    return config !== null;
  }

  /**
   * Get configuration summary for display
   */
  static async getConfigurationSummary(): Promise<{
    userId: string;
    databaseName: string;
    lastUpdated: string;
  } | null> {
    const config = await this.loadConfiguration();
    if (!config) return null;

    return {
      userId: config.userId,
      databaseName: config.databaseName,
      lastUpdated: new Date(config.timestamp).toLocaleString(),
    };
  }
}

export type { AIConfiguration };