import { UserAccessData } from "@/types/api";

export interface UserConfig {
  user_id: string;
  db_id: number;
  table_names: string[];
  access_level: number;
}

export interface Database {
  db_id: number;
  db_name: string;
}

export interface UserStats {
  totalUsers: number;
  mssqlUsers: number;
  vectorDBUsers: number;
  fullAccessUsers: number;
}

export interface UsersManagerHeaderProps {
  onCreateMSSQLAccess: () => void;
  onCreateVectorDBAccess: () => void;
  isDark: boolean;
}

export interface UserSearchInputProps {
  searchTerm: string;
  onSearchChange: (value: string) => void;
  isDark: boolean;
  placeholder?: string;
}

export interface UserStatsCardsProps {
  stats: UserStats;
  isDark: boolean;
}

export interface UserAccessTabsProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
  isDark: boolean;
  children: React.ReactNode;
}

export interface MSSQLUsersListProps {
  users: UserAccessData[];
  onEditUser: (userId: string) => void;
  onCreateAccess: () => void;
  extractNameFromEmail: (email: string) => string;
  getAccessLevelBadge: (config: UserAccessData) => React.ReactNode;
  getDatabaseCount: (config: UserAccessData) => number;
  isDark: boolean;
}

export interface VectorDBUsersListProps {
  users: UserConfig[];
  onEditUser: (userId: string) => void;
  onCreateAccess: () => void;
  onRefresh: () => void;
  extractNameFromEmail: (email: string) => string;
  getAccessLevelBadge: (config: UserConfig) => React.ReactNode;
  getDatabaseName: (dbId: number) => string;
  formatTableNames: (tableNames: string[]) => string;
  isLoading: boolean;
  isDark: boolean;
}

export interface UserCardProps {
  user: UserAccessData | UserConfig;
  type: 'mssql' | 'vector';
  onEdit: (userId: string) => void;
  extractNameFromEmail: (email: string) => string;
  getAccessLevelBadge: (config: UserAccessData | UserConfig) => React.ReactNode;
  getDatabaseCount?: (config: UserAccessData) => number;
  getDatabaseName?: (dbId: number) => string;
  formatTableNames?: (tableNames: string[]) => string;
  isDark: boolean;
}

export interface EmptyStateProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  actionLabel: string;
  onAction: () => void;
  isDark: boolean;
}

export interface UserAccessConfig {
  user_id: string;
  database_access?: {
    parent_databases?: any[];
    sub_databases?: any[];
  };
  sub_company_ids?: string[];
  access_level?: number;
}

