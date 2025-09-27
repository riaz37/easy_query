export interface DatabaseInfo {
  db_id: number;
  db_name: string;
  db_url: string;
  db_type: string;
  is_current: boolean;
  business_rule?: string;
}

export interface UserInfo {
  user_id: string;
  email?: string;
}

export interface BusinessRulesState {
  status: 'idle' | 'loading' | 'loaded' | 'error';
  content: string;
  lastUpdated?: string;
}

export interface DatabaseContextState {
  currentDatabaseId?: number;
  currentDatabaseName?: string;
  availableDatabases: DatabaseInfo[];
  loading: boolean;
  error?: string | null;
}

export interface BusinessRulesEditorState {
  isEditing: boolean;
  editedContent: string;
  hasUnsavedChanges: boolean;
  contentError: string | null;
}

export interface UserConfigurationProps {
  className?: string;
}

export interface OverviewTabProps {
  user: UserInfo | null;
  currentDatabaseName?: string;
  businessRules: BusinessRulesState;
  businessRulesCount: number;
  hasBusinessRules: boolean;
  onNavigateToTab: (tab: string) => void;
}

export interface DatabaseTabProps {
  databases: DatabaseInfo[];
  loading: boolean;
  onDatabaseChange: (databaseId: number) => void;
  businessRules: BusinessRulesState;
}

export interface BusinessRulesTabProps {
  currentDatabaseId?: number;
  currentDatabaseName?: string;
  businessRules: BusinessRulesState;
  businessRulesCount: number;
  hasBusinessRules: boolean;
  editorState: BusinessRulesEditorState;
  onRefresh: () => void;
  onEdit: () => void;
  onSave: () => void;
  onCancel: () => void;
  onContentChange: (content: string) => void;
}

export interface UserInfoCardProps {
  user: UserInfo | null;
}

export interface CurrentStatusCardProps {
  currentDatabaseName?: string;
  businessRules: BusinessRulesState;
  businessRulesCount: number;
  hasBusinessRules: boolean;
}

export interface QuickActionsCardProps {
  onNavigateToTab: (tab: string) => void;
}

export interface DatabaseSelectionCardProps {
  databases: DatabaseInfo[];
  loading: boolean;
  onDatabaseChange: (databaseId: number) => void;
  businessRules: BusinessRulesState;
}

export interface DatabaseCardProps {
  database: DatabaseInfo;
  businessRules: BusinessRulesState;
  onSelect: (databaseId: number) => void;
}

export interface BusinessRulesStatusCardProps {
  currentDatabaseName?: string;
  businessRules: BusinessRulesState;
  businessRulesCount: number;
  hasBusinessRules: boolean;
  editorState: BusinessRulesEditorState;
  onRefresh: () => void;
  onEdit: () => void;
  onSave: () => void;
  onCancel: () => void;
}

export interface BusinessRulesEditorProps {
  currentDatabaseId?: number;
  businessRules: BusinessRulesState;
  editorState: BusinessRulesEditorState;
  onContentChange: (content: string) => void;
  onEdit: () => void;
  onSave: () => void;
  onCancel: () => void;
}

export interface ContextInfoCardProps {
  className?: string;
}
