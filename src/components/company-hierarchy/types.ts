import { MSSQLConfigData } from "@/types/api";

export interface Company {
  id: string;
  name: string;
  description?: string;
  address?: string;
  contactEmail?: string;
  dbId: number;
  parentId?: string;
  children?: Company[];
}

export interface CompanyFormData {
  name: string;
  description: string;
  address: string;
  contactEmail: string;
  dbId: number;
  parentCompanyId?: number;
}

export interface DatabaseFormData {
  db_url: string;
  db_name: string;
  business_rule?: string;
  user_id: string;
}

export type WorkflowStep =
  | "company-info"
  | "database-config"
  | "database-creation"
  | "vector-config"
  | "final-creation";

export interface CompanyTreeViewProps {
  onCompanyCreated?: () => void;
}

export interface CompanyCardProps {
  company: Company;
  onAddSubCompany?: (parentId: string) => void;
  onUpload?: (
    companyId: string,
    companyName: string,
    companyType: "parent" | "sub",
  ) => void;
  isSelected?: boolean;
  onSelect?: () => void;
  level?: number;
}

export interface CompanyCreationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (companyData: CompanyFormData) => Promise<void>;
  type: "parent" | "sub";
  parentCompanyId?: number;
}

export interface EmptyStateProps {
  onAddParentCompany: () => void;
}

export interface CompanyTreeProps {
  companies: Company[];
  onAddSubCompany: (
    name: string,
    description: string,
    contactDatabase: string,
    parentId?: string,
  ) => void;
  selectedCompany: string | null;
  onSelectCompany: (id: string | null) => void;
}

export interface CompanyInfoStepProps {
  companyName: string;
  setCompanyName: (value: string) => void;
  description: string;
  setDescription: (value: string) => void;
  address: string;
  setAddress: (value: string) => void;
  contactEmail: string;
  setContactEmail: (value: string) => void;
  setCurrentStep: (step: WorkflowStep) => void;
  onClose: () => void;
}

export interface DatabaseConfigStepProps {
  selectedDbId: number | null;
  setSelectedDbId: (id: number | null) => void;
  databases: MSSQLConfigData[];
  mssqlLoading: boolean;
  setConfig: any;
  loadDatabases: () => Promise<void>;
  setCurrentStep: (step: WorkflowStep) => void;
  setDatabaseCreationData: (data: any) => void;
  setCurrentTaskId: (taskId: string | null) => void;
  refreshUserConfigs: () => Promise<void>;
}

export interface VectorConfigStepProps {
  currentStep: WorkflowStep;
  setCurrentStep: (step: WorkflowStep) => void;
  selectedUserConfigId: number | null;
  setSelectedUserConfigId: (id: number | null) => void;
  userConfigs: DatabaseConfigData[];
  userConfigLoading: boolean;
  createDatabaseConfig: (config: any) => Promise<any>;
  refreshUserConfigs: () => Promise<void>;
}
