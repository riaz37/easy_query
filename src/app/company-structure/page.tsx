import { CompanyHierarchy } from '@/components/company-hierarchy';
import { PageLayout } from '@/components/layout/PageLayout';

export default function DatabaseHierarchyPage() {
  return (
    <PageLayout 
      background={["frame", "gridframe"]}
      className="min-h-screen"
      container={true}
      maxWidth="full"
    >
      <CompanyHierarchy />
    </PageLayout>
  );
}