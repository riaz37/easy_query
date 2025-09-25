import { CompanyHierarchy } from '@/components/company-hierarchy';
import { PageLayout } from '@/components/layout/PageLayout';

export default function DatabaseHierarchyPage() {
  return (
    <PageLayout 
      background={["frame", "gridframe"]}
      container={false}
      maxWidth="full"
      className="h-screen w-full overflow-hidden"
    >
      <CompanyHierarchy />
    </PageLayout>
  );
}