import { useAuthContext } from '@/components/providers';

export function useUserContext() {
  const { user, isAuthenticated } = useAuthContext();
  return { user, isAuthenticated };
}