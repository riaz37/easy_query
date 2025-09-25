import { useState, useEffect, useRef } from 'react';

/**
 * Hook for managing user settings and tooltip state
 */
export function useUserSettings() {
  const [userId, setUserId] = useState<string>("default");
  const [editingUserId, setEditingUserId] = useState(false);
  const [showUserTooltip, setShowUserTooltip] = useState(false);
  const userTooltipRef = useRef<HTMLDivElement>(null);

  // Handle clicking outside user tooltip
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        userTooltipRef.current &&
        !userTooltipRef.current.contains(event.target as Node)
      ) {
        setShowUserTooltip(false);
        setEditingUserId(false);
      }
    };

    if (showUserTooltip) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [showUserTooltip]);

  const handleUserIdClick = () => {
    setShowUserTooltip(!showUserTooltip);
  };

  const handleSaveUserId = (val: string) => {
    setUserId(val || "default");
    setEditingUserId(false);
    setShowUserTooltip(false);
  };

  const toggleEditUserId = () => {
    setEditingUserId(!editingUserId);
  };

  const closeTooltip = () => {
    setShowUserTooltip(false);
    setEditingUserId(false);
  };

  return {
    userId,
    editingUserId,
    showUserTooltip,
    userTooltipRef,
    handleUserIdClick,
    handleSaveUserId,
    toggleEditUserId,
    closeTooltip,
    setUserId,
  };
}