import { useState, useCallback, useEffect } from "react";
import { CardPosition } from "../types";
import { SYSTEM_NODES } from "../constants";

interface UseDragAndDropProps {
  initialPositions: CardPosition[];
  containerRef: React.RefObject<HTMLDivElement | null>;
}

export const useDragAndDrop = ({
  initialPositions,
  containerRef,
}: UseDragAndDropProps) => {
  const [cardPositions, setCardPositions] =
    useState<CardPosition[]>(initialPositions);
  const [dragging, setDragging] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });

  const handleMouseDown = useCallback(
    (e: React.MouseEvent, nodeId: string) => {
      e.preventDefault();
      setDragging(nodeId);

      const rect = e.currentTarget.getBoundingClientRect();
      const containerRect = containerRef.current?.getBoundingClientRect();
      if (!containerRect) return;

      const offsetX = e.clientX - rect.left - rect.width / 2;
      const offsetY = e.clientY - rect.top - rect.height / 2;
      setDragOffset({ x: offsetX, y: offsetY });
    },
    [containerRef]
  );

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!dragging || !containerRef.current) return;

      const containerRect = containerRef.current.getBoundingClientRect();

      // Calculate new position relative to container
      const newX =
        ((e.clientX - containerRect.left - dragOffset.x) /
          containerRect.width) *
        100;
      const newY =
        ((e.clientY - containerRect.top - dragOffset.y) /
          containerRect.height) *
        100;

      // Constrain to container bounds
      const constrainedX = Math.max(5, Math.min(95, newX));
      const constrainedY = Math.max(5, Math.min(95, newY));

      setCardPositions((prev) => {
        const newPositions = [...prev];
        // Find the index of the dragging node by its ID
        const nodeIndex = SYSTEM_NODES.findIndex(
          (node) => node.id === dragging
        );

        if (nodeIndex !== -1) {
          newPositions[nodeIndex] = { x: constrainedX, y: constrainedY };
        }
        return newPositions;
      });
    },
    [dragging, dragOffset, containerRef]
  );

  const handleMouseUp = useCallback(() => {
    setDragging(null);
    setDragOffset({ x: 0, y: 0 });
  }, []);

  // Effect for drag event listeners
  useEffect(() => {
    if (dragging) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "grabbing";
      document.body.style.userSelect = "none";
    } else {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [dragging, handleMouseMove, handleMouseUp]);

  return {
    cardPositions,
    dragging,
    handleMouseDown,
  };
};
