import { SystemNode, CardPosition } from './types';

export const SYSTEM_NODES: SystemNode[] = [
  {
    id: "search",
    label: "Search Engine",
    title: "Intelligent Search",
    description: "AI-powered semantic search with contextual results.",
    position: [3, 1.5, 0],
    color: "#10b981",
    icon: "search",
    iconPath: "/dashboard/apiconnect.svg",
  },
  {
    id: "analytics",
    label: "Analytics",
    title: "Data Analytics",
    description: "Real-time analytics dashboard with performance insights.",
    position: [-3, 1.5, 0],
    color: "#059669",
    icon: "analytics",
    iconPath: "/dashboard/apiconnect.svg",
  },
  {
    id: "ai-assistant",
    label: "AI Assistant",
    title: "Smart Assistant",
    description: "AI-powered assistant for document processing and guidance.",
    position: [2.5, -1.5, 1.5],
    color: "#34d399",
    icon: "ai",
    iconPath: "/dashboard/apiconnect.svg",
  },
  {
    id: "document-management",
    label: "Document Management",
    title: "Document Hub",
    description: "Centralized document management with version control.",
    position: [-2.5, -1.5, 1.5],
    color: "#6ee7b7",
    icon: "document",
    iconPath: "/dashboard/apiconnect.svg",
  },
  {
    id: "knowledge-graph",
    label: "Knowledge Graph",
    title: "Knowledge Network",
    description: "Visual knowledge mapping with interconnected concepts.",
    position: [-2.5, -3, 0],
    color: "#10b981",
    icon: "graph",
    iconPath: "/dashboard/apiconnect.svg",
  },
  {
    id: "automation",
    label: "Automation",
    title: "Workflow Automation",
    description: "Automated workflows with intelligent process optimization.",
    position: [2.5, -3, 0],
    color: "#059669",
    icon: "automation",
    iconPath: "/dashboard/apiconnect.svg",
  },
];

export const INITIAL_CARD_POSITIONS: CardPosition[] = [
  { x: 25, y: 15 },   // Left top
  { x: 20, y: 45 },   // Left middle (curved inward)
  { x: 75, y: 15 },   // Right top
  { x: 80, y: 45 },   // Right middle (curved outward)
  { x: 25, y: 75 },   // Bottom left
  { x: 75, y: 75 },   // Bottom right
];

export const ANIMATION_CONFIG = {
  PARTICLE_COUNT: 5,
  PARTICLE_SIZE: 0.02,
  PARTICLE_SPEED_BASE: 0.005,
  PARTICLE_SPEED_VARIANCE: 0.003,
  CONNECTION_OPACITY: 0.6,
  PULSE_AMPLITUDE: 0.2,
  CURVE_POINTS: 50,
} as const;

export const CAMERA_CONFIG = {
  FOV: 75,
  NEAR: 0.1,
  FAR: 1000,
  POSITION: [0, 0, 8] as [number, number, number],
} as const;

export const LIGHTING_CONFIG = {
  AMBIENT_INTENSITY: 0.4,
  DIRECTIONAL_INTENSITY: 0.6,
  POINT_INTENSITY: 0.8,
  POINT_DISTANCE: 15,
} as const;