// Type definitions for the dashboard components
import * as THREE from 'three';
export interface SystemNode {
  id: string;
  label: string;
  title: string;
  description: string;
  position: [number, number, number];
  color: string;
  icon: string;
  iconPath: string;
}

export interface CardPosition {
  x: number;
  y: number;
}

export interface ParticleUserData {
  originalPos: THREE.Vector3;
  orbitSpeed: number;
  angle: number;
}

export interface PulseUserData {
  startPos: THREE.Vector3;
  endPos: THREE.Vector3;
  progress: number;
  speed: number;
}

export interface NodeGroupUserData extends SystemNode {}

// Extended THREE.js types for userData
export interface ExtendedObject3D extends THREE.Object3D {
  userData:
    | ParticleUserData
    | PulseUserData
    | NodeGroupUserData
    | Record<string, any>;
}

export interface ExtendedMesh extends THREE.Mesh {
  userData: ParticleUserData | PulseUserData | Record<string, any>;
}

export interface ExtendedGroup extends THREE.Group {
  userData: NodeGroupUserData;
}

export interface ParticleObject {
  mesh: THREE.Mesh;
  startPos: THREE.Vector3;
  endPos: THREE.Vector3;
  progress: number;
  speed: number;
  curve: THREE.QuadraticBezierCurve3;
  connectionIndex: number;
}