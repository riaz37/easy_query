import { useEffect, useRef, useCallback } from 'react';
import * as THREE from 'three';
import { CardPosition, ParticleObject } from '../types';
import { SYSTEM_NODES, ANIMATION_CONFIG, CAMERA_CONFIG, LIGHTING_CONFIG } from '../constants';

interface UseThreeSceneProps {
  mountRef: React.RefObject<HTMLDivElement | null>;
  cardPositions: CardPosition[];
}

export const useThreeScene = ({ mountRef, cardPositions }: UseThreeSceneProps) => {
  const sceneRef = useRef<THREE.Scene | null>(null);
  const animationRef = useRef<number | null>(null);
  const connectionsRef = useRef<THREE.Line[]>([]);
  const particlesRef = useRef<ParticleObject[]>([]);

  const createParticle = useCallback((
    nodeData: typeof SYSTEM_NODES[0], 
    curve: THREE.QuadraticBezierCurve3, 
    startPoint: THREE.Vector3, 
    endPoint: THREE.Vector3, 
    index: number,
    connectionIndex: number
  ): ParticleObject => {
    const particleGeometry = new THREE.SphereGeometry(ANIMATION_CONFIG.PARTICLE_SIZE, 8, 8);
    const particleMaterial = new THREE.MeshBasicMaterial({
      color: nodeData.color,
      transparent: true,
      opacity: 0.8,
    });
    
    const particle = new THREE.Mesh(particleGeometry, particleMaterial.clone());
    const t = index / (ANIMATION_CONFIG.PARTICLE_COUNT - 1);
    const particlePos = curve.getPoint(t);
    particle.position.copy(particlePos);
    
    return {
      mesh: particle,
      startPos: startPoint.clone(),
      endPos: endPoint.clone(),
      progress: t,
      speed: ANIMATION_CONFIG.PARTICLE_SPEED_BASE + Math.random() * ANIMATION_CONFIG.PARTICLE_SPEED_VARIANCE,
      curve: curve,
      connectionIndex: connectionIndex,
    };
  }, []);

  const updateConnections = useCallback(() => {
    if (!sceneRef.current) return;

    connectionsRef.current.forEach((connection, index) => {
      const cardPos = cardPositions[index];
      const nodeData = SYSTEM_NODES[index];
      
      // Convert screen coordinates to 3D world coordinates
      const worldX = ((cardPos.x - 50) / 50) * 6;
      const worldY = -((cardPos.y - 50) / 50) * 4;
      const worldZ = 2;
      
      // Create new curved connection
      const startPoint = new THREE.Vector3(0, 0, 0);
      const endPoint = new THREE.Vector3(worldX, worldY, worldZ);
      const midPoint = new THREE.Vector3(
        worldX * 0.5,
        worldY * 0.5 + 1,
        worldZ * 0.5
      );
      
      const curve = new THREE.QuadraticBezierCurve3(startPoint, midPoint, endPoint);
      const points = curve.getPoints(ANIMATION_CONFIG.CURVE_POINTS);
      
      // Update connection geometry
      const newGeometry = new THREE.BufferGeometry().setFromPoints(points);
      connection.geometry.dispose();
      connection.geometry = newGeometry;
      
      // Update particles along this connection
      const connectionParticles = particlesRef.current.filter(p => p.connectionIndex === index);
      connectionParticles.forEach((particleObj) => {
        particleObj.curve = curve;
        particleObj.startPos = startPoint.clone();
        particleObj.endPos = endPoint.clone();
      });
    });
  }, [cardPositions]);

  const animateScene = useCallback(() => {
    if (!sceneRef.current) return;

    // Animate connection particles
    particlesRef.current.forEach(particleObj => {
      if (!particleObj?.mesh) return;
      
      const particle = particleObj.mesh;
      
      // Update progress
      particleObj.progress += particleObj.speed;
      if (particleObj.progress > 1) particleObj.progress = 0;
      
      // Update position if curve exists
      if (particleObj.curve) {
        try {
          const newPos = particleObj.curve.getPoint(particleObj.progress);
          particle.position.copy(newPos);
          
          // Pulsing effect
          const pulse = Math.sin(particleObj.progress * Math.PI * 2);
          particle.scale.setScalar(0.5 + pulse * 0.5);
          
          const mesh = particle as THREE.Mesh;
          if (mesh.material && 'opacity' in mesh.material) {
            (mesh.material as THREE.Material & { opacity: number }).opacity = 
              0.3 + pulse * 0.5;
          }
        } catch (error) {
          console.warn('Error updating particle position:', error);
        }
      }
    });

    // Animate connection lines with subtle pulsing
    connectionsRef.current.forEach((line, index) => {
      const material = line.material as THREE.LineBasicMaterial;
      material.opacity = 0.4 + Math.sin(Date.now() * 0.001 + index) * ANIMATION_CONFIG.PULSE_AMPLITUDE;
    });
  }, []);

  useEffect(() => {
    if (!mountRef.current) return;

    // Clear any existing content
    mountRef.current.innerHTML = "";

    // Scene setup
    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x0a0a0a, 15, 35);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      CAMERA_CONFIG.FOV,
      window.innerWidth / window.innerHeight,
      CAMERA_CONFIG.NEAR,
      CAMERA_CONFIG.FAR
    );
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });

    // Get actual container dimensions
    const containerWidth = mountRef.current.clientWidth || window.innerWidth;
    const containerHeight = mountRef.current.clientHeight || window.innerHeight;

    renderer.setSize(containerWidth, containerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x000000, 0);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    // Update camera aspect ratio
    camera.aspect = containerWidth / containerHeight;
    camera.updateProjectionMatrix();

    mountRef.current.appendChild(renderer.domElement);

    // Lighting setup
    const ambientLight = new THREE.AmbientLight(0x404040, LIGHTING_CONFIG.AMBIENT_INTENSITY);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, LIGHTING_CONFIG.DIRECTIONAL_INTENSITY);
    directionalLight.position.set(5, 5, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    const pointLight = new THREE.PointLight(0x10b981, LIGHTING_CONFIG.POINT_INTENSITY, LIGHTING_CONFIG.POINT_DISTANCE);
    pointLight.position.set(0, 0, 0);
    scene.add(pointLight);

    // Create connections and particles
    const connections: THREE.Line[] = [];
    const particles: ParticleObject[] = [];

    SYSTEM_NODES.forEach((nodeData, index) => {
      const cardPos = cardPositions[index];
      
      // Convert screen coordinates to 3D world coordinates
      const worldX = ((cardPos.x - 50) / 50) * 6;
      const worldY = -((cardPos.y - 50) / 50) * 4;
      const worldZ = 2;
      
      // Create curved connection line
      const startPoint = new THREE.Vector3(0, 0, 0);
      const endPoint = new THREE.Vector3(worldX, worldY, worldZ);
      const midPoint = new THREE.Vector3(
        worldX * 0.5,
        worldY * 0.5 + 1,
        worldZ * 0.5
      );
      
      const curve = new THREE.QuadraticBezierCurve3(startPoint, midPoint, endPoint);
      const points = curve.getPoints(ANIMATION_CONFIG.CURVE_POINTS);
      
      const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
      const lineMaterial = new THREE.LineBasicMaterial({
        color: nodeData.color,
        transparent: true,
        opacity: ANIMATION_CONFIG.CONNECTION_OPACITY,
        linewidth: 2,
      });
      
      const connectionLine = new THREE.Line(lineGeometry, lineMaterial);
      scene.add(connectionLine);
      connections.push(connectionLine);
      
      // Create particles along the connection
      for (let i = 0; i < ANIMATION_CONFIG.PARTICLE_COUNT; i++) {
        const particleObj = createParticle(nodeData, curve, startPoint, endPoint, i, index);
        particles.push(particleObj);
        scene.add(particleObj.mesh);
      }
    });

    connectionsRef.current = connections;
    particlesRef.current = particles;

    // Camera positioning
    camera.position.set(...CAMERA_CONFIG.POSITION);
    camera.lookAt(0, 0, 0);

    // Animation loop
    let isAnimating = true;
    const animate = (): void => {
      if (!isAnimating) return;

      animationRef.current = requestAnimationFrame(animate);
      animateScene();
      renderer.render(scene, camera);
    };

    animate();

    // Handle resize
    const handleResize = (): void => {
      if (!mountRef.current) return;

      const containerWidth = mountRef.current.clientWidth || window.innerWidth;
      const containerHeight = mountRef.current.clientHeight || window.innerHeight;

      camera.aspect = containerWidth / containerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(containerWidth, containerHeight);
      renderer.setPixelRatio(window.devicePixelRatio);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      isAnimating = false;
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      window.removeEventListener("resize", handleResize);
      if (
        mountRef.current &&
        renderer.domElement &&
        mountRef.current.contains(renderer.domElement)
      ) {
        mountRef.current.removeChild(renderer.domElement);
      }
      scene.clear();
      renderer.dispose();
      renderer.forceContextLoss();
    };
  }, [createParticle, animateScene]);

  return { updateConnections };
};