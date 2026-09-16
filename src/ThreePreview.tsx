import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { safeAssetUrl, VehicleWheel } from './job-utils';

export type ViewerMode = 'material' | 'wireframe' | 'normals';

export interface MeshStats {
  faces: number;
  vertices: number;
  dimensions: [number, number, number];
}

interface PreviewProps {
  url: string | null;
  mode: ViewerMode;
  markers?: VehicleWheel[];
  marking?: boolean;
  onPick?: (point: [number, number, number]) => void;
  onStats?: (stats: MeshStats | null) => void;
}

function eachMaterial(material: THREE.Material | THREE.Material[], fn: (material: THREE.Material) => void) {
  for (const entry of Array.isArray(material) ? material : [material]) if (entry) fn(entry);
}

function disposeMaterial(material: THREE.Material) {
  for (const key of Object.keys(material)) {
    const value = (material as unknown as Record<string, unknown>)[key];
    if (value instanceof THREE.Texture) value.dispose();
  }
  material.dispose();
}

function disposeObject(root: THREE.Object3D) {
  root.traverse(object => {
    const mesh = object as THREE.Mesh;
    if (!mesh.isMesh) return;
    mesh.geometry?.dispose();
    eachMaterial(mesh.userData.swapMaterial ?? mesh.material, disposeMaterial);
    if (mesh.userData.swapMaterial) eachMaterial(mesh.userData.originalMaterial ?? [], disposeMaterial);
  });
}

function applyMode(root: THREE.Object3D, mode: ViewerMode) {
  root.traverse(object => {
    const mesh = object as THREE.Mesh;
    if (!mesh.isMesh) return;
    if (!mesh.userData.originalMaterial) mesh.userData.originalMaterial = mesh.material;
    const original = mesh.userData.originalMaterial as THREE.Material | THREE.Material[];
    if (mesh.userData.swapMaterial) eachMaterial(mesh.userData.swapMaterial, disposeMaterial);
    mesh.userData.swapMaterial = undefined;
    if (mode === 'material') {
      mesh.material = original;
      return;
    }
    const swap = (source: THREE.Material) => {
      if (mode === 'normals') return new THREE.MeshNormalMaterial();
      const color = (source as THREE.MeshStandardMaterial).color;
      return new THREE.MeshBasicMaterial({ color: color ? color.clone() : new THREE.Color('#8aa59b'), wireframe: true });
    };
    const swapped = Array.isArray(original) ? original.map(swap) : swap(original);
    mesh.userData.swapMaterial = swapped;
    mesh.material = swapped;
  });
}

function collectStats(root: THREE.Object3D): MeshStats {
  let faces = 0;
  let vertices = 0;
  root.traverse(object => {
    const mesh = object as THREE.Mesh;
    if (!mesh.isMesh || !mesh.geometry) return;
    const geometry = mesh.geometry as THREE.BufferGeometry;
    vertices += geometry.attributes.position?.count ?? 0;
    faces += Math.round((geometry.index ? geometry.index.count : geometry.attributes.position?.count ?? 0) / 3);
  });
  const size = new THREE.Box3().setFromObject(root).getSize(new THREE.Vector3());
  return { faces, vertices, dimensions: [size.x, size.y, size.z] };
}

export interface ThreePreviewHandle {
  capture: () => string | null;
}

const ThreePreview = forwardRef<ThreePreviewHandle, PreviewProps>(function ThreePreview({ url, mode, markers, marking, onPick, onStats }, ref) {
  const host = useRef<HTMLDivElement>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [rendererError, setRendererError] = useState<string | null>(null);
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const markingRef = useRef(marking);
  markingRef.current = marking;
  const onPickRef = useRef(onPick);
  onPickRef.current = onPick;
  const view = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    controls: OrbitControls;
    loaded: THREE.Object3D | null;
    floor: THREE.Mesh;
    markerGroup: THREE.Group;
    home: { position: THREE.Vector3; target: THREE.Vector3 } | null;
    render: () => void;
  } | null>(null);

  useEffect(() => {
    if (!host.current) return;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#11191a');
    const camera = new THREE.PerspectiveCamera(34, 1, 0.01, 500);
    camera.position.set(3.2, 2.6, 4.2);
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch {
      setRendererError('WebGL is unavailable in this browser — downloads remain available below.');
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.setAttribute('role', 'img');
    renderer.domElement.setAttribute('aria-label', '3D preview. Arrow keys orbit, plus and minus zoom, F fits the model, R resets the view.');
    host.current.appendChild(renderer.domElement);
    scene.add(new THREE.HemisphereLight('#d8fff3', '#172321', 2.1));
    const key = new THREE.DirectionalLight('#fff1d3', 2.2);
    key.position.set(3, 5, 4);
    scene.add(key);
    const floor = new THREE.Mesh(new THREE.CircleGeometry(7, 48), new THREE.MeshStandardMaterial({ color: '#17201f', roughness: 1 }));
    floor.rotation.x = -Math.PI / 2;
    scene.add(floor);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = false;
    const render = () => renderer.render(scene, camera);
    controls.addEventListener('change', render);
    const markerGroup = new THREE.Group();
    scene.add(markerGroup);
    const state = { scene, camera, renderer, controls, loaded: null as THREE.Object3D | null, floor, markerGroup, home: null as { position: THREE.Vector3; target: THREE.Vector3 } | null, render };
    view.current = state;

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const onPointerDown = (event: PointerEvent) => {
      if (!markingRef.current || !onPickRef.current || !state.loaded) return;
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObject(state.loaded, true);
      if (hits.length) onPickRef.current([hits[0].point.x, hits[0].point.y, hits[0].point.z]);
    };
    renderer.domElement.addEventListener('pointerdown', onPointerDown);

    const fit = () => {
      if (!state.loaded) return;
      const box = new THREE.Box3().setFromObject(state.loaded);
      if (box.isEmpty()) return;
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3()).length() || 1;
      const direction = new THREE.Vector3(1, 0.75, 1).normalize();
      camera.position.copy(center).addScaledVector(direction, size * 1.6);
      camera.near = Math.max(size / 1000, 0.001);
      camera.far = size * 100;
      camera.updateProjectionMatrix();
      controls.target.copy(center);
      controls.update();
      state.home = { position: camera.position.clone(), target: center.clone() };
      render();
    };
    const reset = () => {
      if (!state.home) return;
      camera.position.copy(state.home.position);
      controls.target.copy(state.home.target);
      controls.update();
      render();
    };
    const orbit = (theta: number, phi: number) => {
      const offset = camera.position.clone().sub(controls.target);
      const spherical = new THREE.Spherical().setFromVector3(offset);
      spherical.theta -= theta;
      spherical.phi = Math.min(Math.PI - 0.05, Math.max(0.05, spherical.phi - phi));
      camera.position.setFromSpherical(spherical).add(controls.target);
      controls.update();
      render();
    };
    const zoom = (factor: number) => {
      const offset = camera.position.clone().sub(controls.target);
      const distance = Math.max(0.05, offset.length() * factor);
      camera.position.copy(controls.target).addScaledVector(offset.normalize(), distance);
      controls.update();
      render();
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'ArrowLeft') { orbit(0.12, 0); event.preventDefault(); }
      else if (event.key === 'ArrowRight') { orbit(-0.12, 0); event.preventDefault(); }
      else if (event.key === 'ArrowUp') { orbit(0, 0.12); event.preventDefault(); }
      else if (event.key === 'ArrowDown') { orbit(0, -0.12); event.preventDefault(); }
      else if (event.key === '+' || event.key === '=') { zoom(0.85); event.preventDefault(); }
      else if (event.key === '-' || event.key === '_') { zoom(1.18); event.preventDefault(); }
      else if (event.key === 'f' || event.key === '0') { fit(); event.preventDefault(); }
      else if (event.key === 'r') { reset(); event.preventDefault(); }
    };
    const onFit = () => fit();
    const onReset = () => reset();
    renderer.domElement.addEventListener('keydown', onKey);
    renderer.domElement.tabIndex = 0;
    renderer.domElement.addEventListener('preview-fit', onFit);
    renderer.domElement.addEventListener('preview-reset', onReset);
    const resize = () => {
      if (!host.current) return;
      const { width, height } = host.current.getBoundingClientRect();
      if (!width || !height) return;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
      render();
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(host.current);
    render();
    return () => {
      observer.disconnect();
      renderer.domElement.removeEventListener('keydown', onKey);
      renderer.domElement.removeEventListener('preview-fit', onFit);
      renderer.domElement.removeEventListener('preview-reset', onReset);
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      controls.removeEventListener('change', render);
      controls.dispose();
      if (state.loaded) disposeObject(state.loaded);
      disposeObject(floor);
      disposeObject(markerGroup);
      renderer.dispose();
      renderer.domElement.remove();
      view.current = null;
    };
  }, []);

  useEffect(() => {
    const state = view.current;
    if (!state) return;
    if (state.loaded) {
      state.scene.remove(state.loaded);
      disposeObject(state.loaded);
      state.loaded = null;
      state.home = null;
      state.render();
    }
    onStats?.(null);
    setLoadError(null);
    if (!url) {
      state.render();
      return;
    }
    let disposed = false;
    const manager = new THREE.LoadingManager();
    manager.setURLModifier(requested => {
      const safe = safeAssetUrl(requested, url);
      if (safe === null) throw new Error(`Blocked external asset URL: ${requested}`);
      return safe;
    });
    try {
      new GLTFLoader(manager).load(
      url,
      loaded => {
        if (disposed || !view.current) {
          disposeObject(loaded.scene);
          return;
        }
        loaded.scene.traverse(object => {
          const mesh = object as THREE.Mesh;
          if (mesh.isMesh && mesh.geometry?.attributes?.position && !mesh.geometry.attributes.normal) mesh.geometry.computeVertexNormals();
        });
        state.loaded = loaded.scene;
        state.scene.add(loaded.scene);
        applyMode(loaded.scene, modeRef.current);
        onStats?.(collectStats(loaded.scene));
        const box = new THREE.Box3().setFromObject(loaded.scene);
        if (!box.isEmpty()) {
          const center = box.getCenter(new THREE.Vector3());
          const size = box.getSize(new THREE.Vector3()).length() || 1;
          state.floor.position.y = box.min.y - 0.002;
          state.camera.position.copy(center).addScaledVector(new THREE.Vector3(1, 0.75, 1).normalize(), size * 1.6);
          state.camera.near = Math.max(size / 1000, 0.001);
          state.camera.far = size * 100;
          state.camera.updateProjectionMatrix();
          state.controls.target.copy(center);
          state.controls.update();
          state.home = { position: state.camera.position.clone(), target: center.clone() };
        }
        state.render();
      },
      undefined,
      () => {
        /* Keep the procedural preview when an artifact is unavailable. */
        if (disposed || !view.current) return;
        setLoadError('Preview failed to load — downloads remain available below.');
        onStats?.(null);
      },
    );
    } catch {
      setLoadError('Preview failed to load — downloads remain available below.');
      onStats?.(null);
    }
    return () => {
      disposed = true;
    };
  }, [url]);

  useEffect(() => {
    const state = view.current;
    if (!state?.loaded) return;
    applyMode(state.loaded, mode);
    state.render();
  }, [mode, url]);

  useEffect(() => {
    const state = view.current;
    if (!state) return;
    state.controls.enabled = !marking;
    state.render();
  }, [marking]);

  useEffect(() => {
    const state = view.current;
    if (!state) return;
    const group = state.markerGroup;
    while (group.children.length) {
      const child = group.children.pop() as THREE.Object3D;
      disposeObject(child);
    }
    for (const wheel of markers ?? []) {
      const color = wheel.steer ? '#4fd1c5' : '#e8a34f';
      const cylinder = new THREE.Mesh(
        new THREE.CylinderGeometry(wheel.radius, wheel.radius, wheel.half_width * 2, 28, 1, true),
        new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: 0.75, depthTest: false }),
      );
      // CylinderGeometry's axis is Y — rotate onto the wheel's axle axis.
      const axis = new THREE.Vector3(...wheel.axis).normalize();
      cylinder.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis);
      cylinder.position.set(...wheel.center);
      const hub = new THREE.Mesh(new THREE.SphereGeometry(wheel.radius * 0.12, 12, 8), new THREE.MeshBasicMaterial({ color, depthTest: false }));
      hub.position.set(...wheel.center);
      cylinder.renderOrder = 10;
      hub.renderOrder = 10;
      group.add(cylinder, hub);
    }
    state.render();
  }, [markers, url]);

  useImperativeHandle(ref, () => ({
    capture: () => {
      const state = view.current;
      if (!state?.loaded) return null;
      state.render();
      try {
        return state.renderer.domElement.toDataURL('image/png');
      } catch {
        return null;
      }
    },
  }), []);

  const emit = (name: string) => view.current?.renderer.domElement.dispatchEvent(new Event(name));
  return (
    <div className="preview-frame">
      <div ref={host} className="three-preview" aria-label="Interactive 3D asset preview" />
      {rendererError ? <div className="preview-error" role="alert">{rendererError}</div> : null}
      {loadError ? <div className="preview-error" role="alert">{loadError}</div> : null}
      {!url && !loadError ? <div className="preview-empty">Empty preview — generate or select an asset</div> : null}
      <div className="preview-controls">
        <button type="button" className="ghost-button compact" onClick={() => emit('preview-fit')} aria-label="Fit model to view">Fit</button>
        <button type="button" className="ghost-button compact" onClick={() => emit('preview-reset')} aria-label="Reset view">Reset</button>
      </div>
    </div>
  );
});

export default ThreePreview;
