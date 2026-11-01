"use client";
/**
 * 3D TUMOR RECONSTRUCTION SCENE
 * Stacks marching-squares axial contour rings into a 3D wireframe brain with
 * glowing amber edema/whole-tumor and red-orange enhancing tumor core.
 * Includes a sweeping vertical scan-beam shader and interactive OrbitControls.
 *
 * SWAPPING IN REAL MESHES: export marching-cubes meshes from your Python
 * pipeline (`skimage.measure.marching_cubes` -> `.glb` via `trimesh`) and load
 * them here with `useGLTF('/models/case_001_brain.glb')`.
 */
import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import { buildBrainContours, buildTumorContours, type ContourSet } from "@/lib/contours";
import { CRTPass, srgb } from "./CRTPass";

function makeStackMaterial(hex: string, opacity: number) {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uColor: { value: srgb(hex) },
      uBeamY: { value: 0 },
      uAssemble: { value: 1 },
      uExplode: { value: 0 },
      uOpacity: { value: opacity },
    },
    vertexShader: /* glsl */ `
      attribute float aSlice;
      uniform float uBeamY;
      uniform float uAssemble;
      uniform float uExplode;
      varying float vVisible;
      varying float vBeam;
      void main() {
        vVisible = smoothstep(aSlice - 0.03, aSlice + 0.01, uAssemble);
        vec3 pos = position;
        pos.y += (aSlice - 0.5) * uExplode * 0.75;
        vec4 wp = modelMatrix * vec4(pos, 1.0);
        vBeam = smoothstep(0.16, 0.0, abs(wp.y - uBeamY));
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor;
      uniform float uOpacity;
      varying float vVisible;
      varying float vBeam;
      void main() {
        if (vVisible <= 0.01) discard;
        vec3 col = uColor + vec3(0.95, 1.0, 0.85) * vBeam * 0.85;
        float a = (uOpacity + vBeam * 0.55) * vVisible;
        gl_FragColor = vec4(col, a);
      }
    `,
  });
}

function toGeometry(cs: ContourSet) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(cs.positions, 3));
  g.setAttribute("aSlice", new THREE.BufferAttribute(cs.slice, 1));
  return g;
}

export type ReconProps = {
  assemble: number; // 0..1 fraction of axial slices stacked
  explode: number;  // 0..1 vertical slice separation
  autoRotate: boolean;
  showBrain: boolean;
  showEdema: boolean;
  showEnhancing: boolean;
};

function StackModel({ assemble, explode, autoRotate, showBrain, showEdema, showEnhancing }: ReconProps) {
  const root = useRef<THREE.Group>(null);
  const beamPlane = useRef<THREE.Mesh>(null);

  const { brainGeo, wtGeo, etGeo } = useMemo(() => {
    return {
      brainGeo: toGeometry(buildBrainContours(34, 68)),
      wtGeo: toGeometry(buildTumorContours("wt", 38, 84)),
      etGeo: toGeometry(buildTumorContours("et", 36, 84)),
    };
  }, []);

  const brainMat = useMemo(() => makeStackMaterial("#7cff9b", 0.36), []);
  const wtMat = useMemo(() => makeStackMaterial("#ffb000", 0.78), []);
  const etMat = useMemo(() => makeStackMaterial("#ff5a1f", 0.95), []);

  useFrame((state, dt) => {
    const t = state.clock.elapsedTime;
    const beamY = Math.sin(t * 1.15) * 0.65;
    for (const m of [brainMat, wtMat, etMat]) {
      m.uniforms.uBeamY.value = beamY;
      m.uniforms.uAssemble.value = assemble;
      m.uniforms.uExplode.value = explode;
    }
    if (beamPlane.current) {
      beamPlane.current.position.y = beamY;
    }
    if (root.current && autoRotate) {
      root.current.rotation.y += dt * 0.32;
    }
  });

  return (
    <group ref={root}>
      {/* Coordinate cage / stereotactic frame */}
      <mesh>
        <cylinderGeometry args={[0.98, 0.98, 1.38, 32, 1, true]} />
        <meshBasicMaterial color={srgb("#1f3a26")} wireframe transparent opacity={0.35} />
      </mesh>

      {showBrain && <lineSegments geometry={brainGeo} material={brainMat} />}
      {showEdema && <lineSegments geometry={wtGeo} material={wtMat} />}
      {showEnhancing && <lineSegments geometry={etGeo} material={etMat} />}

      {/* Sweeping laser scan plane */}
      <mesh ref={beamPlane} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.05, 0.96, 48]} />
        <meshBasicMaterial
          color={srgb("#7cff9b")}
          transparent
          opacity={0.14}
          side={THREE.DoubleSide}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

export default function ReconScene(props: ReconProps) {
  return (
    <Canvas
      dpr={[1, 1.6]}
      camera={{ position: [1.55, 0.95, 1.85], fov: 38, near: 0.05, far: 30 }}
      gl={{ antialias: true, powerPreference: "high-performance" }}
    >
      <color attach="background" args={["#0b0d0a"]} />
      <StackModel {...props} />
      <OrbitControls
        enablePan={false}
        minDistance={1.3}
        maxDistance={3.4}
        dampingFactor={0.08}
        enableDamping
      />
      <CRTPass />
    </Canvas>
  );
}

