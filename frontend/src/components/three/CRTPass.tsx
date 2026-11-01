"use client";
/**
 * Custom GLSL CRT post-process for the three.js canvases.
 * Renders the scene into an FBO, then draws a fullscreen quad with barrel
 * curvature, scanlines, cheap bloom, chromatic aberration, noise and vignette.
 * Takes over rendering (useFrame priority 1). Intensity follows the CRT toggle.
 */
import { useFBO } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import * as THREE from "three";
import { useViewer } from "@/lib/store";

const vert = /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

const frag = /* glsl */ `
precision highp float;
uniform sampler2D tDiffuse;
uniform vec2 uRes;
uniform float uTime;
uniform float uAmount;   // 0 = off, 0.5 = light, 1 = full
varying vec2 vUv;

vec2 barrel(vec2 uv, float k) {
  vec2 c = uv * 2.0 - 1.0;
  c *= 1.0 + k * dot(c, c);
  return c * 0.5 + 0.5;
}
float rand(vec2 co) { return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453); }

void main() {
  vec2 uv = barrel(vUv, 0.045 * uAmount);
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }

  // chromatic aberration grows towards the edges
  vec2 dir = uv - 0.5;
  float ab = 0.0028 * uAmount * length(dir) * 2.0;
  vec3 col;
  col.r = texture2D(tDiffuse, uv + dir * ab).r;
  col.g = texture2D(tDiffuse, uv).g;
  col.b = texture2D(tDiffuse, uv - dir * ab).b;

  // cheap phosphor bloom (8 taps)
  vec2 px = 1.0 / uRes;
  vec3 bloom = vec3(0.0);
  for (int i = 0; i < 8; i++) {
    float a = float(i) * 0.7853981;
    bloom += texture2D(tDiffuse, uv + vec2(cos(a), sin(a)) * px * 3.5).rgb;
    bloom += texture2D(tDiffuse, uv + vec2(cos(a), sin(a)) * px * 8.0).rgb * 0.6;
  }
  col += bloom / 12.8 * 0.9 * (0.35 + uAmount * 0.65);

  // scanlines + slow rolling bar
  float scan = 0.82 + 0.18 * sin(uv.y * uRes.y * 1.6);
  col *= mix(1.0, scan, uAmount);
  col *= 1.0 + 0.04 * uAmount * sin(uv.y * 6.0 - uTime * 1.3);

  // grain + vignette
  col += (rand(uv * uRes + uTime) - 0.5) * 0.05 * uAmount;
  float vig = smoothstep(1.05, 0.35, length(dir * vec2(1.1, 1.0)));
  col *= mix(1.0, vig, 0.8 * uAmount);

  gl_FragColor = vec4(col, 1.0);
}
`;

export function CRTPass() {
  const { gl, scene, camera, size } = useThree();
  const crtMode = useViewer((s) => s.crtMode);
  const fbo = useFBO({ samples: 0 });

  const { quadScene, quadCam, mat } = useMemo(() => {
    const mat = new THREE.ShaderMaterial({
      vertexShader: vert,
      fragmentShader: frag,
      uniforms: {
        tDiffuse: { value: null },
        uRes: { value: new THREE.Vector2(1, 1) },
        uTime: { value: 0 },
        uAmount: { value: 1 },
      },
      depthTest: false,
      depthWrite: false,
    });
    const quadScene = new THREE.Scene();
    quadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat));
    const quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    return { quadScene, quadCam, mat };
  }, []);

  useEffect(() => () => mat.dispose(), [mat]);

  useFrame((state) => {
    const dpr = gl.getPixelRatio();
    mat.uniforms.uRes.value.set(size.width * dpr, size.height * dpr);
    mat.uniforms.uTime.value = state.clock.elapsedTime;
    const target = crtMode === "full" ? 1 : crtMode === "light" ? 0.45 : 0;
    mat.uniforms.uAmount.value += (target - mat.uniforms.uAmount.value) * 0.15;
    gl.setRenderTarget(fbo);
    gl.render(scene, camera);
    gl.setRenderTarget(null);
    mat.uniforms.tDiffuse.value = fbo.texture;
    gl.render(quadScene, quadCam);
  }, 1);

  return null;
}

/** Raw sRGB colour for custom shaders (bypasses three's linear working space). */
export const srgb = (hex: string) => new THREE.Color(hex).convertLinearToSRGB();

/** Wireframe material whose lines glow near a moving scan plane. */
export function makeScanMaterial(color: string, axis: "x" | "y", opts: { opacity?: number } = {}) {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    wireframe: true,
    uniforms: {
      uColor: { value: srgb(color) },
      uScan: { value: -10 },
      uOpacity: { value: opts.opacity ?? 0.55 },
      uTime: { value: 0 },
    },
    vertexShader: /* glsl */ `
      uniform float uScan;
      varying float vGlow;
      varying float vPassed;
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        float p = wp.${axis};
        vGlow = smoothstep(0.22, 0.0, abs(p - uScan));
        vPassed = step(p, uScan);
        gl_Position = projectionMatrix * viewMatrix * wp;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor;
      uniform float uOpacity;
      varying float vGlow;
      varying float vPassed;
      void main() {
        vec3 c = uColor * (0.45 + 0.35 * vPassed) + vec3(0.85, 1.0, 0.9) * vGlow * 0.9;
        gl_FragColor = vec4(c, uOpacity * (0.6 + 0.4 * vPassed) + vGlow * 0.5);
      }`,
  });
}
