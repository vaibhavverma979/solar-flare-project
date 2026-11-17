import React, { useRef, useMemo, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sparkles, Html, Float } from '@react-three/drei';
import * as THREE from 'three';

/**
 * Animated Central Sun Mesh for Real-Time Nowcasting
 * Features reactive surface turbulence, corona glow, and live SoLEXS/HEL1OS flux response.
 */
function NowcastSunMesh({ solexsFlux = 38.5, isBursting = false }) {
  const sunRef = useRef();
  const coronaRef = useRef();
  const atmosphereRef = useRef();

  // Normalize flux values
  const normalizedFlux = Math.min(Math.max(solexsFlux / 100, 0), 4);
  const isHighActivity = solexsFlux > 80 || isBursting;
  const isExtremeActivity = solexsFlux > 150 || isBursting;

  useFrame((state, delta) => {
    if (sunRef.current) {
      sunRef.current.rotation.y += delta * (0.2 + normalizedFlux * 0.1);
    }
    if (coronaRef.current) {
      coronaRef.current.rotation.y -= delta * 0.35;
      coronaRef.current.rotation.z += delta * 0.15;
    }
    if (atmosphereRef.current) {
      const time = state.clock.getElapsedTime();
      atmosphereRef.current.scale.setScalar(
        1.18 + Math.sin(time * 4) * 0.04 * (1 + normalizedFlux * 0.5)
      );
    }
  });

  const sunColor = isExtremeActivity ? '#ff1100' : (isHighActivity ? '#ff6600' : '#facc15');
  const emissiveColor = isExtremeActivity ? '#ff0033' : (isHighActivity ? '#ea580c' : '#d97706');
  const emissiveIntensity = (0.9 + normalizedFlux * 0.6) * (isBursting ? 1.4 : 1.0);

  return (
    <group>
      {/* Outer Atmosphere Glow */}
      <mesh ref={atmosphereRef}>
        <sphereGeometry args={[1.65, 32, 32]} />
        <meshBasicMaterial
          color={isExtremeActivity ? '#ef4444' : '#f59e0b'}
          transparent
          opacity={isExtremeActivity ? 0.45 : 0.22}
          side={THREE.BackSide}
        />
      </mesh>

      {/* Wireframe Corona Active Region Grid */}
      <mesh ref={coronaRef}>
        <sphereGeometry args={[1.55, 28, 28]} />
        <meshStandardMaterial
          color={isExtremeActivity ? '#f43f5e' : '#38bdf8'}
          wireframe
          transparent
          opacity={isExtremeActivity ? 0.65 : 0.35}
          emissive={isExtremeActivity ? '#e11d48' : '#0284c7'}
          emissiveIntensity={1.3}
        />
      </mesh>

      {/* Central Solid Sun Body */}
      <mesh ref={sunRef}>
        <sphereGeometry args={[1.42, 64, 64]} />
        <meshStandardMaterial
          color={sunColor}
          emissive={emissiveColor}
          emissiveIntensity={emissiveIntensity}
          roughness={0.25}
          metalness={0.75}
        />
      </mesh>
    </group>
  );
}

/**
 * Magnetic Prominence Arch for Active Solar Flare Regions
 */
function MagneticProminenceLoop({ radius = 1.45, position = [0, 0, 0], rotation = [0, 0, 0], active = false }) {
  const loopRef = useRef();

  useFrame((state, delta) => {
    if (loopRef.current) {
      loopRef.current.rotation.z += delta * (active ? 0.8 : 0.2);
    }
  });

  return (
    <group position={position} rotation={rotation}>
      <mesh ref={loopRef}>
        <torusGeometry args={[0.5, 0.04, 16, 32, Math.PI]} />
        <meshStandardMaterial
          color={active ? '#ef4444' : '#38bdf8'}
          emissive={active ? '#dc2626' : '#0284c7'}
          emissiveIntensity={active ? 2.0 : 1.0}
          wireframe={!active}
          transparent
          opacity={active ? 0.9 : 0.6}
        />
      </mesh>
    </group>
  );
}

/**
 * Dynamic Solar Flare Eruption Particles for Real-Time Nowcast
 */
function LiveNowcastEruptions({ hel1osFlux = 18.2, isBursting = false }) {
  const groupRef = useRef();

  const isBurst = hel1osFlux > 50 || isBursting;
  const isExtreme = hel1osFlux > 120 || isBursting;

  const count = isExtreme ? 500 : (isBurst ? 300 : 140);
  const scale = isExtreme ? 4.8 : (isBurst ? 3.8 : 2.6);
  const speed = isExtreme ? 3.0 : (isBurst ? 2.0 : 0.9);
  const particleColor = isExtreme ? '#f43f5e' : (isBurst ? '#fb923c' : '#38bdf8');

  useFrame((state, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * (0.3 + hel1osFlux / 150);
      groupRef.current.rotation.x += delta * 0.1;
    }
  });

  return (
    <group ref={groupRef}>
      <Sparkles
        count={count}
        scale={scale}
        size={isBurst ? 5.5 : 3.2}
        speed={speed}
        color={particleColor}
        opacity={0.9}
        noise={isBurst ? 2.8 : 0.9}
      />

      {/* Surface Prominence Eruption Jets */}
      <MagneticProminenceLoop position={[0.8, 0.9, 0.7]} rotation={[0.4, 0.8, 0.2]} active={isBurst} />
      <MagneticProminenceLoop position={[-0.9, -0.6, 0.8]} rotation={[-0.3, -0.5, 0.7]} active={isExtreme} />

      {/* Polar Eruption Streamers during Hard X-Ray burst */}
      {isBurst && (
        <>
          <group position={[0, 1.45, 0]}>
            <Sparkles count={160} scale={[1.2, 3.2, 1.2]} size={5.0} speed={3.8} color="#f43f5e" />
          </group>
          <group position={[0, -1.45, 0]}>
            <Sparkles count={160} scale={[1.2, 3.2, 1.2]} size={5.0} speed={3.8} color="#f43f5e" />
          </group>
        </>
      )}
    </group>
  );
}

function CanvasLoader() {
  return (
    <Html center>
      <div className="flex flex-col items-center justify-center gap-2 bg-slate-900/90 backdrop-blur-md p-3.5 rounded-xl border border-slate-800 text-slate-300 text-xs font-mono">
        <div className="w-6 h-6 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
        <span>INITIALIZING REAL-TIME NOWCAST 3D...</span>
      </div>
    </Html>
  );
}

/**
 * 3D Nowcasting Telemetry Component
 */
export default function Nowcast3DSolar({
  solexsFlux = 38.5,
  hel1osFlux = 18.2,
  isBursting = false,
  flareClass = 'A'
}) {
  const activeClass = isBursting ? 'X' : flareClass;

  return (
    <div className="w-full h-full min-h-[340px] relative rounded-xl overflow-hidden bg-slate-950/90 border border-slate-800/80">
      {/* Telemetry Status Overlay */}
      <div className="absolute top-3 left-3 z-10 flex flex-col gap-1 text-[11px] font-mono pointer-events-none select-none bg-slate-900/85 backdrop-blur-md p-3 rounded-xl border border-slate-800/80 shadow-lg">
        <div className="flex items-center gap-2 mb-0.5">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span className="text-cyan-400 font-bold tracking-wide">3D NOWCASTING TELEMETRY</span>
        </div>
        
        <div className="flex items-center justify-between gap-4 text-slate-300">
          <span className="text-slate-400">SoLEXS (Soft X-Ray):</span>
          <span className="text-cyan-400 font-bold">{solexsFlux.toFixed(1)} counts/s</span>
        </div>

        <div className="flex items-center justify-between gap-4 text-slate-300">
          <span className="text-slate-400">HEL1OS (Hard X-Ray):</span>
          <span className={hel1osFlux > 50 || isBursting ? "text-rose-400 font-bold animate-pulse" : "text-amber-300 font-bold"}>
            {hel1osFlux.toFixed(1)} counts/s
          </span>
        </div>

        <div className="mt-1 pt-1.5 border-t border-slate-800/80 flex items-center justify-between">
          <span className="text-slate-400">Detected Class:</span>
          <span className={`px-2 py-0.5 rounded text-xs font-bold ${
            activeClass === 'X' ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40 animate-pulse' :
            activeClass === 'M' ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' :
            'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
          }`}>
            CLASS {activeClass}
          </span>
        </div>
      </div>

      {/* 3D Canvas */}
      <Canvas camera={{ position: [0, 0, 5.2], fov: 45 }}>
        <ambientLight intensity={0.4} />
        <pointLight position={[10, 10, 10]} intensity={1.8} color="#fff" />
        <pointLight position={[-10, -10, -10]} intensity={0.6} color="#00f2fe" />
        
        <Suspense fallback={<CanvasLoader />}>
          <Float speed={1.2} rotationIntensity={0.2} floatIntensity={0.2}>
            <NowcastSunMesh solexsFlux={solexsFlux} isBursting={isBursting} />
            <LiveNowcastEruptions hel1osFlux={hel1osFlux} isBursting={isBursting} />
          </Float>
        </Suspense>

        <OrbitControls enableZoom={true} enablePan={false} maxDistance={8} minDistance={3} autoRotate={false} />
      </Canvas>
    </div>
  );
}
