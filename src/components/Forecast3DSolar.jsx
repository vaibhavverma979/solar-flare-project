import React, { useRef, useMemo, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sparkles, Html, Float } from '@react-three/drei';
import * as THREE from 'three';

/**
 * Solar Mesh with Dynamic Risk Corona for Predictive Forecasting
 */
function ForecastSunMesh({ probability = 0.40, riskLevel = 'MEDIUM' }) {
  const sunRef = useRef();
  const riskHaloRef = useRef();
  const auraRingRef = useRef();

  const isHighRisk = riskLevel === 'HIGH' || riskLevel === 'EXTREME' || probability > 0.65;
  const isMediumRisk = riskLevel === 'MEDIUM' || (probability >= 0.3 && probability <= 0.65);

  useFrame((state, delta) => {
    if (sunRef.current) {
      sunRef.current.rotation.y += delta * 0.18;
    }
    if (riskHaloRef.current) {
      riskHaloRef.current.rotation.y -= delta * 0.3;
      const pulse = Math.sin(state.clock.getElapsedTime() * 3) * 0.05;
      riskHaloRef.current.scale.setScalar(1.2 + pulse * (1 + probability));
    }
    if (auraRingRef.current) {
      auraRingRef.current.rotation.z += delta * 0.4;
      auraRingRef.current.rotation.x = Math.sin(state.clock.getElapsedTime() * 0.5) * 0.2;
    }
  });

  // Color dynamics according to risk
  const haloColor = isHighRisk ? '#f43f5e' : (isMediumRisk ? '#f59e0b' : '#10b981');
  const emissiveColor = isHighRisk ? '#e11d48' : (isMediumRisk ? '#d97706' : '#059669');

  return (
    <group>
      {/* Outer Predictive Risk Halo Atmosphere */}
      <mesh ref={riskHaloRef}>
        <sphereGeometry args={[1.5, 32, 32]} />
        <meshBasicMaterial
          color={haloColor}
          transparent
          opacity={0.3 + probability * 0.4}
          side={THREE.BackSide}
        />
      </mesh>

      {/* Orbiting Risk Boundary Ring */}
      <mesh ref={auraRingRef}>
        <torusGeometry args={[1.9, 0.02, 16, 64]} />
        <meshStandardMaterial
          color={haloColor}
          emissive={emissiveColor}
          emissiveIntensity={1.5}
          wireframe
        />
      </mesh>

      {/* Sun Body with Predictive Temperature Gradient */}
      <mesh ref={sunRef}>
        <sphereGeometry args={[1.38, 64, 64]} />
        <meshStandardMaterial
          color={isHighRisk ? '#ea580c' : (isMediumRisk ? '#d97706' : '#0284c7')}
          emissive={emissiveColor}
          emissiveIntensity={0.8 + probability * 0.8}
          roughness={0.3}
          metalness={0.7}
        />
      </mesh>
    </group>
  );
}

/**
 * Neupert Effect Precursor Vector Stream Cones
 * Visualizes non-thermal electron lead streams radiating into interplanetary space.
 */
function SolarWindPrecursorCone({ angle = 0, probability = 0.4, horizonMin = 30 }) {
  const coneRef = useRef();

  useFrame((state, delta) => {
    if (coneRef.current) {
      coneRef.current.rotation.z = angle + Math.sin(state.clock.getElapsedTime() * 2) * 0.1;
    }
  });

  const coneLength = 2.2 + probability * 1.5;
  const isHighRisk = probability > 0.6;

  return (
    <group ref={coneRef} rotation={[0, 0, angle]}>
      <mesh position={[0, coneLength / 2 + 1.3, 0]}>
        <coneGeometry args={[0.45 + probability * 0.3, coneLength, 16, 1, true]} />
        <meshStandardMaterial
          color={isHighRisk ? '#f43f5e' : '#38bdf8'}
          emissive={isHighRisk ? '#e11d48' : '#0284c7'}
          emissiveIntensity={1.4}
          wireframe
          transparent
          opacity={0.4 + probability * 0.4}
        />
      </mesh>

      <Sparkles
        count={Math.round(80 * (1 + probability))}
        scale={[1.0, coneLength, 1.0]}
        size={4.0}
        speed={1.5 + probability * 2.0}
        color={isHighRisk ? '#fb7185' : '#38bdf8'}
      />
    </group>
  );
}

function CanvasLoader() {
  return (
    <Html center>
      <div className="flex flex-col items-center justify-center gap-2 bg-slate-900/90 backdrop-blur-md p-3.5 rounded-xl border border-slate-800 text-slate-300 text-xs font-mono">
        <div className="w-6 h-6 border-2 border-amber-400 border-t-transparent rounded-full animate-spin"></div>
        <span>INITIALIZING PREDICTIVE FORECAST 3D...</span>
      </div>
    </Html>
  );
}

/**
 * 3D Forecasting Telemetry Component
 */
export default function Forecast3DSolar({
  forecastData = {
    current_probability: 0.24,
    peak_window_probability: 0.40,
    risk_level: 'MEDIUM',
    horizon_min: 30
  },
  isBursting = false
}) {
  const effectiveProb = isBursting ? 0.88 : (forecastData.peak_window_probability || 0.40);
  const effectiveRisk = isBursting ? 'HIGH' : (forecastData.risk_level || 'MEDIUM');

  return (
    <div className="w-full h-full min-h-[340px] relative rounded-xl overflow-hidden bg-slate-950/90 border border-slate-800/80">
      {/* Telemetry Status Overlay */}
      <div className="absolute top-3 left-3 z-10 flex flex-col gap-1 text-[11px] font-mono pointer-events-none select-none bg-slate-900/85 backdrop-blur-md p-3 rounded-xl border border-slate-800/80 shadow-lg">
        <div className="flex items-center gap-2 mb-0.5">
          <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span>
          <span className="text-amber-400 font-bold tracking-wide">3D FORECASTING TELEMETRY</span>
        </div>

        <div className="flex items-center justify-between gap-4 text-slate-300">
          <span className="text-slate-400">Peak Window Prob:</span>
          <span className="text-amber-400 font-bold">{(effectiveProb * 100).toFixed(0)}%</span>
        </div>

        <div className="flex items-center justify-between gap-4 text-slate-300">
          <span className="text-slate-400">Forecast Horizon:</span>
          <span className="text-cyan-400 font-bold">{forecastData.horizon_min || 30} Minutes</span>
        </div>

        <div className="mt-1 pt-1.5 border-t border-slate-800/80 flex items-center justify-between">
          <span className="text-slate-400">Risk Assessment:</span>
          <span className={`px-2 py-0.5 rounded text-xs font-bold ${
            effectiveRisk === 'HIGH' || effectiveRisk === 'EXTREME' ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40 animate-pulse' :
            effectiveRisk === 'MEDIUM' ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' :
            'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
          }`}>
            {effectiveRisk} RISK
          </span>
        </div>
      </div>

      {/* 3D Canvas */}
      <Canvas camera={{ position: [0, 0, 5.5], fov: 45 }}>
        <ambientLight intensity={0.4} />
        <pointLight position={[10, 10, 10]} intensity={1.8} color="#fff" />
        <pointLight position={[-10, -10, -10]} intensity={0.8} color="#f59e0b" />

        <Suspense fallback={<CanvasLoader />}>
          <Float speed={1.0} rotationIntensity={0.15} floatIntensity={0.2}>
            <ForecastSunMesh probability={effectiveProb} riskLevel={effectiveRisk} />
            <SolarWindPrecursorCone angle={Math.PI / 4} probability={effectiveProb} horizonMin={forecastData.horizon_min} />
            <SolarWindPrecursorCone angle={-Math.PI / 3} probability={effectiveProb * 0.8} horizonMin={forecastData.horizon_min} />
          </Float>
        </Suspense>

        <OrbitControls enableZoom={true} enablePan={false} maxDistance={8} minDistance={3} autoRotate={false} />
      </Canvas>
    </div>
  );
}
