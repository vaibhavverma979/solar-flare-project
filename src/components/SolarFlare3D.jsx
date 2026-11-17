import React, { useRef, useState, useMemo, Suspense } from 'react';
import { Canvas, useFrame, useLoader } from '@react-three/fiber';
import { OrbitControls, Stars, Sparkles, Html, Float } from '@react-three/drei';
import * as THREE from 'three';

const EARTH_TEXTURE_URL = 'https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg';
const CLOUDS_TEXTURE_URL = 'https://unpkg.com/three-globe/example/img/earth-clouds1024.png';

/**
 * 3D Sun Celestial Body at Position [-3, 0, 0]
 */
function SunBody({ solexsFlux = 38.5, hel1osFlux = 18.2, isBursting = false }) {
  const sunMeshRef = useRef();
  const coronaMeshRef = useRef();
  const atmosphereMeshRef = useRef();

  const isSpiking = hel1osFlux > 50 || isBursting;
  const isExtreme = hel1osFlux > 100 || isBursting;

  useFrame((state, delta) => {
    if (sunMeshRef.current) {
      sunMeshRef.current.rotation.y += delta * 0.25;
    }
    if (coronaMeshRef.current) {
      coronaMeshRef.current.rotation.y -= delta * 0.4;
      coronaMeshRef.current.rotation.z += delta * 0.1;
    }
    if (atmosphereMeshRef.current) {
      const time = state.clock.getElapsedTime();
      atmosphereMeshRef.current.scale.setScalar(
        1.15 + Math.sin(time * 3) * 0.03 * (1 + (solexsFlux / 100))
      );
    }
  });

  const sunColor = isExtreme ? '#ff1100' : (isSpiking ? '#ff6600' : '#facc15');
  const emissiveColor = isExtreme ? '#ff0033' : (isSpiking ? '#ea580c' : '#d97706');
  const emissiveIntensity = (0.8 + (solexsFlux / 100) * 0.5) * (isBursting ? 1.4 : 1.0);

  return (
    <group position={[-3, 0, 0]}>
      {/* Outer Atmosphere Atmosphere Glow */}
      <mesh ref={atmosphereMeshRef}>
        <sphereGeometry args={[1.35, 32, 32]} />
        <meshBasicMaterial
          color={isExtreme ? '#ef4444' : '#f59e0b'}
          transparent
          opacity={isExtreme ? 0.45 : 0.25}
          side={THREE.BackSide}
        />
      </mesh>

      {/* Wireframe Corona Active Regions */}
      <mesh ref={coronaMeshRef}>
        <sphereGeometry args={[1.25, 24, 24]} />
        <meshStandardMaterial
          color={isExtreme ? '#f43f5e' : '#38bdf8'}
          wireframe
          transparent
          opacity={isExtreme ? 0.65 : 0.35}
          emissive={isExtreme ? '#e11d48' : '#0284c7'}
          emissiveIntensity={1.2}
        />
      </mesh>

      {/* Solid Sun Core */}
      <mesh ref={sunMeshRef}>
        <sphereGeometry args={[1.15, 64, 64]} />
        <meshStandardMaterial
          color={sunColor}
          emissive={emissiveColor}
          emissiveIntensity={emissiveIntensity}
          roughness={0.2}
          metalness={0.8}
        />
      </mesh>

      {/* Flare Prominence Sparkles */}
      <Sparkles
        count={isExtreme ? 350 : (isSpiking ? 200 : 100)}
        scale={isExtreme ? 3.8 : (isSpiking ? 2.8 : 2.2)}
        size={isSpiking ? 5.0 : 3.0}
        speed={isSpiking ? 2.5 : 1.0}
        color={isExtreme ? '#f43f5e' : (isSpiking ? '#fb923c' : '#38bdf8')}
        opacity={0.9}
      />
    </group>
  );
}

/**
 * 3D Earth Celestial Body at Position [3, 0, 0] with NASA Textures & Fallback Shader
 */
function EarthBody({ isNowcastActive = false, hel1osFlux = 18.2, mag }) {
  const earthRef = useRef();
  const cloudsRef = useRef();
  const shieldRef = useRef();

  const isShieldHitting = isNowcastActive || hel1osFlux > 50;
  const bz = mag?.b_z ?? 5.0;
  const isSouthward = bz < 0;

  // Load Earth & Clouds textures with useLoader
  let earthTexture = null;
  let cloudsTexture = null;

  try {
    [earthTexture, cloudsTexture] = useLoader(THREE.TextureLoader, [
      EARTH_TEXTURE_URL,
      CLOUDS_TEXTURE_URL,
    ]);
  } catch (err) {
    console.warn("Earth texture load notice, using fallback shaders:", err);
  }

  useFrame((state, delta) => {
    if (earthRef.current) {
      earthRef.current.rotation.y += delta * 0.12;
    }
    if (cloudsRef.current) {
      cloudsRef.current.rotation.y += delta * 0.18;
    }
    if (shieldRef.current) {
      const time = state.clock.getElapsedTime();
      shieldRef.current.rotation.z += delta * 0.3;
      const pulseScalar = 1.25 + Math.sin(time * 6) * (isShieldHitting ? 0.08 : 0.02);
      
      // Dynamic Pressure Compression (Sunward side X-axis)
      let compressX = 1.0;
      if (isSouthward && isShieldHitting) {
         compressX = 0.65; // Compress the shield facing the sun
      }
      
      shieldRef.current.scale.set(pulseScalar * compressX, pulseScalar, pulseScalar);
    }
  });

  const shieldColor = isSouthward && isShieldHitting ? '#f43f5e' : (isShieldHitting ? '#f97316' : '#22d3ee');
  const shieldEmissive = isSouthward && isShieldHitting ? '#ef4444' : (isShieldHitting ? '#ef4444' : '#0284c7');

  return (
    <group position={[3, 0, 0]}>
      {/* Atmosphere Glow */}
      <mesh>
        <sphereGeometry args={[0.92, 32, 32]} />
        <meshBasicMaterial
          color={isShieldHitting ? '#f97316' : '#38bdf8'}
          transparent
          opacity={isShieldHitting ? 0.45 : 0.25}
          side={THREE.BackSide}
        />
      </mesh>

      {/* Rotating Clouds Layer */}
      <mesh ref={cloudsRef}>
        <sphereGeometry args={[0.82, 48, 48]} />
        {cloudsTexture ? (
          <meshStandardMaterial
            map={cloudsTexture}
            transparent
            opacity={0.45}
            blending={THREE.AdditiveBlending}
          />
        ) : (
          <meshStandardMaterial
            color="#ffffff"
            wireframe
            transparent
            opacity={0.15}
          />
        )}
      </mesh>

      {/* Earth Solid Globe */}
      <mesh ref={earthRef}>
        <sphereGeometry args={[0.8, 64, 64]} />
        {earthTexture ? (
          <meshStandardMaterial
            map={earthTexture}
            roughness={0.6}
            metalness={0.1}
          />
        ) : (
          <meshStandardMaterial
            color="#1e3a8a"
            emissive="#0284c7"
            emissiveIntensity={0.4}
            roughness={0.4}
            metalness={0.3}
          />
        )}
      </mesh>

      {/* Magnetosphere Shield (Pulses from Cyan -> Crimson Orange on Impact) */}
      <mesh ref={shieldRef}>
        <sphereGeometry args={[0.98, 28, 28]} />
        <meshStandardMaterial
          color={shieldColor}
          emissive={shieldEmissive}
          emissiveIntensity={isShieldHitting ? 2.5 : 0.8}
          wireframe
          transparent
          opacity={isShieldHitting ? 0.75 : 0.3}
        />
      </mesh>
    </group>
  );
}

/**
 * High-Performance InstancedMesh Particle System for 5000+ CME Plasma Particles
 */
function PlasmaParticleSystem({ active = false, intensity = 1.0 }) {
  const meshRef = useRef();
  const count = 5000;
  
  // Setup buffers for instanced physics
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const particles = useMemo(() => {
    const temp = [];
    for (let i = 0; i < count; i++) {
      temp.push({
        t: Math.random() * 100, // random start time offset
        factor: 20 + Math.random() * 100, // speed factor
        speed: 0.01 + Math.random() / 200, // drift speed
        xOffset: (Math.random() - 0.5) * 1.5,
        yOffset: (Math.random() - 0.5) * 1.5,
        zOffset: (Math.random() - 0.5) * 1.5,
      });
    }
    return temp;
  }, [count]);

  const colorArray = useMemo(() => {
    const arr = new Float32Array(count * 3);
    const color = new THREE.Color();
    for (let i = 0; i < count; i++) {
      // Base colors mix orange and red
      if (Math.random() > 0.5) {
        color.set('#f43f5e'); // Rose/Red
      } else {
        color.set('#f97316'); // Orange
      }
      arr[i * 3] = color.r;
      arr[i * 3 + 1] = color.g;
      arr[i * 3 + 2] = color.b;
    }
    return arr;
  }, [count]);

  useFrame((state) => {
    if (!meshRef.current || !active) return;
    
    // Animate 5000 particles at 60 FPS directly via InstancedMesh matrix
    particles.forEach((particle, i) => {
      let t = (particle.t += particle.speed);
      
      // Calculate particle position: 
      // Flow from Sun [-3, 0, 0] to Earth [3, 0, 0]
      // Use modulo to loop the particles from sun to earth
      const progress = (t * intensity * 2.5) % 6.0;
      
      const x = -3 + progress + (Math.sin(t * 10) * particle.xOffset * 0.5);
      const y = (Math.cos(t * 15) * particle.yOffset * 0.5);
      const z = (Math.sin(t * 20) * particle.zOffset * 0.5);

      dummy.position.set(x, y, z);
      
      // Scale particles based on intensity and progress
      const s = Math.max(0.01, 0.08 * intensity * (1.0 - Math.abs(x)/4)); 
      dummy.scale.set(s, s, s);
      dummy.updateMatrix();
      
      meshRef.current.setMatrixAt(i, dummy.matrix);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  if (!active) return null;

  return (
    <instancedMesh ref={meshRef} args={[null, null, count]}>
      <dodecahedronGeometry args={[0.2, 0]}>
        <instancedBufferAttribute attach="attributes-color" args={[colorArray, 3]} />
      </dodecahedronGeometry>
      <meshStandardMaterial 
        vertexColors={true}
        emissive="#ef4444"
        emissiveIntensity={intensity * 2.0}
        transparent
        opacity={0.8}
        roughness={0.2}
      />
    </instancedMesh>
  );
}

/**
 * Predictive Trajectory Vector Arc (Direct Hit vs Deflected)
 */
function PredictiveTrajectoryVector({ active = false, trajectory = 'Direct Hit' }) {
  const arcRef = useRef();

  const isHit = trajectory === 'Direct Hit';
  const targetPos = isHit ? [3, 0, 0] : [3, 3.5, -2];

  useFrame((state, delta) => {
    if (arcRef.current) {
      const time = state.clock.getElapsedTime();
      arcRef.current.rotation.x = Math.sin(time * 2) * 0.05;
    }
  });

  if (!active) return null;

  // Compute quadratic bezier curve from Sun [-3,0,0] to target
  const midControlPos = isHit ? [0, 1.2, 0] : [0, 2.5, -1];
  const curve = new THREE.QuadraticBezierCurve3(
    new THREE.Vector3(-3, 0, 0),
    new THREE.Vector3(...midControlPos),
    new THREE.Vector3(...targetPos)
  );

  const points = curve.getPoints(40);
  const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);

  return (
    <group ref={arcRef}>
      {/* Glowing Trajectory Line */}
      <primitive object={new THREE.Line(
        lineGeometry,
        new THREE.LineBasicMaterial({
          color: isHit ? 0x38bdf8 : 0xf59e0b,
          linewidth: 3,
          transparent: true,
          opacity: 0.85
        })
      )} />

      {/* Target Indicator Ring around Earth if Direct Hit */}
      {isHit ? (
        <group position={[3, 0, 0]}>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[1.25, 0.03, 16, 48]} />
            <meshStandardMaterial
              color="#ef4444"
              emissive="#f43f5e"
              emissiveIntensity={2.5}
              transparent
              opacity={0.9}
            />
          </mesh>
          <Html position={[0, 1.4, 0]} center>
            <div className="bg-rose-500/20 text-rose-300 border border-rose-500/50 backdrop-blur-md px-2 py-0.5 rounded text-[10px] font-mono font-bold animate-pulse whitespace-nowrap">
              ⚠️ DIRECT GEOMAGNETIC IMPACT TARGET
            </div>
          </Html>
        </group>
      ) : (
        <group position={targetPos}>
          <Html position={[0, 0.5, 0]} center>
            <div className="bg-amber-500/20 text-amber-300 border border-amber-500/50 backdrop-blur-md px-2 py-0.5 rounded text-[10px] font-mono font-bold whitespace-nowrap">
              ↪️ OFF-TARGET / DEFLECTED TRAJECTORY
            </div>
          </Html>
        </group>
      )}
    </group>
  );
}

/**
 * Camera Controller Component for Smooth Lerping View States
 */
function CameraRig({ viewMode = 'system' }) {
  useFrame((state, delta) => {
    let targetPos = new THREE.Vector3(0, 0, 8.5);
    let lookAtPos = new THREE.Vector3(0, 0, 0);

    if (viewMode === 'sun') {
      targetPos.set(-3, 0, 4.2);
      lookAtPos.set(-3, 0, 0);
    } else if (viewMode === 'earth') {
      targetPos.set(3, 0, 4.2);
      lookAtPos.set(3, 0, 0);
    }

    state.camera.position.lerp(targetPos, delta * 3.0);
    state.camera.lookAt(lookAtPos);
  });

  return null;
}

function CanvasLoader() {
  return (
    <Html center>
      <div className="flex flex-col items-center justify-center gap-2 bg-slate-900/90 backdrop-blur-md p-3.5 rounded-xl border border-slate-800 text-slate-300 text-xs font-mono">
        <div className="w-6 h-6 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
        <span>LOADING 3D SPACE WEATHER SIMULATION...</span>
      </div>
    </Html>
  );
}

/**
 * IMF Vector Arrow (Points Up for Northward Bz, Down for Southward Bz)
 */
function IMFVectorArrow({ mag }) {
  const bz = mag?.b_z ?? 5.0;
  const isSouthward = bz < 0;
  const arrowColor = isSouthward ? '#f97316' : '#22d3ee';
  const arrowRef = useRef();

  useFrame((state) => {
    if (arrowRef.current) {
      const time = state.clock.getElapsedTime();
      arrowRef.current.position.y = Math.sin(time * 2) * 0.1;
    }
  });

  return (
    <group position={[0, 1.5, 0]} ref={arrowRef}>
      <group rotation={[isSouthward ? Math.PI : 0, 0, 0]}>
        <mesh position={[0, 0, 0]}>
          <cylinderGeometry args={[0.02, 0.02, 0.6, 8]} />
          <meshStandardMaterial color={arrowColor} emissive={arrowColor} emissiveIntensity={1.5} />
        </mesh>
        <mesh position={[0, 0.35, 0]}>
          <coneGeometry args={[0.1, 0.2, 8]} />
          <meshStandardMaterial color={arrowColor} emissive={arrowColor} emissiveIntensity={1.5} />
        </mesh>
      </group>
      <Html position={[0, 0.8, 0]} center>
        <div className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold backdrop-blur-md border whitespace-nowrap ${isSouthward ? 'bg-rose-500/20 text-rose-300 border-rose-500/50' : 'bg-cyan-500/20 text-cyan-300 border-cyan-500/50'}`}>
          IMF B_z: {bz.toFixed(1)} nT
        </div>
      </Html>
    </group>
  );
}

/**
 * Main 3D Space Weather Solar & Earth Component
 */
export default function SolarFlare3D({
  hel1osFlux = 18.2,
  solexsFlux = 38.5,
  isNowcastActive = false,
  isForecastActive = false,
  forecastTrajectory = 'Direct Hit',
  isBursting = false,
  mag,
  aspex
}) {
  const [cameraView, setCameraView] = useState('system'); // 'system', 'sun', 'earth'
  const [activeTrajectory, setActiveTrajectory] = useState(forecastTrajectory);

  const effectiveNowcast = isNowcastActive || hel1osFlux > 50 || isBursting;
  const effectiveForecast = isForecastActive || isBursting;

  return (
    <div className="w-full h-full min-h-[440px] relative rounded-2xl overflow-hidden bg-slate-950/95 border border-slate-800/80 shadow-2xl flex flex-col justify-between select-none">
      {/* Sleek Top HUD Overlay Bar */}
      <div className="absolute top-3 left-3 right-3 z-10 flex flex-col sm:flex-row items-center justify-between gap-3 pointer-events-auto">
        {/* Left: Camera Focus Controls */}
        <div className="flex items-center bg-slate-900/90 backdrop-blur-md p-1 rounded-xl border border-slate-800 text-[11px] font-mono shadow-md">
          <button
            onClick={() => setCameraView('system')}
            className={`px-3 py-1 rounded-lg transition-all ${
              cameraView === 'system'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            [Full System View]
          </button>
          <button
            onClick={() => setCameraView('sun')}
            className={`px-3 py-1 rounded-lg transition-all ${
              cameraView === 'sun'
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 font-bold shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            [Focus Sun]
          </button>
          <button
            onClick={() => setCameraView('earth')}
            className={`px-3 py-1 rounded-lg transition-all ${
              cameraView === 'earth'
                ? 'bg-blue-500/20 text-blue-300 border border-blue-500/40 font-bold shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            [Focus Earth]
          </button>
        </div>

        {/* Right: Trajectory Switcher & Status Badges */}
        <div className="flex items-center gap-2">
          <div className="flex items-center bg-slate-900/90 backdrop-blur-md p-1 rounded-xl border border-slate-800 text-[11px] font-mono">
            <button
              onClick={() => setActiveTrajectory('Direct Hit')}
              className={`px-2.5 py-1 rounded-lg transition-all ${
                activeTrajectory === 'Direct Hit'
                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Direct Hit
            </button>
            <button
              onClick={() => setActiveTrajectory('Deflected / Off-Target')}
              className={`px-2.5 py-1 rounded-lg transition-all ${
                activeTrajectory !== 'Direct Hit'
                  ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Off-Target
            </button>
          </div>

          {effectiveNowcast && (
            <span className="bg-rose-500/20 text-rose-300 border border-rose-500/40 px-2.5 py-1 rounded-xl text-[10px] font-mono font-bold animate-pulse">
              ⚡ NOWCAST IMPACT ACTIVE
            </span>
          )}
        </div>
      </div>

      {/* 3D WebGL Canvas */}
      <Canvas camera={{ position: [0, 0, 8.5], fov: 45 }}>
        <ambientLight intensity={0.4} />
        <pointLight position={[-3, 2, 4]} intensity={2.2} color="#fff" />
        <pointLight position={[3, 2, 4]} intensity={0.9} color="#38bdf8" />

        <Suspense fallback={<CanvasLoader />}>
          {/* Deep Space Background Stars */}
          <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade />

          {/* Camera View Rig */}
          <CameraRig viewMode={cameraView} />

          {/* Celestial Body 1: The Sun [-3, 0, 0] */}
          <Float speed={1.0} rotationIntensity={0.1} floatIntensity={0.15}>
            <SunBody solexsFlux={solexsFlux} hel1osFlux={hel1osFlux} isBursting={isBursting} />
          </Float>

          {/* Celestial Body 2: 3D Earth [3, 0, 0] */}
          <Float speed={0.8} rotationIntensity={0.1} floatIntensity={0.1}>
            <EarthBody isNowcastActive={effectiveNowcast} hel1osFlux={hel1osFlux} mag={mag} />
          </Float>

          {/* IMF Vector Indicator */}
          <IMFVectorArrow mag={mag} />

          {/* Nowcasting Radiation Stream (Sun -> Earth Axis) */}
          <PlasmaParticleSystem active={effectiveNowcast} intensity={isBursting ? 1.5 : (hel1osFlux / 100)} />

          {/* Predictive Trajectory Arc */}
          <PredictiveTrajectoryVector active={effectiveForecast} trajectory={activeTrajectory} />
        </Suspense>

        <OrbitControls enableDamping={true} dampingFactor={0.05} maxDistance={14} minDistance={3} />
      </Canvas>

      {/* Bottom Telemetry Legend */}
      <div className="absolute bottom-3 left-3 right-3 z-10 flex items-center justify-between text-[11px] font-mono text-slate-400 bg-slate-900/80 backdrop-blur-md px-3 py-1.5 rounded-xl border border-slate-800/80 pointer-events-none">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-amber-400"></span>
            Sun Position: [-3, 0, 0]
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
            Earth Position: [3, 0, 0]
          </span>
        </div>
        <span className="text-cyan-400 font-semibold">Space Weather Simulation Engine Active</span>
      </div>
    </div>
  );
}
