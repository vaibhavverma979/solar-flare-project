import React, { useState, useEffect, useCallback } from 'react';
import SolarFlare3D from './components/SolarFlare3D';
import Nowcast3DSolar from './components/Nowcast3DSolar';
import Forecast3DSolar from './components/Forecast3DSolar';
import GeomagneticGauge from './components/GeomagneticGauge';
import AICoPilot from './components/AICoPilot';
import useTelemetryWebSocket from './hooks/useTelemetryWebSocket';
import { 
  Activity, 
  Zap, 
  Flame, 
  ShieldAlert, 
  RefreshCw, 
  Radio, 
  BarChart3, 
  Layers,
  Eye,
  Sparkles,
  Globe,
  Wifi,
  FileText,
  Download,
  Loader2
} from 'lucide-react';

export default function App() {
  const [currentSolexs, setCurrentSolexs] = useState(38.5);
  const [currentHel1os, setCurrentHel1os] = useState(18.2);
  const [isSimulatingBurst, setIsSimulatingBurst] = useState(false);
  const [forecastTrajectory, setForecastTrajectory] = useState('Direct Hit');
  
  // New Geomagnetic State
  const [magData, setMagData] = useState({ b_z: 5.0, b_total: 6.0, is_southward: false });
  const [aspexData, setAspexData] = useState({ velocity: 400.0, density: 5.0, pressure: 2.0 });
  const [geomagneticForecast, setGeomagneticForecast] = useState({ predicted_dst: -10, predicted_kp: 1.5, storm_scale: 'Quiet', eta_earth_mins: 0 });
  
  // 3D Telemetry View Mode: 'space-weather' (Sun-Earth 3D), 'dual' (Side-by-Side), 'nowcast', or 'forecast'
  const [telemetryViewMode, setTelemetryViewMode] = useState('space-weather');

  const [nowcastData, setNowcastData] = useState([]);
  const [forecastData, setForecastData] = useState({
    current_probability: 0.24,
    peak_window_probability: 0.40,
    risk_level: 'MEDIUM',
    horizon_min: 30
  });

  const [isLoading, setIsLoading] = useState(false);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);

  const handleGenerateReport = async () => {
    setIsGeneratingReport(true);
    try {
      const response = await fetch('/api/copilot/generate-report', {
        method: 'POST'
      });
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = 'ISRO_Advisory.pdf';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      } else {
        alert('Failed to generate report (Server returned error). Please check the backend logs.');
      }
    } catch (error) {
      alert('Error downloading report: ' + error.message);
    } finally {
      setIsGeneratingReport(false);
    }
  };

  // WebSocket Integration for 10Hz Real-Time Streaming
  const handleWebSocketMessage = useCallback((data) => {
    // Check if it's the new unified JSON frame
    if (data.timestamp && data.solexs) {
      setCurrentSolexs(data.solexs.flux * 1e6); // Scale up for visual parity with old values
      setCurrentHel1os(data.hel1os.flux * 1e6);
      setIsSimulatingBurst(data.hel1os.flux * 1e6 > 50); // Infer bursting from high flux

      if (data.mag) setMagData(data.mag);
      if (data.aspex) setAspexData(data.aspex);
      if (data.geomagnetic_forecast) setGeomagneticForecast(data.geomagnetic_forecast);
    } 
    // Fallback for old WS format just in case
    else if (data.type === 'telemetry_frame' && data.payload) {
      setCurrentSolexs(data.payload.solexs_flux);
      setCurrentHel1os(data.payload.hel1os_flux);
      setIsSimulatingBurst(data.payload.is_bursting);
    }
  }, []);

  const { isConnected, latency } = useTelemetryWebSocket('ws://localhost:8000/ws/telemetry', handleWebSocketMessage);

  // Fetch telemetry & ML data from API endpoints
  const fetchTelemetry = async () => {
    setIsLoading(true);
    try {
      // 1. Fetch Nowcast Catalogue
      const resNowcast = await fetch('/api/nowcast/catalogue');
      if (resNowcast.ok) {
        const jsonNowcast = await resNowcast.json();
        if (jsonNowcast.status === 'ok') {
          setNowcastData(jsonNowcast.catalogue || []);
          if (!isConnected && jsonNowcast.catalogue && jsonNowcast.catalogue.length > 0) {
            const topPeak = jsonNowcast.catalogue[0].peak_flux_soft || 45.0;
            setCurrentSolexs(topPeak);
          }
        }
      }

      // 2. Fetch Forecast Probability
      const resForecast = await fetch('/api/forecast/predict');
      if (resForecast.ok) {
        const jsonForecast = await resForecast.json();
        if (jsonForecast.status === 'ok') {
          setForecastData(jsonForecast);
        }
      }
    } catch (err) {
      console.warn("API fetch notice:", err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchTelemetry();
    // Use fallback polling if WS is not connected
    const interval = setInterval(() => {
      if (!isConnected) {
        fetchTelemetry();
      }
    }, 15000);
    return () => clearInterval(interval);
  }, [isConnected]);

  // Compute active values based on simulation toggle
  const activeHel1os = currentHel1os;
  const activeSolexs = currentSolexs;
  const activeFlareClass = nowcastData.length > 0 ? (nowcastData[0].class || 'A') : 'A';

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-6 lg:p-8 font-sans">
      {/* Header Bar */}
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 mb-6 border-b border-slate-800/80">
        <div className="flex items-center gap-3">
          <div className="bg-gradient-to-r from-cyan-500 via-blue-600 to-indigo-600 text-slate-950 font-black text-lg px-3.5 py-1.5 rounded-xl shadow-lg shadow-cyan-500/20 font-mono tracking-wider">
            ADITYA-L1
          </div>
          <div>
            <h1 className="text-xl md:text-2xl font-bold tracking-tight bg-gradient-to-r from-slate-100 via-slate-200 to-slate-400 bg-clip-text text-transparent">
              Solar Flare Detection & Prediction Dashboard
            </h1>
            <p className="text-xs text-slate-400 mt-0.5">
              ISRO PRADAN Telemetry · SoLEXS (Soft X-Ray) + HEL1OS (Hard X-Ray) Nowcasting & Forecasting
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {isConnected ? (
            <div className="flex items-center gap-2 bg-emerald-900/40 border border-emerald-500/50 px-3.5 py-1.5 rounded-full text-xs font-mono text-emerald-300">
              <Wifi className="w-3.5 h-3.5 animate-pulse" />
              <span>WS CONNECTED - 10Hz STREAM ({latency}ms)</span>
            </div>
          ) : (
            <div className="flex items-center gap-2 bg-rose-900/40 border border-rose-500/50 px-3.5 py-1.5 rounded-full text-xs font-mono text-rose-300">
              <Wifi className="w-3.5 h-3.5" />
              <span>WS DISCONNECTED - POLLING (15s)</span>
            </div>
          )}
          
          <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 px-3.5 py-1.5 rounded-full text-xs font-mono text-slate-300">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
            <span>L1 Orbit Status: ONLINE</span>
          </div>
          <button 
            onClick={handleGenerateReport}
            disabled={isGeneratingReport}
            className="flex items-center gap-1.5 bg-orange-500/10 hover:bg-orange-500/20 text-orange-400 border border-orange-500/30 px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all disabled:opacity-50"
          >
            {isGeneratingReport ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <FileText className="w-3.5 h-3.5" />}
            Generate Advisory (PDF)
          </button>
          <button 
            onClick={fetchTelemetry}
            disabled={isLoading}
            className="flex items-center gap-1.5 bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </header>

      {/* Top Metric Cards (5 Column Grid to include Geomagnetic) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
        <div className="bg-slate-900/90 backdrop-blur-md border border-slate-800/80 rounded-2xl p-5 hover:border-slate-700 transition-all shadow-md">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>ACTIVE PAYLOADS</span>
            <Radio className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold font-mono mt-2 text-slate-100">SoLEXS + HEL1OS</div>
          <div className="text-xs text-cyan-400 mt-1 flex items-center gap-1">
            <span>Synced Cadence ~1s</span>
          </div>
        </div>

        <div className="bg-slate-900/90 backdrop-blur-md border border-slate-800/80 rounded-2xl p-5 hover:border-slate-700 transition-all shadow-md">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>DETECTED NOWCASTS</span>
            <Flame className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold font-mono mt-2 text-slate-100">
            {nowcastData.length} Events
          </div>
          <div className="text-xs text-amber-400 mt-1">5σ Background Excess</div>
        </div>

        <div className="bg-slate-900/90 backdrop-blur-md border border-slate-800/80 rounded-2xl p-5 hover:border-slate-700 transition-all shadow-md">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>PEAK FLARE CLASS</span>
            <Activity className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold font-mono mt-2 text-slate-100">
            {isSimulatingBurst ? 'CLASS X' : (nowcastData.length > 0 ? (nowcastData[0].class || 'A') : 'A (Quiet)')}
          </div>
          <div className="text-xs text-emerald-400 mt-1">GOES-Analogue Scale</div>
        </div>

        <div className="bg-slate-900/90 backdrop-blur-md border border-slate-800/80 rounded-2xl p-5 hover:border-slate-700 transition-all shadow-md">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>30-MIN FORECAST PROB</span>
            <ShieldAlert className="w-4 h-4 text-rose-400" />
          </div>
          <div className="text-2xl font-bold font-mono mt-2 text-slate-100">
            {isSimulatingBurst ? '88%' : `${Math.round((forecastData.peak_window_probability || 0.4) * 100)}%`}
          </div>
          <div className="text-xs text-rose-400 mt-1 flex items-center justify-between">
            <span>Risk: {isSimulatingBurst ? 'HIGH' : (forecastData.risk_level || 'MEDIUM')}</span>
            <span>Horizon: 30 Min</span>
          </div>
        </div>

        <GeomagneticGauge 
          mag={magData} 
          aspex={aspexData} 
          forecast={geomagneticForecast} 
        />
      </div>

      {/* 3D Telemetry Control & View Selector Bar */}
      <div className="bg-slate-900/90 backdrop-blur-md border border-slate-800/80 rounded-2xl p-4 mb-6 flex flex-col md:flex-row items-center justify-between gap-4 shadow-lg">
        <div className="flex items-center gap-3 w-full md:w-auto overflow-x-auto">
          <Eye className="w-4 h-4 text-cyan-400 hidden sm:block shrink-0" />
          <span className="text-xs font-mono font-bold text-slate-300 uppercase tracking-wider shrink-0">
            3D Scene Mode:
          </span>
          <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs font-mono font-semibold shrink-0">
            <button
              onClick={() => setTelemetryViewMode('space-weather')}
              className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
                telemetryViewMode === 'space-weather'
                  ? 'bg-gradient-to-r from-cyan-500 to-indigo-600 text-slate-950 font-bold shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Globe className="w-3.5 h-3.5" />
              <span>3D Sun-Earth Simulation</span>
            </button>
            <button
              onClick={() => setTelemetryViewMode('dual')}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                telemetryViewMode === 'dual'
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-slate-950 font-bold shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Dual Split View
            </button>
            <button
              onClick={() => setTelemetryViewMode('nowcast')}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                telemetryViewMode === 'nowcast'
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-slate-950 font-bold shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Nowcast 3D
            </button>
            <button
              onClick={() => setTelemetryViewMode('forecast')}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                telemetryViewMode === 'forecast'
                  ? 'bg-gradient-to-r from-amber-500 to-orange-600 text-slate-950 font-bold shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Forecast 3D
            </button>
          </div>
        </div>

        {/* Flare Burst Simulator Toggle */}
        <button
          onClick={() => setIsSimulatingBurst(!isSimulatingBurst)}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-mono font-bold transition-all border shadow-lg ${
            isSimulatingBurst
              ? 'bg-rose-500/20 text-rose-300 border-rose-500/50 shadow-rose-500/20 animate-pulse'
              : 'bg-slate-800/90 text-slate-300 border-slate-700 hover:bg-slate-800 hover:border-slate-600'
          }`}
        >
          <Flame className={`w-4 h-4 ${isSimulatingBurst ? 'text-rose-400' : 'text-amber-400'}`} />
          <span>{isSimulatingBurst ? 'SIMULATING HARD X-RAY BURST' : 'SIMULATE SOLAR FLARE BURST'}</span>
        </button>
      </div>

      {/* 3D Telemetry Visualizers Container */}
      {telemetryViewMode === 'space-weather' && (
        <div className="bg-slate-900/90 backdrop-blur-md border border-slate-800/80 rounded-2xl p-5 mb-6 shadow-xl">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-cyan-400" />
              <h2 className="text-base font-bold text-slate-100 tracking-wide font-mono uppercase">
                3D Sun-Earth Space Weather & Flare Trajectory Simulation
              </h2>
            </div>
            <span className="text-xs font-mono bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 px-3 py-1 rounded-full font-semibold">
              DUAL CELESTIAL IMPACT ENGINE
            </span>
          </div>

          <div className="w-full h-[480px]">
            <SolarFlare3D
              hel1osFlux={activeHel1os}
              solexsFlux={activeSolexs}
              isNowcastActive={isSimulatingBurst || activeHel1os > 50}
              isForecastActive={true}
              forecastTrajectory={forecastTrajectory}
              isBursting={isSimulatingBurst}
              mag={magData}
              aspex={aspexData}
            />
          </div>
        </div>
      )}

      {telemetryViewMode === 'dual' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* NOWCAST 3D CONTAINER */}
          <div className="bg-slate-900/90 backdrop-blur-md border border-slate-800/80 rounded-2xl p-5 shadow-xl flex flex-col justify-between">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-cyan-400" />
                <h2 className="text-sm font-bold text-slate-200 tracking-wide font-mono uppercase">
                  3D Live Nowcast Solar Telemetry
                </h2>
              </div>
              <span className="text-[10px] font-mono bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 px-2.5 py-0.5 rounded-full">
                REAL-TIME FLARE DETECTION
              </span>
            </div>

            <div className="w-full h-[380px] rounded-xl overflow-hidden">
              <Nowcast3DSolar
                solexsFlux={activeSolexs}
                hel1osFlux={activeHel1os}
                isBursting={isSimulatingBurst}
                flareClass={activeFlareClass}
              />
            </div>

            <div className="mt-3 flex items-center justify-between text-xs text-slate-400 font-mono">
              <span>Detects Active Surface Prominences & Flux Spikes</span>
              <span className="text-cyan-400">Interactive 3D Orbit</span>
            </div>
          </div>

          {/* FORECAST 3D CONTAINER */}
          <div className="bg-slate-900/90 backdrop-blur-md border border-slate-800/80 rounded-2xl p-5 shadow-xl flex flex-col justify-between">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-amber-400" />
                <h2 className="text-sm font-bold text-slate-200 tracking-wide font-mono uppercase">
                  3D Live Forecast Solar Telemetry
                </h2>
              </div>
              <span className="text-[10px] font-mono bg-amber-500/10 text-amber-400 border border-amber-500/30 px-2.5 py-0.5 rounded-full">
                30-MIN RISK & PRECURSORS
              </span>
            </div>

            <div className="w-full h-[380px] rounded-xl overflow-hidden">
              <Forecast3DSolar
                forecastData={forecastData}
                isBursting={isSimulatingBurst}
              />
            </div>

            <div className="mt-3 flex items-center justify-between text-xs text-slate-400 font-mono">
              <span>Predicts Neupert Effect Lead Vectors & Solar Wind Cones</span>
              <span className="text-amber-400">PyTorch LSTM + TCN Engine</span>
            </div>
          </div>
        </div>
      )}

      {telemetryViewMode === 'nowcast' && (
        <div className="bg-slate-900/90 backdrop-blur-md border border-slate-800/80 rounded-2xl p-6 mb-6 shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-cyan-400" />
              <h2 className="text-lg font-bold text-slate-100 tracking-wide font-mono">
                3D NOWCASTING LIVE SOLAR TELEMETRY (REAL-TIME FLARE MONITOR)
              </h2>
            </div>
            <span className="text-xs font-mono bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 px-3 py-1 rounded-full font-semibold">
              FULL-SCREEN REAL-TIME SCENE
            </span>
          </div>

          <div className="w-full h-[520px] rounded-xl overflow-hidden">
            <Nowcast3DSolar
              solexsFlux={activeSolexs}
              hel1osFlux={activeHel1os}
              isBursting={isSimulatingBurst}
              flareClass={activeFlareClass}
            />
          </div>
        </div>
      )}

      {telemetryViewMode === 'forecast' && (
        <div className="bg-slate-900/90 backdrop-blur-md border border-slate-800/80 rounded-2xl p-6 mb-6 shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-amber-400" />
              <h2 className="text-lg font-bold text-slate-100 tracking-wide font-mono">
                3D FORECASTING LIVE SOLAR TELEMETRY (30-MIN HORIZON PREDICTIVE SCENE)
              </h2>
            </div>
            <span className="text-xs font-mono bg-amber-500/20 text-amber-300 border border-amber-500/40 px-3 py-1 rounded-full font-semibold">
              FULL-SCREEN PREDICTIVE SCENE
            </span>
          </div>

          <div className="w-full h-[520px] rounded-xl overflow-hidden">
            <Forecast3DSolar
              forecastData={forecastData}
              isBursting={isSimulatingBurst}
            />
          </div>
        </div>
      )}

      {/* Physics Precursor Analysis & Models Section */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-6">
        <div className="lg:col-span-12 bg-slate-900/90 backdrop-blur-md border border-slate-800/80 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center gap-2 mb-4">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            <h2 className="text-base font-bold text-slate-200 tracking-wide font-mono">
              PHYSICS PRECURSOR ANALYSIS & DUAL TELEMETRY FUSION
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
            <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800/80">
              <div className="text-slate-400 mb-1">Neupert Effect Precursor</div>
              <div className="text-slate-200 text-sm font-semibold">
                HXR Non-Thermal Lead Window: ~60s
              </div>
              <p className="text-slate-400 text-[11px] mt-1 leading-relaxed">
                Hard X-ray (HEL1OS) bremsstrahlung emission traces accelerated electrons prior to Soft X-ray (SoLEXS) thermal peak.
              </p>
            </div>

            <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800/80">
              <div className="text-slate-400 mb-1">Spectral Hardness Ratio</div>
              <div className="flex items-center justify-between">
                <span className="text-slate-200 text-sm font-semibold">
                  {(activeHel1os / (activeSolexs + 1e-9)).toFixed(3)}
                </span>
                <span className="text-cyan-400 font-bold">
                  {isSimulatingBurst ? 'HARDNESS SPIKE (+180%)' : 'STABLE BASELINE'}
                </span>
              </div>
              <p className="text-slate-400 text-[11px] mt-1 leading-relaxed">
                Real-time ratio of HEL1OS (8-150 keV) to SoLEXS (1-30 keV) counts.
              </p>
            </div>

            <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800/80">
              <div className="text-slate-400 mb-1">ML Nowcast & Forecast Model</div>
              <div className="text-emerald-400 text-sm font-semibold">
                RandomForest + PyTorch LSTM Fusion
              </div>
              <p className="text-slate-400 text-[11px] mt-1 leading-relaxed">
                Suppresses false positives and predicts flare probability on a 30-minute rolling horizon without temporal leakage.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Master Flare Nowcast Catalogue Table */}
      <div className="bg-slate-900/90 backdrop-blur-md border border-slate-800/80 rounded-2xl p-5 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            <h2 className="text-base font-bold text-slate-200 tracking-wide font-mono">
              MASTER FLARE NOWCAST CATALOGUE
            </h2>
          </div>
          <div className="text-xs text-slate-400 font-mono">
            {nowcastData.length} Coincidence-Matched Events
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="text-slate-400 border-b border-slate-800">
                <th className="pb-3 px-3">EVENT ID</th>
                <th className="pb-3 px-3">CLASS</th>
                <th className="pb-3 px-3">START TIME (UTC)</th>
                <th className="pb-3 px-3">PEAK TIME (UTC)</th>
                <th className="pb-3 px-3">END TIME (UTC)</th>
                <th className="pb-3 px-3">SoLEXS PEAK</th>
                <th className="pb-3 px-3">CONFIDENCE</th>
                <th className="pb-3 px-3">NEUPERT CONSISTENT</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {nowcastData.length > 0 ? (
                nowcastData.map((row, i) => (
                  <tr key={i} className="hover:bg-slate-800/30 transition-colors">
                    <td className="py-3 px-3 font-bold text-slate-200">{row.event_id}</td>
                    <td className="py-3 px-3">
                      <span className="bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 px-2 py-0.5 rounded font-bold">
                        {row.class || 'A'}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-slate-300">{row.start_time}</td>
                    <td className="py-3 px-3 text-slate-300">{row.peak_time}</td>
                    <td className="py-3 px-3 text-slate-300">{row.end_time}</td>
                    <td className="py-3 px-3 text-slate-200">{(row.peak_flux_soft || 0).toFixed(1)} cts/s</td>
                    <td className="py-3 px-3 font-bold text-emerald-400">{Math.round((row.confidence || 0.5) * 100)}%</td>
                    <td className="py-3 px-3">
                      <span className={row.neupert_consistent ? "text-emerald-400 font-bold" : "text-slate-500"}>
                        {row.neupert_consistent ? '✓ YES' : '✗ NO'}
                      </span>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={8} className="py-6 text-center text-slate-500">
                    No 5σ threshold flare events detected in active telemetry window.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
      
      <AICoPilot />
    </div>
  );
}
