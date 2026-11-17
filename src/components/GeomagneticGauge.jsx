import React from 'react';
import { ShieldAlert, Activity, Navigation, Clock } from 'lucide-react';

export default function GeomagneticGauge({ mag, aspex, forecast }) {
  // Default values to prevent crashing
  const b_z = mag?.b_z ?? 0.0;
  const is_southward = b_z < 0;
  
  const kp = forecast?.predicted_kp ?? 0.0;
  const dst = forecast?.predicted_dst ?? 0.0;
  const scale = forecast?.storm_scale ?? "Quiet";
  const eta = forecast?.eta_earth_mins ?? 0.0;

  // Determine Kp gauge color
  let gaugeColor = 'bg-emerald-400';
  let textColor = 'text-emerald-400';
  let glowColor = 'shadow-emerald-500/20';

  if (kp >= 8) {
    gaugeColor = 'bg-rose-500';
    textColor = 'text-rose-400';
    glowColor = 'shadow-rose-500/30';
  } else if (kp >= 6) {
    gaugeColor = 'bg-orange-500';
    textColor = 'text-orange-400';
    glowColor = 'shadow-orange-500/20';
  } else if (kp >= 5) {
    gaugeColor = 'bg-amber-400';
    textColor = 'text-amber-400';
    glowColor = 'shadow-amber-400/20';
  }

  // Calculate percentage for Kp bar (0 to 9)
  const kpPercent = Math.min(Math.max((kp / 9) * 100, 0), 100);

  // Format ETA nicely
  const etaMins = Math.floor(eta);
  const etaSecs = Math.floor((eta - etaMins) * 60);

  return (
    <div className={`bg-slate-900/90 backdrop-blur-md border border-slate-800/80 rounded-2xl p-5 shadow-lg ${glowColor} transition-all duration-300`}>
      <div className="flex items-center justify-between mb-4 border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <ShieldAlert className={`w-5 h-5 ${textColor}`} />
          <h2 className="text-sm font-bold text-slate-200 tracking-wide font-mono">
            GEOMAGNETIC STORM ADVISORY
          </h2>
        </div>
        <span className={`text-[10px] font-mono px-2.5 py-0.5 rounded-full border bg-opacity-20 font-bold ${textColor} border-current`}>
          {scale.toUpperCase()}
        </span>
      </div>

      <div className="space-y-5">
        {/* Kp Gauge */}
        <div>
          <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-2">
            <span>Kp Index (Planetary)</span>
            <span className={`font-bold ${textColor}`}>{kp.toFixed(1)} / 9.0</span>
          </div>
          <div className="w-full bg-slate-800 h-3 rounded-full overflow-hidden">
            <div 
              className={`h-full ${gaugeColor} transition-all duration-300 ease-out`}
              style={{ width: `${kpPercent}%` }}
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          {/* Dst Index */}
          <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-800/50">
            <div className="flex items-center gap-1.5 text-xs font-mono text-slate-400 mb-1">
              <Activity className="w-3.5 h-3.5" />
              <span>Dst Index</span>
            </div>
            <div className="text-lg font-bold text-slate-200 font-mono">
              {dst.toFixed(1)} <span className="text-xs font-normal text-slate-500">nT</span>
            </div>
          </div>

          {/* Bz Orientation */}
          <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-800/50">
            <div className="flex items-center gap-1.5 text-xs font-mono text-slate-400 mb-1">
              <Navigation className={`w-3.5 h-3.5 ${is_southward ? 'text-rose-400 animate-pulse transform rotate-180' : 'text-cyan-400'}`} />
              <span>IMF B_z</span>
            </div>
            <div className={`text-lg font-bold font-mono ${is_southward ? 'text-rose-400' : 'text-cyan-400'}`}>
              {b_z.toFixed(2)} <span className="text-xs font-normal opacity-70">nT</span>
            </div>
          </div>
        </div>

        {/* ETA */}
        <div className="flex items-center justify-between bg-slate-950 p-3 rounded-xl border border-slate-800">
          <div className="flex items-center gap-2 text-xs font-mono text-slate-300">
            <Clock className="w-4 h-4 text-indigo-400" />
            <span>Earth Impact ETA:</span>
          </div>
          <div className="text-sm font-bold font-mono text-indigo-300">
            {eta > 0 ? `${etaMins}m ${etaSecs}s` : 'IMMINENT'}
          </div>
        </div>
      </div>
    </div>
  );
}
