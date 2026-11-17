# Aditya-L1 Solar Flare Detection & Prediction System

> **ISRO Hackathon MVP** | Built on Aditya-L1 SoLEXS + HEL1OS data from PRADAN

---

## Project Overview

A full-stack sycxfstem that ingests X-ray light-curve data from two Aditya-L1 payloads and:

- **Nowcasts** flares: detects and classifies events in near-real-time
- **Forecasts** flares: predicts the probability of a flare in the next N minutes
- **Visualises** everything in an interactive dashboard with live alert triggers

## Instruments

| Instrument | Band | Physics |
|---|---|---|
| **SoLEXS** | 1–30 keV (soft X-ray) | Thermal plasma diagnostic; GOES-equivalent flux |
| **HEL1OS** | 8–150 keV (hard X-ray) | Non-thermal bremsstrahlung; Neupert-effect precursor |

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env to set paths if needed

# 3. Inspect a PRADAN file (before running readers)
python scripts/inspect_fits.py data/raw/your_file.fits

# 4. Run the Phase 1 ingestion pipeline
python scripts/run_pipeline.py \
    --solexs data/raw/solexs_*.fits \
    --hel1os data/raw/hel1os_*.fits

# 5. Run Nowcasting & Forecasting end-to-end pipeline on PRADAN data
python scripts/run_nowcast_forecast.py

# 6. Launch Web Dashboard & REST API
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
# Open http://localhost:8000 or http://127.0.0.1:8000 in your browser to view the interactive dashboard

# 7. Run test suite across all modules
python -m pytest tests/ -v --tb=short
```

## Repository Structure

```
aditya-flare-system/
├── data/                    # raw, processed, catalogues, vectordb
├── src/
│   ├── ingestion/           # FITS/CSV readers + alignment (Phase 1 ✓)
│   ├── preprocessing/       # background, cleaning, features (Phase 2)
│   ├── nowcast/             # threshold + ML detector + fusion + multimodal (Phase 3+7)
│   ├── forecast/            # dataset, baseline, LSTM/TCN, PatchTST, SHAP (Phase 4+7)
│   ├── vision/              # SUIT UV image analysis: CNN + visual features (Phase 7)
│   ├── vectordb/            # Historical flare pattern matching via embeddings (Phase 7)
│   ├── api/                 # FastAPI REST + WebSocket + multi-modal endpoints (Phase 5+7)
│   └── db/                  # SQLAlchemy models + repository (Phase 5)
├── frontend/                # React + Plotly.js dashboard (Phase 6)
├── tests/                   # Unit tests (103 tests, synthetic fixtures, no real data needed)
├── scripts/                 # inspect_fits.py, run_pipeline.py
├── config.py                # pydantic-settings configuration
├── .env.example
└── requirements.txt
```

## Build Phases

| Phase | Status | Description |
|---|---|---|
| 1 — Ingestion | ✅ **Complete** | SoLEXS/HEL1OS readers + align.py |
| 2 — Preprocessing | ✅ **Complete** | Background subtraction, cleaning, features |
| 3 — Nowcasting | ✅ **Complete** | Threshold detector + ML FP suppression + fusion |
| 4 — Forecasting | ✅ **Complete** | Baseline + LSTM/TCN + SHAP |
| 5 — API | ✅ **Complete** | FastAPI REST + WebSocket |
| 6 — Dashboard | ✅ **Complete** | React + Plotly.js |
| 7 — Multi-Modal AI | ✅ **Complete** | SUIT CV + VectorDB + PatchTST + Fusion |

## Multi-Modal AI Fusion (Phase 7)

The system now operates as a multi-modal intelligence platform combining four signal sources:

| Module | Signal Source | Technology | Fallback |
|---|---|---|---|
| **SUIT Vision** | Solar UV images (11 channels, 200–400 nm) | ResNet-18 CNN | Heuristic (gradient + variance) |
| **VectorDB** | Historical flare pattern matching (50 events, 1859–2025) | ChromaDB + 128-d embeddings | In-memory brute-force cosine search |
| **PatchTST** | Long-horizon time-series (6-hour windows) | Patch Transformer (Nie et al., 2023) | BiLSTM fallback |
| **Neupert Effect** | SoLEXS/HEL1OS cross-correlation | Physics heuristic | — |

### Multi-Modal API Endpoints

```
GET /api/multimodal/status            — Module availability check
GET /api/multimodal/fusion            — Fused 4-modality prediction with per-channel breakdown
GET /api/multimodal/historical-match  — Top-5 historical flare pattern matches
GET /api/multimodal/suit-analysis     — SUIT image analysis (magnetic classification + visual features)
```

### Fusion Strategy

Weighted ensemble with confidence gating:
```
effective_weight_i = base_weight_i × confidence_i
fused_probability = Σ(effective_weight_i × probability_i) / Σ(effective_weight_i)
```
Default weights: `[time_series=0.45, vision=0.20, historical=0.20, neupert=0.15]`

## Standard Data Schema

All internal DataFrames share this schema:

| Column | Type | Description |
|---|---|---|
| `timestamp` | `datetime64[ns, UTC]` | Event time, always UTC |
| `flux` | `float64` | counts/s |
| `flux_err` | `float64` | 1-σ uncertainty |
| `energy_low` | `float32` | Lower energy bound (keV) |
| `energy_high` | `float32` | Upper energy bound (keV) |
| `energy_channel` | `str` | e.g. "1-30keV" |
| `instrument` | `str` | "SoLEXS" or "HEL1OS" |
| `quality_flag` | `int8` | 0=good, 1=suspect, 2=bad |

## Configuration

All parameters live in `.env` (see `.env.example`).  No hardcoded paths exist anywhere in the codebase.

Key settings:
- `RESAMPLE_CADENCE_S` — common time cadence (0 = auto)
- `SIGMA_THRESHOLD` — nowcast detection threshold (σ)
- `FORECAST_HORIZON_MIN` — forecast lead time window
- `DATABASE_URL` — swap SQLite → TimescaleDB here
- `SUIT_IMAGE_DIR` — path to SUIT FITS images
- `VECTORDB_PATH` — ChromaDB persistence directory
- `PATCHTST_CONTEXT_LEN` — PatchTST attention window (default 360 = 6 hours)
- `EMBEDDING_DIM` — vector embedding dimensionality (default 128)

## Testing

Tests use synthetic FRED-profile flare fixtures (`tests/fixtures/synthetic_lc.py`)
and require no real PRADAN files:

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run only multi-modal AI tests
python -m pytest tests/vision/ tests/vectordb/ tests/forecast/test_patchtst.py -v
```

## Physics Notes

- **Neupert Effect**: HXR emission from non-thermal electrons (HEL1OS) *precedes* thermal SXR emission (SoLEXS).  This is the key cross-instrument precursor signal implemented in `src/nowcast/fusion.py::neupert_consistent()`.
- **Spectral Hardness**: `flux_HEL1OS / flux_SoLEXS` rises before impulsive flares and is a key ML feature.
- **Time-based splits**: All ML training uses time-ordered splits.  No shuffling.  No temporal leakage.
- **McIntosh/Mt. Wilson Classification**: Sunspot magnetic complexity (α/β/βγ/βγδ) correlates strongly with flare productivity.  The SUIT CNN classifies active regions into these categories for fusion with X-ray forecasts.
- **PatchTST Attention**: Patching reduces token count from T to T/P, enabling multi-hour attention over solar cycles rather than 60-minute windows.

