"""
DeepHealthX — Flask Demo App
Run:  python webapp/demo_app.py   (from project root)
Open: http://localhost:5000
"""

import io, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
from PIL import Image

import tensorflow as tf
from tensorflow import keras

tf.get_logger().setLevel('ERROR')

app  = Flask(__name__)
ROOT = Path(__file__).parent.parent
MDIR = ROOT / 'models'

# ── Load once at startup ──────────────────────────────────────────────────────
print("Loading models...", flush=True)

_clinical = keras.models.load_model(str(MDIR / 'clinical_model.h5'))
_ecg      = keras.models.load_model(str(MDIR / 'ecg_model.h5'))
_mri      = keras.models.load_model(str(MDIR / 'mri_model.h5'))
_fusion   = keras.models.load_model(str(MDIR / 'fusion_model.h5'))

_c_emb = keras.Model(_clinical.input, _clinical.get_layer('clinical_embedding').output)
_e_emb = keras.Model(_ecg.input,      _ecg.get_layer('ecg_embedding').output)
_m_emb = keras.Model(_mri.input,      _mri.get_layer('mri_embedding').output)

_scaler = None
sp = MDIR / 'clinical_scaler.pkl'
if sp.exists():
    with open(str(sp), 'rb') as f:
        _scaler = pickle.load(f)

print("All models ready. Visit http://localhost:5000", flush=True)

FEAT = ['age','sex','cp','trestbps','chol','fbs',
        'restecg','thalach','exang','oldpeak','slope','ca','thal']

RANGES = {
    'age':(29,77),'sex':(0,1),'cp':(0,3),'trestbps':(90,200),
    'chol':(126,564),'fbs':(0,1),'restecg':(0,2),'thalach':(60,202),
    'exang':(0,1),'oldpeak':(0.0,6.2),'slope':(0,2),'ca':(0,4),'thal':(0,3)
}

def norm_clinical(d):
    arr = np.array([d.get(k,0) for k in FEAT], dtype=np.float32).reshape(1,-1)
    if _scaler:
        return _scaler.transform(arr).astype(np.float32)
    n = arr.copy()
    for i, k in enumerate(FEAT):
        lo, hi = RANGES[k]
        n[0,i] = (arr[0,i] - lo) / max(hi - lo, 1e-8)
    return n

def norm_ecg(raw):
    df  = pd.read_csv(io.StringIO(raw.decode()), header=None)
    sig = (df.iloc[0,:187].values if df.shape[1] >= 187
           else df.iloc[:187,0].values).astype(np.float32)
    if len(sig) < 187:
        sig = np.pad(sig, (0, 187-len(sig)))
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    return sig.reshape(1, 187, 1)

def norm_mri(raw):
    img = Image.open(io.BytesIO(raw)).convert('RGB').resize((224,224))
    return (np.array(img, dtype=np.float32) / 255.0).reshape(1, 224, 224, 3)

# ─────────────────────────────────────────────────────────────────────────────
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeepHealthX — Cardiac Risk Demo</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root {
  --bg-main: #0B1120;
  --bg-card: #141E33;
  --bg-header: #0F172A;
  --text-main: #F8FAFC;
  --text-muted: #94A3B8;
  --border: #1E293B;
  --accent: #3B82F6;
  --accent-hover: #2563EB;
  --red: #EF4444;
  --red-dim: #7F1D1D;
  --green: #10B981;
  --green-dim: #064E3B;
  --amber: #F59E0B;
  --amber-dim: #78350F;
  --pink: #EC4899;
  --purple: #A78BFA;
}
* { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
body { background: var(--bg-main); color: var(--text-main); min-height: 100vh; overflow-x: hidden; }

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }

/* Header */
header {
  background: var(--bg-header);
  border-bottom: 1px solid var(--border);
  padding: 20px 40px;
  display: flex; align-items: center; gap: 20px;
  box-shadow: 0 4px 30px rgba(0,0,0,0.5);
}
.logo-icon {
  width: 48px; height: 48px; background: linear-gradient(135deg, var(--red), #B91C1C);
  border-radius: 12px; display: flex; align-items: center; justify-content: center;
  font-size: 1.6rem; box-shadow: 0 0 20px rgba(239,68,68,0.4);
  animation: pulse-glow 3s infinite;
}
@keyframes pulse-glow { 0% { box-shadow: 0 0 15px rgba(239,68,68,0.4); } 50% { box-shadow: 0 0 25px rgba(239,68,68,0.6); } 100% { box-shadow: 0 0 15px rgba(239,68,68,0.4); } }
.logo-text h1 { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.03em; }
.logo-text p { font-size: 0.85rem; color: var(--text-muted); font-weight: 500; }
.header-badge {
  margin-left: auto; background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.3);
  color: #93C5FD; padding: 6px 16px; border-radius: 20px; font-size: 0.75rem;
  font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
}

.wrap { max-width: 1280px; margin: 30px auto; padding: 0 24px; }

/* Cards */
.card {
  background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px;
  padding: 24px; margin-bottom: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
  transition: transform 0.2s, box-shadow 0.2s;
}
.card:hover { border-color: #334155; box-shadow: 0 10px 40px rgba(0,0,0,0.3); }
.card-title {
  font-size: 0.8rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--text-muted); margin-bottom: 20px; padding-bottom: 12px;
  border-bottom: 1px solid rgba(255,255,255,0.05); display: flex; align-items: center; gap: 8px;
}
.card-title::before { content: ''; width: 4px; height: 16px; background: var(--accent); border-radius: 2px; }

/* Preset bar */
.preset-bar { display: flex; gap: 12px; flex-wrap: wrap; }
.btn-preset {
  padding: 10px 24px; border-radius: 24px; font-size: 0.85rem; font-weight: 600;
  cursor: pointer; border: 1px solid; transition: all 0.2s ease; display: flex; align-items: center; gap: 8px;
}
.btn-d { background: rgba(239,68,68,0.1); color: #FCA5A5; border-color: rgba(239,68,68,0.3); }
.btn-d:hover { background: var(--red); color: #fff; border-color: var(--red); transform: translateY(-2px); box-shadow: 0 4px 15px rgba(239,68,68,0.3); }
.btn-h { background: rgba(16,185,129,0.1); color: #6EE7B7; border-color: rgba(16,185,129,0.3); }
.btn-h:hover { background: var(--green); color: #fff; border-color: var(--green); transform: translateY(-2px); box-shadow: 0 4px 15px rgba(16,185,129,0.3); }
.btn-c { background: rgba(148,163,184,0.1); color: #CBD5E1; border-color: rgba(148,163,184,0.3); }
.btn-c:hover { background: #475569; color: #fff; border-color: #64748B; transform: translateY(-2px); }

/* Layout Grid */
.cols2 { display: grid; grid-template-columns: 340px 1fr; gap: 24px; }

/* Upload zones */
.upload-zone {
  border: 2px dashed #334155; border-radius: 12px; padding: 24px; text-align: center;
  cursor: pointer; position: relative; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  background: rgba(15,23,42,0.5); margin-bottom: 16px; overflow: hidden;
}
.upload-zone::after {
  content: ''; position: absolute; top: 0; left: -100%; width: 50%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
  transform: skewX(-20deg); transition: 0.5s;
}
.upload-zone:hover::after { left: 150%; }
.upload-zone:hover, .upload-zone.has-file { border-color: var(--accent); background: rgba(59,130,246,0.05); }
.upload-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; z-index: 2; }
.uz-icon { font-size: 2.5rem; margin-bottom: 12px; filter: drop-shadow(0 4px 6px rgba(0,0,0,0.3)); }
.uz-title { font-size: 0.95rem; font-weight: 600; color: #F1F5F9; }
.uz-sub { font-size: 0.75rem; color: var(--text-muted); margin-top: 6px; }
.file-badge {
  display: inline-flex; align-items: center; gap: 6px; margin-top: 12px;
  background: rgba(59,130,246,0.2); color: #93C5FD; padding: 6px 14px;
  border-radius: 12px; font-size: 0.75rem; font-weight: 600; border: 1px solid rgba(59,130,246,0.3);
}
.preview-img {
  width: 100%; max-height: 180px; object-fit: contain; border-radius: 12px;
  margin-top: 12px; border: 1px solid var(--border); display: none; background: #000;
  box-shadow: 0 4px 15px rgba(0,0,0,0.4);
}

/* Clinical grid */
.g-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 16px; margin-bottom: 20px; }
.field { position: relative; }
.field label {
  display: block; font-size: 0.7rem; font-weight: 600; color: var(--text-muted);
  letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 8px;
}
.field input, .field select {
  width: 100%; height: 44px; background: rgba(15,23,42,0.8); border: 1px solid var(--border);
  border-radius: 8px; padding: 0 14px; font-size: 0.9rem; color: #F8FAFC; font-weight: 500;
  outline: none; transition: all 0.2s; box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
}
.field input:focus, .field select:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(59,130,246,0.2), inset 0 2px 4px rgba(0,0,0,0.2); }
.field select option { background: var(--bg-card); color: #fff; }
.field input[type="number"]::-webkit-inner-spin-button { opacity: 0.5; }

/* Predict button */
.btn-predict {
  width: 100%; padding: 18px; margin-top: 10px;
  background: linear-gradient(135deg, var(--accent), #4F46E5);
  color: #fff; border: none; border-radius: 12px; font-size: 1.1rem; font-weight: 700;
  cursor: pointer; letter-spacing: 0.5px; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 6px 20px rgba(59,130,246,0.4); display: flex; align-items: center; justify-content: center; gap: 12px;
}
.btn-predict:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(59,130,246,0.5); }
.btn-predict:active { transform: translateY(1px); box-shadow: 0 2px 10px rgba(59,130,246,0.4); }
.btn-predict:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }

/* Result Area */
#result { display: none; animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards; opacity: 0; transform: translateY(20px); }
@keyframes slideUp { to { opacity: 1; transform: translateY(0); } }

.risk-banner {
  border-radius: 16px; padding: 32px 40px; display: grid; grid-template-columns: auto 1fr; gap: 32px;
  align-items: center; margin-bottom: 24px; position: relative; overflow: hidden;
  box-shadow: 0 10px 40px rgba(0,0,0,0.3); border: 1px solid transparent; backdrop-filter: blur(10px);
}
.risk-banner::before {
  content: ''; position: absolute; inset: 0; background: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0IiBoZWlnaHQ9IjQiPgo8cmVjdCB3aWR0aD0iNCIgaGVpZ2h0PSI0IiBmaWxsPSIjZmZmIiBmaWxsLW9wYWNpdHk9IjAuMDUiLz4KPC9zdmc+') repeat;
  opacity: 0.3; pointer-events: none;
}

/* Gauge */
.gauge-wrap { position: relative; width: 140px; height: 140px; filter: drop-shadow(0 4px 10px rgba(0,0,0,0.4)); }
.gauge-wrap svg { transform: rotate(-90deg); width: 100%; height: 100%; }
.gauge-bg { fill: none; stroke: rgba(255,255,255,0.1); stroke-width: 8; }
.gauge-fill {
  fill: none; stroke-width: 8; stroke-linecap: round;
  transition: stroke-dashoffset 1.5s cubic-bezier(0.65, 0, 0.05, 1);
}
.gauge-text {
  position: absolute; inset: 0; display: flex; flex-direction: column;
  align-items: center; justify-content: center;
}
.gauge-pct { font-size: 2.2rem; font-weight: 800; color: #fff; line-height: 1; text-shadow: 0 2px 10px rgba(0,0,0,0.5); }
.gauge-sub { font-size: 0.7rem; color: rgba(255,255,255,0.8); letter-spacing: 0.1em; text-transform: uppercase; margin-top: 4px; font-weight: 600; }

.risk-info { z-index: 1; }
.risk-info-title { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; color: rgba(255,255,255,0.7); font-weight: 600; margin-bottom: 4px; }
.risk-info h2 { font-size: 2.2rem; font-weight: 800; color: #fff; letter-spacing: -0.02em; text-shadow: 0 2px 10px rgba(0,0,0,0.3); }
.risk-info p { color: rgba(255,255,255,0.9); font-size: 1rem; margin-top: 10px; line-height: 1.6; font-weight: 500; }

/* Score bars */
.result-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.score-item { margin-bottom: 18px; }
.score-header { display: flex; justify-content: space-between; align-items: center; font-size: 0.9rem; margin-bottom: 8px; }
.score-name { font-weight: 600; color: #E2E8F0; display: flex; align-items: center; gap: 8px; }
.score-name .badge { font-size: 0.65rem; padding: 2px 6px; border-radius: 4px; background: rgba(255,255,255,0.1); }
.score-val { font-weight: 800; letter-spacing: 0.5px; }
.score-track { height: 10px; background: rgba(0,0,0,0.3); border-radius: 5px; overflow: hidden; border: 1px solid rgba(255,255,255,0.05); }
.score-fill { height: 100%; border-radius: 5px; transition: width 1.5s cubic-bezier(0.65, 0, 0.05, 1); position: relative; }
.score-fill::after {
  content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.3) 50%, rgba(255,255,255,0) 100%);
  animation: shimmer 2s infinite;
}
@keyframes shimmer { 100% { transform: translateX(100%); } }

/* Interpretation */
.interp {
  padding: 24px; border-radius: 12px; font-size: 1rem; line-height: 1.6; border-left: 6px solid;
  margin-top: 10px; font-weight: 500; display: flex; gap: 16px;
}
.interp-icon { font-size: 2rem; line-height: 1; }
.i-low { background: linear-gradient(to right, rgba(16,185,129,0.15), rgba(16,185,129,0.05)); border-color: var(--green); color: #A7F3D0; }
.i-med { background: linear-gradient(to right, rgba(245,158,11,0.15), rgba(245,158,11,0.05)); border-color: var(--amber); color: #FDE68A; }
.i-high{ background: linear-gradient(to right, rgba(239,68,68,0.15), rgba(239,68,68,0.05)); border-color: var(--red); color: #FECACA; }

/* Findings summary */
.findings-grid { display: grid; grid-template-columns: 1fr; gap: 12px; margin-top: 20px; }
.finding-card {
  background: rgba(0,0,0,0.2); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 16px; display: flex; align-items: flex-start; gap: 12px;
}
.finding-dot { width: 12px; height: 12px; border-radius: 50%; margin-top: 4px; flex-shrink: 0; }
.f-clinical { background: var(--accent); box-shadow: 0 0 10px var(--accent); }
.f-ecg { background: var(--pink); box-shadow: 0 0 10px var(--pink); }
.f-mri { background: var(--green); box-shadow: 0 0 10px var(--green); }
.finding-text h4 { font-size: 0.8rem; color: var(--text-muted); margin-bottom: 2px; text-transform: uppercase; letter-spacing: 0.05em; }
.finding-text p { font-size: 0.9rem; font-weight: 500; }

.formula-note {
  margin-top: 24px; padding: 16px; background: rgba(59,130,246,0.1); border-radius: 8px;
  border: 1px dashed rgba(59,130,246,0.3); text-align: center; color: #93C5FD;
  font-family: monospace; font-size: 0.95rem; font-weight: 600;
}

/* Spinner Overlay */
.spinner-overlay {
  position: fixed; inset: 0; background: rgba(11,17,32,0.8); backdrop-filter: blur(5px);
  display: none; align-items: center; justify-content: center; z-index: 100; flex-direction: column;
}
.loader {
  width: 60px; height: 60px; border: 4px solid rgba(255,255,255,0.1);
  border-left-color: var(--accent); border-radius: 50%; animation: spin 1s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.spinner-text { margin-top: 20px; font-size: 1.1rem; font-weight: 600; color: #fff; letter-spacing: 0.5px; }

@media(max-width: 1024px) {
  .cols2 { grid-template-columns: 1fr; }
  .result-cols { grid-template-columns: 1fr; }
  .g-row { grid-template-columns: repeat(2, 1fr); }
}
@media(max-width: 600px) {
  .g-row { grid-template-columns: 1fr; }
  .risk-banner { grid-template-columns: 1fr; text-align: center; justify-items: center; padding: 24px; gap: 20px; }
}
</style>
</head>
<body>

<header>
  <div class="logo-icon">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.42 4.58a5.4 5.4 0 0 0-7.65 0l-.77.78-.77-.78a5.4 5.4 0 0 0-7.65 0C1.46 6.7 1.33 10.28 4 13l8 8 8-8c2.67-2.72 2.54-6.3.42-8.42z"></path></svg>
  </div>
  <div class="logo-text">
    <h1>DeepHealthX</h1>
    <p>Multi-Modal Cardiac Risk Assessment System</p>
  </div>
  <div class="header-badge">Two-Stage Fusion Workflow ·</div>
</header>

<div class="wrap">
  <!-- Preset bar -->
  <div class="card">
    <div class="card-title">Demo Control Panel</div>
    <div class="preset-bar">
      <button class="btn-preset btn-d" onclick="fill('disease')">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>
        Load High-Risk Patient Profile
      </button>
      <button class="btn-preset btn-h" onclick="fill('healthy')">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
        Load Low-Risk Patient Profile
      </button>
      <button class="btn-preset btn-c" onclick="fill('clear')">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M2.13 15.57a10 10 0 1 0 3.43-5.57L8 14"></path></svg>
        Reset to Defaults
      </button>
    </div>
  </div>

  <div class="cols2">
    <!-- LEFT: uploads -->
    <div>
      <div class="card">
        <div class="card-title">Modality 2: ECG Signal</div>
        <div class="upload-zone" id="ecg-zone">
          <input type="file" accept=".csv" id="ecg-file" onchange="onEcg(this)">
          <div class="uz-icon">📈</div>
          <div class="uz-title">Upload ECG Recording (.csv)</div>
          <div class="uz-sub">MIT-BIH Protocol · 187 timestep array</div>
          <div id="ecg-badge"></div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">Modality 3: Cardiac MRI</div>
        <div class="upload-zone" id="mri-zone">
          <input type="file" accept=".jpg,.jpeg,.png" id="mri-file" onchange="onMri(this)">
          <div class="uz-icon">🫀</div>
          <div class="uz-title">Upload MRI Scan (.jpg/.png)</div>
          <div class="uz-sub">Axial cardiac slice · Automatic resizing to 224x224</div>
          <div id="mri-badge"></div>
        </div>
        <img id="mri-prev" class="preview-img">
      </div>
    </div>

    <!-- RIGHT: clinical + predict -->
    <div class="card">
      <div class="card-title">Modality 1: Clinical Features</div>

      <div class="g-row">
        <div class="field"><label>Age (years)</label>
          <input type="number" id="age" min="29" max="77" value="50"></div>
        <div class="field"><label>Biological Sex</label>
          <select id="sex">
            <option value="0">Female (0)</option>
            <option value="1" selected>Male (1)</option>
          </select></div>
        <div class="field"><label>Chest Pain (0-3)</label>
          <input type="number" id="cp" min="0" max="3" value="0"
                 title="0=Typical angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic. Note: 0 is highest risk in this dataset."></div>
        <div class="field"><label>Resting BP (mmHg)</label>
          <input type="number" id="trestbps" min="90" max="200" value="120"></div>
      </div>

      <div class="g-row">
        <div class="field"><label>Cholesterol (mg/dl)</label>
          <input type="number" id="chol" min="126" max="564" value="200"></div>
        <div class="field"><label>Fasting Blood Sugar > 120</label>
          <select id="fbs">
            <option value="0" selected>False (0)</option>
            <option value="1">True (1)</option>
          </select></div>
        <div class="field"><label>Resting ECG (0-2)</label>
          <input type="number" id="restecg" min="0" max="2" value="0"></div>
        <div class="field"><label>Max Heart Rate</label>
          <input type="number" id="thalach" min="60" max="202" value="150"></div>
      </div>

      <div class="g-row">
        <div class="field"><label>Exercise Angina</label>
          <select id="exang">
            <option value="0" selected>No (0)</option>
            <option value="1">Yes (1)</option>
          </select></div>
        <div class="field"><label>ST Depression (Oldpeak)</label>
          <input type="number" id="oldpeak" min="0" max="6.2" step="0.1" value="0.0"></div>
        <div class="field"><label>ST Slope (0-2)</label>
          <input type="number" id="slope" min="0" max="2" value="1"></div>
        <div class="field"><label>Fluoroscopy Vessels (0-4)</label>
          <input type="number" id="ca" min="0" max="4" value="0"></div>
        <div class="field"><label>Thalassemia (0-3)</label>
          <input type="number" id="thal" min="0" max="3" value="1"></div>
      </div>

      <button class="btn-predict" id="go-btn" onclick="predict()">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
        Execute Multi-Modal Inference
      </button>
    </div>
  </div>

  <!-- Result Area -->
  <div id="result">
    <div class="risk-banner" id="banner">
      <div class="gauge-wrap">
        <svg viewBox="0 0 120 120">
          <circle class="gauge-bg" cx="60" cy="60" r="50"/>
          <circle class="gauge-fill" cx="60" cy="60" r="50" id="gauge-arc" stroke-dasharray="314" stroke-dashoffset="314" stroke="#fff"/>
        </svg>
        <div class="gauge-text">
          <div class="gauge-pct" id="g-pct">0%</div>
          <div class="gauge-sub">Risk Score</div>
        </div>
      </div>
      <div class="risk-info">
        <div class="risk-info-title">System Diagnosis</div>
        <h2 id="r-label">—</h2>
        <p id="r-conf">—</p>
      </div>
    </div>

    <div class="result-cols">
      <div class="card">
        <div class="card-title">Modal Branch Analysis</div>
        <div id="bars"></div>
        <div class="formula-note">
          Combined Risk = (Clinical × 0.30) + (ECG × 0.35) + (MRI × 0.35)
          <div style="font-size: 0.8rem; font-weight: normal; margin-top: 8px;">
            Clinical weight reduced due to small training dataset (303 patients).<br>
            ECG and MRI weighted equally as primary diagnostic signals.
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-title">Clinical Interpretation</div>
        <div id="interp"></div>
        
        <div class="findings-grid">
          <div class="finding-card">
            <div class="finding-dot f-clinical"></div>
            <div class="finding-text">
              <h4>Clinical Profile</h4>
              <p id="f-clin-text">—</p>
            </div>
          </div>
          <div class="finding-card">
            <div class="finding-dot f-ecg"></div>
            <div class="finding-text">
              <h4>Electrocardiogram</h4>
              <p id="f-ecg-text">—</p>
            </div>
          </div>
          <div class="finding-card">
            <div class="finding-dot f-mri"></div>
            <div class="finding-text">
              <h4>Cardiac MRI</h4>
              <p id="f-mri-text">—</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Fullscreen Spinner -->
<div class="spinner-overlay" id="spin">
  <div class="loader"></div>
  <div class="spinner-text">Running multi-stage neural network inference...</div>
</div>

<script>
// Exact presets requested
const PRESETS = {
  disease: { age:67, sex:1, cp:0, trestbps:120, chol:229, fbs:0, restecg:0, thalach:129, exang:1, oldpeak:2.6, slope:1, ca:2, thal:3 },
  healthy: { age:56, sex:1, cp:1, trestbps:120, chol:236, fbs:0, restecg:1, thalach:178, exang:0, oldpeak:0.8, slope:2, ca:0, thal:2 },
  clear:   { age:50, sex:1, cp:0, trestbps:120, chol:200, fbs:0, restecg:0, thalach:150, exang:0, oldpeak:0.0, slope:1, ca:0, thal:1 }
};

function fill(name) {
  for(const[k,v] of Object.entries(PRESETS[name])) {
    const el = document.getElementById(k);
    if(el) {
      el.value = v;
      // Add a small flash effect to show data was populated
      el.style.transition = 'background 0.3s';
      el.style.background = 'rgba(59,130,246,0.3)';
      setTimeout(() => el.style.background = '', 300);
    }
  }
}

function onEcg(inp) {
  const b = document.getElementById('ecg-badge');
  if(inp.files[0]) {
    document.getElementById('ecg-zone').classList.add('has-file');
    b.innerHTML = `<span class="file-badge"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"></polyline></svg> ${inp.files[0].name}</span>`;
  }
}

function onMri(inp) {
  const b = document.getElementById('mri-badge');
  const p = document.getElementById('mri-prev');
  if(inp.files[0]) {
    document.getElementById('mri-zone').classList.add('has-file');
    b.innerHTML = `<span class="file-badge"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"></polyline></svg> ${inp.files[0].name}</span>`;
    const r = new FileReader();
    r.onload = e => { p.src = e.target.result; p.style.display = 'block'; };
    r.readAsDataURL(inp.files[0]);
  }
}

function riskColor(s) {
  if(s < 25) return 'var(--green)';
  if(s <= 50) return 'var(--amber)';
  return 'var(--red)';
}

function riskConfig(s) {
  if(s < 25) return { 
    label: 'Low Risk', 
    cls: 'i-low', 
    icon: '🟢',
    msg: 'Patient shows healthy cardiac function across multi-modal indicators. No immediate intervention required. Routine annual monitoring recommended.' 
  };
  if(s <= 50) return { 
    label: 'Medium Risk', 
    cls: 'i-med', 
    icon: '🟡',
    msg: 'Some concerning indicators detected requiring clinical correlation. Follow-up within 4-6 weeks recommended. Consider stress test and lipid panel.' 
  };
  return { 
    label: 'High Risk', 
    cls: 'i-high', 
    icon: '🔴',
    msg: 'Significant cardiac risk indicators detected across modalities! Immediate cardiology consultation and intervention strongly recommended.' 
  };
}

function getFindingText(score, type) {
  if(score < 25) return `Normal ${type} biomarkers detected.`;
  if(score <= 50) return `Borderline abnormal ${type} patterns observed.`;
  return `Critical abnormalities in ${type} analysis.`;
}

async function predict() {
  const ecgF = document.getElementById('ecg-file').files[0];
  const mriF = document.getElementById('mri-file').files[0];
  
  if(!ecgF) { alert('Validation Error: Please upload an ECG CSV file.'); return; }
  if(!mriF) { alert('Validation Error: Please upload an MRI image file.'); return; }

  const btn = document.getElementById('go-btn');
  const spin = document.getElementById('spin');
  const result = document.getElementById('result');
  
  btn.disabled = true;
  spin.style.display = 'flex';
  result.style.display = 'none';

  const fd = new FormData();
  ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
    .forEach(k => fd.append(k, document.getElementById(k).value));
  fd.append('ecg_file', ecgF);
  fd.append('mri_file', mriF);

  try {
    const res = await fetch('/predict', { method: 'POST', body: fd });
    const d = await res.json();
    
    if(d.error) { throw new Error(d.error); }

    // Use Combine Score as the MAIN metric
    const cs = d.combined;
    const rc = riskColor(cs);
    const conf = riskConfig(cs);

    // Update Banner
    const banner = document.getElementById('banner');
    banner.style.background = `linear-gradient(135deg, ${rc}33, rgba(15,23,42,0.9))`;
    banner.style.borderColor = `${rc}55`;
    banner.style.boxShadow = `0 10px 40px ${rc}22`;

    // Animate Gauge
    document.getElementById('g-pct').textContent = `${cs.toFixed(1)}%`;
    document.getElementById('g-pct').style.color = rc;
    
    const arc = document.getElementById('gauge-arc');
    arc.style.stroke = rc;
    const offset = 314 - (cs / 100) * 314; // SVG circle length is ~314 (2 * pi * 50)
    
    // reset gauge first, then animate
    arc.style.transition = 'none';
    arc.style.strokeDashoffset = 314;
    setTimeout(() => {
      arc.style.transition = 'stroke-dashoffset 1.5s cubic-bezier(0.65, 0, 0.05, 1)';
      arc.style.strokeDashoffset = offset;
    }, 50);

    document.getElementById('r-label').textContent = `${conf.icon} ${conf.label}`;
    document.getElementById('r-label').style.color = rc;
    document.getElementById('r-conf').innerHTML = `<span style="opacity:0.7">Network Consensus Confidence:</span> <b>${Math.max(cs, 100-cs).toFixed(1)}%</b>`;

    // Render Bars
    document.getElementById('bars').innerHTML = [
      ['Clinical Branch', d.clinical, 'var(--accent)', '0.30w'],
      ['ECG 1D-CNN Branch', d.ecg, 'var(--pink)', '0.35w'],
      ['MRI 2D-CNN Branch', d.mri, 'var(--green)', '0.35w'],
      ['Combined Risk Score', cs, rc, 'MAIN'],
      ['Legacy Fusion Head', d.fusion, 'var(--purple)', 'REF'],
    ].map(([n, v, c, b]) => `
      <div class="score-item">
        <div class="score-header">
          <span class="score-name">${n} <span class="badge" style="color:${c}">${b}</span></span>
          <span class="score-val" style="color:${c}">${v.toFixed(1)}%</span>
        </div>
        <div class="score-track">
          <div class="score-fill" style="width:${v}%; background:${c}; box-shadow: 0 0 10px ${c}88"></div>
        </div>
      </div>`).join('');

    // Update Interpretation
    document.getElementById('interp').innerHTML = 
      `<div class="interp ${conf.cls}">
         <div class="interp-icon">${conf.icon}</div>
         <div><b>${conf.label} Assessment:</b><br/>${conf.msg}</div>
       </div>`;

    // Update Modality Specific Findings
    document.getElementById('f-clin-text').textContent = getFindingText(d.clinical, 'clinical');
    document.getElementById('f-ecg-text').textContent = getFindingText(d.ecg, 'electrocardiographic');
    document.getElementById('f-mri-text').textContent = getFindingText(d.mri, 'myocardial structural');

    // Show result visually
    result.style.display = 'block';
    
    // Small delay to ensure render before scroll
    setTimeout(() => {
      result.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

  } catch(e) {
    alert('Inference Pipeline Error: ' + e.message);
  } finally {
    btn.disabled = false;
    spin.style.display = 'none';
  }
}
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(PAGE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        clin = {k: float(request.form.get(k, 0)) for k in FEAT}
        X_c  = norm_clinical(clin)

        ecg_file = request.files.get('ecg_file')
        if not ecg_file:
            return jsonify({'error': 'ECG file missing'}), 400
        X_e = norm_ecg(ecg_file.read())

        mri_file = request.files.get('mri_file')
        if not mri_file:
            return jsonify({'error': 'MRI file missing'}), 400
        X_m = norm_mri(mri_file.read())

        # Flip clinical score probability because 1=healthy in the dataset
        c_s = 1.0 - float(_clinical.predict(X_c, verbose=0)[0][0])
        e_s = float(_ecg.predict(X_e,      verbose=0)[0][0])
        m_s = float(_mri.predict(X_m,      verbose=0)[0][0])

        fused = np.concatenate([
            _c_emb.predict(X_c, verbose=0),
            _e_emb.predict(X_e, verbose=0),
            _m_emb.predict(X_m, verbose=0),
        ], axis=1)
        # Flip fusion score probability identically to clinical because it inherited the same target labels
        f_s = 1.0 - float(_fusion.predict(fused, verbose=0)[0][0])

        # Core logic: Combined risk score computation based on weights
        combined_s = (c_s * 0.30) + (e_s * 0.35) + (m_s * 0.35)

        return jsonify({
            'clinical': round(c_s * 100, 2),
            'ecg':      round(e_s * 100, 2),
            'mri':      round(m_s * 100, 2),
            'fusion':   round(f_s * 100, 2),
            'combined': round(combined_s * 100, 2)
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5000, threaded=False)
