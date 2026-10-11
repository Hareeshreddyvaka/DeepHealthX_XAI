"""
DeepHealthX — Streamlit Web Application
Run: streamlit run webapp/app.py  (from project root)
Environment: Python 3.10 + TensorFlow 2.13 + Streamlit 1.28
"""

import io, sys, pickle, warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from scipy.signal import find_peaks

st.set_page_config(
    page_title="DeepHealthX",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.main { padding-top: 1rem; }
div[data-testid="metric-container"] {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 12px 16px;
}
.stNumberInput input { font-size: 14px; }
.section-header {
    background: #1a3a5c;
    color: white;
    padding: 8px 14px;
    border-radius: 6px;
    margin-bottom: 10px;
    font-weight: 600;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'

FEATURE_KEYS = [
    'age','sex','cp','trestbps','chol','fbs',
    'restecg','thalach','exang','oldpeak','slope','ca','thal'
]

FEATURE_RANGES = {
    'age':(29,77),'sex':(0,1),'cp':(0,3),'trestbps':(90,200),
    'chol':(126,564),'fbs':(0,1),'restecg':(0,2),'thalach':(60,202),
    'exang':(0,1),'oldpeak':(0.0,6.2),'slope':(0,2),'ca':(0,4),'thal':(0,3)
}

# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource
def load_models():
    missing = [f for f in
        ['clinical_model.h5','ecg_model.h5','mri_model.h5','fusion_model.h5']
        if not (MODELS_DIR / f).exists()]
    if missing:
        st.error("❌ Run training notebook first. Missing: " + ", ".join(missing))
        st.stop()
    try:
        clinical = keras.models.load_model(str(MODELS_DIR / 'clinical_model.h5'))
        ecg      = keras.models.load_model(str(MODELS_DIR / 'ecg_model.h5'))
        mri      = keras.models.load_model(str(MODELS_DIR / 'mri_model.h5'))
        fusion   = keras.models.load_model(str(MODELS_DIR / 'fusion_model.h5'))
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        st.exception(e); st.stop()

    c_emb = keras.Model(clinical.input, clinical.get_layer('clinical_embedding').output)
    e_emb = keras.Model(ecg.input,      ecg.get_layer('ecg_embedding').output)
    m_emb = keras.Model(mri.input,      mri.get_layer('mri_embedding').output)
    for m in [c_emb, e_emb, m_emb]:
        m.trainable = False

    scaler = None
    sp = MODELS_DIR / 'clinical_scaler.pkl'
    if sp.exists():
        with open(str(sp), 'rb') as f:
            scaler = pickle.load(f)
    else:
        st.warning("⚠ clinical_scaler.pkl not found — using fallback normalisation")

    return clinical, ecg, mri, c_emb, e_emb, m_emb, fusion, scaler


(clinical_model, ecg_model, mri_model,
 c_emb_model, e_emb_model, m_emb_model,
 fusion_model, clinical_scaler) = load_models()

# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess_clinical(features):
    arr = np.array(
        [features.get(k, 0) for k in FEATURE_KEYS],
        dtype=np.float32
    ).reshape(1, -1)
    if clinical_scaler is not None:
        return clinical_scaler.transform(arr).astype(np.float32)
    norm = arr.copy()
    for i, k in enumerate(FEATURE_KEYS):
        lo, hi = FEATURE_RANGES[k]
        norm[0, i] = (arr[0, i] - lo) / max(hi - lo, 1e-8)
    return norm


def preprocess_ecg(csv_bytes):
    try:
        df     = pd.read_csv(io.StringIO(csv_bytes), header=None)
        signal = (df.iloc[0, :187].values if df.shape[1] >= 187
                  else df.iloc[:187, 0].values).astype(np.float32)
        if len(signal) < 187:
            signal = np.pad(signal, (0, 187 - len(signal)))
        if len(signal) > 400:
            peaks, _ = find_peaks(signal, distance=100, height=np.mean(signal))
            beats = []
            for p in peaks:
                s, e = p - 93, p + 94
                if s >= 0 and e <= len(signal):
                    beats.append(signal[s:e])
            if beats:
                signal = np.mean(beats, axis=0).astype(np.float32)
                st.info(f"✓ {len(beats)} heartbeats detected and averaged")
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        return signal.reshape(1, 187, 1)
    except Exception as e:
        st.error(f"ECG error: {e}"); return None


def preprocess_mri(image_file):
    try:
        img = Image.open(image_file).convert('RGB').resize((224, 224))
        return (np.array(img, dtype=np.float32) / 255.0).reshape(1, 224, 224, 3)
    except Exception as e:
        st.error(f"MRI error: {e}"); return None


def run_prediction(clinical_data, ecg_signal, mri_image):
    c_pred = float(clinical_model.predict(clinical_data, verbose=0)[0][0])
    e_pred = float(ecg_model.predict(ecg_signal,         verbose=0)[0][0])
    m_pred = float(mri_model.predict(mri_image,          verbose=0)[0][0])
    fused  = np.concatenate([
        c_emb_model.predict(clinical_data, verbose=0),
        e_emb_model.predict(ecg_signal,    verbose=0),
        m_emb_model.predict(mri_image,     verbose=0),
    ], axis=1)
    f_pred = float(fusion_model.predict(fused, verbose=0)[0][0])
    return {'fusion': f_pred, 'clinical': c_pred, 'ecg': e_pred, 'mri': m_pred}


def risk_label(score):
    if score < 0.33: return "🟢 Low Risk"
    if score < 0.67: return "🟡 Medium Risk"
    return "🔴 High Risk"


def risk_color(score):
    if score < 0.33: return '#27ae60'
    if score < 0.67: return '#f39c12'
    return '#e74c3c'

# =============================================================================
# UI
# =============================================================================

st.markdown("""
<div style='text-align:center;padding:1rem 0 0.5rem'>
<h1 style='margin:0'>❤️ DeepHealthX</h1>
<p style='color:#666;margin:4px 0'>Multi-Modal Deep Learning for Heart Disease Risk Assessment</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2 = st.tabs(["📋 Risk Assessment", "ℹ️ About"])

with tab1:

    # ── TOP ROW: two upload cards side by side ────────────────────────────────
    up_col1, up_col2 = st.columns(2)

    with up_col1:
        st.markdown('<div class="section-header">📊 ECG Signal</div>', unsafe_allow_html=True)
        ecg_method = st.radio("Source", ["Upload CSV", "Demo Signal"],
                              key="ecg_m", horizontal=True)
        if ecg_method == "Upload CSV":
            ecg_file = st.file_uploader(
                "MIT-BIH CSV — 187 columns per row",
                type=["csv"], key="ecg_f", label_visibility="collapsed"
            )
            if ecg_file:
                ecg_signal = preprocess_ecg(ecg_file.read().decode('utf-8'))
                if ecg_signal is not None:
                    st.success("✓ ECG loaded successfully")
                    # Plot with matplotlib — no external chart library needed
                    fig, ax = plt.subplots(figsize=(6, 1.8))
                    ax.plot(ecg_signal[0, :, 0], color='#e74c3c', lw=1.2)
                    ax.set_xlabel('Sample', fontsize=8)
                    ax.set_ylabel('Amplitude', fontsize=8)
                    ax.tick_params(labelsize=7)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
            else:
                ecg_signal = None
        else:
            t = np.linspace(0, 1, 187)
            demo_ecg = (0.5*np.sin(2*np.pi*t) + 0.3*np.sin(4*np.pi*t) +
                        0.2*np.cos(6*np.pi*t) + 0.1*np.random.randn(187))
            ecg_signal = demo_ecg.reshape(1, 187, 1).astype(np.float32)
            fig, ax = plt.subplots(figsize=(6, 1.8))
            ax.plot(demo_ecg, color='#3498db', lw=1.2)
            ax.set_xlabel('Sample', fontsize=8); ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.info("Using synthetic demo ECG signal")

    with up_col2:
        st.markdown('<div class="section-header">🖼️ Cardiac MRI Scan</div>', unsafe_allow_html=True)
        mri_method = st.radio("Source", ["Upload Image", "Demo Image"],
                              key="mri_m", horizontal=True)
        if mri_method == "Upload Image":
            mri_file = st.file_uploader(
                "JPG or PNG cardiac MRI",
                type=["jpg","jpeg","png"], key="mri_f", label_visibility="collapsed"
            )
            if mri_file:
                mri_image = preprocess_mri(mri_file)
                if mri_image is not None:
                    st.success("✓ MRI loaded successfully")
                    st.image(mri_file, width=220)
            else:
                mri_image = None
        else:
            from PIL import ImageDraw
            demo_img = Image.new('RGB', (224, 224), 'black')
            d = ImageDraw.Draw(demo_img)
            d.ellipse([60,40,164,144], fill='red',  outline='red')
            d.ellipse([100,80,124,120], fill='blue', outline='blue')
            st.image(demo_img, caption="Synthetic demo MRI", width=220)
            mri_image = (np.array(demo_img,dtype=np.float32)/255.0).reshape(1,224,224,3)

    st.markdown("---")

    # ── CLINICAL FEATURES — number inputs in a clean grid ────────────────────
    st.markdown('<div class="section-header">👤 Clinical Features — enter values directly</div>',
                unsafe_allow_html=True)

    clin = {}

    # Row 1
    r1 = st.columns(5)
    with r1[0]:
        clin['age'] = st.number_input(
            "Age (years)", min_value=29, max_value=77, value=50, step=1)
    with r1[1]:
        clin['sex'] = st.selectbox(
            "Sex", [0, 1],
            format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
    with r1[2]:
        clin['cp'] = st.number_input(
            "Chest Pain (0–3)", min_value=0, max_value=3, value=0, step=1,
            help="0=Typical angina  1=Atypical  2=Non-anginal  3=Asymptomatic")
    with r1[3]:
        clin['trestbps'] = st.number_input(
            "Resting BP (mmHg)", min_value=90, max_value=200, value=120, step=1)
    with r1[4]:
        clin['chol'] = st.number_input(
            "Cholesterol (mg/dl)", min_value=126, max_value=564, value=200, step=1)

    # Row 2
    r2 = st.columns(5)
    with r2[0]:
        clin['fbs'] = st.selectbox(
            "Fasting BS > 120", [0, 1],
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
    with r2[1]:
        clin['restecg'] = st.number_input(
            "Resting ECG (0–2)", min_value=0, max_value=2, value=0, step=1,
            help="0=Normal  1=ST-T abnorm.  2=LV hypertrophy")
    with r2[2]:
        clin['thalach'] = st.number_input(
            "Max Heart Rate", min_value=60, max_value=202, value=150, step=1)
    with r2[3]:
        clin['exang'] = st.selectbox(
            "Exercise Angina", [0, 1],
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
    with r2[4]:
        clin['oldpeak'] = st.number_input(
            "ST Depression", min_value=0.0, max_value=6.2,
            value=0.0, step=0.1, format="%.1f")

    # Row 3
    r3 = st.columns(5)
    with r3[0]:
        clin['slope'] = st.number_input(
            "ST Slope (0–2)", min_value=0, max_value=2, value=1, step=1,
            help="0=Upsloping  1=Flat  2=Downsloping")
    with r3[1]:
        clin['ca'] = st.number_input(
            "Major Vessels (0–4)", min_value=0, max_value=4, value=0, step=1)
    with r3[2]:
        clin['thal'] = st.number_input(
            "Thalassemia (0–3)", min_value=0, max_value=3, value=1, step=1,
            help="0=Normal  1=Fixed defect  2=Reversible  3=Unknown")
    with r3[3]:
        st.markdown("")   # spacer
    with r3[4]:
        st.markdown("")   # spacer

    st.markdown("---")

    # ── Quick-fill presets ────────────────────────────────────────────────────
    st.markdown("**Quick presets** — click to fill demo values, then adjust if needed:")
    p1, p2, p3 = st.columns(3)
    preset_disease = p1.button("🔴 High-risk patient values",  use_container_width=True)
    preset_healthy = p2.button("🟢 Low-risk patient values",   use_container_width=True)
    preset_clear   = p3.button("⬜ Clear to defaults",          use_container_width=True)

    if preset_disease:
        st.session_state.update({
            'age':63,'sex':1,'cp':3,'trestbps':148,'chol':310,'fbs':1,
            'restecg':1,'thalach':118,'exang':1,'oldpeak':2.5,
            'slope':2,'ca':2,'thal':2
        })
        st.rerun()

    if preset_healthy:
        st.session_state.update({
            'age':42,'sex':0,'cp':0,'trestbps':118,'chol':185,'fbs':0,
            'restecg':0,'thalach':172,'exang':0,'oldpeak':0.0,
            'slope':0,'ca':0,'thal':1
        })
        st.rerun()

    if preset_clear:
        st.session_state.update({
            'age':50,'sex':1,'cp':0,'trestbps':120,'chol':200,'fbs':0,
            'restecg':0,'thalach':150,'exang':0,'oldpeak':0.0,
            'slope':1,'ca':0,'thal':1
        })
        st.rerun()

    # Apply session state values to clin dict
    for k in FEATURE_KEYS:
        if k in st.session_state:
            clin[k] = st.session_state[k]

    st.markdown("")

    # ── Predict button ────────────────────────────────────────────────────────
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        go = st.button("🔮  Run Cardiac Risk Assessment",
                       use_container_width=True, type="primary")

    # ── Results ───────────────────────────────────────────────────────────────
    if go:
        errors = []
        if ecg_method == "Upload CSV" and ecg_signal is None:
            errors.append("❌ Upload an ECG CSV file or switch to Demo Signal")
        if mri_method == "Upload Image" and mri_image is None:
            errors.append("❌ Upload an MRI image or switch to Demo Image")
        for e in errors:
            st.error(e)

        if not errors:
            clinical_data = preprocess_clinical(clin)
            with st.spinner("Running tri-modal inference…"):
                preds = run_prediction(clinical_data, ecg_signal, mri_image)

            st.success("✓ Assessment complete")
            st.markdown("---")

            # ── Top: overall score ────────────────────────────────────────────
            fs = preds['fusion']
            rc = risk_color(fs)
            rl = risk_label(fs)

            top1, top2, top3 = st.columns([1, 2, 1])
            with top2:
                st.markdown(f"""
                <div style='text-align:center;background:{rc}22;border:2px solid {rc};
                            border-radius:12px;padding:20px;margin-bottom:1rem'>
                    <div style='font-size:2.8rem;font-weight:700;color:{rc}'>
                        {fs*100:.1f}%
                    </div>
                    <div style='font-size:1.3rem;font-weight:600;color:{rc}'>
                        {rl}
                    </div>
                    <div style='color:#555;margin-top:6px;font-size:0.9rem'>
                        Confidence: {max(fs, 1-fs)*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Middle: branch scores as matplotlib bar chart ─────────────────
            res1, res2 = st.columns(2)

            with res1:
                st.markdown("**Individual branch scores**")
                fig, ax = plt.subplots(figsize=(5, 2.8))
                names  = ['Clinical', 'ECG', 'MRI', 'Fusion']
                vals   = [preds['clinical'], preds['ecg'], preds['mri'], preds['fusion']]
                colors = ['#3498db', '#e91e63', '#4caf50', rc]
                bars   = ax.bar(names, [v*100 for v in vals],
                                color=colors, alpha=0.85, edgecolor='white')
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 1,
                            f'{v*100:.1f}%', ha='center', va='bottom',
                            fontsize=9, fontweight='bold')
                ax.set_ylim(0, 115)
                ax.set_ylabel('Risk Score (%)', fontsize=9)
                ax.axhline(50, color='gray', linestyle='--', lw=0.8, alpha=0.5)
                ax.tick_params(labelsize=9)
                ax.grid(True, alpha=0.2, axis='y')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with res2:
                st.markdown("**Score breakdown**")
                for name, key, color in [
                    ('Clinical', 'clinical', '#3498db'),
                    ('ECG',      'ecg',      '#e91e63'),
                    ('MRI',      'mri',      '#4caf50'),
                    ('Fusion',   'fusion',   rc),
                ]:
                    v = preds[key]
                    st.markdown(f"""
                    <div style='margin:6px 0'>
                        <div style='display:flex;justify-content:space-between;
                                    font-size:13px;margin-bottom:3px'>
                            <span><b>{name}</b></span>
                            <span style='color:{color}'><b>{v*100:.1f}%</b>
                            &nbsp;{risk_label(v)}</span>
                        </div>
                        <div style='background:#eee;border-radius:4px;height:8px'>
                            <div style='width:{v*100:.1f}%;background:{color};
                                        height:100%;border-radius:4px'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Clinical interpretation ───────────────────────────────────────
            st.markdown("---")
            st.markdown("**Clinical Interpretation**")
            if fs < 0.33:
                st.info("✅ **Low Risk** — Indicators across all three modalities suggest low cardiac risk. Routine monitoring recommended.")
            elif fs < 0.67:
                st.warning("⚠️ **Medium Risk** — Some concerning indicators detected. Clinical follow-up and additional testing recommended.")
            else:
                st.error("🚨 **High Risk** — Significant risk indicators detected across modalities. Immediate cardiology consultation strongly recommended.")

            # ── Input summary ─────────────────────────────────────────────────
            with st.expander("📋 View full input summary"):
                s1, s2 = st.columns(2)
                with s1:
                    st.write("**Clinical values entered:**")
                    labels_map = {
                        'age':'Age','sex':'Sex','cp':'Chest Pain',
                        'trestbps':'Resting BP','chol':'Cholesterol',
                        'fbs':'Fasting BS','restecg':'Resting ECG',
                        'thalach':'Max Heart Rate','exang':'Exercise Angina',
                        'oldpeak':'ST Depression','slope':'ST Slope',
                        'ca':'Major Vessels','thal':'Thalassemia'
                    }
                    df_clin = pd.DataFrame(
                        [(labels_map.get(k,k), v) for k, v in clin.items()],
                        columns=['Feature', 'Value']
                    )
                    st.dataframe(df_clin, hide_index=True, use_container_width=True)
                with s2:
                    st.write("**Prediction scores:**")
                    df_pred = pd.DataFrame({
                        'Branch': ['Clinical','ECG','MRI','Fusion'],
                        'Score':  [f"{preds[k]*100:.2f}%" for k in
                                   ['clinical','ecg','mri','fusion']],
                        'Risk':   [risk_label(preds[k]) for k in
                                   ['clinical','ecg','mri','fusion']]
                    })
                    st.dataframe(df_pred, hide_index=True, use_container_width=True)

with tab2:
    st.markdown("""
## About DeepHealthX

Multi-modal deep learning for heart disease risk assessment.

### Data Modalities
| # | Modality | Dataset | Architecture |
|---|---|---|---|
| 1 | Clinical (13 features) | UCI Heart Disease | Dense NN |
| 2 | ECG (187-point waveform) | MIT-BIH Heartbeat | 1D-CNN |
| 3 | Cardiac MRI (224×224) | ACDC Challenge | Custom CNN |

### Two-Stage Architecture
**Stage 1** — Each branch trains independently on its own dataset with real labels.

**Stage 2** — All branch weights are frozen. Embeddings (32+64+64=160 dims) are extracted and randomly paired. Only the fusion head trains on these embeddings.

### Clinical Feature Reference
| Feature | Range | Description |
|---|---|---|
| Age | 29–77 | Years |
| Sex | 0/1 | 0=Female, 1=Male |
| CP | 0–3 | 0=Typical angina, 3=Asymptomatic |
| Trestbps | 90–200 | Resting blood pressure mmHg |
| Chol | 126–564 | Serum cholesterol mg/dl |
| FBS | 0/1 | Fasting blood sugar > 120 mg/dl |
| Restecg | 0–2 | 0=Normal, 1=ST-T abnormality, 2=LV hypertrophy |
| Thalach | 60–202 | Maximum heart rate achieved |
| Exang | 0/1 | Exercise-induced angina |
| Oldpeak | 0–6.2 | ST depression induced by exercise |
| Slope | 0–2 | 0=Upsloping, 1=Flat, 2=Downsloping |
| CA | 0–4 | Number of major vessels coloured |
| Thal | 0–3 | 0=Normal, 1=Fixed defect, 2=Reversible, 3=Unknown |

### ⚠️ Disclaimer
For **research and educational purposes only**. Not a substitute for professional medical diagnosis.

---
**Environment**: TF 2.13 · Streamlit 1.28 · Python 3.10
""")

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:gray;padding:0.5rem'>
DeepHealthX © 2026 &nbsp;|&nbsp; VIT Vellore &nbsp;|&nbsp; Research use only
</div>
""", unsafe_allow_html=True)
