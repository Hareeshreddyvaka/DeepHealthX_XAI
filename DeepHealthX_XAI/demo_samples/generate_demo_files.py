"""
DeepHealthX — Demo File Generator (Fixed)
Place in project root and run:  python generate_demo_files.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
import cv2
from PIL import Image

OUTPUT_DIR = Path('demo_files')
OUTPUT_DIR.mkdir(exist_ok=True)

# Try every likely location for the ACDC training folder
POSSIBLE_MRI_PATHS = [
    Path('data/testing/testing'),
    Path('data/testing'),
    Path('../data/testing/testing'),
    Path('../data/testing'),
]

MRI_BASE = None
for p in POSSIBLE_MRI_PATHS:
    if p.exists():
        has_patients = any(
            (child / 'Info.cfg').exists()
            for child in p.iterdir() if child.is_dir()
        )
        if has_patients:
            MRI_BASE = p
            print(f"Found ACDC data at: {MRI_BASE.resolve()}")
            break

if MRI_BASE is None:
    print("ERROR: Cannot find ACDC training folder.")
    print("Looked in:", [str(p) for p in POSSIBLE_MRI_PATHS])
    raise SystemExit(1)

# ECG files
print("\nGenerating ECG files...")
ECG_PATHS = [Path('data/mitbih_test.csv'), Path('../data/mitbih_test.csv')]
ecg_path = next((p for p in ECG_PATHS if p.exists()), None)

if ecg_path:
    df = pd.read_csv(ecg_path, header=None)
    for label, cls, row_num, fname in [
        ('Normal',             0, 5, 'ecg_normal.csv'),
        ('Ventricular arrhy.', 2, 3, 'ecg_arrhythmia.csv'),
        ('Supraventricular',   1, 2, 'ecg_supraventricular.csv'),
    ]:
        rows = df[df.iloc[:, 187] == cls]
        if len(rows) == 0:
            continue
        beat = rows.iloc[min(row_num, len(rows)-1), :187]
        beat.to_csv(OUTPUT_DIR / fname, header=False, index=False)
        print(f"  Saved {fname}  ({label})")
else:
    print("  mitbih_test.csv not found")

# MRI to JPG
print("\nConverting ACDC NIfTI files to JPG...")

def extract_label(cfg_path):
    with open(cfg_path) as f:
        for line in f:
            if 'Group' in line:
                g = line.split(':')[1].strip()
                return g, (0 if g == 'NOR' else 1)
    return None, None

patient_dirs = sorted([p for p in MRI_BASE.iterdir() if p.is_dir()])
saved = 0

for patient_dir in patient_dirs:
    if saved >= 6:
        break
    cfg_file = patient_dir / 'Info.cfg'
    if not cfg_file.exists():
        continue
    group, label = extract_label(cfg_file)
    if label is None:
        continue
    nii_files = sorted([f for f in patient_dir.glob('*frame*.nii.gz') if '_gt' not in f.name])
    if not nii_files:
        continue
    try:
        data = nib.load(str(nii_files[0])).get_fdata()
        mid_z = data.shape[2] // 2
        slc   = data[:, :, mid_z].astype(np.float32)
        s_min, s_max = slc.min(), slc.max()
        if s_max - s_min < 1e-8:
            continue
        slc = (slc - s_min) / (s_max - s_min)
        slc_uint8 = (slc * 255).astype(np.uint8)
        slc_224   = cv2.resize(slc_uint8, (224, 224), interpolation=cv2.INTER_LINEAR)
        slc_rgb   = cv2.cvtColor(slc_224, cv2.COLOR_GRAY2RGB)
        tag       = 'NOR' if label == 0 else 'disease'
        out_name  = f"mri_{patient_dir.name}_{tag}.jpg"
        Image.fromarray(slc_rgb).save(OUTPUT_DIR / out_name, quality=95)
        print(f"  Saved {out_name}  (Group: {group})")
        saved += 1
    except Exception as e:
        print(f"  Skipping {patient_dir.name}: {e}")

print(f"\nDone — {saved} MRI JPGs + ECG CSVs saved to demo_files/")
print("""
Demo combinations:
  Healthy : ecg_normal.csv      + mri_patientXXX_NOR.jpg
            Age=42 Chol=185 CP=0 CA=0 Thalach=170

  Disease : ecg_arrhythmia.csv  + mri_patientXXX_disease.jpg
            Age=63 Chol=310 CP=3 CA=2 Thalach=120 Oldpeak=2.5
""")
