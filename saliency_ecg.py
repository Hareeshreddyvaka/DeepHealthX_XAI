import numpy as np
import tensorflow as tf
from tensorflow import keras

class DummyECGModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv1D(16, kernel_size=7, padding='same', activation='relu')
        self.conv2 = keras.layers.Conv1D(32, kernel_size=5, padding='same', activation='relu')
        self.flatten = keras.layers.Flatten()
        self.fc = keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, x):
        x = self.conv2(self.conv1(x))
        x = self.flatten(x)
        return self.fc(x)

class TemporalSaliencyECG:
    def __init__(self, model):
        self.model = model
        
    def compute_saliency(self, input_tensor):
        x = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(x)
            output = self.model(x)
            
            if output.shape[1] > 1:
                score = output[:, 1]
                prob = output[0, 1].numpy()
            else:
                score = output[:, 0]
                prob = output[0, 0].numpy()
                
        grads = tape.gradient(score, x)
        
        saliency = tf.abs(grads).numpy().squeeze()
        
        window = np.ones(10) / 10
        saliency_smoothed = np.convolve(saliency, window, mode='same')
        
        s_min, s_max = saliency_smoothed.min(), saliency_smoothed.max()
        if s_max - s_min > 1e-8:
            saliency_normalized = (saliency_smoothed - s_min) / (s_max - s_min)
        else:
            saliency_normalized = np.zeros_like(saliency_smoothed)
            
        return saliency_normalized, float(prob)

    def find_critical_regions(self, saliency, threshold=0.6):
        mask = saliency >= threshold
        
        regions = []
        in_region = False
        start_idx = 0
        
        for i, val in enumerate(mask):
            if val and not in_region:
                in_region = True
                start_idx = i
            elif not val and in_region:
                in_region = False
                end_idx = i
                importance = float(saliency[start_idx:end_idx].max())
                regions.append({
                    "start": start_idx,
                    "end": end_idx,
                    "importance": importance
                })
        
        if in_region:
            regions.append({
                "start": start_idx,
                "end": len(saliency),
                "importance": float(saliency[start_idx:].max())
            })
            
        regions.sort(key=lambda x: x["importance"], reverse=True)
        return regions

    def interpret_ecg(self, region, total_samples=187):
        width = region["end"] - region["start"]
        
        if width < 15:
            phase = "Sharp spike (potential artifact or isolated complex)"
        elif width <= 35:
            phase = "QRS-like complex region"
        elif width <= 70:
            phase = "T-wave or prolonged phase"
        else:
            phase = "Extended arrhythmic segment or baseline shift"
            
        interpretation = f"High model reliance on {phase}."
        
        if "T-wave" in phase:
            interpretation += " Potential indicator of repolarization abnormality (e.g., ischemia)."
        elif "QRS" in phase:
            interpretation += " Potential indicator of ventricular depolarization abnormality."
        elif "artifact" in phase:
            interpretation += " Potential indicator of premature complex or signal noise."
            
        return interpretation

def explain_ecg(ecg_signal_array, model_path='models/ecg_model.h5'):
    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    
    try:
        model = keras.models.load_model(model_path, compile=False)
    except OSError:
        dummy = DummyECGModel()
        dummy.build(input_shape=(1, 500, 1))
        model = dummy
        
    saliency_analyzer = TemporalSaliencyECG(model)
    
    if len(ecg_signal_array.shape) == 1:
        ecg_tensor = ecg_signal_array[np.newaxis, ..., np.newaxis]
    elif ecg_signal_array.shape[0] == 1 and ecg_signal_array.shape[1] == 1:
        ecg_tensor = np.transpose(ecg_signal_array, (0, 2, 1))
    else:
        ecg_tensor = ecg_signal_array
        
    saliency, prob = saliency_analyzer.compute_saliency(ecg_tensor)
    regions = saliency_analyzer.find_critical_regions(saliency, threshold=0.6)
    
    peak_regions = []
    for reg in regions[:3]:
        interp = saliency_analyzer.interpret_ecg(reg)
        peak_regions.append({
            "start": reg["start"],
            "end": reg["end"],
            "importance": reg["importance"],
            "interpretation": interp
        })
        
    if peak_regions:
        best = peak_regions[0]
        start_ms = best["start"] * 4
        end_ms = best["end"] * 4
        crit_win = f"{start_ms}-{end_ms}ms"
    else:
        crit_win = "No clear critical window identified."
        
    return {
        "ecg_signal": ecg_signal_array.tolist(),
        "saliency": saliency.tolist(),
        "peak_regions": peak_regions,
        "prediction": prob,
        "critical_window": crit_win
    }

if __name__ == "__main__":
    import json
    try:
        # Load the real patient data given in the demo directory
        dummy_ecg = np.loadtxt('demo_samples/demo_files/ecg_arrhythmia.csv', delimiter=',')
        dummy_ecg = dummy_ecg.flatten()
        
        # Enforce exact dimension limits to avoid Keras crashes
        if len(dummy_ecg) > 187:
            dummy_ecg = dummy_ecg[:187]
        elif len(dummy_ecg) < 187:
            dummy_ecg = np.pad(dummy_ecg, (0, 187 - len(dummy_ecg)), 'constant')
            
    except Exception as e:
        print(f"Failed to load CSV, using random noise: {e}")
        dummy_ecg = np.random.randn(187)

    res = explain_ecg(dummy_ecg, 'models/ecg_model.h5')
    
    print(json.dumps({
        "prediction": res["prediction"],
        "critical_window": res["critical_window"],
        "peak_regions": res["peak_regions"],
        "saliency_length": len(res["saliency"])
    }, indent=2))

    import matplotlib.pyplot as plt
    signal = np.array(res["ecg_signal"])
    saliency = np.array(res["saliency"])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(signal, color='black')
    ax1.set_title(f'Simulated ECG Signal (Risk Prediction: {res["prediction"]:.2f})')
    ax1.set_ylabel('Amplitude')
    
    ax2.plot(saliency, color='red')
    ax2.fill_between(range(len(saliency)), saliency, color='red', alpha=0.3)
    
    for reg in res['peak_regions']:
        ax2.axvspan(reg['start'], reg['end'], color='yellow', alpha=0.3)
        
    ax2.set_title('Temporal Saliency (Critical Zones Highlighted)')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Importance')
    
    plt.tight_layout()
    plt.savefig('demo_ecg_saliency.png')
    plt.close()
    print("\n[+] Saved visualization to 'demo_ecg_saliency.png'")
