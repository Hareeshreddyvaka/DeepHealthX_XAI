import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import shap
import pickle

POPULATION_MEDIAN = {
    'age': 55.0, 'sex': 1.0, 'cp': 1.0, 'trestbps': 130.0, 'chol': 240.0,
    'fbs': 0.0, 'restecg': 1.0, 'thalach': 150.0, 'exang': 0.0, 'oldpeak': 0.8,
    'slope': 1.0, 'ca': 0.0, 'thal': 2.0, 'bmi': 25.0, 'smoking': 0.0
}

FEATURE_RANGES = {
    'age': (20.0, 100.0), 'sex': (0.0, 1.0), 'cp': (0.0, 3.0), 'trestbps': (80.0, 200.0), 'chol': (100.0, 400.0),
    'fbs': (0.0, 1.0), 'restecg': (0.0, 2.0), 'thalach': (60.0, 220.0), 'exang': (0.0, 1.0), 'oldpeak': (0.0, 6.0),
    'slope': (0.0, 2.0), 'ca': (0.0, 4.0), 'thal': (0.0, 3.0), 'bmi': (15.0, 45.0), 'smoking': (0.0, 1.0)
}

DEFAULT_FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'bmi', 'smoking'
]

class ClinicalShapExplainer:
    def __init__(self, model_path, scaler_path=None, background_data=None, feature_names=None):
        import logging
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        
        try:
            self.model = keras.models.load_model(model_path, compile=False)
        except OSError:
            self.model = self._create_dummy_model()
            
        self.scaler = None
        if scaler_path:
            # Scalers might be saved via joblib. If joblib missing, use pickle
            try:
                import joblib
                self.scaler = joblib.load(scaler_path)
            except ImportError:
                try:
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                except FileNotFoundError:
                    pass
            except FileNotFoundError:
                pass
                
        self.expected_input_size = self.model.input_shape[-1] if self.model.input_shape else 14
        
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = DEFAULT_FEATURE_NAMES[:self.expected_input_size]
            
        if background_data is None:
            background_data = np.array([[POPULATION_MEDIAN.get(f, 0.0) for f in self.feature_names]])
            
        def model_predict(x):
            if self.scaler:
                try:
                    x_processed = self.scaler.transform(x)
                except Exception:
                    x_processed = x
            else:
                x_processed = x
                
            out = self.model.predict(x_processed, verbose=0)
            if out.shape[1] > 1:
                return out[:, 1]
            return out.flatten()
            
        self.explainer = shap.KernelExplainer(model_predict, background_data)

    def _create_dummy_model(self):
        model = keras.Sequential([
            keras.layers.Dense(16, activation='relu', input_shape=(14,)),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def _preprocess_input(self, patient_data):
        processed = []
        for feat in self.feature_names:
            val = patient_data.get(feat)
            if val is None or np.isnan(float(val) if val is not None else np.nan):
                val = POPULATION_MEDIAN.get(feat, 0.0)
                
            bounds = FEATURE_RANGES.get(feat, (-np.inf, np.inf))
            val = np.clip(val, bounds[0], bounds[1])
            processed.append(float(val))
            
        return np.array(processed).reshape(1, -1)
        
    def explain_prediction(self, patient_data):
        x = self._preprocess_input(patient_data)
        
        x_pred = self.scaler.transform(x) if self.scaler else x
        out = self.model.predict(x_pred, verbose=0)
        
        if out.shape[1] > 1:
            pred = out[0, 1].item()
        else:
            pred = out[0, 0].item()
                
        # Calculate SHAP values for the ORIGINAL unscaled inputs
        # so interpretations make sense (e.g. "chol (+0.12)" not StandardScaled(chol))
        shap_values = self.explainer.shap_values(x, nsamples=100)
        
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values
            
        sv = sv.flatten().tolist()
        expected_val = float(self.explainer.expected_value[1]) if isinstance(self.explainer.expected_value, list) else float(self.explainer.expected_value)
        
        return {
            'shap_values': sv,
            'base_value': expected_val,
            'prediction': pred,
            'feature_names': self.feature_names,
            'input_features': x.flatten().tolist()
        }

    def get_force_plot_data(self, explanation):
        shap_vals = explanation['shap_values']
        features = []
        for i, sv in enumerate(shap_vals):
            if abs(sv) > 1e-4:
                features.append({
                    'name': self.feature_names[i],
                    'value': explanation['input_features'][i],
                    'contribution': sv,
                    'direction': 'up' if sv > 0 else 'down'
                })
                
        return {
            'base_value': explanation['base_value'],
            'prediction': explanation['prediction'],
            'features': features
        }

    def get_bar_plot_data(self, explanation, top_k=10):
        shap_vals = explanation['shap_values']
        names = explanation['feature_names']
        
        importance = [(names[i], abs(shap_vals[i]), shap_vals[i]) for i in range(len(shap_vals))]
        importance.sort(key=lambda x: x[1], reverse=True)
        top_features = importance[:top_k]
        
        return {
            'features': [f[0] for f in top_features],
            'magnitudes': [f[1] for f in top_features],
            'contributions': [f[2] for f in top_features]
        }

def get_clinical_explanation(patient_features_dict, model_path='models/clinical_model.h5', scaler_path='models/clinical_scaler.pkl'):
    explainer = ClinicalShapExplainer(model_path, scaler_path)
    explanation = explainer.explain_prediction(patient_features_dict)
    
    shap_vals = explanation['shap_values']
    names = explanation['feature_names']
    pred = explanation['prediction']
    
    contributions = [(names[i], shap_vals[i]) for i in range(len(shap_vals))]
    contributions.sort(key=lambda x: x[1], reverse=True)
    
    top_pushing_up = [c for c in contributions if c[1] > 0][:3]
    top_pushing_down = [c for c in contributions if c[1] < 0][:3]
    
    risk_level = "HIGH RISK" if pred > 0.5 else "LOW RISK"
    
    if pred > 0.5 and top_pushing_up:
        reasons = [f"{c[0]} (+{c[1]:.2f})" for c in top_pushing_up]
        interp = f"Patient at {risk_level} primarily because: " + " + ".join(reasons)
    elif pred <= 0.5 and top_pushing_down:
        reasons = [f"{c[0]} ({c[1]:.2f})" for c in top_pushing_down]
        interp = f"Patient at {risk_level} primarily because: " + " + ".join(reasons)
    else:
        interp = f"Patient at {risk_level} with competing factors."
        
    return {
        'shap_values': explanation['shap_values'],
        'base_value': explanation['base_value'],
        'prediction': explanation['prediction'],
        'feature_names': explanation['feature_names'],
        'input_features': explanation['input_features'],
        'interpretation': interp
    }

if __name__ == "__main__":
    sample_patient = {
        'age': 65.0,
        'sex': 1.0,
        'chol': 280.0,
        'trestbps': 160.0,
        'thalach': 120.0
    }
    
    res = get_clinical_explanation(sample_patient)
    explainer = ClinicalShapExplainer('models/clinical_model.h5', 'models/clinical_scaler.pkl')
    force_data = explainer.get_force_plot_data(res)
    bar_data = explainer.get_bar_plot_data(res)
    
    print(json.dumps({
        'explanation': res,
        'force_plot': force_data,
        'bar_plot': bar_data
    }, indent=2))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    features = bar_data['features'][::-1]
    contributions = bar_data['contributions'][::-1]
    colors = ['red' if c > 0 else 'blue' for c in contributions]
    
    plt.barh(features, contributions, color=colors)
    plt.title(f'Clinical SHAP Contributions (Risk Prediction: {res["prediction"]:.2f})')
    plt.xlabel('SHAP Value (Impact on Prediction)')
    plt.tight_layout()
    plt.savefig('demo_clinical_shap.png')
    plt.close()
    print("\n[+] Saved visualization to 'demo_clinical_shap.png'")
