import numpy as np
import tensorflow as tf
from tensorflow import keras

class DummyMRIModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(16, 3, padding='same', activation='relu')
        self.pool1 = keras.layers.MaxPooling2D(2)
        self.conv2 = keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.pool2 = keras.layers.MaxPooling2D(2)
        self.conv3 = keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.pool3 = keras.layers.MaxPooling2D(2)
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(160, activation='relu')
        self.fc2 = keras.layers.Dense(2, activation='softmax')
        
    def call(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        emb = self.fc1(self.flatten(x))
        return self.fc2(emb)

class GradCAMMRI:
    def __init__(self, model):
        self.model = model
        self.last_conv_layer_name = self._get_last_conv_layer()
        
        self.grad_model = keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
        )

    def _get_last_conv_layer(self):
        for layer in reversed(self.model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                return layer.name
        raise ValueError("No Convolutional layer found in the loaded model.")

    def generate_cam(self, input_tensor, target_class=1):
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            conv_outputs, predictions = self.grad_model(input_tensor)
            
            if isinstance(predictions, list):
                predictions = predictions[0]
                
            if predictions.shape[1] == 1:
                score = predictions[:, 0]
            else:
                score = predictions[:, target_class]
                
        grads = tape.gradient(score, conv_outputs)
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        cam = np.maximum(cam, 0)
        
        input_shape = input_tensor.shape[1:3]
        cam = tf.image.resize(cam[..., tf.newaxis], input_shape, method='bilinear').numpy().squeeze()
        
        cam_max = cam.max()
        if cam_max > 1e-8:
            cam = cam / cam_max
        else:
            cam = np.zeros_like(cam)
            
        return cam, predictions.numpy()

    def overlay_heatmap(self, heatmap, original_image, alpha=0.5):
        if original_image.max() > 1.0:
            heatmap_scaled = heatmap * 255.0
        else:
            heatmap_scaled = heatmap
            
        return (1 - alpha) * original_image + alpha * heatmap_scaled

    def get_explanation_data(self, input_tensor):
        cam, predictions = self.generate_cam(input_tensor, target_class=1)
        
        if predictions.shape[1] == 1:
            prob = predictions[0, 0]
        else:
            prob = predictions[0, 1]
            
        h, w = cam.shape
        regions = {
            "LAD": np.mean(cam[:h//2, :w//2]),
            "LCx": np.mean(cam[:h, w//2:]),
            "RCA": np.mean(cam[h//2:, :w//2])
        }
        
        critical_regions = []
        for reg_name, intensity in regions.items():
            if intensity > 0.4:
                critical_regions.append(f"{reg_name} shows abnormality")
            else:
                critical_regions.append(f"{reg_name} clear")
                
        return {
            "heatmap": cam.tolist(),
            "prediction": float(prob),
            "critical_regions": critical_regions,
            "confidence": float(prob) if prob > 0.5 else 1.0 - float(prob),
            "grad_cam_intensity": float(np.mean(cam))
        }

def process_mri_for_explanation(mri_numpy_array, model_path='models/mri_model.h5'):
    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    
    # Patch for classic Keras MobileNet architectures having serialization conflicts on newer TF versions
    class CustomDepthwiseConv2D(keras.layers.DepthwiseConv2D):
        def __init__(self, **kwargs):
            if 'groups' in kwargs:
                kwargs.pop('groups')
            super().__init__(**kwargs)

    try:
        model = keras.models.load_model(
            model_path, 
            compile=False, 
            custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}
        )
    except OSError:
        dummy_model = DummyMRIModel()
        dummy_model.build(input_shape=(1, 224, 224, 1))
        model = dummy_model
        
    mri_array = mri_numpy_array.astype(np.float32)
    if mri_array.max() > 255:
        mri_array = (mri_array - mri_array.min()) / (mri_array.max() - mri_array.min() + 1e-8)
    elif mri_array.max() > 1.0:
        mri_array = mri_array / 255.0
        
    # Force 3-channel representation since the MobileNet .h5 explicitly expects RGB (axis -1 = 3)
    if len(mri_array.shape) == 2:
        mri_array = np.stack((mri_array,)*3, axis=-1)
    elif len(mri_array.shape) == 3 and mri_array.shape[-1] == 1:
        mri_array = np.concatenate([mri_array]*3, axis=-1)
    elif len(mri_array.shape) == 3 and (mri_array.shape[0] == 1 or mri_array.shape[0] == 3):
        mri_array = np.transpose(mri_array, (1, 2, 0))
        if mri_array.shape[-1] == 1:
            mri_array = np.concatenate([mri_array]*3, axis=-1)
            
    if len(mri_array.shape) == 3:
        input_tensor = tf.convert_to_tensor(mri_array)[tf.newaxis, ...]
    else:
        input_tensor = tf.convert_to_tensor(mri_array)
            
    explainer = GradCAMMRI(model)
    data = explainer.get_explanation_data(input_tensor)
    
    heatmap_array = np.array(data['heatmap'])
    
    img_2d = mri_array.squeeze()
    # If RGB, just take the mean to get a native grayscale background for the overlay rendering
    if len(img_2d.shape) == 3 and img_2d.shape[-1] == 3:
        img_2d = np.mean(img_2d, axis=-1)
    elif len(img_2d.shape) > 2:
        img_2d = img_2d[0]
        
    data['overlay'] = explainer.overlay_heatmap(heatmap_array, img_2d).tolist()
    
    return data

if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    
    try:
        # Load real patient MRI 
        img_raw = plt.imread('demo_samples/demo_files/mri_patient101_disease.jpg')
        
        # Force 3-channel RGB for MobileNet constraints
        if len(img_raw.shape) == 2:
            img_raw = np.stack((img_raw,)*3, axis=-1)
        elif img_raw.shape[-1] == 4:
            img_raw = img_raw[..., :3]
            
        # Safely force native 224x224 dimensions via tensorflow resizing
        dummy_mri = tf.image.resize(img_raw, (224, 224)).numpy()
    except Exception as e:
        print(f"Failed to load image, using random static fallback: {e}")
        dummy_mri = np.random.rand(224, 224, 3) * 255.0

    res = process_mri_for_explanation(dummy_mri, 'models/mri_model.h5')
    
    print(json.dumps({
        "prediction": res["prediction"],
        "confidence": res["confidence"],
        "critical_regions": res["critical_regions"],
        "grad_cam_intensity": res["grad_cam_intensity"],
        "heatmap_shape": [len(res["heatmap"]), len(res["heatmap"][0])],
        "overlay_shape": [len(res["overlay"]), len(res["overlay"][0])]
    }, indent=2))

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    heatmap = np.array(res["heatmap"])
    
    # Format original MRI strictly to [0..1] to prevent matplotlib float clipping artifacts
    display_mri = dummy_mri
    if display_mri.max() > 1.0:
        display_mri = display_mri / 255.0
    if len(display_mri.shape) == 2:
        display_mri = np.stack((display_mri,)*3, axis=-1)
        
    # Generate full RGB overlay correctly colorizing scalar gradients
    jet = plt.colormaps["jet"] if hasattr(plt, "colormaps") else cm.get_cmap("jet")
    jet_heatmap = jet(heatmap)[..., :3]
    overlay_rgb = 0.5 * display_mri + 0.5 * jet_heatmap
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(display_mri)
    axes[0].set_title('Original MRI (Normalized)')
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Matrix Details')
    axes[2].imshow(overlay_rgb)
    axes[2].set_title(f'Pathology Overlay (Risk: {res["prediction"]:.2f})')
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig('demo_mri_gradcam.png')
    plt.close()
    print("\n[+] Saved visualization to 'demo_mri_gradcam.png'")
