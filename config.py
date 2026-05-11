"""
DeepHealthX Configuration File

Customize training hyperparameters, model architectures, and paths here.

This file centralizes all configuration to make the project easily customizable
without modifying the main notebook.
"""

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PATHS = {
    # Data directories
    'data_dir': '../data',
    'clinical_data': '../data/heart.csv',
    'ecg_train': '../data/mitbih_train.csv',
    'ecg_test': '../data/mitbih_test.csv',
    'mri_dir': '../data/acdc_mri',

    # Model save directories
    'models_dir': '../models',
    'clinical_model': '../models/clinical_model.h5',
    'ecg_model': '../models/ecg_model.h5',
    'mri_model': '../models/mri_model.h5',
    'fusion_model': '../models/fusion_model.h5',

    # Visualization outputs
    'stage1_curves': '../models/stage1_training_curves.png',
    'fusion_curves': '../models/fusion_training_curves.png',
    'confusion_matrices': '../models/confusion_matrices.png',
    'roc_curves': '../models/roc_curves.png',
    'metrics_comparison': '../models/metrics_comparison.png',
}

# =============================================================================
# MODEL ARCHITECTURE CONFIGURATION
# =============================================================================

ARCHITECTURE = {
    # CLINICAL BRANCH
    'clinical': {
        'input_features': 13,
        'dense_layers': [128, 64, 32],
        'embedding_dim': 32,
        'dropout_rate': 0.3,
        'activation': 'relu',
    },

    # ECG BRANCH
    'ecg': {
        'signal_length': 187,
        'conv1d_filters': [32, 64, 128],
        'kernel_size': 3,
        'pool_size': 2,
        'embedding_dim': 64,
        'dropout_rate': 0.3,
        'activation': 'relu',
    },

    # MRI BRANCH
    'mri': {
        'image_size': (224, 224),
        'backbone': 'EfficientNetB0',
        'pretrained_weights': 'imagenet',
        'embedding_dim': 64,
        'dropout_rate': 0.3,
        'activation': 'relu',
    },

    # FUSION HEAD
    'fusion': {
        'concatenated_dim': 160,  # 32 + 64 + 64
        'hidden_dim': 128,
        'dropout_rate': 0.4,
        'activation': 'relu',
        'output_activation': 'sigmoid',
    }
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

TRAINING = {
    # Stage 1 Training
    'stage1': {
        'batch_size': 32,
        'epochs': 30,
        'validation_split': 0.2,
        'optimizer': 'adam',
        'loss': 'binary_crossentropy',
        'metrics': ['accuracy', 'precision', 'recall', 'auc'],
    },

    # Stage 2 Fusion Training
    'stage2': {
        'batch_size': 32,
        'epochs': 30,
        'validation_split': 0.2,
        'optimizer': 'adam',
        'loss': 'binary_crossentropy',
        'metrics': ['accuracy', 'precision', 'recall', 'auc'],
        'repeat_factor': 3,  # Each sample paired 3 times in random pairing
    }
}

# =============================================================================
# DATA PREPROCESSING CONFIGURATION
# =============================================================================

PREPROCESSING = {
    # Clinical data preprocessing
    'clinical': {
        'missing_value_strategy': 'mean',  # Options: 'mean', 'median', 'drop'
        'scaling_method': 'minmax',  # Options: 'minmax', 'standard', 'robust'
        'test_split': 0.2,
    },

    # ECG signal preprocessing
    'ecg': {
        'normalization': 'standardizer',  # Options: 'standardizer', 'minmax'
        'expected_length': 187,
        'padding_strategy': 'truncate',  # If not 187: 'pad' or 'truncate'
    },

    # MRI image preprocessing
    'mri': {
        'target_size': (224, 224),
        'normalization_method': 'divide_by_255',  # 0-1 range
        'color_mode': 'rgb',  # Convert all to RGB
    }
}

# =============================================================================
# DATA SPLIT CONFIGURATION
# =============================================================================

DATA_SPLIT = {
    'random_seed': 42,
    'train_ratio': 0.8,
    'validation_ratio': 0.2,
    'stratified': True,  # Maintain class balance in splits
}

# =============================================================================
# REGULARIZATION & OPTIMIZATION
# =============================================================================

REGULARIZATION = {
    'dropout_rate_clinical': 0.3,
    'dropout_rate_ecg': 0.3,
    'dropout_rate_mri': 0.3,
    'dropout_rate_fusion': 0.4,

    # L1/L2 regularization
    'l1_rate': 0.0,
    'l2_rate': 1e-5,

    # Early stopping
    'early_stopping': {
        'enabled': True,
        'monitor': 'val_loss',
        'patience': 5,
        'restore_best': True,
    }
}

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

EVALUATION = {
    'metrics': [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'roc_auc',
        'confusion_matrix',
    ],

    'visualization': {
        'plot_training_curves': True,
        'plot_confusion_matrix': True,
        'plot_roc_curve': True,
        'plot_metrics_comparison': True,
        'save_high_res': True,  # Save at 150 DPI
    },

    'threshold_tuning': {
        'enable': False,
        'thresholds': [0.3, 0.4, 0.5, 0.6, 0.7],
    }
}

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================

INFERENCE = {
    'batch_inference': True,
    'return_individual_scores': True,  # Clinical, ECG, MRI scores
    'confidence_threshold': 0.5,

    # Risk categorization
    'risk_categories': {
        'low': (0.0, 0.33),    # Green
        'medium': (0.33, 0.67),  # Yellow
        'high': (0.67, 1.0),    # Red
    }
}

# =============================================================================
# LOGGING & MONITORING
# =============================================================================

LOGGING = {
    'verbosity': 1,  # 0=silent, 1=progress, 2=detailed
    'log_to_file': False,
    'log_file': '../logs/training.log',

    'tensorboard': {
        'enabled': False,
        'log_dir': '../logs/tensorboard',
    }
}

# =============================================================================
# ADVANCED OPTIONS
# =============================================================================

ADVANCED = {
    # GPU Configuration
    'gpu': {
        'enabled': True,
        'memory_growth': True,  # Allocate GPU memory as needed
        'device_id': 0,  # Which GPU to use
    },

    # Data augmentation (if enabled)
    'augmentation': {
        'enabled': False,
        'clinical_jitter': 0.01,
        'ecg_noise': 0.02,
        'mri_rotation': 10,  # degrees
    },

    # Ensemble options (future)
    'ensemble': {
        'use_ensemble': False,
        'num_models': 3,
    },

    # Model compression
    'compression': {
        'enable_pruning': False,
        'enable_quantization': False,
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config():
    """Get complete configuration dictionary"""
    return {
        'paths': PATHS,
        'architecture': ARCHITECTURE,
        'training': TRAINING,
        'preprocessing': PREPROCESSING,
        'data_split': DATA_SPLIT,
        'regularization': REGULARIZATION,
        'evaluation': EVALUATION,
        'inference': INFERENCE,
        'logging': LOGGING,
        'advanced': ADVANCED,
    }

def validate_config():
    """Validate configuration consistency"""
    config = get_config()

    # Check embedding dims sum to total
    total_embedding = (
        config['architecture']['clinical']['embedding_dim'] +
        config['architecture']['ecg']['embedding_dim'] +
        config['architecture']['mri']['embedding_dim']
    )
    expected_fusion_input = config['architecture']['fusion']['concatenated_dim']

    assert total_embedding == expected_fusion_input, \
        f"Embedding dimensions don't match: {total_embedding} != {expected_fusion_input}"

    print("✓ Configuration validated successfully")
    return True

if __name__ == '__main__':
    import json

    print("\n" + "="*70)
    print("DeepHealthX Configuration")
    print("="*70)

    config = get_config()

    # Pretty print configuration
    for section, values in config.items():
        print(f"\n[{section.upper()}]")
        print(json.dumps(values, indent=2))

    print("\n" + "="*70)
    validate_config()
    print("="*70)
