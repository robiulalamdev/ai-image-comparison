# config/settings.py

import os

class Settings:
    """Application settings and configuration."""
    
    def __init__(self):
        # Base directories
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.MODELS_DIR = os.path.join(self.BASE_DIR, "models")
        self.PRETRAINED_MODELS_DIR = os.path.join(self.MODELS_DIR, "pretrained")
        self.UPLOADS_DIR = os.path.join(self.BASE_DIR, "uploads")
        self.RESULTS_DIR = os.path.join(self.BASE_DIR, "results")
        self.CACHE_DIR = os.path.join(self.BASE_DIR, "cache")
        self.LOGS_DIR = os.path.join(self.BASE_DIR, "logs")

        # Ensure directories exist
        os.makedirs(self.PRETRAINED_MODELS_DIR, exist_ok=True)
        os.makedirs(self.UPLOADS_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.LOGS_DIR, exist_ok=True)

        # Model weights paths
        self.superpoint_weights_path = os.path.join(self.PRETRAINED_MODELS_DIR, "superpoint_v1.pth")
        self.superglue_weights_path = os.path.join(self.PRETRAINED_MODELS_DIR, "superglue_indoor.pth")
        self.changeformer_weights_path = os.path.join(self.PRETRAINED_MODELS_DIR, "changeformer_levir.pth")
        
        # Model configuration parameters
        self.superpoint_config = {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024,
            'remove_borders': 4,
        }
        
        self.superglue_config = {
            'weights': 'indoor', # 'indoor' or 'outdoor'
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }

        # ChangeFormer config - these should ideally match the pre-trained model's architecture
        # If you are cloning the original ChangeFormer repo, its `config.py` would provide these.
        # This is a placeholder; adjust if your ChangeFormer model has different parameters.
        self.changeformer_config = {
            'input_nc': 3, # Input channels for each image (RGB)
            'output_nc': 2, # Number of output classes (e.g., 2 for binary change/no-change)
            'embed_dim': 256,
            'depths': [2, 2, 2, 2],
            'num_heads': [4, 8, 16, 32],
            'mlp_ratio': 4.,
            'qkv_bias': True,
            'qk_scale': None,
            'drop_rate': 0.,
            'attn_drop_rate': 0.,
            'drop_path_rate': 0.1,
            'norm_layer': 'nn.LayerNorm', # Use string to avoid circular import if nn is not available yet
            'patch_norm': True,
            'use_checkpoint': True,
            'decoder_embed_dim': 256,
            'decoder_depths': [2, 2, 2, 2],
            'decoder_num_heads': [4, 8, 16, 32],
            'align_corners': True
        }
        
        # API settings
        self.API_HOST = "0.0.0.0"
        self.API_PORT = 8000
        self.API_RELOAD = True # Set to False for production
        self.API_LOG_LEVEL = "info"