# models/superglue.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Helper functions from SuperGlue's original implementation (simplified)
def MLP(channels: list, do_bn: bool=True):
    """Multi-layer perceptron."""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Linear(channels[i - 1], channels[i]))
        if i < (n - 1) and do_bn:
            layers.append(nn.BatchNorm1d(channels[i]))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def normalize_keypoints(keypoints, image_size):
    """Normalize keypoints to [-1, 1] range based on image size."""
    h, w = image_size
    # Scale keypoints to [0, 1]
    keypoints_norm = keypoints / torch.tensor([[w - 1, h - 1]], dtype=keypoints.dtype, device=keypoints.device)
    # Scale to [-1, 1]
    keypoints_norm = keypoints_norm * 2 - 1
    return keypoints_norm

class AttentionalPropagation(nn.Module):
    """Attentional Propagation Module for SuperGlue."""
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        assert feature_dim % num_heads == 0
        self.head_dim = feature_dim // num_heads

        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, query, key, value):
        """
        Args:
            query (torch.Tensor): (B, N, D)
            key (torch.Tensor): (B, M, D)
            value (torch.Tensor): (B, M, D)
        """
        B, N, D = query.shape
        M = key.shape[1]

        # Project and reshape for multi-head attention
        query_h = self.q_proj(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, N, D/H)
        key_h = self.k_proj(key).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)     # (B, H, M, D/H)
        value_h = self.v_proj(value).view(B, M, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, M, D/H)

        # Scaled Dot-Product Attention
        scores = torch.matmul(query_h, key_h.transpose(-2, -1)) / (self.head_dim ** 0.5) # (B, H, N, M)
        attn = F.softmax(scores, dim=-1)

        # Weighted sum of values
        out = torch.matmul(attn, value_h) # (B, H, N, D/H)
        out = out.transpose(1, 2).contiguous().view(B, N, D) # (B, N, D)

        # Output projection and residual connection
        out = self.out_proj(out)
        out = self.norm(query + out) # Residual connection and LayerNorm
        return out

class SuperGlue(nn.Module):
    """SuperGlue Graph Neural Network for Feature Matching."""
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor', # 'indoor' or 'outdoor'
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross', 'self', 'cross'], # Example GNN layers
        'num_gnn_layers': 9, # Number of GNN layers (total)
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
        'num_heads': 4, # For multi-head attention
    }

    def __init__(self, config={}):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.descriptor_dim = self.config['descriptor_dim']
        self.num_gnn_layers = self.config['num_gnn_layers']
        self.sinkhorn_iterations = self.config['sinkhorn_iterations']
        self.match_threshold = self.config['match_threshold']

        # Keypoint MLP encoder
        self.kpt_encoder = MLP(self.config['keypoint_encoder'] + [self.descriptor_dim])

        # Attentional propagation modules
        self.self_attn = AttentionalPropagation(self.descriptor_dim, self.config['num_heads'])
        self.cross_attn = AttentionalPropagation(self.descriptor_dim, self.config['num_heads'])

        # Final MLP for scores
        self.final_mlp = MLP([self.descriptor_dim, self.descriptor_dim, 1])

        # Load weights (simplified - in a real scenario, you'd load from a .pth file)
        # The actual weights are loaded in main.py based on settings.
        print(f"SuperGlue model initialized with weights: {self.config['weights']}")

    def forward(self, data):
        """
        Perform feature matching using SuperGlue.
        
        Args:
            data (dict): Dictionary containing:
                - 'image0': (B, 1, H, W) grayscale image 0 tensor
                - 'image1': (B, 1, H, W) grayscale image 1 tensor
                - 'keypoints0': list of (N, 2) keypoint tensors for image 0
                - 'scores0': list of (N,) score tensors for image 0
                - 'descriptors0': list of (D, N) descriptor tensors for image 0
                - 'keypoints1': list of (M, 2) keypoint tensors for image 1
                - 'scores1': list of (M,) score tensors for image 1
                - 'descriptors1': list of (D, M) descriptor tensors for image 1
        
        Returns:
            dict: Dictionary containing:
                - 'matches0': (B, N) tensor of matches (index of matched keypoint in image1, or -1)
                - 'matching_scores0': (B, N) tensor of matching scores
        """
        # Assume batch size is 1 for simplicity in this example
        # For batch processing, you'd need padding or a more complex graph representation.
        assert data['keypoints0'][0].shape[0] > 0 and data['keypoints1'][0].shape[0] > 0, \
            "SuperGlue requires at least one keypoint in each image."

        kpts0 = data['keypoints0'][0].float() # (N, 2)
        kpts1 = data['keypoints1'][0].float() # (M, 2)
        desc0 = data['descriptors0'][0].transpose(0, 1).float() # (N, D)
        desc1 = data['descriptors1'][0].transpose(0, 1).float() # (M, D)
        
        # Normalize keypoints
        h0, w0 = data['image0'].shape[2:]
        h1, w1 = data['image1'].shape[2:]
        kpts0_norm = normalize_keypoints(kpts0, (h0, w0))
        kpts1_norm = normalize_keypoints(kpts1, (h1, w1))

        # Encode keypoints with MLP
        kpts0_enc = self.kpt_encoder(kpts0_norm)
        kpts1_enc = self.kpt_encoder(kpts1_norm)

        # Add keypoint encoding to descriptors
        feat0 = desc0 + kpts0_enc
        feat1 = desc1 + kpts1_enc

        # Attentional Propagation
        for i in range(self.num_gnn_layers):
            # Self-attention
            feat0 = self.self_attn(feat0, feat0, feat0)
            feat1 = self.self_attn(feat1, feat1, feat1)
            
            # Cross-attention
            new_feat0 = self.cross_attn(feat0, feat1, feat1)
            new_feat1 = self.cross_attn(feat1, feat0, feat0)
            feat0 = new_feat0
            feat1 = new_feat1

        # Compute affinity matrix
        scores = torch.matmul(feat0, feat1.transpose(-2, -1)) / (self.descriptor_dim ** 0.5)

        # Sinkhorn algorithm for optimal transport
        # Add dummy rows/cols for unmatched points
        scores = F.pad(scores, (0, 1, 0, 1), 'constant', 0) # Pad with zeros
        scores[:, -1, :] = self.match_threshold # Last row for unmatched in image0
        scores[:, :, -1] = self.match_threshold # Last col for unmatched in image1

        # Apply Sinkhorn algorithm (simplified)
        # A full Sinkhorn implementation is complex; here we use a basic iterative softmax
        # For a true Sinkhorn, refer to the original SuperGlue repo.
        for _ in range(self.sinkhorn_iterations):
            scores = F.softmax(scores, dim=1) # Row-wise softmax
            scores = F.softmax(scores, dim=2) # Column-wise softmax

        # Extract matches
        # The last row/column represent unmatched points
        matches0 = torch.argmax(scores[:, :-1, :-1], dim=2) # Matches for image0
        matching_scores0 = scores[:, :-1, :-1].max(dim=2)[0] # Scores for image0 matches

        # Filter out low-confidence matches and unmatched points
        unmatched_mask = (matching_scores0 < self.match_threshold)
        matches0[unmatched_mask] = -1 # Mark as unmatched

        return {
            'matches0': matches0, # (B, N)
            'matching_scores0': matching_scores0 # (B, N)
        }