# models/superpoint.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def simple_nms(scores, nms_radius: int):
    """Apply non-maximum suppression to keypoints."""
    assert(nms_radius >= 0)

    def max_pool(x):
        return F.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2): # Apply twice for better suppression
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """Remove keypoints close to borders."""
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    """Keep only the k keypoints with highest scores."""
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Sample descriptors at keypoint locations."""
    # Keypoints are (N, 2) in (x, y) format
    # Descriptors are (B, C, H, W)
    b, c, h, w = descriptors.shape
    
    # Normalize keypoints to [-1, 1] for F.grid_sample
    # Keypoints are in (x, y) = (col, row)
    # grid_sample expects (x, y) where x is horizontal, y is vertical
    # and coordinates are in [-1, 1]
    
    # Adjust keypoints to be relative to the feature map grid
    # Add 0.5 to center on pixel, then scale to [-1, 1]
    keypoints_normalized = keypoints.clone()
    keypoints_normalized[:, 0] = (keypoints_normalized[:, 0] / (w * s / 2)) * 2 - 1
    keypoints_normalized[:, 1] = (keypoints_normalized[:, 1] / (h * s / 2)) * 2 - 1

    # Reshape keypoints for grid_sample: (B, 1, N, 2)
    keypoints_normalized = keypoints_normalized.view(b, 1, -1, 2)

    # Sample descriptors
    descriptors = F.grid_sample(
        descriptors, keypoints_normalized, mode='bilinear', align_corners=True)
    
    # Reshape back to (B, C, N) and normalize
    descriptors = F.normalize(descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor
    
    SuperPoint: Self-Supervised Interest Point Detection and Description.
    Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. In CVPRW, 2019.
    https://arxiv.org/abs/1712.07629
    """
    
    default_config = {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1, # -1 means no limit
        'remove_borders': 4,
    }

    def __init__(self, config={}):
        super().__init__()
        self.config = {**self.default_config, **config}
        
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        # Shared encoder
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        # Detector head
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0) # 64+1 for pixel-wise softmax

        # Descriptor head
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, 256, kernel_size=1, stride=1, padding=0) # 256-dim descriptors

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('SuperPoint model initialized.')

    def forward(self, data):
        """
        Compute keypoints, scores, and descriptors for an image.
        
        Args:
            data (dict): A dictionary containing 'image' tensor (B, 1, H, W).
                         If image has 3 channels, it will be converted to grayscale.
        
        Returns:
            dict: A dictionary containing:
                - 'keypoints': (B, N, 2) tensor of keypoint coordinates (x, y)
                - 'scores': (B, N) tensor of keypoint scores
                - 'descriptors': (B, D, N) tensor of descriptors
        """
        # Shared encoder
        image = data['image']
        if image.shape[1] == 3:  # RGB to grayscale
            image = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
            image = image.unsqueeze(1) # Add channel dimension back
        
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Detector head
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa) # (B, 65, H/8, W/8)
        scores = F.softmax(scores, 1)[:, :-1] # Remove last channel (no interest point)
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8) # Reshape to (B, H/8, W/8, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8) # Reshape to (B, H, W)
        scores = scores.unsqueeze(1) # Add channel dimension (B, 1, H, W)

        # Descriptor head
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa) # (B, 256, H/8, W/8)
        descriptors = F.normalize(descriptors, p=2, dim=1) # L2 normalize descriptors

        # Extract keypoints
        keypoints = []
        scores_list = []
        descriptors_list = []

        for i in range(b):
            # Apply NMS to scores
            kpts_scores = simple_nms(scores[i], self.config['nms_radius'])
            
            # Threshold scores
            mask = kpts_scores > self.config['keypoint_threshold']
            kpts_coords = mask.nonzero(as_tuple=False) # (N, 2) in (row, col) format
            kpts_scores = kpts_scores[mask] # (N,)

            # Convert (row, col) to (x, y)
            kpts_coords = kpts_coords[:, [1, 0]] # (N, 2) in (x, y) format

            # Remove border keypoints
            if self.config['remove_borders'] > 0:
                kpts_coords, kpts_scores = remove_borders(
                    kpts_coords, kpts_scores, self.config['remove_borders'],
                    image.shape[2], image.shape[3]) # Use original image dimensions

            # Keep top-k keypoints
            if self.config['max_keypoints'] >= 0:
                kpts_coords, kpts_scores = top_k_keypoints(
                    kpts_coords, kpts_scores, self.config['max_keypoints'])
            
            # Sample descriptors at keypoint locations
            if len(kpts_coords) > 0:
                desc = sample_descriptors(
                    kpts_coords[None].float(), descriptors[i:i+1], s=8) # s=8 is the stride of the encoder
                descriptors_list.append(desc.squeeze(0)) # Remove batch dim
            else:
                descriptors_list.append(torch.empty(256, 0, device=self.device)) # Empty tensor if no keypoints

            keypoints.append(kpts_coords)
            scores_list.append(kpts_scores)

        # Pad keypoints, scores, descriptors to the same size for batching (if needed)
        # For simplicity, we'll return lists of tensors, which can be handled by SuperGlue
        return {
            'keypoints': keypoints, # List of (N, 2) tensors
            'scores': scores_list, # List of (N,) tensors
            'descriptors': descriptors_list # List of (D, N) tensors
        }