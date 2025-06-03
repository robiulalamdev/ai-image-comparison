# utils/alignment.py

import cv2
import numpy as np
import torch
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class ImageAligner:
    """Handles image alignment using various methods."""

    def __init__(self, superpoint_model=None, superglue_model=None, device=None):
        self.superpoint_model = superpoint_model
        self.superglue_model = superglue_model
        self.device = device
        self.image_processor = None # Will be set by main.py after initialization

    def set_image_processor(self, processor):
        self.image_processor = processor

    def align_images_basic(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Aligns two images using basic feature matching (ORB/SIFT + RANSAC).
        This serves as a fallback or a simpler alternative to SuperGlue.
        
        Args:
            img1: Reference image (HWC, BGR).
            img2: Image to be aligned (HWC, BGR).
            
        Returns:
            Tuple: (img1, aligned_img2, confidence)
        """
        logger.info("Performing basic image alignment (ORB/SIFT)...")
        # Convert to grayscale
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=2000)

        # Find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1_gray, None)
        kp2, des2 = orb.detectAndCompute(img2_gray, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            logger.warning("Not enough features found for basic alignment. Returning original images.")
            return img1, img2, 0.0

        # Use BFMatcher to find the best matches
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        # Find homography
        if len(matches) < 4:
            logger.warning("Not enough good matches for homography. Returning original images.")
            return img1, img2, 0.0

        H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

        if H is None:
            logger.warning("Homography estimation failed for basic alignment. Returning original images.")
            return img1, img2, 0.0

        # Warp image2 to align with image1
        h, w = img1.shape[:2]
        aligned_img2 = cv2.warpPerspective(img2, H, (w, h))

        # Calculate alignment confidence (e.g., ratio of inliers)
        inlier_ratio = np.sum(mask) / len(matches) if mask is not None else 0.0
        
        logger.info(f"Basic alignment complete with confidence: {inlier_ratio:.2f}")
        return img1, aligned_img2, float(inlier_ratio)

    # Note: SuperPoint/SuperGlue alignment logic is now primarily in main.py
    # because it needs direct access to the models which are initialized there.
    # The `main.py` will call `self.align_with_superglue` if models are loaded.
    # This `ImageAligner` class remains for basic alignment and future expansion.