# utils/detection.py

import cv2
import numpy as np
import logging
from typing import Tuple, Dict
from skimage.metrics import structural_similarity as ssim # Ensure this import is present

logger = logging.getLogger(__name__)

class ChangeDetector:
    """Handles change detection using various methods."""

    def __init__(self, changeformer_model=None, device=None):
        self.changeformer_model = changeformer_model
        self.device = device
        self.image_processor = None # Will be set by main.py after initialization

    def set_image_processor(self, processor):
        self.image_processor = processor

    def detect_changes_basic(self, img1: np.ndarray, img2: np.ndarray, sensitivity: float = 0.5, min_area: int = 50) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Detects changes between two images using a basic pixel difference method.
        This serves as a fallback or simpler alternative to ChangeFormer.
        
        Args:
            img1: Reference image (HWC, BGR).
            img2: Comparison image (HWC, BGR).
            sensitivity: Threshold for pixel difference (0.0 to 1.0).
            min_area: Minimum area for a detected change region.
            
        Returns:
            Tuple: (result_image_with_boxes, binary_mask, confidence_scores)
        """
        logger.info("Performing basic change detection (pixel difference)...")
        
        # Convert to grayscale for difference calculation
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Compute absolute difference
        diff = cv2.absdiff(img1_gray, img2_gray)

        # Apply threshold to get a binary mask
        # Scale sensitivity from 0-1 to 0-255 for OpenCV threshold
        threshold_val = int(sensitivity * 255)
        _, binary_mask = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)
        
        # Convert to 0/1 mask for consistency with advanced models
        binary_mask = (binary_mask / 255).astype(np.uint8)

        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # Find contours and draw bounding boxes
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result_img_with_boxes = img2.copy() # Draw on the comparison image
        confidence_scores_list = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence (e.g., mean pixel difference in the region)
                region_diff_mean = np.mean(diff[y:y+h, x:x+w]) / 255.0
                confidence_scores_list.append(float(region_diff_mean))
                
                # Draw bounding box
                cv2.rectangle(result_img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2) # Red color
                
                # Add confidence text
                cv2.putText(result_img_with_boxes, f'{region_diff_mean:.2f}', 
                            (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            
                # Semi-transparent overlay
                overlay = result_img_with_boxes.copy()
                cv2.fillPoly(overlay, [contour], (0, 0, 255))
                result_img_with_boxes = cv2.addWeighted(result_img_with_boxes, 0.8, overlay, 0.2, 0)
        
        confidence_summary = {
            "mean_confidence": float(np.mean(confidence_scores_list)) if confidence_scores_list else 0.0,
            "max_confidence": float(np.max(confidence_scores_list)) if confidence_scores_list else 0.0,
            "num_changes": len(confidence_scores_list)
        }
        
        logger.info(f"Basic change detection complete. Found {confidence_summary['num_changes']} changes.")
        return result_img_with_boxes, binary_mask, confidence_summary



    def detect_changes_ssim(self, img1: np.ndarray, img2: np.ndarray, sensitivity: float = 0.5, min_area: int = 50) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Detects changes between two images using the Structural Similarity Index (SSIM).
        This version is adapted from user's known-working SSIM logic.
        Requires images to be aligned.

        Args:
            img1: Reference image (HWC, BGR).
            img2: Comparison image (HWC, BGR).
            sensitivity: This parameter is currently ignored for thresholding,
                         as the threshold is fixed to 200 to match user's working code.
                         For min_area, use the provided min_area.
            min_area: Minimum area for a detected change region.

        Returns:
            Tuple: (result_image_with_boxes, binary_mask, confidence_scores)
        """
        logger.info("Performing SSIM change detection (using user's working logic)...")

        # Convert to grayscale for SSIM calculation
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Calculate SSIM. full=True returns the difference map.
        # data_range specifies the range of pixel values (0-255 for 8-bit grayscale)
        # Make sure skimage is installed: pip install scikit-image
        (score, diff_map) = ssim(img1_gray, img2_gray, full=True, data_range=255)

        # Scale diff_map to 0-255
        diff_map = (diff_map * 255).astype("uint8")

        # --- ADAPTING USER'S WORKING THRESHOLD LOGIC ---
        # Your code: thresh = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY_INV)[1]
        # This means pixels <= 200 are marked as changes.
        _, binary_mask = cv2.threshold(diff_map, 200, 255, cv2.THRESH_BINARY_INV)

        # Convert to 0/1 mask
        binary_mask = (binary_mask / 255).astype(np.uint8)

        # Apply morphological operations (keeping a minimal (3,3) kernel as in my previous version)
        # Your original working SSIM code did not have explicit morphology here.
        # If you find too much noise, try removing these two lines:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel) # Still might remove very small features

        # Find contours and draw bounding boxes
        contours, _ = cv2.findContours(binary_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result_img_with_boxes = img2.copy()
        confidence_scores_list = []

        for contour in contours:
            area = cv2.contourArea(contour)
            # Use the min_area provided by the user in the API call
            if area > min_area: # Changed from your hardcoded 40 to use the API parameter
                x, y, w, h = cv2.boundingRect(contour)

                # Confidence can be based on mean SSIM difference in the region
                region_ssim_diff_mean = np.mean(diff_map[y:y+h, x:x+w]) / 255.0
                confidence_scores_list.append(float(region_ssim_diff_mean))

                cv2.rectangle(result_img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green color for SSIM changes
                cv2.putText(result_img_with_boxes, f'{region_ssim_diff_mean:.2f}', 
                                (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                overlay = result_img_with_boxes.copy()
                cv2.fillPoly(overlay, [contour], (0, 255, 0)) # Green overlay
                result_img_with_boxes = cv2.addWeighted(result_img_with_boxes, 0.8, overlay, 0.2, 0)

        confidence_summary = {
            "mean_confidence": float(np.mean(confidence_scores_list)) if confidence_scores_list else 0.0,
            "max_confidence": float(np.max(confidence_scores_list)) if confidence_scores_list else 0.0,
            "num_changes": len(confidence_scores_list)
        }

        logger.info(f"SSIM change detection complete. Found {confidence_summary['num_changes']} changes.")
        return result_img_with_boxes, binary_mask, confidence_summary

    # Note: ChangeFormer detection logic is now primarily in main.py
    # because it needs direct access to the model which is initialized there.
    # The `main.py` will call `self.detect_changes_advanced` if the model is loaded.
    # This `ChangeDetector` class remains for basic detection and future expansion.