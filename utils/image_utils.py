# utils/image_utils.py

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
from skimage.measure import label, regionprops
from typing import Tuple

class ImageProcessor:
    """Utility class for image preprocessing and postprocessing."""

    def __init__(self):
        self.normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.denormalize_transform = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

    def bytes_to_opencv(self, image_bytes: bytes) -> np.ndarray:
        """Converts image bytes to OpenCV (BGR) format."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_np is None:
            raise ValueError("Could not decode image bytes. Invalid image format.")
        return img_np

    def opencv_to_bytes(self, image_np: np.ndarray, format: str = 'jpeg') -> bytes:
        """Converts OpenCV image (BGR) to bytes."""
        is_success, buffer = cv2.imencode(f".{format}", image_np)
        if not is_success:
            raise ValueError("Could not encode image to bytes.")
        return io.BytesIO(buffer).getvalue()

    def preprocess_image(self, image_np: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """
        Resizes and converts image to RGB.
        Args:
            image_np: OpenCV image (HWC, BGR).
            target_size: Desired (width, height) for resizing.
        Returns:
            Processed image (HWC, RGB).
        """
        # Resize
        resized_img = cv2.resize(image_np, target_size, interpolation=cv2.INTER_LINEAR)
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        return rgb_img

    def numpy_to_tensor_grayscale(self, image_np: np.ndarray) -> torch.Tensor:
        """Converts a grayscale NumPy array (H, W) to a PyTorch tensor (1, 1, H, W)."""
        # Add channel dimension, then batch dimension
        tensor = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0) / 255.0
        return tensor

    def numpy_to_tensor_rgb(self, image_np: np.ndarray) -> torch.Tensor:
        """Converts an RGB NumPy array (H, W, 3) to a PyTorch tensor (1, 3, H, W) and normalizes."""
        # Convert HWC to CHW, then add batch dimension
        tensor = torch.from_numpy(image_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = self.normalize_transform(tensor)
        return tensor

    def tensor_to_numpy_rgb(self, tensor: torch.Tensor) -> np.ndarray:
        """Converts a PyTorch tensor (1, 3, H, W) to an RGB NumPy array (H, W, 3)."""
        tensor = self.denormalize_transform(tensor.squeeze(0).cpu()) # Remove batch dim and denormalize
        tensor = torch.clamp(tensor, 0, 1) # Clamp values to [0, 1]
        numpy_img = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return numpy_img

    def prepare_changeformer_input(self, img1_np: np.ndarray, img2_np: np.ndarray, device: torch.device) -> torch.Tensor:
        """
        Prepares input tensor for ChangeFormer.
        Assumes img1_np and img2_np are already preprocessed (e.g., resized to 512x512 RGB).
        ChangeFormer often takes concatenated RGB images.
        """
        # Convert to PIL for standard transforms
        img1_pil = Image.fromarray(img1_np)
        img2_pil = Image.fromarray(img2_np)

        transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize_transform # Apply normalization
        ])

        tensor1 = transform(img1_pil).unsqueeze(0) # (1, 3, H, W)
        tensor2 = transform(img2_pil).unsqueeze(0) # (1, 3, H, W)

        # Concatenate along channel dimension (B, 6, H, W)
        combined_tensor = torch.cat([tensor1, tensor2], dim=1).to(device)
        return combined_tensor

    def draw_differences(self, image_pil: Image.Image, change_mask_np: np.ndarray, min_area: int = 50) -> Image.Image:
        """
        Draws red bounding boxes around detected changes on a PIL Image.
        Args:
            image_pil: The original PIL Image (reference image) to draw on.
            change_mask_np: Binary NumPy array (H, W) where 1 indicates change.
            min_area: Minimum area for a detected change region to be drawn.
        Returns:
            PIL Image with drawn bounding boxes.
        """
        draw = ImageDraw.Draw(image_pil)
        # Convert mask to uint8 for OpenCV operations (label expects integer mask)
        mask_uint8 = (change_mask_np * 255).astype(np.uint8)

        # Find connected components and their bounding boxes
        labeled_mask = label(mask_uint8, connectivity=2) # Use 8-connectivity
        for region in regionprops(labeled_mask):
            if region.area > min_area: # Filter small noisy regions
                min_row, min_col, max_row, max_col = region.bbox
                # Draw rectangle (PIL uses (left, upper, right, lower))
                draw.rectangle([(min_col, min_row), (max_col, max_row)], outline="red", width=3)
        return image_pil