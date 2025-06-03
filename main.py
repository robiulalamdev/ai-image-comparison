#!/usr/bin/env python3
"""
Advanced AI Image Comparison API
Includes SuperPoint, SuperGlue, and ChangeFormer integration
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import requests
import io
import base64
import logging
import asyncio
from typing import Tuple, List, Dict, Optional
import json
import time
from pathlib import Path
import hashlib

# Add project root to sys.path to allow importing custom modules
# This assumes main.py is at the root of the project
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from config.settings import Settings
from models.superpoint import SuperPoint
from models.superglue import SuperGlue
from models.changeformer import ChangeFormer
from utils.image_utils import ImageProcessor
from utils.alignment import ImageAligner
from utils.detection import ChangeDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedImageComparisonAPI:
    """Advanced AI-powered image comparison API"""
    
    def __init__(self):
        self.settings = Settings()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize directories
        self.setup_directories()
        
        # Initialize models
        self.superpoint = None
        self.superglue = None
        self.changeformer = None
        self.load_models() # This method will load models or fall back
        
        # Initialize processors
        self.image_processor = ImageProcessor()
        self.aligner = ImageAligner(self.superpoint, self.superglue, self.device)
        self.detector = ChangeDetector(self.changeformer, self.device)
        
        # Initialize FastAPI
        self.app = FastAPI(
            title="Advanced AI Image Comparison API",
            description="Professional-grade image comparison with SuperPoint, SuperGlue, and ChangeFormer",
            version="2.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"], # Allows all origins for development
            allow_credentials=True,
            allow_methods=["*"], # Allows all methods
            allow_headers=["*"], # Allows all headers
        )
        
        self.setup_routes()
        
        # Processing cache
        self.cache = {}
        self.max_cache_size = 100
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            'models/pretrained',
            'uploads',
            'results',
            'cache',
            'logs'
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_models(self):
        """Load all AI models"""
        logger.info("Loading AI models...")
        
        try:
            # Load SuperPoint
            self.superpoint = SuperPoint(self.settings.superpoint_config).eval().to(self.device)
            superpoint_path = self.settings.superpoint_weights_path
            if os.path.exists(superpoint_path):
                self.superpoint.load_state_dict(torch.load(superpoint_path, map_location=self.device))
                logger.info("✅ SuperPoint loaded from pretrained weights")
            else:
                logger.warning("⚠️ SuperPoint pretrained weights not found. Please run download_models.py.")
                self.superpoint = None # Set to None if not loaded
            
            # Load SuperGlue
            self.superglue = SuperGlue(self.settings.superglue_config).eval().to(self.device)
            superglue_path = self.settings.superglue_weights_path
            if os.path.exists(superglue_path):
                self.superglue.load_state_dict(torch.load(superglue_path, map_location=self.device))
                logger.info("✅ SuperGlue loaded from pretrained weights")
            else:
                logger.warning("⚠️ SuperGlue pretrained weights not found. Please run download_models.py.")
                self.superglue = None # Set to None if not loaded
            
            # Load ChangeFormer
            # ChangeFormer model initialization needs to match the architecture from the original repo
            # For simplicity, we'll use a basic initialization.
            # In a real-world scenario, you'd integrate the full ChangeFormer model definition
            # from its source (e.g., by cloning its repo and importing its specific classes).
            # The `download_models.py` will handle cloning the ChangeFormer repo.
            # Here, we assume a simplified `ChangeFormer` class is defined in `models/changeformer.py`
            # that can load the weights.
            
            # Placeholder for ChangeFormer config if it's not directly from a `get_config` function
            # You might need to adjust these parameters based on the actual ChangeFormer model you use
            changeformer_model_params = {
                'input_nc': 3, # Assuming two 3-channel images are concatenated before the model, resulting in 6 channels for the first conv layer.
                               # However, the `prepare_changeformer_input` function concatenates them to 6 channels.
                               # So, the input_nc for ChangeFormer should be 6.
                'output_nc': 2, # Binary change detection (change/no-change)
                'embed_dim': 256,
                'depths': [2, 2, 2, 2], # Example depths, adjust based on actual model
                'num_heads': [4, 8, 16, 32], # Example num_heads, adjust based on actual model
                'mlp_ratio': 4.,
                'qkv_bias': True,
                'qk_scale': None,
                'drop_rate': 0.,
                'attn_drop_rate': 0.,
                'drop_path_rate': 0.1,
                'norm_layer': nn.LayerNorm,
                'patch_norm': True,
                'use_checkpoint': False,
                'decoder_embed_dim': 256,
                'decoder_depths': [2, 2, 2, 2],
                'decoder_num_heads': [4, 8, 16, 32],
                'align_corners': True
            }
            self.changeformer = ChangeFormer(**changeformer_model_params).eval().to(self.device)

            changeformer_path = self.settings.changeformer_weights_path
            if os.path.exists(changeformer_path):
                state_dict = torch.load(changeformer_path, map_location=self.device)
                # Remove 'module.' prefix if it exists (common for DataParallel trained models)
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                self.changeformer.load_state_dict(new_state_dict, strict=True)
                logger.info("✅ ChangeFormer loaded from pretrained weights")
            else:
                logger.warning("⚠️ ChangeFormer pretrained weights not found. Please run download_models.py.")
                self.changeformer = None # Set to None if not loaded
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.load_fallback_models() # Attempt to load fallback models if an error occurs
    
    def load_fallback_models(self):
        """Load fallback models if advanced models fail"""
        logger.info("Loading fallback models...")
        
        # Basic feature detector (e.g., OpenCV's SIFT/ORB if SuperPoint/SuperGlue fail)
        # For this example, we'll just set them to None, which will trigger basic alignment/detection
        self.superpoint = None
        self.superglue = None
        self.changeformer = None
        
        logger.info("✅ Fallback models loaded (basic functionality will be used)")
    
    def setup_routes(self):
        """Setup all API routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root_html():
            """Root endpoint providing a simple HTML form to upload images."""
            return """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Advanced AI Image Comparison</title>
                <script src="https://cdn.tailwindcss.com"></script>
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
                <style>
                    body {
                        font-family: 'Inter', sans-serif;
                        background-color: #f3f4f6;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                        padding: 20px;
                    }
                    .container {
                        background-color: #ffffff;
                        border-radius: 12px;
                        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
                        padding: 30px;
                        max-width: 800px;
                        width: 100%;
                        text-align: center;
                    }
                    .button-primary {
                        background-color: #4f46e5;
                        color: white;
                        padding: 12px 24px;
                        border-radius: 8px;
                        font-weight: 600;
                        transition: background-color 0.3s ease;
                        cursor: pointer;
                    }
                    .button-primary:hover {
                        background-color: #4338ca;
                    }
                    .input-file {
                        display: block;
                        width: 100%;
                        padding: 10px;
                        border: 2px dashed #d1d5db;
                        border-radius: 8px;
                        text-align: center;
                        cursor: pointer;
                        background-color: #f9fafb;
                        color: #4b5563;
                        transition: border-color 0.3s ease;
                    }
                    .input-file:hover {
                        border-color: #9ca3af;
                    }
                    .image-preview {
                        max-width: 100%;
                        height: auto;
                        border-radius: 8px;
                        border: 1px solid #e5e7eb;
                        margin-top: 15px;
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                    }
                    .spinner {
                        border: 4px solid rgba(0, 0, 0, 0.1);
                        border-left-color: #4f46e5;
                        border-radius: 50%;
                        width: 30px;
                        height: 30px;
                        animation: spin 1s linear infinite;
                        margin: 20px auto;
                    }
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1 class="text-3xl font-bold text-gray-800 mb-6">Advanced AI Image Comparison Tool</h1>
                    <p class="text-gray-600 mb-8">Upload two images (reference and comparison) to detect and highlight differences.</p>

                    <form id="uploadForm" class="space-y-6">
                        <div>
                            <label for="image1" class="block text-lg font-medium text-gray-700 mb-2">Reference Image (Image A):</label>
                            <input type="file" id="image1" name="image1" accept="image/*" required class="input-file">
                            <img id="preview1" class="image-preview hidden" src="#" alt="Image A Preview">
                        </div>
                        <div>
                            <label for="image2" class="block text-lg font-medium text-gray-700 mb-2">Comparison Image (Image B):</label>
                            <input type="file" id="image2" name="image2" accept="image/*" required class="input-file">
                            <img id="preview2" class="image-preview hidden" src="#" alt="Image B Preview">
                        </div>
                        <div class="flex flex-col md:flex-row md:space-x-4 space-y-4 md:space-y-0">
                            <div class="flex-1">
                                <label for="sensitivity" class="block text-lg font-medium text-gray-700 mb-2">Sensitivity (0.0 - 1.0):</label>
                                <input type="number" id="sensitivity" name="sensitivity" value="0.5" step="0.01" min="0" max="1" class="w-full p-2 border border-gray-300 rounded-md">
                            </div>
                            <div class="flex-1">
                                <label for="min_area" class="block text-lg font-medium text-gray-700 mb-2">Min Change Area (pixels):</label>
                                <input type="number" id="min_area" name="min_area" value="100" step="1" min="0" class="w-full p-2 border border-gray-300 rounded-md">
                            </div>
                            <div class="flex-1">
                                <label for="alignment_method" class="block text-lg font-medium text-gray-700 mb-2">Alignment Method:</label>
                                <select id="alignment_method" name="alignment_method" class="w-full p-2 border border-gray-300 rounded-md">
                                    <option value="superglue">SuperGlue (Advanced)</option>
                                    <option value="basic">Basic (ORB/SIFT)</option>
                                </select>
                            </div>
                            <div class="flex-1">
                                <label for="detect_method" class="block text-lg font-medium text-gray-700 mb-2">Detection Method:</label>
                                <select id="detect_method" name="detect_method" class="w-full p-2 border border-gray-300 rounded-md">
                                    <option value="Advance">Advance</option>
                                    <option value="Basic">Basic</option>
                                </select>
                            </div>
                        </div>
                        <button type="submit" class="button-primary w-full flex items-center justify-center mt-6">
                            <span id="buttonText">Compare Images</span>
                            <div id="spinner" class="spinner hidden ml-3"></div>
                        </button>
                    </form>

                    <div id="resultContainer" class="mt-8 hidden">
                        <h2 class="text-2xl font-semibold text-gray-800 mb-4">Comparison Result:</h2>
                        <img id="resultImage" class="image-preview" src="#" alt="Comparison Result">
                        <div id="metadata" class="mt-4 text-left text-gray-700 text-sm"></div>
                    </div>

                    <div id="errorMessage" class="mt-4 text-red-600 font-medium hidden"></div>
                </div>

                <script>
                    function previewImage(event, previewId) {
                        const [file] = event.target.files;
                        if (file) {
                            document.getElementById(previewId).src = URL.createObjectURL(file);
                            document.getElementById(previewId).classList.remove('hidden');
                        }
                    }

                    document.getElementById('image1').addEventListener('change', (e) => previewImage(e, 'preview1'));
                    document.getElementById('image2').addEventListener('change', (e) => previewImage(e, 'preview2'));

                    document.getElementById('uploadForm').addEventListener('submit', async function(event) {
                        event.preventDefault();

                        const image1File = document.getElementById('image1').files[0];
                        const image2File = document.getElementById('image2').files[0];
                        const sensitivity = document.getElementById('sensitivity').value;
                        const minArea = document.getElementById('min_area').value;
                        const alignmentMethod = document.getElementById('alignment_method').value;
                        const detectMethod = document.getElementById('detect_method').value;

                        const resultContainer = document.getElementById('resultContainer');
                        const resultImage = document.getElementById('resultImage');
                        const metadataDiv = document.getElementById('metadata');
                        const errorMessage = document.getElementById('errorMessage');
                        const buttonText = document.getElementById('buttonText');
                        const spinner = document.getElementById('spinner');

                        resultContainer.classList.add('hidden');
                        errorMessage.classList.add('hidden');
                        metadataDiv.innerHTML = '';
                        buttonText.textContent = 'Processing...';
                        spinner.classList.remove('hidden');
                        document.querySelector('button[type="submit"]').disabled = true;

                        if (!image1File || !image2File) {
                            errorMessage.textContent = 'Please select both images.';
                            errorMessage.classList.remove('hidden');
                            buttonText.textContent = 'Compare Images';
                            spinner.classList.add('hidden');
                            document.querySelector('button[type="submit"]').disabled = false;
                            return;
                        }

                        const formData = new FormData();
                        formData.append('reference_image', image1File);
                        formData.append('comparison_image', image2File);
                        formData.append('sensitivity', sensitivity);
                        formData.append('min_area', minArea);
                        formData.append('alignment_method', alignmentMethod);
                        formData.append('detect_method', detectMethod);

                        try {
                            const response = await fetch('/compare', {
                                method: 'POST',
                                body: formData,
                            });

                            if (!response.ok) {
                                const errorData = await response.json();
                                throw new Error(errorData.detail || 'An error occurred during comparison.');
                            }

                            const blob = await response.blob();
                            const imageUrl = URL.createObjectURL(blob);
                            resultImage.src = imageUrl;
                            resultContainer.classList.remove('hidden');

                            // Extract metadata from headers
                            const metadataHeader = response.headers.get('X-Metadata');
                            if (metadataHeader) {
                                try {
                                    const metadata = JSON.parse(metadataHeader);
                                    let metadataHtml = '<strong>Comparison Details:</strong><br>';
                                    for (const key in metadata) {
                                        if (metadata.hasOwnProperty(key)) {
                                            let value = metadata[key];
                                            if (typeof value === 'number') {
                                                value = value.toFixed(3); // Format numbers
                                            } else if (typeof value === 'object' && value !== null) {
                                                value = JSON.stringify(value, null, 2); // Pretty print objects
                                            }
                                            metadataHtml += `<strong>${key.replace(/_/g, ' ')}:</strong> ${value}<br>`;
                                        }
                                    }
                                    metadataDiv.innerHTML = metadataHtml;
                                } catch (e) {
                                    console.error("Failed to parse metadata header:", e);
                                    metadataDiv.textContent = "Metadata available but could not be parsed.";
                                }
                            } else {
                                metadataDiv.textContent = "No metadata available.";
                            }

                        } catch (error) {
                            errorMessage.textContent = 'Error: ' + error.message;
                            errorMessage.classList.remove('hidden');
                        } finally {
                            buttonText.textContent = 'Compare Images';
                            spinner.classList.add('hidden');
                            document.querySelector('button[type="submit"]').disabled = false;
                        }
                    });
                </script>
            </body>
            </html>
            """
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "models_loaded": {
                    "superpoint": self.superpoint is not None,
                    "superglue": self.superglue is not None,
                    "changeformer": self.changeformer is not None
                }
            }
        
        @self.app.post("/compare")
        async def compare_images_api(
            reference_image: UploadFile = File(...),
            comparison_image: UploadFile = File(...),
            sensitivity: float = 0.5,
            min_area: int = 50,
            alignment_method: str = "superglue", # "superglue" or "basic"
            detect_method: str = "Basic" # "Advance" or "Basic"
        ):
            """Advanced image comparison with multiple options"""


            print(f"Detect Method: {detect_method}")
            
            try:
                # Validate inputs
                if not self.validate_image_file(reference_image):
                    raise HTTPException(status_code=400, detail="Invalid reference image format.")
                if not self.validate_image_file(comparison_image):
                    raise HTTPException(status_code=400, detail="Invalid comparison image format.")
                
                # Generate unique request ID for caching
                request_id = await self.generate_request_id(reference_image, comparison_image, sensitivity, min_area, alignment_method)
                
                # Check cache
                cached_result_path = Path("cache") / f"result_{request_id}.jpg"
                if cached_result_path.exists():
                    logger.info(f"Returning cached result for {request_id}")
                    # Read metadata from a companion JSON file if it exists
                    metadata_path = Path("cache") / f"metadata_{request_id}.json"
                    metadata = {}
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                        except Exception as e:
                            logger.warning(f"Could not load metadata for cached result {request_id}: {e}")
                    
                    return FileResponse(
                        cached_result_path,
                        media_type="image/jpeg",
                        filename="comparison_result.jpg",
                        headers={"X-Metadata": json.dumps(metadata)}
                    )
                
                # Read images
                ref_bytes = await reference_image.read()
                comp_bytes = await comparison_image.read()
                
                # Process images (PIL to OpenCV for consistency)
                ref_img_pil = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
                comp_img_pil = Image.open(io.BytesIO(comp_bytes)).convert("RGB")

                # Keep original sizes for final drawing
                original_ref_size = ref_img_pil.size # (width, height)
                
                # Convert to OpenCV format (HWC, BGR)
                ref_img_cv = cv2.cvtColor(np.array(ref_img_pil), cv2.COLOR_RGB2BGR)
                comp_img_cv = cv2.cvtColor(np.array(comp_img_pil), cv2.COLOR_RGB2BGR)
                
                # Preprocess images for models (e.g., resize to a common inference size)
                # ChangeFormer is typically trained on 256x256 or 512x512
                inference_size = (512, 512)
                ref_processed = cv2.resize(ref_img_cv, inference_size, interpolation=cv2.INTER_LINEAR)
                comp_processed = cv2.resize(comp_img_cv, inference_size, interpolation=cv2.INTER_LINEAR)
                
                # Align images using selected method
                logger.info(f"Aligning images using {alignment_method}")
                if alignment_method == "superglue" and self.superpoint and self.superglue:
                    ref_aligned, comp_aligned, alignment_confidence = await self.align_with_superglue(
                        ref_processed, comp_processed
                    )
                else:
                    # Fallback to basic alignment if SuperGlue is not selected or not loaded
                    ref_aligned, comp_aligned, alignment_confidence = self.aligner.align_images_basic(
                        ref_processed, comp_processed
                    )
                
                # Detect changes
                logger.info("Detecting changes with advanced AI")
                if self.changeformer:
                    logger.info("Using ChangeFormer for change detection")
                    result_img_aligned, change_mask_aligned, confidence_scores = await self.detect_changes_advanced(
                        ref_aligned, comp_aligned, sensitivity, min_area
                    )
                else:
                    logger.info("Using Basic Detection for change detection")
                    # Fallback to basic detection if ChangeFormer is not loaded
                    if detect_method == "Advance":
                        result_img_aligned, change_mask_aligned, confidence_scores = self.detector.detect_changes_ssim(
                            ref_aligned, comp_aligned, sensitivity, min_area
                        )
                    else:
                        result_img_aligned, change_mask_aligned, confidence_scores = self.detector.detect_changes_basic(
                            ref_aligned, comp_aligned, sensitivity, min_area
                        )


                    

                
                # Resize the final result image and mask back to the original reference image's size
                # This ensures the output image has the same dimensions as the original reference image
                result_img_final = cv2.resize(result_img_aligned, original_ref_size, interpolation=cv2.INTER_LINEAR)
                change_mask_final = cv2.resize(change_mask_aligned, original_ref_size, interpolation=cv2.INTER_NEAREST)

                # Redraw differences on the *original* reference image using the *final* mask
                # This ensures the output image is based on the original reference, not the resized/aligned one
                result_pil_image = Image.fromarray(cv2.cvtColor(ref_img_cv, cv2.COLOR_BGR2RGB))
                result_pil_image = self.image_processor.draw_differences(result_pil_image, change_mask_final, min_area)


                # Save result
                result_path = Path("results") / f"result_{request_id}.jpg"
                result_pil_image.save(result_path, format="JPEG")
                
                # Prepare response metadata
                metadata = {
                    "alignment_confidence": float(alignment_confidence),
                    "change_confidence": confidence_scores,
                    "processing_time": time.time(),
                    "method_used": alignment_method,
                    "device_used": str(self.device),
                    "sensitivity": sensitivity,
                    "min_area": min_area
                }
                
                # Cache result and metadata
                self.cache[request_id] = str(result_path) # Store path as string
                if len(self.cache) > self.max_cache_size:
                    # Remove oldest entry from cache and delete files
                    oldest_key = next(iter(self.cache))
                    old_result_path = Path(self.cache.pop(oldest_key))
                    old_metadata_path = Path("cache") / f"metadata_{oldest_key}.json"
                    if old_result_path.exists():
                        os.remove(old_result_path)
                    if old_metadata_path.exists():
                        os.remove(old_metadata_path)
                
                # Save metadata to a companion JSON file in the cache directory
                metadata_cache_path = Path("cache") / f"metadata_{request_id}.json"
                with open(metadata_cache_path, 'w') as f:
                    json.dump(metadata, f)

                # Add metadata to response headers
                return FileResponse(
                    result_path,
                    media_type="image/jpeg",
                    filename="comparison_result.jpg",
                    headers={"X-Metadata": json.dumps(metadata)}
                )
                
            except Exception as e:
                logger.error(f"Error in compare_images API endpoint: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
        
        @self.app.get("/models/status")
        async def model_status():
            """Get detailed model status"""
            return {
                "superpoint": {
                    "loaded": self.superpoint is not None,
                    "device": str(self.device),
                    "weights_path": self.settings.superpoint_weights_path
                },
                "superglue": {
                    "loaded": self.superglue is not None,
                    "device": str(self.device),
                    "weights_path": self.settings.superglue_weights_path
                },
                "changeformer": {
                    "loaded": self.changeformer is not None,
                    "device": str(self.device),
                    "weights_path": self.settings.changeformer_weights_path
                }
            }
    
    async def align_with_superglue(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Align images using SuperPoint + SuperGlue"""
        try:
            # Convert to grayscale for SuperPoint/SuperGlue if not already
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Convert to tensors
            tensor1 = self.image_processor.numpy_to_tensor_grayscale(img1_gray).to(self.device)
            tensor2 = self.image_processor.numpy_to_tensor_grayscale(img2_gray).to(self.device)
            
            # Extract keypoints and descriptors with SuperPoint
            with torch.no_grad():
                pred1 = self.superpoint({'image': tensor1})
                pred2 = self.superpoint({'image': tensor2})
            
            # Prepare inputs for SuperGlue
            data_sg = {
                'image0': tensor1,
                'image1': tensor2,
                'keypoints0': pred1['keypoints'],
                'scores0': pred1['scores'],
                'descriptors0': pred1['descriptors'],
                'keypoints1': pred2['keypoints'],
                'scores1': pred2['scores'],
                'descriptors1': pred2['descriptors'],
            }

            with torch.no_grad():
                pred = self.superglue(data_sg)
            
            # Extract matches
            matches = pred['matches0'][0].cpu().numpy()
            confidence = pred['matching_scores0'][0].cpu().numpy()
            
            # Get valid matches
            valid = matches > -1
            mkpts0 = pred1['keypoints'][0][valid].cpu().numpy()
            mkpts1 = pred2['keypoints'][0][matches[valid]].cpu().numpy()
            
            if len(mkpts0) < 4:
                logger.warning("Not enough matches for homography, using basic alignment")
                return self.aligner.align_images_basic(img1, img2)
            
            # Compute homography
            H, mask = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5.0)
            
            if H is None:
                logger.warning("Homography estimation failed, using basic alignment.")
                return img1, img2, 0.0 # Return original images and 0 confidence
            
            # Warp image
            h, w = img1.shape[:2]
            img2_aligned = cv2.warpPerspective(img2, H, (w, h))
            
            # Calculate alignment confidence (mean of matching scores for inliers)
            alignment_confidence = np.mean(confidence[valid]) if len(confidence[valid]) > 0 else 0.0
            
            return img1, img2_aligned, float(alignment_confidence)
            
        except Exception as e:
            logger.error(f"Error in SuperGlue alignment: {e}", exc_info=True)
            return self.aligner.align_images_basic(img1, img2)
    
    async def detect_changes_advanced(self, img1: np.ndarray, img2: np.ndarray, 
                                     sensitivity: float, min_area: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Advanced change detection using ChangeFormer"""
        try:
            # Preprocess for ChangeFormer (ensure RGB, normalize, resize)
            # ChangeFormer expects input_nc=3, so we need to concatenate outside if it's a Siamese network
            # Or, if it expects 6 channels, `prepare_changeformer_input` will handle it.
            # Assuming ChangeFormer takes two 3-channel inputs and processes them.
            # The `prepare_changeformer_input` function already concatenates to 6 channels.
            
            input_tensor = self.image_processor.prepare_changeformer_input(img1, img2, self.device)
            
            # Run ChangeFormer
            with torch.no_grad():
                output = self.changeformer(input_tensor)
                # Assuming output is logits for 2 classes (no-change, change)
                # We want the probability of the 'change' class (index 1)
                change_probability = F.softmax(output, dim=1)[:, 1, :, :].squeeze(0)
                change_mask = change_probability.cpu().numpy()
            
            # Post-process change mask
            # Thresholding based on sensitivity
            binary_mask = (change_mask > sensitivity).astype(np.uint8)
            
            # Apply morphological operations to clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours and draw results
            # The original images (img1, img2) are already resized to inference_size
            # We'll draw on the *aligned* comparison image (img2)
            result_img_with_boxes = img2.copy()
            confidence_scores_list = []
            
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate confidence for this region (mean probability within the bounding box)
                    region_confidence = np.mean(change_mask[y:y+h, x:x+w])
                    confidence_scores_list.append(float(region_confidence))
                    
                    # Draw bounding box
                    cv2.rectangle(result_img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2) # Red color, 2px thickness
                    
                    # Add confidence text
                    cv2.putText(result_img_with_boxes, f'{region_confidence:.2f}', 
                                (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                                
                    # Semi-transparent overlay (optional, but good for visualization)
                    overlay = result_img_with_boxes.copy()
                    cv2.fillPoly(overlay, [contour], (0, 0, 255)) # Fill with red
                    result_img_with_boxes = cv2.addWeighted(result_img_with_boxes, 0.8, overlay, 0.2, 0)
            
            confidence_summary = {
                "mean_confidence": float(np.mean(confidence_scores_list)) if confidence_scores_list else 0.0,
                "max_confidence": float(np.max(confidence_scores_list)) if confidence_scores_list else 0.0,
                "num_changes": len(confidence_scores_list)
            }
            
            return result_img_with_boxes, binary_mask, confidence_summary
            
        except Exception as e:
            logger.error(f"Error in advanced change detection: {e}", exc_info=True)
            # Fallback to basic detection if ChangeFormer fails
            return self.detector.detect_changes_basic(img1, img2, sensitivity, min_area)
    
    def validate_image_file(self, file: UploadFile) -> bool:
        """Validate uploaded image file by content type."""
        allowed_types = ["image/jpeg", "image/png", "image/jpg"]
        return file.content_type in allowed_types



    async def generate_request_id(self, file1: UploadFile, file2: UploadFile, sensitivity: float, min_area: int, alignment_method: str) -> str:
        """Generate unique request ID based on file contents and parameters."""
        
        # Reset file pointers to the beginning before reading
        # These are asynchronous operations, so they must be awaited
        await file1.seek(0)
        await file2.seek(0)
        
        hasher = hashlib.md5()
        
        # Read a portion of the file content asynchronously
        # The result of .read() is bytes, which hasher.update() expects
        hasher.update(await file1.read(1024 * 1024)) # Read first 1MB
        hasher.update(await file2.read(1024 * 1024)) # Read first 1MB
        
        # Reset file pointers again, so the files can be read from the beginning
        # by other parts of your application (e.g., for actual image processing)
        await file1.seek(0)
        await file2.seek(0)

        # Include parameters in hash to ensure unique cache for different settings
        params_str = f"{sensitivity}_{min_area}_{alignment_method}"
        hasher.update(params_str.encode('utf-8'))

        return hasher.hexdigest()[:16]

# Initialize the API
api = AdvancedImageComparisonAPI()
app = api.app

if __name__ == "__main__":
    print("🚀 Starting Advanced AI Image Comparison API...")
    print("🧠 Loading AI models (SuperPoint, SuperGlue, ChangeFormer)...")
    print("📊 API available at: http://localhost:8000")
    print("📚 Documentation at: http://localhost:8000/docs")
    print("🔧 Model status at: http://localhost:8000/models/status")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True, # Set to False for production
        log_level="info"
    )