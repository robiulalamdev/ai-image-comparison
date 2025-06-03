# Advanced AI Image Comparison API

This project provides a professional-grade AI-powered image comparison API using state-of-the-art models for image alignment and semantic change detection. It's designed to handle variations in camera angle, rotation, and zoom, and highlight meaningful differences with red bounding boxes.

## Features

- **Advanced Image Alignment:** Utilizes SuperPoint and SuperGlue for robust keypoint detection and matching, enabling accurate alignment even with slight perspective changes. A basic ORB/SIFT fallback is also available.
- **Semantic Change Detection:** Employs the ChangeFormer model, a Transformer-based network, for high-accuracy semantic change detection, identifying actual changes rather than minor pixel noise. A basic pixel-difference fallback is also available.
- **FastAPI Backend:** A high-performance, easy-to-use RESTful API for image uploads and comparison results.
- **Web Interface:** A simple, responsive web interface for easy demonstration and testing.
- **Configurable Parameters:** Adjust sensitivity and minimum change area for detection.
- **Caching:** Caches results for repeated identical requests to improve performance.
- **Detailed Logging:** Provides comprehensive logging for monitoring and debugging.

## Project Structure
