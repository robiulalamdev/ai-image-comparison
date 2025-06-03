# # download_models.py

# import os
# import requests
# import subprocess
# import sys
# from pathlib import Path

# # Define paths relative to this script
# BASE_DIR = Path(__file__).resolve().parent
# CHECKPOINTS_DIR = BASE_DIR / "models" / "pretrained"
# CHANGEFORMER_REPO_DIR = BASE_DIR / "ChangeFormer"

# # Create directories if they don't exist
# CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# # 1. Clone ChangeFormer repository
# print("1. Cloning ChangeFormer repository...")
# if not CHANGEFORMER_REPO_DIR.exists():
#     try:
#         subprocess.run(
#             ["git", "clone", "https://github.com/wgcban/ChangeFormer.git", str(CHANGEFORMER_REPO_DIR)],
#             check=True,
#             capture_output=True,
#             text=True
#         )
#         print("ChangeFormer repository cloned successfully.")
#     except subprocess.CalledProcessError as e:
#         print(f"Error cloning ChangeFormer repository: {e}")
#         print(f"Stdout: {e.stdout}")
#         print(f"Stderr: {e.stderr}")
#         sys.exit(1)
# else:
#     print("ChangeFormer repository already exists. Skipping clone.")

# # 2. Download SuperPoint pre-trained weights
# print("2. Downloading SuperPoint pre-trained weights...")
# SUPERPOINT_WEIGHTS_URL = "https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth"
# SUPERPOINT_WEIGHTS_PATH = CHECKPOINTS_DIR / "superpoint_v1.pth"

# if not SUPERPOINT_WEIGHTS_PATH.exists():
#     try:
#         print(f"Downloading SuperPoint weights from {SUPERPOINT_WEIGHTS_URL}...")
#         response = requests.get(SUPERPOINT_WEIGHTS_URL, stream=True)
#         response.raise_for_status() # Raise an exception for bad status codes
#         with open(SUPERPOINT_WEIGHTS_PATH, 'wb') as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)
#         print("SuperPoint weights downloaded successfully.")
#     except requests.exceptions.RequestException as e:
#         print(f"Error downloading SuperPoint weights: {e}")
#         print("Please ensure you have an internet connection and the URL is correct.")
#         sys.exit(1)
# else:
#     print("SuperPoint weights already exist. Skipping download.")

# # 3. Download SuperGlue pre-trained weights (indoor)
# print("3. Downloading SuperGlue pre-trained weights (indoor)...")
# SUPERGLUE_WEIGHTS_URL = "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_indoor.pth"
# SUPERGLUE_WEIGHTS_PATH = CHECKPOINTS_DIR / "superglue_indoor.pth"

# if not SUPERGLUE_WEIGHTS_PATH.exists():
#     try:
#         print(f"Downloading SuperGlue weights from {SUPERGLUE_WEIGHTS_URL}...")
#         response = requests.get(SUPERGLUE_WEIGHTS_URL, stream=True)
#         response.raise_for_status()
#         with open(SUPERGLUE_WEIGHTS_PATH, 'wb') as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)
#         print("SuperGlue weights downloaded successfully.")
#     except requests.exceptions.RequestException as e:
#         print(f"Error downloading SuperGlue weights: {e}")
#         print("Please ensure you have an internet connection and the URL is correct.")
#         sys.exit(1)
# else:
#     print("SuperGlue weights already exist. Skipping download.")

# # 4. Download ChangeFormer pre-trained weights (LEVIR-CD dataset)
# print("4. Downloading ChangeFormer pre-trained weights (LEVIR-CD)...")
# CHANGEFORMER_WEIGHTS_URL = "https://github.com/wgcban/ChangeFormer/releases/download/v1.0/ChangeFormerV6_LEVIR.pth"
# CHANGEFORMER_WEIGHTS_PATH = CHECKPOINTS_DIR / "changeformer_levir.pth"

# if not CHANGEFORMER_WEIGHTS_PATH.exists():
#     print(f"Downloading ChangeFormer weights from {CHANGEFORMER_WEIGHTS_URL}...")
#     try:
#         response = requests.get(CHANGEFORMER_WEIGHTS_URL, stream=True)
#         response.raise_for_status()
#         with open(CHANGEFORMER_WEIGHTS_PATH, 'wb') as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)
#         print("ChangeFormer weights downloaded successfully.")
#     except requests.exceptions.RequestException as e:
#         print(f"Error downloading ChangeFormer weights: {e}")
#         print("Please ensure the URL is correct or download manually.")
#         sys.exit(1)
# else:
#     print("ChangeFormer weights already exist. Skipping download.")

# print("\nAll models and repositories are set up.")
# print("You can now proceed to install dependencies and run the FastAPI application.")











import requests
import sys
from pathlib import Path
import zipfile # Import zipfile for handling zip archives
import io # Optional, for handling zip file in memory if needed, but not strictly necessary here

# Assuming BASE_DIR and CHECKPOINTS_DIR are defined earlier in your script
# Example (adjust based on your actual script's structure):
BASE_DIR = Path(__file__).resolve().parent.parent # This assumes download_models.py is in 'scripts' and checkpoints is in parent of scripts
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists

# 4. Download ChangeFormer pre-trained weights (LEVIR-CD dataset)
print("4. Downloading ChangeFormer pre-trained weights (LEVIR-CD)...")

# --- CORRECTED URL AND FILENAMES ---
# This is the actual URL for the LEVIR model zip file on GitHub releases v0.1.0
CHANGEFORMER_ZIP_URL = "https://github.com/wgcban/ChangeFormer/releases/download/v0.1.0/CD_ChangeFormerV6_LEVIR_B16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256.zip"
CHANGEFORMER_ZIP_FILENAME = "CD_ChangeFormerV6_LEVIR_B16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256.zip" # Full name for local storage

# Define the local path for the downloaded zip file
CHANGEFORMER_ZIP_PATH = CHECKPOINTS_DIR / CHANGEFORMER_ZIP_FILENAME

# Define the target name for the extracted .pth file (e.g., 'changeformer_levir.pth')
TARGET_PTH_FILENAME = "changeformer_levir.pth"
TARGET_PTH_PATH = CHECKPOINTS_DIR / TARGET_PTH_FILENAME

# Check if the final .pth file already exists to avoid re-downloading and re-extracting
if TARGET_PTH_PATH.exists():
    print("ChangeFormer weights (final .pth) already exist. Skipping download and extraction.")
else:
    # 4.1. Download the .zip file if it doesn't already exist
    if not CHANGEFORMER_ZIP_PATH.exists():
        print(f"Downloading ChangeFormer weights (ZIP) from {CHANGEFORMER_ZIP_URL}...")
        try:
            response = requests.get(CHANGEFORMER_ZIP_URL, stream=True)
            response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
            with open(CHANGEFORMER_ZIP_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("ChangeFormer ZIP file downloaded successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading ChangeFormer weights ZIP: {e}")
            print("Please ensure the URL is correct or download manually.")
            sys.exit(1)
    else:
        print("ChangeFormer ZIP file already exists. Skipping download.")

    # 4.2. Extract the desired .pth file from the downloaded zip
    print(f"Extracting '{TARGET_PTH_FILENAME}' from '{CHANGEFORMER_ZIP_FILENAME}'...")
    try:
        with zipfile.ZipFile(CHANGEFORMER_ZIP_PATH, 'r') as zip_ref:
            # Check if 'best_ckpt.pt' or 'last_ckpt.pt' exists within the zip
            model_file_to_extract = None
            if 'best_ckpt.pt' in zip_ref.namelist():
                model_file_to_extract = 'best_ckpt.pt'
            elif 'last_ckpt.pt' in zip_ref.namelist():
                model_file_to_extract = 'last_ckpt.pt'
            else:
                print(f"Error: Neither 'best_ckpt.pt' nor 'last_ckpt.pt' found in the zip file '{CHANGEFORMER_ZIP_FILENAME}'.")
                sys.exit(1)

            # Extract the chosen model file to the CHECKPOINTS_DIR
            zip_ref.extract(model_file_to_extract, path=CHECKPOINTS_DIR)

            # Rename the extracted file to the desired target name
            (CHECKPOINTS_DIR / model_file_to_extract).rename(TARGET_PTH_PATH)
            print(f"'{model_file_to_extract}' extracted and renamed to '{TARGET_PTH_FILENAME}' successfully.")

    except zipfile.BadZipFile:
        print(f"Error: The downloaded file '{CHANGEFORMER_ZIP_FILENAME}' is not a valid ZIP file. Please try downloading manually and verify.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during extraction/renaming: {e}")
        sys.exit(1)

print("\nAll models and repositories are set up.")
print("You can now proceed to install dependencies and run the FastAPI application.")