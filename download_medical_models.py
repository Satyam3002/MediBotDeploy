#!/usr/bin/env python3
"""
Medical Models Download Script
Downloads and caches medical AI models for skin and X-ray analysis
"""

import os
import sys
import json
from pathlib import Path
import requests
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import gdown  # for Google Drive downloads

MODEL_CACHE_DIR = Path("./model_cache")
MODEL_CONFIG_FILE = MODEL_CACHE_DIR / "model_config.json"

# Model configurations with cloud storage URLs
MODEL_CONFIGS = {
    "skin": {
        "url": "https://drive.google.com/drive/folders/1v_Ma_xi1UIFzFJYAjA9LDLaQEVIEBnwU",
        "save_path": "backend/saved_skin_model",
        "model_file": "skinconvnext_scripted.pt"
    },
    "xray": {
        "url": "https://drive.google.com/drive/folders/15c2mrKZM41YPJ68tmfQuCqVgCsdBvBgi",
        "save_path": "backend/saved_xray_model"
    },
    "diagnosis": {
        "url": "YOUR_GOOGLE_DRIVE_DIAGNOSIS_MODEL_URL",  # Replace with your Google Drive URL
        "save_path": ".",
        "model_file": "diagnosis_model.joblib"
    }
}

def create_directory(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    print(f"📁 Created/verified directory: {path}")
    return path

def download_from_cloud(url, save_path, filename=None):
    """Download model from cloud storage"""
    try:
        print(f"🔄 Downloading from {url}...")
        save_path = create_directory(save_path)
        
        if filename:
            output_path = os.path.join(save_path, filename)
        else:
            output_path = save_path
            
        # Download using gdown for Google Drive
        gdown.download_folder(url, output=output_path, quiet=False)
        
        print(f"✅ Model downloaded successfully to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download model: {str(e)}")
        return False

def ensure_models_available():
    """Ensure all required models are downloaded"""
    # Create cache directory
    create_directory(MODEL_CACHE_DIR)
    
    # Download each model if not present
    for model_type, model_info in MODEL_CONFIGS.items():
        save_path = Path(model_info['save_path'])
        model_file = model_info.get('model_file')
        
        # Check if model exists
        if model_file:
            model_path = save_path / model_file
            if not model_path.exists():
                print(f"Model {model_type} not found, downloading...")
                download_from_cloud(
                    model_info['url'],
                    str(save_path),
                    model_file
                )
        else:
            if not save_path.exists() or not any(save_path.iterdir()):
                print(f"Model {model_type} not found, downloading...")
                download_from_cloud(
                    model_info['url'],
                    str(save_path)
                )

if __name__ == "__main__":
    ensure_models_available()