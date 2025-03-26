import os
import shutil
from typing import Optional

def moveImagesToRoot(download_dir: str = "original_dataset") -> None:
    """Collect all images to root directory"""
    for root, dirs, files in os.walk(download_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(download_dir, file)
                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(file)
                    i = 1
                    while os.path.exists(dst_path):
                        dst_path = os.path.join(download_dir, f"{base}_{i}{ext}")
                        i += 1
                shutil.move(src_path, dst_path)

def removeSubdirs(download_dir: str = "original_dataset") -> None:
    """Remove all subdirectories"""
    for root, dirs, files in os.walk(download_dir, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if dir_path != download_dir:
                shutil.rmtree(dir_path)

def createEmptyDirectory(dir_path: str) -> None:
    """Create empty directory, removing existing if needed"""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path) 
    os.makedirs(dir_path)
