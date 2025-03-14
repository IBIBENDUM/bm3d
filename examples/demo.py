#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BM3D image denoising demonstration file

Usage:  python3 demo.py --input_image=example.png --sigma=5
"""

import argparse
import logging
import os
from datetime import datetime

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import bm3d
import cv2

def setup_logging(log_level: str = "INFO"):
    """
    Configure logging to save into file and print into console
    Logs saved in "logs" directory with timestamp in filename
    :param logLevel: log level from "logging"
    """
    # Create a logs directory if it doesn't exist
    logsDir = "logs"
    if not os.path.exists(logsDir):
        os.makedirs(logsDir)

    # Log file name is the program start timestamp
    logFilename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    logFilepath = os.path.join(logsDir, logFilename)

    # Configure logging with file and console handlers
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(logFilepath), logging.StreamHandler()],
    )


def loadImage(image_path: str) -> np.ndarray:
    """
    Load image by path and return as ndarray
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        logging.info(f"Loading image {image_path}")
        # Open grayscaled image
        return img
    except Exception as e:
        logging.error(f"Error during image loading: {e}")
        raise Exception(f"Error during image loading: {e}")


def main(imagePath: str = "data/cameraman256.png", noiseVariance: int = 25) -> None:
    """
    Load image, apply noise, denoise by BM3D and calculate metrics
    """
    try:
        originalImage: np.ndarray = loadImage(imagePath)

        noisyImage: np.ndarray = bm3d.addNoise(originalImage, noiseVariance)

        denoisedImage: np.ndarray = bm3d.bm3d(noisyImage, noiseVariance)

        psnr: float = bm3d.calculatePSNR(originalImage, denoisedImage)

    except Exception as e:
        logging.error(f"An error occurred during the process: {e}")
        raise Exception(f"An error occurred during the process: {e}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="BM3D image denoising demonstration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        default="data/cameraman256.png",
        help="Path to the input image",
        metavar="\b",
    )
    parser.add_argument(
        "-s",
        "--sigma",
        type=int,
        default=25,
        help="Noise variance",
        metavar="\b",
    )
    logLevels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    parser.add_argument(
        "-l",
        "--logLevel",
        type=str,
        default="INFO",
        choices=logLevels,
        help= f"Logging level {logLevels}",
        metavar="\b",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    setup_logging(args.logLevel)

    main(args.image, args.sigma)
