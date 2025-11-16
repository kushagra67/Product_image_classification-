"""
Image preprocessing and OpenCV utilities
"""
import cv2
import numpy as np
from PIL import Image


class ImagePreprocessor:
    """Utilities for image preprocessing"""
    
    @staticmethod
    def resize_image(image_path, target_size=(224, 224), keep_aspect=False):
        """
        Resize image to target size
        
        Args:
            image_path (str): Path to image
            target_size (tuple): Target (height, width)
            keep_aspect (bool): Whether to maintain aspect ratio
        
        Returns:
            np.ndarray: Resized image
        """
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        if keep_aspect:
            # Resize while maintaining aspect ratio
            h, w = image.shape[:2]
            aspect = w / h
            
            if aspect > 1:
                new_w = target_size[1]
                new_h = int(new_w / aspect)
            else:
                new_h = target_size[0]
                new_w = int(new_h * aspect)
            
            image = cv2.resize(image, (new_w, new_h))
            
            # Pad to target size
            top = (target_size[0] - new_h) // 2
            bottom = target_size[0] - new_h - top
            left = (target_size[1] - new_w) // 2
            right = target_size[1] - new_w - left
            
            image = cv2.copyMakeBorder(
                image, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        else:
            image = cv2.resize(image, (target_size[1], target_size[0]))
        
        return image
    
    @staticmethod
    def enhance_contrast(image, clip_limit=2.0, tile_size=(8, 8)):
        """
        Enhance image contrast using CLAHE
        
        Args:
            image (np.ndarray): Input image
            clip_limit (float): Clipping limit
            tile_size (tuple): Tile size
        
        Returns:
            np.ndarray: Enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            l = clahe.apply(l)
            
            # Merge back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            enhanced = clahe.apply(image)
        
        return enhanced
    
    @staticmethod
    def detect_edges(image, method='canny', threshold1=100, threshold2=200):
        """
        Detect edges in image
        
        Args:
            image (np.ndarray): Input image
            method (str): Edge detection method ('canny' or 'sobel')
            threshold1, threshold2 (int): Thresholds for Canny
        
        Returns:
            np.ndarray: Edge map
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        if method == 'canny':
            edges = cv2.Canny(gray, threshold1, threshold2)
        elif method == 'sobel':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            edges = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return edges
    
    @staticmethod
    def normalize_image(image, mean=None, std=None):
        """
        Normalize image values
        
        Args:
            image (np.ndarray): Input image
            mean (tuple): Mean values for normalization
            std (tuple): Std values for normalization
        
        Returns:
            np.ndarray: Normalized image
        """
        image = image.astype(np.float32) / 255.0
        
        if mean is not None and std is not None:
            mean = np.array(mean)
            std = np.array(std)
            image = (image - mean) / std
        
        return image


def remove_background(image_path, method='morphological'):
    """
    Remove background from product image
    
    Args:
        image_path (str): Path to image
        method (str): Background removal method
    
    Returns:
        np.ndarray: Image with removed background
    """
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    if method == 'morphological':
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for white (background)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_white, upper_white)
        mask = cv2.bitwise_not(mask)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
    elif method == 'grabcut':
        # Use GrabCut algorithm
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        rect = (10, 10, image.shape[1]-10, image.shape[0]-10)
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Apply mask
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return result
