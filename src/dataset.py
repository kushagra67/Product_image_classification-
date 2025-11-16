"""
Dataset utilities for loading and preprocessing product images
"""
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ProductImageDataset(Dataset):
    """Custom Dataset for product images"""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): List of image file paths
            labels (list): List of corresponding labels
            transform (callable, optional): Optional image transformations
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Load and return image and label"""
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image using PIL
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_default_transforms(image_size=224, augment=False):
    """
    Get image transformation pipeline
    
    Args:
        image_size (int): Target image size
        augment (bool): Whether to apply data augmentation
    
    Returns:
        transforms.Compose: Transformation pipeline
    """
    if augment:
        return transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def load_images_from_directory(directory_path, label, valid_extensions=('.jpg', '.jpeg', '.png')):
    """
    Load all images from a directory
    
    Args:
        directory_path (str): Path to directory containing images
        label: Label for all images in directory
        valid_extensions (tuple): Valid image file extensions
    
    Returns:
        tuple: (image_paths, labels)
    """
    image_paths = []
    labels = []
    
    if not os.path.exists(directory_path):
        print(f"Warning: Directory {directory_path} does not exist")
        return image_paths, labels
    
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(valid_extensions):
            full_path = os.path.join(directory_path, filename)
            image_paths.append(full_path)
            labels.append(label)
    
    return image_paths, labels
