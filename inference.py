"""
Inference script for product image classification
"""
import os
import sys
import argparse
import torch
import json
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import create_model
from dataset import get_default_transforms
from preprocessing import ImagePreprocessor


class ImageClassifier:
    """Classifier for product images"""
    
    def __init__(self, model_path, label_map_path, device='cpu'):
        """
        Initialize classifier
        
        Args:
            model_path (str): Path to saved model
            label_map_path (str): Path to label mapping JSON
            device (str): Device to use
        """
        self.device = device
        
        # Load label mapping
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
        
        self.idx_to_label = {v: k for k, v in self.label_map.items()}
        
        # Load model
        num_classes = len(self.label_map)
        self.model = create_model(
            model_name='resnet50',
            num_classes=num_classes,
            pretrained=False,
            device=device
        )
        
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        # Transform pipeline
        self.transform = get_default_transforms(image_size=224, augment=False)
    
    def predict(self, image_path, return_top_k=1):
        """
        Predict class for image
        
        Args:
            image_path (str): Path to image
            return_top_k (int): Return top-k predictions
        
        Returns:
            list: List of (class_name, confidence) tuples
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Get top-k predictions
        top_k_probs, top_k_indices = torch.topk(probabilities, return_top_k)
        
        results = []
        for prob, idx in zip(top_k_probs, top_k_indices):
            class_name = self.idx_to_label[idx.item()]
            confidence = prob.item()
            results.append((class_name, confidence))
        
        return results
    
    def predict_batch(self, image_paths, return_top_k=1):
        """
        Predict classes for batch of images
        
        Args:
            image_paths (list): List of image paths
            return_top_k (int): Return top-k predictions
        
        Returns:
            dict: Mapping from image path to predictions
        """
        results = {}
        
        for image_path in image_paths:
            results[image_path] = self.predict(image_path, return_top_k)
        
        return results


def main(args):
    """Main inference function"""
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.label_map):
        print(f"Error: Label map not found at {args.label_map}")
        sys.exit(1)
    
    # Initialize classifier
    device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    print(f"Using device: {device}")
    
    classifier = ImageClassifier(
        model_path=args.model,
        label_map_path=args.label_map,
        device=device
    )
    
    # Single image prediction
    if args.image:
        print(f"\nPredicting for: {args.image}")
        predictions = classifier.predict(args.image, return_top_k=args.top_k)
        
        print("\nPredictions:")
        for i, (class_name, confidence) in enumerate(predictions, 1):
            print(f"  {i}. {class_name}: {confidence:.4f}")
    
    # Directory prediction
    elif args.image_dir:
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_paths = []
        
        # Search for images recursively in subdirectories
        for root, dirs, files in os.walk(args.image_dir):
            for f in files:
                if f.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, f))
        
        if not image_paths:
            print(f"No images found in {args.image_dir}")
            sys.exit(1)
        
        print(f"\nFound {len(image_paths)} images")
        results = classifier.predict_batch(image_paths, return_top_k=args.top_k)
        
        # Save results
        output_data = {}
        for image_path, predictions in results.items():
            output_data[os.path.basename(image_path)] = [
                {'class': class_name, 'confidence': float(confidence)}
                for class_name, confidence in predictions
            ]
        
        with open('predictions.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print("\nPredictions saved to predictions.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify product images')
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/saved/best_model.pth',
        help='Path to saved model'
    )
    parser.add_argument(
        '--label-map',
        type=str,
        default='models/saved/label_map.json',
        help='Path to label mapping JSON'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to single image for prediction'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Path to directory of images for batch prediction'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=1,
        help='Return top-k predictions'
    )
    parser.add_argument(
        '--use-cuda',
        action='store_true',
        help='Use CUDA if available'
    )
    
    args = parser.parse_args()
    
    if not args.image and not args.image_dir:
        print("Error: Please provide either --image or --image-dir")
        parser.print_help()
        sys.exit(1)
    
    main(args)
