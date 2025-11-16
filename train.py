"""
Main training script for product image classification
"""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, random_split
import json
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataset import ProductImageDataset, get_default_transforms, load_images_from_directory
from models import create_model
from training import Trainer


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        # Create default config
        default_config = {
            'model': {
                'name': 'resnet50',
                'num_classes': 10,
                'pretrained': True
            },
            'training': {
                'batch_size': 32,
                'num_epochs': 20,
                'learning_rate': 1e-3,
                'train_split': 0.8
            },
            'data': {
                'image_size': 224,
                'augment': True
            },
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
        print(f"Created default config at {config_path}")
        return default_config
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_dataset(data_dir, config):
    """Load dataset from directory structure"""
    image_paths = []
    labels = []
    label_to_idx = {}
    
    # Assume directory structure: data_dir/class_name/image.jpg
    class_idx = 0
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        label_to_idx[class_name] = class_idx
        paths, lbls = load_images_from_directory(class_path, class_idx)
        image_paths.extend(paths)
        labels.extend(lbls)
        
        print(f"Loaded class '{class_name}': {len(paths)} images")
        class_idx += 1
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    print(f"\nTotal images: {len(image_paths)}")
    print(f"Number of classes: {class_idx}")
    
    # Create dataset
    transform_train = get_default_transforms(
        image_size=config['data']['image_size'],
        augment=config['data']['augment']
    )
    transform_val = get_default_transforms(
        image_size=config['data']['image_size'],
        augment=False
    )
    
    # Split dataset
    dataset = ProductImageDataset(image_paths, labels, transform=None)
    train_size = int(len(dataset) * config['training']['train_split'])
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Apply transforms to datasets
    train_dataset.dataset.transform = transform_train
    val_dataset.dataset.transform = transform_val
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, label_to_idx


def main(args):
    """Main training function"""
    # Load configuration
    config = load_config(args.config)
    print("\nConfiguration:")
    print(json.dumps(config, indent=2, default=str))
    
    # Create output directories
    os.makedirs('models/saved', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Device
    device = config['device']
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\nLoading dataset...")
    train_loader, val_loader, label_to_idx = load_dataset(args.data_dir, config)
    
    # Save label mapping
    with open('models/saved/label_map.json', 'w') as f:
        json.dump(label_to_idx, f, indent=2)
    print("Label mapping saved to models/saved/label_map.json")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        device=device
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=config['training']['learning_rate'],
        num_classes=config['model']['num_classes']
    )
    
    # Train model
    print("\nStarting training...")
    save_path = 'models/saved/best_model.pth'
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        save_path=save_path
    )
    
    # Save history
    with open('logs/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\nTraining completed!")
    print(f"Best model saved to {save_path}")
    print(f"Training history saved to logs/training_history.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train product image classifier')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Path to dataset directory (default: data/raw)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found")
        print("Please organize your images in: data/raw/class_name/image.jpg")
        sys.exit(1)
    
    main(args)
