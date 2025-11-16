"""
Generate sample product images for each category
This script creates synthetic images for demonstration purposes
"""

import numpy as np
import cv2
import os
from pathlib import Path

def create_sample_images():
    """Create sample product images for each category"""
    
    data_dir = Path('data/raw')
    categories = {
        'electronics': {
            'color': (50, 100, 200),  # Blue-ish
            'shapes': ['rectangles', 'circles'],
            'count': 15
        },
        'furniture': {
            'color': (139, 69, 19),  # Brown-ish
            'shapes': ['rectangles', 'polygons'],
            'count': 15
        },
        'clothing': {
            'color': (200, 100, 150),  # Pink-ish
            'shapes': ['irregular', 'circles'],
            'count': 15
        },
        'books': {
            'color': (180, 140, 70),  # Tan-ish
            'shapes': ['rectangles', 'text'],
            'count': 15
        }
    }
    
    for category, props in categories.items():
        category_dir = data_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating {props['count']} sample images for {category}...")
        
        for i in range(props['count']):
            # Create a random image
            img = np.random.randint(200, 256, (256, 256, 3), dtype=np.uint8)
            
            # Add category-specific features
            color = props['color']
            
            # Draw some shapes
            for _ in range(np.random.randint(2, 5)):
                x1, y1 = np.random.randint(20, 200, 2)
                x2, y2 = np.random.randint(x1+20, 236, 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            
            # Add some circles
            for _ in range(np.random.randint(1, 3)):
                center = tuple(np.random.randint(30, 226, 2))
                radius = np.random.randint(10, 30)
                cv2.circle(img, center, radius, color, -1)
            
            # Add noise for variation
            noise = np.random.normal(0, 15, img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            # Save image
            img_path = category_dir / f'{category}_{i:03d}.jpg'
            cv2.imwrite(str(img_path), img)
            print(f"  ✓ Created {img_path.name}")
    
    print("\n" + "="*60)
    print("✓ Sample dataset created successfully!")
    print("="*60)
    print("\nDataset structure:")
    for category in categories.keys():
        category_dir = data_dir / category
        files = list(category_dir.glob('*.jpg'))
        print(f"  - {category}: {len(files)} images")

if __name__ == '__main__':
    create_sample_images()
