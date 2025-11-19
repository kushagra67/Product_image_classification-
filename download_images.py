"""
Script to download real product images from multiple sources
Supports: Unsplash API, Pexels API, and direct URLs as fallback
"""
import os
import requests
import json
from pathlib import Path
from urllib.parse import urlencode
import time
import io
from PIL import Image

# Configuration
UNSPLASH_API_URL = "https://api.unsplash.com/search/photos"
PEXELS_API_URL = "https://api.pexels.com/v1/search"

# For API keys:
# Unsplash: https://unsplash.com/developers
# Pexels: https://www.pexels.com/api/
UNSPLASH_API_KEY = "YOUR_UNSPLASH_API_KEY_HERE"
PEXELS_API_KEY = "YOUR_PEXELS_API_KEY_HERE"

# Product categories with multiple search queries for better variety
CATEGORIES = {
    'books': ['book', 'bookshelf', 'library', 'textbook', 'novel', 'hardcover', 'paperback'],
    'clothing': ['fashion', 'clothes', 'apparel', 'shirt', 'dress', 'pants', 'jacket', 'sweater'],
    'electronics': ['electronics', 'gadgets', 'devices', 'laptop', 'smartphone', 'tablet', 'headphones', 'camera'],
    'furniture': ['furniture', 'chair', 'desk', 'table', 'sofa', 'couch', 'bed', 'cabinet']
}

OUTPUT_DIR = Path('data/raw')


def download_from_unsplash(category, queries):
    """Download images from Unsplash API"""
    if UNSPLASH_API_KEY == "YOUR_UNSPLASH_API_KEY_HERE":
        return 0
    
    downloaded = 0
    
    for query in queries[:3]:
        params = {
            'query': query,
            'per_page': 5,
            'client_id': UNSPLASH_API_KEY,
            'orientation': 'squarish'
        }
        
        try:
            response = requests.get(UNSPLASH_API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'results' not in data or not data['results']:
                continue
            
            for photo in data['results'][:3]:
                try:
                    image_url = photo['urls']['regular']
                    filename = f"{category}_{downloaded:03d}.jpg"
                    filepath = OUTPUT_DIR / category / filename
                    
                    # Skip if file already exists
                    if filepath.exists():
                        continue
                    
                    img_response = requests.get(image_url, timeout=10)
                    img_response.raise_for_status()
                    
                    # Validate and save image
                    img = Image.open(io.BytesIO(img_response.content))
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    img.save(filepath, 'JPEG', quality=85)
                    
                    downloaded += 1
                    print("✓", end=" ", flush=True)
                    time.sleep(0.2)
                    
                except Exception:
                    print("✗", end=" ", flush=True)
            
        except Exception:
            pass
    
    return downloaded


def download_from_pexels(category, queries):
    """Download images from Pexels API"""
    if PEXELS_API_KEY == "YOUR_PEXELS_API_KEY_HERE":
        return 0
    
    downloaded = 0
    headers = {'Authorization': PEXELS_API_KEY}
    
    for query in queries[:3]:
        params = {
            'query': query,
            'per_page': 5,
            'size': 'large',
            'orientation': 'square'
        }
        
        try:
            response = requests.get(PEXELS_API_URL, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'photos' not in data or not data['photos']:
                continue
            
            for photo in data['photos'][:3]:
                try:
                    image_url = photo['src']['large']
                    filename = f"{category}_{downloaded:03d}.jpg"
                    filepath = OUTPUT_DIR / category / filename
                    
                    if filepath.exists():
                        continue
                    
                    img_response = requests.get(image_url, timeout=10)
                    img_response.raise_for_status()
                    
                    img = Image.open(io.BytesIO(img_response.content))
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    img.save(filepath, 'JPEG', quality=85)
                    
                    downloaded += 1
                    print("✓", end=" ", flush=True)
                    time.sleep(0.2)
                    
                except Exception:
                    print("✗", end=" ", flush=True)
            
        except Exception:
            pass
    
    return downloaded


def download_fallback_images(category):
    """
    Fallback: Download from curated direct URLs
    These are high-quality product images from Unsplash
    """
    image_urls = {
        'books': [
            'https://images.unsplash.com/photo-1507842072343-583f20270319?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1512820790803-83ca734da794?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1543002588-d83ceddc8d0e?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1495446815901-a7297e8b7f1a?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1524995997946-a1c2e315a42f?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1488160481651-0c06cbf119b2?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1534445867742-c5049db021d2?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1506880018603-83d5b814b5a6?w=500&h=500&fit=crop',
        ],
        'clothing': [
            'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1542272604-787c62d465d1?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1523293182086-7651a899d37f?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1509631179647-0177331693ae?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1508319941329-a76448ba5e81?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1505088169282-b7b62408b0a0?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1502630859cebcb021f923bc1c8a1671?w=500&h=500&fit=crop',
        ],
        'electronics': [
            'https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1516321318423-f06f70d504f0?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1483058712412-4245e9b90334?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1491553895911-0055eca6402d?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1498050108023-c5249f4df085?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1519389950473-47ba0277781c?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1486312338219-ce68d2c6f44d?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1574144611937-0df059b5ef3e?w=500&h=500&fit=crop',
        ],
        'furniture': [
            'https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1506439773649-6e0eb8cfb237?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1555597673-b21d5c3737f4?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1586023492125-27b2c045b122?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1550355291-bbee04a92027?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1526923147efb546f88e1b36b05fe1a7b?w=500&h=500&fit=crop',
            'https://images.unsplash.com/photo-1549887534-f3cde46db900?w=500&h=500&fit=crop',
        ]
    }
    
    downloaded = 0
    urls = image_urls.get(category, [])
    
    for idx, url in enumerate(urls):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            filename = f"{category}_{idx:03d}.jpg"
            filepath = OUTPUT_DIR / category / filename
            
            if filepath.exists():
                continue
            
            # Validate and save
            img = Image.open(io.BytesIO(response.content))
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(filepath, 'JPEG', quality=85)
            
            downloaded += 1
            print("✓", end=" ", flush=True)
            time.sleep(0.2)
            
        except Exception:
            print("✗", end=" ", flush=True)
    
    return downloaded


def main():
    """Main download function"""
    print("🚀 Product Image Downloader")
    print("="*60)
    print("Downloading from: Unsplash API, Pexels API, and Direct URLs")
    print("="*60)
    
    # Create directories
    for category in CATEGORIES.keys():
        (OUTPUT_DIR / category).mkdir(parents=True, exist_ok=True)
    
    total_downloaded = {cat: 0 for cat in CATEGORIES}
    
    for category, queries in CATEGORIES.items():
        print(f"\n📥 {category.upper()}")
        
        # Try Unsplash API
        print("  Unsplash: ", end="", flush=True)
        unsplash_count = download_from_unsplash(category, queries)
        total_downloaded[category] += unsplash_count
        print(f" ({unsplash_count})")
        
        # Try Pexels API
        print("  Pexels:   ", end="", flush=True)
        pexels_count = download_from_pexels(category, queries)
        total_downloaded[category] += pexels_count
        print(f" ({pexels_count})")
        
        # Fallback to direct URLs
        print("  Direct:   ", end="", flush=True)
        direct_count = download_fallback_images(category)
        total_downloaded[category] += direct_count
        print(f" ({direct_count})")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 Download Summary")
    print("="*60)
    
    grand_total = 0
    for category, count in total_downloaded.items():
        print(f"  {category.capitalize():<15} {count:>3} images")
        grand_total += count
    
    print("-"*60)
    print(f"  {'Total':<15} {grand_total:>3} images")
    print("="*60)
    
    print("\n✅ Download completed!")
    print(f"📁 Images saved to: {OUTPUT_DIR.absolute()}")
    
    print("\n📝 Next steps:")
    print("  1. Review images: ls data/raw/*/")
    print("  2. Delete low-quality images if any")
    print("  3. Retrain model: python train.py")
    print("\n💡 Tips:")
    print("  - To get more images, configure API keys:")
    print("    * UNSPLASH_API_KEY from https://unsplash.com/developers")
    print("    * PEXELS_API_KEY from https://www.pexels.com/api/")


if __name__ == '__main__':
    main()
