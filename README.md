# Product Image Classification using PyTorch & OpenCV

A comprehensive deep learning project for classifying product images using ResNet50 backbone with PyTorch and advanced preprocessing using OpenCV.

## Features

- **Deep Learning Model**: ResNet50 and EfficientNet-based classifiers with transfer learning
- **Image Preprocessing**: Advanced preprocessing with OpenCV including:
  - Contrast enhancement (CLAHE)
  - Edge detection (Canny & Sobel)
  - Background removal
  - Automatic resizing with aspect ratio preservation
- **Data Augmentation**: Comprehensive augmentation pipeline for robust training
- **Training Framework**: Complete training loop with learning rate scheduling and early stopping
- **Inference Pipeline**: Easy-to-use prediction interface for single and batch predictions
- **Jupyter Notebook**: Interactive exploration and analysis tools

## Project Structure

```
Product_image_classification/
├── src/
│   ├── dataset.py          # Dataset utilities and data loading
│   ├── models.py           # Neural network architectures
│   ├── training.py         # Training and evaluation logic
│   └── preprocessing.py    # Image preprocessing with OpenCV
├── data/
│   ├── raw/                # Raw product images organized by class
│   └── processed/          # Processed images (optional)
├── models/
│   └── saved/              # Trained model checkpoints
├── notebooks/
│   └── product_classification.ipynb  # Interactive exploration
├── train.py                # Main training script
├── inference.py            # Prediction script
├── config.yaml             # Training configuration
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU support)

### Setup

1. Clone the repository:
```bash
cd Product_image_classification
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

Organize your images in the following structure:
```
data/raw/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── classN/
    └── ...
```

## Usage

### Training

1. **Configure training** (optional):
   Edit `config.yaml` to customize:
   - Model architecture
   - Learning rate
   - Batch size
   - Number of epochs
   - Image size

2. **Run training**:
```bash
python train.py --data-dir data/raw --config config.yaml
```

The best model will be saved to `models/saved/best_model.pth`.

### Inference

**Single image prediction**:
```bash
python inference.py --image path/to/image.jpg
```

**Batch prediction**:
```bash
python inference.py --image-dir path/to/images/
```

**With GPU**:
```bash
python inference.py --image path/to/image.jpg --use-cuda
```

### Jupyter Notebook

Launch the interactive notebook for exploration:
```bash
jupyter notebook notebooks/product_classification.ipynb
```

The notebook includes:
- Dataset analysis and visualization
- Image preprocessing demonstrations
- Model architecture inspection
- Training visualization
- Performance metrics
- Inference examples

## Configuration

Edit `config.yaml` to customize:

```yaml
model:
  name: 'resnet50'           # Model architecture
  num_classes: 10            # Number of product classes
  pretrained: true           # Use pretrained weights

training:
  batch_size: 32             # Batch size
  num_epochs: 20             # Number of epochs
  learning_rate: 1e-3        # Learning rate
  train_split: 0.8           # Train/val split ratio

data:
  image_size: 224            # Input image size
  augment: true              # Enable data augmentation

device: 'cuda'               # 'cuda' or 'cpu'
```

## Model Architecture

### ResNet50-based Classifier
- Backbone: ResNet50 (pretrained on ImageNet)
- Custom head:
  - Dropout (0.5)
  - Linear layer (2048 → 512)
  - ReLU activation
  - Dropout (0.3)
  - Classification layer (512 → num_classes)

### EfficientNet-based Classifier (Lightweight)
- Backbone: EfficientNet-B0 (pretrained)
- Optimized for mobile and edge devices

## Training Features

- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
- **Early Stopping**: Prevents overfitting with configurable patience
- **Checkpointing**: Saves best model based on validation loss
- **Data Augmentation**: Random rotation, flip, color jitter, and resizing
- **Progress Tracking**: Real-time metrics with tqdm progress bars

## Image Preprocessing

The `preprocessing.py` module provides:

1. **Resize**: Smart resizing with aspect ratio preservation
2. **Contrast Enhancement**: CLAHE for improved visual features
3. **Edge Detection**: Canny and Sobel edge detection
4. **Background Removal**: Morphological operations and GrabCut
5. **Normalization**: ImageNet normalization constants

Example usage:
```python
from src.preprocessing import ImagePreprocessor

preprocessor = ImagePreprocessor()

# Resize image
resized = preprocessor.resize_image('image.jpg', target_size=(224, 224))

# Enhance contrast
enhanced = preprocessor.enhance_contrast(resized, clip_limit=2.0)

# Detect edges
edges = preprocessor.detect_edges(enhanced, method='canny')

# Remove background
no_bg = preprocessor.remove_background('image.jpg', method='grabcut')
```

## API Reference

### Dataset
```python
from src.dataset import ProductImageDataset, get_default_transforms

# Create dataset
dataset = ProductImageDataset(image_paths, labels, transform=transform)

# Get transforms
transform = get_default_transforms(image_size=224, augment=True)
```

### Models
```python
from src.models import create_model

# Create model
model = create_model(
    model_name='resnet50',
    num_classes=10,
    pretrained=True,
    device='cuda'
)
```

### Training
```python
from src.training import Trainer

# Create trainer
trainer = Trainer(model, device='cuda', learning_rate=1e-3)

# Train
history = trainer.train(train_loader, val_loader, num_epochs=20)
```

### Inference
```python
from inference import ImageClassifier

# Initialize
classifier = ImageClassifier(
    'models/saved/best_model.pth',
    'models/saved/label_map.json'
)

# Predict
predictions = classifier.predict('image.jpg', return_top_k=3)
```

## Performance

Expected performance depends on:
- Dataset size (minimum 100 images per class recommended)
- Image quality and diversity
- Number of classes
- Training duration and hyperparameters

Typical results on well-balanced datasets:
- **Validation Accuracy**: 85-95%
- **Training Time**: 2-10 minutes per epoch (depending on GPU)

## Troubleshooting

### Out of Memory Error
- Reduce batch size in `config.yaml`
- Use EfficientNet instead of ResNet50
- Reduce image size

### Low Accuracy
- Ensure sufficient training data (100+ per class)
- Increase training epochs
- Check data quality and class balance
- Use data augmentation

### CUDA Not Available
- Install CUDA 11.0+ and cuDNN
- Or set `device: 'cpu'` in config.yaml

## Dependencies

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- OpenCV >= 4.8.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Matplotlib >= 3.8.0
- scikit-learn >= 1.3.0
- Pillow >= 10.0.0
- tqdm >= 4.66.0

See `requirements.txt` for complete list.

## License

This project is provided as-is for educational and commercial use.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## References

- ResNet: He et al., 2015 - "Deep Residual Learning for Image Recognition"
- EfficientNet: Tan & Le, 2019 - "EfficientNet: Rethinking Model Scaling"
- PyTorch: https://pytorch.org/
- OpenCV: https://opencv.org/

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the Jupyter notebook examples
3. Examine the API documentation in source files

## Next Steps

- Experiment with different architectures
- Fine-tune hyperparameters
- Deploy using ONNX or TorchServe
- Integrate with production pipelines
- Add real-time inference capabilities
