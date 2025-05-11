# Indian Classical Dance Classification

This project implements a deep learning model to classify different forms of Indian classical dance using computer vision and neural networks.

## Project Overview

The project uses a Convolutional Neural Network (CNN) based on ResNet18 architecture to classify eight different forms of Indian classical dance:

- Bharatanatyam
- Kathak
- Kathakali
- Kuchipudi
- Manipuri
- Mohiniyattam
- Odissi
- Sattriya

## Project Structure

```
indiandanceclassification/
├── CNN.ipynb                              # Main training notebook
├── dance_model.pth                        # Trained model weights
├── gradcam_*.jpg                          # Grad-CAM visualizations
├── resnet18-gradcam-visualization-flat.py # Visualization script
├── README.md                              # Project documentation
└── final/                                 # Dataset directory 
    ├── dataset.py                         # Dataset handling code
    ├── test.csv                          # Test set metadata
    ├── train.csv                         # Training set metadata
    ├── csv2/                             # Additional metadata
    ├── test/                             # Test images
    ├── train/                            # Training images
    └── validation/                       # Validation images
```

## Technical Details

### Model Architecture
- Base model: ResNet18 (pretrained)
- Modified final fully connected layer for 8-class classification
- Input size: 224x224 pixels
- Data augmentation: Random cropping, horizontal flipping

### Training Configuration
- Optimizer: SGD with momentum (0.9)
- Learning rate: 0.001 with step decay
- Batch size: 32
- Number of epochs: 24
- Loss function: Cross Entropy Loss

### Performance
- The model achieves 100% validation accuracy
- Training completed in approximately 63 minutes
- Uses CUDA for GPU acceleration when available

### Explainability
The project includes Grad-CAM visualizations to provide interpretability of model decisions, showing which parts of the dance images the model focuses on for classification.

## Requirements
- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- grad-cam

## Usage

1. Training the model:
```python
python CNN.ipynb
```

2. Generating visualizations:
```python
python resnet18-gradcam-visualization-flat.py
```

## Model Artifacts
- The trained model weights are saved in `dance_model.pth`
- Grad-CAM visualizations are saved as `gradcam_*.jpg`

## Dataset Organization
- Training data is organized in class-specific folders
- Images are preprocessed and normalized
- Data is split into train, validation, and test sets

## Citation
If you use this project, please cite our work:
[Project citation to be added]