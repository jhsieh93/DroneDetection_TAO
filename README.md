# Drone Detection System using NVIDIA TAO Toolkit

End-to-end object detection pipeline for autonomous drone detection using NVIDIA's Transfer Learning Toolkit (TAO) and DetectNet_v2 architecture.

## Project Overview

This project demonstrates building a production-ready object detection system for drone identification using:
- **NVIDIA TAO Toolkit** for transfer learning
- **DetectNet_v2** architecture (ResNet-18 backbone)
- **KITTI format** dataset processing
- **TensorRT** optimization for deployment

## Performance

- **Mean Average Precision (mAP):** 84%
- **Training Time:** 80 epochs on NVIDIA GPU
- **Inference:** Real-time detection with TensorRT

## Technical Stack

- NVIDIA TAO Toolkit 4.0.0
- Docker containerization
- Python data preprocessing
- KITTI dataset format
- ResNet-18 backbone

## Dataset Preparation

Dataset organized in KITTI format:
```
Dataset/
├── kitti/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   └── test/
└── tfrecords/
```

## Training Configuration

Key hyperparameters:
- Architecture: ResNet-18
- Input resolution: 1248x384
- Batch size: 8
- Optimizer: Adam (lr=0.0001)
- Augmentation: Random flip, brightness/contrast adjustment
- Training epochs: 80

## Project Structure
```
DroneDetection/
├── detectnet_v2_official.txt    # Training spec file
├── inference_spec.txt            # Inference configuration
├── Dataset/                      # Training/validation/test data
├── output/                       # Model weights and results
└── README.md
```

## Results

Achieved 84% mAP on validation set, demonstrating effective transfer learning for drone detection tasks.

## Skills Demonstrated

- Deep learning model training and optimization
- Docker containerization for ML workflows
- Dataset preparation and format conversion
- Transfer learning with pre-trained models
- GPU-accelerated training
- Production ML pipeline development

## Author

Jordan Hsieh - Technical Solutions Manager
Transitioning into AI/ML product development roles

## Acknowledgments

Built using NVIDIA TAO Toolkit and DetectNet_v2 architecture.