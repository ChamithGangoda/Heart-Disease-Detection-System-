# Chest X-ray Disease Classifier (ResNet-34)

This repository contains a PyTorch implementation of a deep learning classifier based on **ResNet-34** to identify six major thoracic conditions from chest X-ray images:
* Pneumonia
* Effusion
* Atelectasis
* Nodule
* Cardiomegaly
* Mass

## Features
- **Transfer Learning:** Fine-tunes a pretrained ResNet-34 backbone with a custom MLP classifier head.
- **Layer Unfreezing:** Starts with frozen early layers for stable training, then unfreezes all layers after epoch 10 for full network fine-tuning.
- **Data Pipeline:** Utilizes custom PyTorch dataloaders with heavy data augmentation (rotation, cropping, color jitter, affine transforms).
- **Optimization:** Employs the `AdamW` optimizer, `CosineAnnealingWarmRestarts` learning rate scheduling, and automatic mixed precision (AMP) training for fast convergence.
