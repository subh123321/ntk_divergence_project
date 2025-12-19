# RTX 3050 Setup Guide

## 1. System requirements
- NVIDIA RTX 3050 (6GB VRAM)
- Recent NVIDIA driver (supporting CUDA 12.x)
- Python 3.10

## 2. Install CUDA-compatible PyTorch
(Commands as shown in README / environment section.)

## 3. VRAM management tips
- Use `mixed_precision` where possible.
- Keep batch_size small when computing NTK.
- Run eigenvalue computations on CPU.
