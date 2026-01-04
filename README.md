**NTK Divergence in Classification**
This repository contains a reproducibility study and extension of the ICLR 2025 paper “Divergence of Empirical Neural Tangent Kernel in Classification Problems” by Yu, Tian, and Chen.
The project reproduces the paper’s main experiments (synthetic circle and MNIST odd–even classification) and adds ablations on width, depth, loss functions, initialization, dataset separability, training horizon, and architecture.
ntk_divergence_project/
├── data/                 # Circle dataset + MNIST odd/even loaders
├── models/               # FCN and ResNet-style MLPs
├── ntk/                  # Empirical NTK computation and eigenvalue analysis
├── training/             # Training and evaluation loops, metrics
├── experiments/          # Reproduction + ablation scripts
├── config/               # RTX 3050–friendly configs (YAML)
├── results/              # Logs and plots

Key components:

data/generate_circle.py: 6‑point unit circle dataset with alternating labels.
data/mnist_loader.py: MNIST odd vs even binary classification loader.​
models/fcn.py: Fully connected networks used in all baseline experiments.
models/resnet.py: Simple ResNet‑style MLP for architecture ablations.
​ntk/ntk_computation.py: Empirical NTK computation using PyTorch autograd / torch.func.
​experiments/reproduce_circle.py: Reproduces the circle divergence experiment.
experiments/reproduce_mnist.py: Reproduces the MNIST NTK divergence experiment.

Installation
This repo is designed for Python 3.10 and CUDA‑enabled PyTorch (cu121) on an RTX 3050.

Create and activate environment:

bash
conda create -n ntk-divergence python=3.10
conda activate ntk-divergence
Install PyTorch with CUDA 12.1 and dependencies:

bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy matplotlib seaborn jupyter tqdm pyyaml
Clone this repository:

bash
git clone https://github.com/subh123321/ntk_divergence_project.git
cd ntk_divergence_project
Usage
All commands assume you are in the project root with the ntk-divergence environment active.

Running Experiments
Circle (FCN)
python -m experiments.reproduce_circle

Circle (ResNet)
python -m experiments.reproduce_circle_resnet

MNIST ResNet (Minimal NTK)
python -m experiments.reproduce_mnist_resnet

Ablations
python -m experiments.ablations.ablation_width
python -m experiments.ablations.ablation_loss
python -m experiments.ablations.ablation_depth
...

Generated plots are saved to:

results/plots/


Experiments Implemented
1️⃣ Synthetic Circle Dataset

FCN and ResNet architectures

Output divergence: 
<img width="179" height="46" alt="image" src="https://github.com/user-attachments/assets/821e7a8f-277d-4314-86ae-0be0cf8d1f92" />


NTK eigenvalue evolution

NTK divergence measurement

2️⃣ MNIST (Binary Classification)

Fully Connected ResNet

Fixed sample output divergence

NTK diagonal divergence:
<img width="342" height="95" alt="image" src="https://github.com/user-attachments/assets/a1e2f510-2d2e-4bcf-ba57-b03d68f43a99" />


🔬 Ablation Studies

The following ablations are fully implemented:

Ablation	Description
Ablation 1	Width scaling (64 → 2048)
Ablation 2	Loss functions (BCE, MSE, Hinge, Focal)
Ablation 3	Network depth (2–5 layers)
Ablation 4	Initialization scale sensitivity
Ablation 5	Dataset separability (easy → hard)
Ablation 6	Training duration (10k → 50k epochs)
Ablation 7	Activation functions (ReLU, GELU, Tanh)

Each ablation measures NTK divergence:
<img width="268" height="100" alt="image" src="https://github.com/user-attachments/assets/9c977a03-c39a-40e5-b4ba-7c42009fe88d" />


Key Findings

Empirical NTK diverges under cross-entropy loss

Divergence increases with depth

Divergence decreases with width

MSE loss preserves near-constant NTK

ResNets delay but do not prevent divergence

MNIST exhibits NTK divergence similar to synthetic data

These results match the claims in the ICLR 2025 paper.

Reference

If you use this repository, please cite:

@inproceedings{ntk_divergence_2025,
  title={Divergence of Empirical Neural Tangent Kernel in Classification Problems},
  booktitle={ICLR},
  year={2025}
}

Author

Subhra Jyoti Das
IIT Roorkee
Reproducibility Track Project

	​

	​
