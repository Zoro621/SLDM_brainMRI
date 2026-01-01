# SBLDM: Slice-By-Slice Lightweight Latent Diffusion Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Kaggle-Ready-20BEFF.svg)](https://www.kaggle.com/)

A **Kaggle-ready PyTorch implementation** of a scaled-down Slice-By-Slice Lightweight Latent Diffusion Model for 2D medical image synthesis. Optimized for T4/P100 GPUs (16-32GB VRAM).

## 🎯 Key Features

- **Lightweight VAE**: Configurable encoder-decoder with residual blocks for 4× latent compression
- **Efficient UNet**: Score prediction network with attention layers and time embeddings
- **Flexible Sampling**: DDPM, DDIM, and Adaptive sampling strategies
- **Novel Contributions**:
  - 🆕 **Noise Rebalancing Schedule**: Gamma-based schedule for enhanced structural preservation
  - 🆕 **Frequency-Aware Diffusion Loss**: FFT-based loss balancing low/high frequency features
  - 🆕 **Latent CutMix**: Augmentation strategy in latent space
  - 🆕 **Adaptive Sampling**: Early-stop sampling based on convergence detection

## 📁 Project Structure

```
genAI_project/
├── configs/                    # Training configurations
│   ├── config_debug.yaml      # Quick smoke test (64×64, 5 epochs)
│   ├── config_medium.yaml     # Medium run (128×128, 50 epochs)
│   └── config_full.yaml       # Full training (128×128, 200 epochs)
├── data/                       # Data loading and preprocessing
│   ├── preprocess.py          # BraTS/custom dataset preprocessing
│   └── dataset.py             # PyTorch datasets and dataloaders
├── models/                     # Model architectures
│   ├── vae.py                 # Variational Autoencoder
│   ├── unet_score.py          # UNet for score prediction
│   └── ema.py                 # Exponential Moving Average
├── diffusion/                  # Diffusion utilities
│   ├── schedules.py           # Noise schedules (linear, cosine, gamma-rebalanced)
│   ├── loss.py                # Loss functions (simple, frequency-aware, cutmix)
│   └── samplers.py            # DDPM, DDIM, Adaptive samplers
├── eval/                       # Evaluation metrics
│   └── metrics.py             # SSIM, PSNR, FID computation
├── tests/                      # Validation tests
│   └── smoke_test.py          # GPU OOM and functionality tests
├── notebooks/                  # Jupyter notebooks
│   └── kaggle_sbldm.ipynb     # End-to-end Kaggle notebook
├── report/                     # Research report template
│   └── report_template.md     # LaTeX-style report structure
├── train_vae.py               # VAE training script
├── train_diffusion.py         # Diffusion model training script
├── sample.py                  # Image generation script
├── requirements.txt           # Python dependencies
└── paper_analysis.md          # Paper analysis and gaps
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sbldm-medical.git
cd sbldm-medical

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running on Kaggle

1. **Upload to Kaggle**: Create a new notebook and upload the repository
2. **Open notebook**: Use `notebooks/kaggle_sbldm.ipynb`
3. **Enable GPU**: Settings → Accelerator → GPU T4 ×2 (or P100)
4. **Run all cells**: The notebook handles setup, training, and evaluation

### Local Training

```bash
# Quick smoke test (5 minutes)
python train_vae.py --config configs/config_debug.yaml
python train_diffusion.py --config configs/config_debug.yaml

# Medium run (2-3 hours on T4)
python train_vae.py --config configs/config_medium.yaml
python train_diffusion.py --config configs/config_medium.yaml

# Full training (8-12 hours on T4)
python train_vae.py --config configs/config_full.yaml
python train_diffusion.py --config configs/config_full.yaml
```

### Generate Samples

```bash
# Using DDIM (faster)
python sample.py --config configs/config_medium.yaml --sampler ddim --num_samples 16

# Using DDPM
python sample.py --config configs/config_medium.yaml --sampler ddpm --num_samples 16

# Using Adaptive sampling
python sample.py --config configs/config_medium.yaml --sampler adaptive --num_samples 16
```

## 📊 Priority Experiments

### Experiment 1: Debug Run (5-10 minutes on T4)
**Goal**: Verify pipeline works end-to-end
```bash
python train_vae.py --config configs/config_debug.yaml
python train_diffusion.py --config configs/config_debug.yaml
python sample.py --config configs/config_debug.yaml --num_samples 4
```
- **Resolution**: 64×64
- **Epochs**: VAE=5, Diffusion=5
- **Expected**: Blurry but recognizable reconstructions

### Experiment 2: Medium Run (2-4 hours on T4)
**Goal**: Train reasonable quality model
```bash
python train_vae.py --config configs/config_medium.yaml
python train_diffusion.py --config configs/config_medium.yaml
python sample.py --config configs/config_medium.yaml --sampler ddim --num_samples 16
python -c "from eval.metrics import compute_metrics_on_folder; compute_metrics_on_folder('outputs/samples', 'data/processed/val')"
```
- **Resolution**: 128×128
- **Epochs**: VAE=30, Diffusion=50
- **Expected**: SSIM > 0.75, PSNR > 22dB

### Experiment 3: Full Run with Ablations (8-12 hours on T4)
**Goal**: Compare novel contributions
```bash
# Baseline (linear schedule, simple loss)
python train_diffusion.py --config configs/config_full.yaml \
    --schedule linear --loss simple

# Novel: Gamma-rebalanced schedule
python train_diffusion.py --config configs/config_full.yaml \
    --schedule gamma_rebalanced --loss simple

# Novel: Frequency-aware loss
python train_diffusion.py --config configs/config_full.yaml \
    --schedule linear --loss frequency_aware

# Novel: Both contributions
python train_diffusion.py --config configs/config_full.yaml \
    --schedule gamma_rebalanced --loss frequency_aware
```
- **Expected improvements**: +2-5% SSIM, +1-2dB PSNR with novel contributions

## ⏱️ Runtime Estimates (Kaggle T4 GPU)

| Configuration | VAE Training | Diffusion Training | Sampling (16 imgs) | Total |
|--------------|--------------|-------------------|-------------------|-------|
| Debug | 2-3 min | 3-5 min | 30 sec | ~10 min |
| Medium | 20-30 min | 1.5-2.5 hrs | 1 min | ~3 hrs |
| Full | 1-1.5 hrs | 6-10 hrs | 2 min | ~12 hrs |

## 💾 Checkpoint and Resume

All training scripts support checkpoint/resume:

```bash
# Resume VAE training from checkpoint
python train_vae.py --config configs/config_full.yaml \
    --resume checkpoints/vae_epoch_50.pth

# Resume diffusion training
python train_diffusion.py --config configs/config_full.yaml \
    --resume checkpoints/diffusion_epoch_100.pth
```

Checkpoints are saved every N epochs (configurable in YAML) and include:
- Model state dict
- Optimizer state dict
- EMA weights
- Current epoch
- Best validation loss
- Training configuration

## 📈 Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **SSIM** | Structural Similarity Index | > 0.80 |
| **PSNR** | Peak Signal-to-Noise Ratio | > 25 dB |
| **FID** | Fréchet Inception Distance | < 50 |

```python
from eval.metrics import compute_ssim, compute_psnr, compute_fid

# Compute metrics
ssim = compute_ssim(generated_images, real_images)
psnr = compute_psnr(generated_images, real_images)
fid = compute_fid(generated_images, real_images, device='cuda')
```

## 🔬 Novel Contributions

### 1. Noise Rebalancing Schedule (Gamma-based)
Modifies the standard linear/cosine schedule with a gamma correction to preserve structural information at critical timesteps:

```python
from diffusion.schedules import get_schedule

schedule = get_schedule(
    schedule_type='gamma_rebalanced',
    num_timesteps=1000,
    gamma=0.75  # Lower = more structure preservation
)
```

### 2. Frequency-Aware Diffusion Loss
Balances low-frequency (structure) and high-frequency (detail) components using FFT:

```python
from diffusion.loss import FrequencyAwareLoss

loss_fn = FrequencyAwareLoss(
    low_freq_weight=1.0,
    high_freq_weight=0.5,
    cutoff_ratio=0.25  # Fraction of spectrum considered "low frequency"
)
```

### 3. Latent CutMix Augmentation
Augments latent representations during training:

```python
from diffusion.loss import LatentCutMix

cutmix = LatentCutMix(alpha=1.0, prob=0.5)
mixed_latents, mixed_targets = cutmix(latents, targets)
```

### 4. Adaptive Sampling
Early-stops sampling when convergence is detected:

```python
from diffusion.samplers import AdaptiveSampler

sampler = AdaptiveSampler(
    model, schedule,
    threshold=1e-4,  # Convergence threshold
    patience=5       # Steps below threshold to stop
)
```

## 📋 Reproducibility Checklist

- [ ] **Environment**: Python 3.8+, PyTorch 2.0+, CUDA 11.7+
- [ ] **Random Seeds**: Set in config (default: 42)
- [ ] **Data**: Preprocessed to 128×128 grayscale PNG
- [ ] **Hardware**: T4/P100 GPU with 16GB+ VRAM
- [ ] **Mixed Precision**: Enabled by default for memory efficiency
- [ ] **Gradient Accumulation**: Supported for larger effective batch sizes

```bash
# Verify GPU setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# Run smoke tests
python tests/smoke_test.py
```

## 🗂️ Dataset Preparation

### BraTS Dataset
```bash
# Download BraTS 2020/2021 from Kaggle
# Place in data/raw/brats/

python data/preprocess.py \
    --input_dir data/raw/brats \
    --output_dir data/processed \
    --size 128 \
    --modality t1
```

### Custom Dataset
Place 2D medical images (PNG/JPG) in `data/raw/custom/`:
```
data/raw/custom/
├── train/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── val/
    ├── image_101.png
    └── ...
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{sbldm_pytorch,
  author = {Your Name},
  title = {SBLDM: Slice-By-Slice Lightweight Latent Diffusion Model},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/sbldm-medical}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original SBLDM paper authors
- PyTorch team
- Hugging Face diffusers library (for reference implementations)
- BraTS challenge organizers
