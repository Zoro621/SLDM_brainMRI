# SBLDM Paper Analysis: Slice-By-Slice Lightweight Latent Diffusion Model

## 1. Method Summary (Plain Language)

SBLDM (Slice-By-Slice Lightweight Latent Diffusion Model) is a two-stage generative model designed for medical image synthesis:

### Stage 1: VAE Training
- A Variational Autoencoder compresses high-dimensional medical images (e.g., MRI slices) into a compact latent space
- The encoder learns a downsampled representation (typically 4x-8x spatial reduction)
- The decoder reconstructs images from latent codes
- Uses KL divergence regularization to ensure smooth latent space

### Stage 2: Latent Diffusion Training
- A lightweight UNet learns to denoise in the compressed latent space
- Forward process: gradually adds Gaussian noise to latent codes following a noise schedule
- Reverse process: learns to predict and remove noise, enabling generation
- Operates on much smaller tensors than pixel-space diffusion (faster training, less memory)

### Inference (Sampling)
- Start from pure Gaussian noise in latent space
- Iteratively denoise using the trained UNet
- Decode final latent code through VAE decoder to get synthetic image
- Supports DDPM (slow, high quality) and DDIM (fast, configurable steps)

---

## 2. Exact Components and Training Stages

### Architecture Components:
1. **VAE Encoder**: Conv layers with downsampling (stride 2), outputs μ and σ for latent distribution
2. **VAE Decoder**: Transposed convolutions for upsampling, reconstructs image
3. **UNet Score Network**: 
   - Time embedding (sinusoidal positional encoding)
   - Encoder path: 3 down blocks with residual connections
   - Bottleneck: attention or residual block
   - Decoder path: 3 up blocks with skip connections
4. **Noise Schedule**: Linear or cosine β schedule
5. **Sampler**: DDPM or DDIM reverse diffusion

### Training Stages:
| Stage | Input | Output | Loss | Typical Duration |
|-------|-------|--------|------|------------------|
| 1. VAE | Raw images | Reconstructed images + latents | MSE + KL | 50-100 epochs |
| 2. Diffusion | Latent codes | Predicted noise | MSE (ε-prediction) | 100-500 epochs |

### Key Hyperparameters from Paper (ASSUMPTION: typical LDM settings):
- Latent channels: 4
- Downsampling factor: 4x (128×128 → 32×32 latent)
- UNet channels: [64, 128, 256]
- Diffusion timesteps: 1000
- β schedule: linear [1e-4, 0.02] or cosine

---

## 3. Identified Gaps, Weaknesses, and Unstated Assumptions

### 3.1 Architectural Gaps
| Gap | Description | Impact |
|-----|-------------|--------|
| **3D vs 2D limitation** | ASSUMPTION: Paper likely focuses on 3D volumes; our 2D slice approach loses inter-slice context | May affect anatomical consistency |
| **Missing attention scales** | Unclear if attention is used at single or multiple resolutions | Affects quality vs speed tradeoff |
| **VAE bottleneck size** | Optimal latent dimension not ablated | Too small = blurry; too large = slow diffusion |

### 3.2 Training Gaps
| Gap | Description | Proposed Fix |
|-----|-------------|--------------|
| **Noise schedule choice** | Linear vs cosine not compared | We implement both + gamma-rebalancing |
| **Loss weighting** | SNR weighting not discussed | Add frequency-aware loss option |
| **Data augmentation** | Augmentation in latent space unexplored | Implement Latent CutMix |
| **Early stopping criteria** | No adaptive sampling discussed | Implement score-norm threshold |

### 3.3 Evaluation Gaps
| Gap | Description | Our Addition |
|-----|-------------|--------------|
| **Small-sample FID** | FID unreliable with <2048 samples | Implement FID with confidence intervals |
| **Reconstruction heatmaps** | Missing error visualization | Add per-pixel error maps |
| **Downstream utility** | Generation quality ≠ clinical utility | Add segmentation augmentation test |
| **Perceptual metrics** | Only pixel-wise metrics | Add LPIPS if feasible |

### 3.4 Reproducibility Gaps
| Gap | Description |
|-----|-------------|
| **Memory footprint** | Actual VRAM usage not reported |
| **Training time** | Wall-clock time per epoch missing |
| **Random seeds** | Reproducibility seeds not provided |
| **Dataset splits** | Exact train/val/test splits unclear |

---

## 4. Proposed Experiments for Kaggle Reproduction

### Priority 1: Core Validation (Quick Runs - 1-2 hours)
| Experiment | Goal | Config |
|------------|------|--------|
| E1.1 VAE Reconstruction | Verify VAE learns meaningful latents | 20 epochs, 128×128, bs=16 |
| E1.2 Diffusion Sanity | Verify loss decreases | 500 steps, 4 latent channels |
| E1.3 DDIM Sampling | Verify can generate images | 50 DDIM steps |

### Priority 2: Novel Contributions (Medium Runs - 4-6 hours)
| Experiment | Goal | Config |
|------------|------|--------|
| E2.1 Noise Schedule Ablation | Compare linear vs cosine vs γ-rebalanced | γ ∈ {0.5, 1.0, 2.0} |
| E2.2 Frequency-Aware Loss | Test FFT loss impact on sharpness | λ_freq ∈ {0.01, 0.1, 0.5} |
| E2.3 Latent CutMix | Test augmentation effect | cutmix_prob ∈ {0.0, 0.25, 0.5} |
| E2.4 Adaptive Sampling | Test early-stop quality | threshold ∈ {0.01, 0.05, 0.1} |

### Priority 3: Full Reproduction (Extended - 8+ hours)
| Experiment | Goal | Config |
|------------|------|--------|
| E3.1 Full VAE Training | Match paper reconstruction | 100 epochs, best LR |
| E3.2 Full Diffusion Training | Match paper generation | 50K+ steps |
| E3.3 FID/SSIM Benchmark | Compare to paper metrics | 1000 generated samples |

---

## 5. Recommended Hyperparameters for Kaggle T4/P100

### Hardware Constraints:
- T4: 16GB VRAM, ~8.1 TFLOPS FP32
- P100: 16GB VRAM, ~9.3 TFLOPS FP32
- Kaggle session: 9-12 hour limit, intermittent disconnects

### VAE Configuration (128×128):
```yaml
vae:
  in_channels: 1  # grayscale MRI
  latent_channels: 4
  hidden_dims: [32, 64, 128, 256]
  downsample_factor: 4  # 128→32 latent spatial
  batch_size: 32  # fits T4 comfortably
  learning_rate: 1e-4
  epochs: 50
  kl_weight: 1e-5  # start low, anneal up
  
  # Memory optimization
  mixed_precision: true
  gradient_checkpointing: false  # not needed for small VAE
```

### UNet Diffusion Configuration:
```yaml
unet:
  in_channels: 4  # latent channels
  model_channels: 64
  channel_mult: [1, 2, 4]  # [64, 128, 256]
  num_res_blocks: 2
  attention_resolutions: [8]  # attention at 8×8 only
  dropout: 0.1
  
diffusion:
  timesteps: 1000
  beta_schedule: "cosine"  # or "linear", "gamma_rebalanced"
  beta_start: 1e-4
  beta_end: 0.02
  gamma: 1.0  # for noise rebalancing
  
  # Training
  batch_size: 16  # latent batches fit easily
  learning_rate: 2e-4
  warmup_steps: 500
  training_steps: 50000
  
  # Sampling
  ddim_steps: 50
  eta: 0.0  # deterministic DDIM
```

### Memory Budget Estimate:
| Component | VRAM (FP16) | VRAM (FP32) |
|-----------|-------------|-------------|
| VAE (train) | ~2 GB | ~4 GB |
| UNet (train) | ~4 GB | ~8 GB |
| Latent batch (32×4×32×32) | ~0.5 GB | ~1 GB |
| **Total headroom** | ~10 GB | ~13 GB |

### Quick Debug Configuration (64×64):
```yaml
# For fast iteration
resolution: 64
vae_hidden_dims: [32, 64, 128]
unet_channels: 32
channel_mult: [1, 2, 2]
batch_size: 64
training_steps: 1000
ddim_steps: 20
```

---

## 6. Novel Contributions Implementation Plan

### Contribution 1: Noise Rebalancing Schedule (γ-schedule)
**Motivation**: Standard linear/cosine schedules may not be optimal for medical images which have different noise characteristics than natural images.

**Implementation**:
```python
β(t) = β_min + (β_max - β_min) * (t/T)^γ

# γ < 1: More noise early, less late (gentler denoising)
# γ = 1: Linear schedule
# γ > 1: Less noise early, more late (aggressive early denoising)
```

**Hypothesis**: Medical images with fine anatomical details may benefit from γ < 1 (preserving structure longer).

### Contribution 2: Frequency-Aware Diffusion Loss
**Motivation**: Standard MSE treats all frequencies equally, but medical images have important high-frequency anatomical edges.

**Implementation**:
```python
L_total = L_mse(ε_pred, ε_true) + λ * L_freq(ε_pred, ε_true)

L_freq = MSE(FFT(ε_pred), FFT(ε_true))  # or weighted by frequency
```

**Hypothesis**: Adding frequency loss improves edge sharpness in generated images.

---

## 7. Assumptions Log

| ID | Assumption | Rationale | Risk |
|----|------------|-----------|------|
| A1 | Paper uses 2D slices independently | Common for LDM medical imaging | Low |
| A2 | β schedule is linear [1e-4, 0.02] | Standard LDM defaults | Low |
| A3 | UNet uses sinusoidal time embedding | Universal in diffusion models | Very Low |
| A4 | VAE uses MSE + KL loss | Standard VAE training | Very Low |
| A5 | Latent channels = 4 | Stable Diffusion default | Medium |
| A6 | No class conditioning | Unconditional generation | Medium |
| A7 | Grayscale input (1 channel) | Brain MRI typically grayscale | Low |

---

## 8. Expected Outputs and Success Criteria

| Metric | Baseline (Untrained) | Target (Trained) |
|--------|---------------------|------------------|
| VAE Reconstruction SSIM | ~0.3 | >0.85 |
| VAE Reconstruction PSNR | ~15 dB | >25 dB |
| Generated FID | ~300 | <100 (ideally <50) |
| DDIM 50-step generation time | - | <5 sec/image |
| Training step time | - | <200 ms/step |

---

*Document generated for Kaggle SBLDM reproduction project*
*Last updated: 2024*
