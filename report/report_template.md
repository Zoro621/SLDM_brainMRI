# Slice-By-Slice Lightweight Latent Diffusion for Medical Image Synthesis

**Authors**: [Your Name]  
**Date**: [Date]  
**Institution**: [Your Institution]

---

## Abstract

*[200-300 words summarizing the work]*

This paper presents an efficient implementation of Slice-By-Slice Lightweight Latent Diffusion Model (SBLDM) for 2D medical image synthesis. We introduce two novel contributions: (1) a **Noise Rebalancing Schedule** using gamma correction to preserve structural information during diffusion, and (2) a **Frequency-Aware Diffusion Loss** that balances low and high-frequency components using FFT decomposition. Our approach achieves competitive SSIM/PSNR scores while being optimized for resource-constrained environments (Kaggle T4 GPU). Experiments on [BraTS/custom dataset] demonstrate [key findings]. The complete implementation is publicly available.

**Keywords**: Diffusion Models, Medical Imaging, Latent Diffusion, MRI Synthesis, Deep Learning

---

## 1. Introduction

### 1.1 Motivation

Medical image synthesis is crucial for [applications]. Traditional methods suffer from [limitations]. Diffusion models have shown promising results but face computational challenges.

### 1.2 Problem Statement

Given [input specification], our goal is to [objective]. The key challenges include:
- Memory constraints on consumer GPUs
- Preserving anatomical structure fidelity
- Balancing generation speed with quality

### 1.3 Contributions

1. **Noise Rebalancing Schedule**: A gamma-based modification to standard diffusion schedules that improves structural preservation at critical timesteps.

2. **Frequency-Aware Diffusion Loss**: An FFT-based loss function that separately weights low-frequency (structure) and high-frequency (detail) components.

3. **Kaggle-Ready Implementation**: A complete, reproducible codebase optimized for T4/P100 GPUs with mixed precision training.

4. *[Additional contributions: Latent CutMix, Adaptive Sampling]*

---

## 2. Related Work

### 2.1 Diffusion Models

*[Review of DDPM, DDIM, Score-based models]*

Denoising Diffusion Probabilistic Models (DDPM) [Ho et al., 2020] established the foundation for modern diffusion-based generative models. Key subsequent works include:
- DDIM [Song et al., 2020]: Deterministic sampling for faster generation
- Score-based models [Song & Ermon, 2019]: Alternative formulation via score matching
- Latent Diffusion [Rombach et al., 2022]: Diffusion in compressed latent space

### 2.2 Medical Image Synthesis

*[Review of GAN-based and diffusion-based medical imaging works]*

Traditional approaches:
- GANs: Pix2Pix, CycleGAN for modality translation
- VAEs: For reconstruction and generation
- Recent diffusion models: [List relevant medical imaging diffusion papers]

### 2.3 Efficient Diffusion Models

*[Review of lightweight and efficient diffusion approaches]*

- Knowledge distillation
- Progressive distillation
- Latent space diffusion
- Efficient attention mechanisms

---

## 3. Methods

### 3.1 Problem Formulation

Let $x_0 \in \mathbb{R}^{H \times W \times C}$ be a medical image. The goal is to learn a generative model $p_\theta(x_0)$ that can sample new images from the data distribution.

### 3.2 Architecture Overview

```
Input Image (128×128×1)
        ↓
   VAE Encoder
        ↓
  Latent (32×32×4)
        ↓
   UNet + Time
        ↓
   Noise Prediction
        ↓
   DDIM/DDPM Sampler
        ↓
   VAE Decoder
        ↓
Output Image (128×128×1)
```

### 3.3 Variational Autoencoder (VAE)

The VAE compresses images to a 4× smaller latent space:

$$z = \text{Enc}(x), \quad \hat{x} = \text{Dec}(z)$$

Training objective:
$$\mathcal{L}_{\text{VAE}} = \mathbb{E}[\|x - \hat{x}\|_1] + \beta \cdot D_{\text{KL}}(q(z|x) \| p(z))$$

**Architecture Details**:
- Encoder: 4 downsampling blocks with residual connections
- Latent channels: 4
- Decoder: 4 upsampling blocks with skip connections

### 3.4 UNet Score Network

The UNet predicts noise $\epsilon_\theta(z_t, t)$ given noisy latent $z_t$ and timestep $t$:

**Time Embedding**:
$$\gamma(t) = [\sin(2\pi \omega_1 t), \cos(2\pi \omega_1 t), ..., \sin(2\pi \omega_d t), \cos(2\pi \omega_d t)]$$

**Architecture**:
- 3 downsampling blocks (64→128→256 channels)
- Middle block with self-attention
- 3 upsampling blocks with skip connections
- Attention at 16×16 resolution

### 3.5 Novel Contribution 1: Noise Rebalancing Schedule

Standard linear schedule: $\beta_t = \beta_{\min} + \frac{t}{T}(\beta_{\max} - \beta_{\min})$

**Our gamma-rebalanced schedule**:
$$\beta_t^{\text{rebalanced}} = \beta_t \cdot \left(\frac{t}{T}\right)^\gamma$$

where $\gamma < 1$ reduces noise at early timesteps (preserving structure) and increases it at later timesteps (focusing on details).

**Intuition**: Medical images have critical structural components (organs, boundaries) that should be preserved. By reducing noise injection at early timesteps, we maintain structural fidelity.

### 3.6 Novel Contribution 2: Frequency-Aware Diffusion Loss

Standard diffusion loss: $\mathcal{L}_{\text{simple}} = \|\epsilon - \epsilon_\theta(z_t, t)\|^2$

**Our frequency-aware loss**:

$$\mathcal{L}_{\text{freq}} = \lambda_{\text{low}} \|\mathcal{F}_{\text{low}}(\epsilon - \epsilon_\theta)\|^2 + \lambda_{\text{high}} \|\mathcal{F}_{\text{high}}(\epsilon - \epsilon_\theta)\|^2$$

where $\mathcal{F}_{\text{low}}$ and $\mathcal{F}_{\text{high}}$ are low-pass and high-pass filters via FFT:

```python
def frequency_decompose(x, cutoff=0.25):
    fft = torch.fft.fft2(x)
    H, W = x.shape[-2:]
    mask = create_circular_mask(H, W, cutoff)
    low_freq = torch.fft.ifft2(fft * mask).real
    high_freq = torch.fft.ifft2(fft * (1 - mask)).real
    return low_freq, high_freq
```

### 3.7 Sampling Strategies

**DDPM**: Full stochastic sampling over T steps
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right) + \sigma_t z$$

**DDIM**: Deterministic sampling with configurable steps
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\left(\frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta$$

**Adaptive**: Early-stop when $\|\epsilon_\theta(x_t, t)\|$ converges

---

## 4. Experiments

### 4.1 Dataset

| Dataset | Train | Val | Test | Resolution | Modality |
|---------|-------|-----|------|------------|----------|
| BraTS 2020 | [N] | [N] | [N] | 128×128 | T1/T2/FLAIR |
| Custom | [N] | [N] | [N] | 128×128 | [Type] |

**Preprocessing**:
1. Extract 2D axial slices
2. Resize to 128×128
3. Normalize to [-1, 1]
4. Filter slices with >10% brain content

### 4.2 Implementation Details

| Hyperparameter | Value |
|----------------|-------|
| VAE latent channels | 4 |
| UNet base channels | 64 |
| Diffusion timesteps | 1000 |
| Batch size | 8 |
| Learning rate | 1e-4 |
| Optimizer | AdamW |
| Mixed precision | FP16 |
| EMA decay | 0.9999 |

**Training Schedule**:
- VAE: 100 epochs (~1.5 hours on T4)
- Diffusion: 200 epochs (~10 hours on T4)

### 4.3 Evaluation Metrics

- **SSIM**: Structural Similarity Index (higher is better)
- **PSNR**: Peak Signal-to-Noise Ratio in dB (higher is better)
- **FID**: Fréchet Inception Distance (lower is better)

### 4.4 Ablation Studies

#### 4.4.1 Noise Schedule Comparison

| Schedule | SSIM ↑ | PSNR ↑ | FID ↓ |
|----------|--------|--------|-------|
| Linear | [.XX] | [XX.X] | [XX] |
| Cosine | [.XX] | [XX.X] | [XX] |
| **Gamma-rebalanced** | [.XX] | [XX.X] | [XX] |

#### 4.4.2 Loss Function Comparison

| Loss | SSIM ↑ | PSNR ↑ | FID ↓ |
|------|--------|--------|-------|
| Simple (L2) | [.XX] | [XX.X] | [XX] |
| **Frequency-aware** | [.XX] | [XX.X] | [XX] |

#### 4.4.3 Combined Contributions

| Configuration | SSIM ↑ | PSNR ↑ | FID ↓ | Training Time |
|---------------|--------|--------|-------|---------------|
| Baseline | [.XX] | [XX.X] | [XX] | [X.X hrs] |
| + Gamma schedule | [.XX] | [XX.X] | [XX] | [X.X hrs] |
| + Freq. loss | [.XX] | [XX.X] | [XX] | [X.X hrs] |
| + Both | [.XX] | [XX.X] | [XX] | [X.X hrs] |

### 4.5 Comparison with Baselines

| Method | SSIM ↑ | PSNR ↑ | FID ↓ | Params | Time |
|--------|--------|--------|-------|--------|------|
| VAE only | [.XX] | [XX.X] | [XX] | [X.XM] | [Xs] |
| GAN | [.XX] | [XX.X] | [XX] | [X.XM] | [Xs] |
| DDPM | [.XX] | [XX.X] | [XX] | [X.XM] | [Xs] |
| **Ours** | [.XX] | [XX.X] | [XX] | [X.XM] | [Xs] |

### 4.6 Qualitative Results

*[Insert figure grid showing: Input → VAE Recon → Generated samples]*

```
+-------------------+-------------------+-------------------+
|   Ground Truth    |  VAE Reconstruction|   Generated      |
+-------------------+-------------------+-------------------+
|  [Image 1]        |  [Image 1]        |  [Image 1]       |
|  [Image 2]        |  [Image 2]        |  [Image 2]       |
|  [Image 3]        |  [Image 3]        |  [Image 3]       |
+-------------------+-------------------+-------------------+
```

**Figure 1**: Qualitative comparison of ground truth, VAE reconstruction, and diffusion-generated images. Our method preserves anatomical structures while generating realistic variations.

---

## 5. Discussion

### 5.1 Analysis of Noise Rebalancing

*[Discuss why gamma-rebalanced schedule works for medical images]*

The gamma-rebalanced schedule reduces noise at early timesteps (t close to T), which corresponds to preserving coarse structure. This is particularly important for medical images where anatomical boundaries must be preserved.

### 5.2 Analysis of Frequency-Aware Loss

*[Discuss the role of frequency decomposition]*

Medical images contain important structural information in low frequencies and fine details (edges, textures) in high frequencies. By separately weighting these components, we achieve better balance between global structure and local details.

### 5.3 Limitations

1. **Resolution**: Current implementation limited to 128×128
2. **3D Extension**: Does not utilize inter-slice consistency
3. **Modality**: Focused on single-modality; multi-modal fusion not explored
4. **Diversity**: Trade-off between fidelity and diversity

### 5.4 Future Work

- Extend to 3D volumetric synthesis
- Incorporate multi-modality conditioning
- Explore progressive resolution training
- Apply to downstream tasks (segmentation, registration)

---

## 6. Conclusion

We presented an efficient implementation of slice-by-slice latent diffusion for medical image synthesis. Our two novel contributions—Noise Rebalancing Schedule and Frequency-Aware Diffusion Loss—improve structural fidelity and detail preservation respectively. The complete implementation is optimized for Kaggle T4 GPUs and publicly available for reproducibility.

---

## References

```
[1] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.

[2] Song, J., Meng, C., & Ermon, S. (2020). Denoising Diffusion Implicit Models. ICLR.

[3] Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR.

[4] Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. NeurIPS.

[5] [BraTS Challenge Reference]

[6] [Additional references]
```

---

## Appendix

### A. Additional Qualitative Results

*[Additional sample grids]*

### B. Hyperparameter Sensitivity

| Parameter | Range Tested | Best Value |
|-----------|--------------|------------|
| Gamma (schedule) | 0.5-1.0 | [X.XX] |
| Low freq weight | 0.5-2.0 | [X.XX] |
| High freq weight | 0.1-1.0 | [X.XX] |
| Cutoff ratio | 0.1-0.5 | [X.XX] |

### C. Training Curves

*[Insert loss curves for VAE and diffusion training]*

### D. Computational Resources

| Resource | Specification |
|----------|--------------|
| GPU | NVIDIA T4 (16GB) |
| Training Time | ~12 hours total |
| Memory Usage | ~14GB peak |
| Storage | ~2GB checkpoints |

### E. Code Availability

The complete implementation is available at:
- GitHub: https://github.com/[username]/sbldm-medical
- Kaggle: https://kaggle.com/code/[username]/sbldm-medical

---

## Checklist

- [ ] Abstract summarizes key contributions
- [ ] Introduction clearly states problem and contributions
- [ ] Related work covers relevant literature
- [ ] Methods section includes all technical details
- [ ] Experiments include ablations and comparisons
- [ ] Figures are clear and properly captioned
- [ ] Tables are formatted consistently
- [ ] Limitations are honestly discussed
- [ ] Code is publicly available
- [ ] Results are reproducible
