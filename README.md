# Neural Style Transfer for Anime Stylization


This repository contains the implementation, experiments, and benchmarks for a Final Year Project investigating optimisation-based Neural Style Transfer (NST) for anime stylisation. The work adapts the classical Gatys et al. (2016) framework with three additional loss functions — **Edge Loss**, **Surface Smoothing Loss**, and **Laplacian Loss** — and evaluates the resulting system against the GAN-based benchmark AnimeGANv2.

---

## 1. Project Overview

Standard NST is poorly suited to anime stylisation: it produces ghosting artefacts in flat regions, fails to preserve clean structural boundaries, and does not reproduce the characteristic flat colour palette of anime artwork. This project investigates whether targeted, interpretable loss-function adaptations can address these limitations within an optimisation-based framework.

**Three loss extensions are introduced:**

| Loss | Mechanism | Purpose |
|------|-----------|---------|
| Edge Loss | Differentiable Sobel operators on luminance | Preserves structural boundaries; eliminates ghosting |
| Surface Smoothing Loss | Guided Filter on generated image | Encourages flat colour regions |
| Laplacian Loss | Second-order derivative penalty | Suppresses high-frequency texture |

**Five progressive configurations (V1–V5)** are evaluated, alongside three ablation studies validating the choice of optimiser, edge detector, and feature extractor. A video-domain extension with temporal consistency loss is also provided.

---

## 2. Repository Contents

| File | Description |
|------|-------------|
| `NST_FYP.ipynb` | Main implementation. Full NST pipeline with Edge, Surface, and Laplacian losses, and the five experimental configurations (V1 baseline → V5 full system). |
| `AnimeGAN_Benchmark.ipynb` | AnimeGANv2 inference notebook. Run independently to generate benchmark outputs for comparison against the main pipeline. |
| `Experiment_A_Optimizers.ipynb` | Ablation Study A: Optimiser comparison (L-BFGS vs. Adam, SGD, SGD+Momentum, RMSprop across 10 configurations). |
| `Experiment_B_EdgeDetectors.ipynb` | Ablation Study B: Edge detector comparison (No-Edge, Sobel, Prewitt, Scharr, Laplacian, Laplacian-Diagonal). |
| `Experiment_C_FeatureExtractors.ipynb` | Ablation Study C: Feature extractor comparison (VGG19, VGG16, ResNet50). |
| `Video_NST.ipynb` | Video-domain extension. Implements a temporal consistency loss to suppress frame-to-frame flickering. |
| `final_report.pdf` | Full written project report (see Section 6). |
| `requirements.txt` | Python dependencies. |
| `LICENSE` | MIT License. |
| `README.md` | This file. |

### 2.1 Note on Framework Separation

The main NST pipeline is implemented in **TensorFlow**, while the AnimeGANv2 benchmark requires **PyTorch**. The two notebooks are deliberately separated to avoid framework conflicts within a single runtime.

---

## 3. Methodology Summary

### 3.1 Total Loss Formulation

The full objective function combines the original NST losses with the three proposed extensions:

```
L_total = α·L_content + β·L_style + γ·L_TV + δ·L_edge + ε·L_surface + ζ·L_laplacian
```

where α, β, γ, δ, ε, ζ are tunable weights and L_TV denotes total variation regularisation.

### 3.2 Experimental Configurations

| Version | Configuration | Purpose |
|---------|---------------|---------|
| V1 | Baseline (content + style only) | Reference point |
| V2 | + Edge Loss | Tests structure preservation |
| V3 | + Surface Loss | Tests colour flattening |
| V4 | Edge + Surface combined | Joint structure-and-flatness |
| V5 | + Laplacian Loss | Full proposed system |

### 3.3 Evaluation Metrics

- **Edge SSIM** — structural similarity of edge maps relative to content image
- **Flatness Score** — proportion of image area within low colour-variance regions
- **Colour Variance** — global colour distribution spread
- **Gradient Magnitude** — mean image gradient (proxy for texture density)
- **User Study (N=12)** — perceived "best balanced anime look"

---

## 4. Key Findings

A condensed summary of results; full quantitative tables and figures are provided in the report.

- **Edge Loss** improves Edge SSIM by **+22.4%** over the no-edge baseline, validating Sobel-based edge preservation as effective against ghosting artefacts.
- **L-BFGS** converges to a final loss approximately **800× lower** than gradient-based optimisers (Adam, SGD, RMSprop) under matched configurations.
- **Sobel** is validated as the optimal edge detector (ESSIM = 0.994); Prewitt is a near-equivalent alternative.
- **VGG19** achieves the best loss convergence and flatness; ResNet50 is 42% faster with a marginal (~1.8%) flatness penalty.
- **Metric–perception gap:** V5 attained the highest quantitative flatness (0.639, exceeding AnimeGAN's 0.585), yet **67% of survey respondents preferred V3/V4** outputs, which were described as more balanced.
- **Video extension:** Temporal consistency loss reduced frame-to-frame variance by approximately 94.5% on the test sequence; results are based on a single clip and have not been perceptually validated.

---

## 5. Reproducing the Experiments

### 5.1 Environment Setup

Python 3.9+ is recommended. Install dependencies via:

```bash
pip install -r requirements.txt
```

GPU acceleration is strongly recommended; CPU-only runs of `NST_FYP.ipynb` may take several minutes per output image.

### 5.2 Recommended Execution Order

1. `NST_FYP.ipynb` — main pipeline and V1–V5 outputs
2. `AnimeGAN_Benchmark.ipynb` — generates benchmark outputs for comparison
3. `Experiment_A_Optimizers.ipynb`
4. `Experiment_B_EdgeDetectors.ipynb`
5. `Experiment_C_FeatureExtractors.ipynb`
6. `Video_NST.ipynb` — requires a short video clip (see notebook for details)

Each notebook is self-contained and documents its required inputs (content image, style image, or video clip) in its first configuration cell.

---

## 6. Written Report

The full Final Year Project report — including literature review, design rationale, implementation details, evaluation, and discussion of limitations — is available as:

📄 Final Project Report-NST_Anime_Style.pdf

The report covers:

- **Chapter 2** — Literature Review (NST, AnimeGAN, perceptual losses)
- **Chapter 3** — Project Design
- **Chapter 4** — Implementation
- **Chapter 5** — Evaluation (V1–V5 results, ablation studies, video extension)
- **Chapter 6** — Discussion, limitations, future work
- **Appendices A–D** — Source code references, experimental configurations, additional figures, environment setup

---

## 7. Limitations and Honest Reporting

In line with academic best practice, the following limitations are explicitly acknowledged:

- The user study is small in scale (N=12) and was conducted informally.
- The video extension was tested on a single 7-second clip; results may not generalise.
- The optimisation-based system is substantially slower than feed-forward GAN approaches and is not suitable for real-time applications.
- Quantitative metrics (Flatness, Edge SSIM, Colour Variance) do not always align with perceived anime quality — this is itself a finding rather than a defect, but it limits the strength of metric-only claims.

These caveats are discussed in greater depth in Chapter 6 of the report.

---

## 8. References

Primary references for the methodology:

- Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). *Image Style Transfer Using Convolutional Neural Networks.* CVPR.
- Chen, J., Liu, G., & Chen, X. (2020). *AnimeGAN: A Novel Lightweight GAN for Photo Animation.* ISICA.
- He, K., Sun, J., & Tang, X. (2013). *Guided Image Filtering.* TPAMI.
- Simonyan, K. & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* ICLR.

A complete reference list is provided in the report.

---

## 9. License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for full terms.
