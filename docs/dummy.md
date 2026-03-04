# Fast MRI Reconstruction

## Goal
This document summarizes a replication/evaluation workflow for **brain MRI reconstruction** on fastMRI data using baseline methods from the official ecosystem:

- Zero-filled reconstruction
- ESPIRiT (classical baseline)
- U-Net (learned baseline)

Optional extensions:

- End-to-end Variational Network (VarNet)
- Domain adaptation / domain learning approaches

---

## References

- Main datasets (brain, knee, breast, prostate): https://fastmri.med.nyu.edu/
- fastMRI paper (2018): https://arxiv.org/abs/1811.08839
- fastMRI+ paper (2021): https://arxiv.org/abs/2109.03812
- fastMRI repository (main): https://github.com/facebookresearch/fastMRI/tree/main
- fastMRI examples: https://github.com/facebookresearch/fastMRI/tree/main/fastmri_examples

---

## Problem Setup (Brain MRI)

In accelerated MRI, only a subset of k-space is acquired to reduce scan time. Reconstruction must recover a high-fidelity image from undersampled k-space.

Given fully sampled target image $x$ and undersampled measurements $y = PFx + n$:

- $F$: Fourier operator
- $P$: undersampling mask
- $n$: noise

The reconstruction objective is to estimate $\hat{x}$ such that quality metrics (NMSE, PSNR, SSIM) improve while preserving anatomical detail.

---

## Methods to Compare

### 1) Zero-filled
Direct inverse FFT from undersampled k-space (missing frequencies set to zero).

- Very fast
- Strong aliasing artifacts
- Lower PSNR/SSIM expected

### 2) ESPIRiT
Calibration-based parallel imaging using sensitivity map estimation.

- Classical model-based baseline
- Better than zero-filled in many cases
- Computationally heavier than zero-filled

### 3) U-Net (pretrained baseline)
Learned post-processing/reconstruction network from fastMRI ecosystem.

- Typically strongest among the three baselines
- Better perceptual quality and higher SSIM/PSNR expected

### Optional 4) VarNet / domain learning
If available in your local setup, add:

- End-to-end variational network results
- Domain adaptation/domain learning results

---

## Data and Split

- Primary focus: **Brain multicoil** data
- Suggested split for development: validation subset
- Use a shared file list (`selected_samples.txt`) so all methods are evaluated on identical cases

Example: generate/select subset first, then run all baselines on the same subset.

---

## Reproduction Workflow (in this repo)

> Note: most scripts are under `mri-reconstruction-scripts/` and use the local `fastMRI/` checkout.

### A) Zero-filled baseline

```bash
python experiments/run_zero_filled.py \
  --fastmri_repo fastMRI \
  --challenge multicoil \
  --data_path data/multicoil_val \
  --file_list selected_samples.txt \
  --output_path results/zero_filled_subset
```

### B) ESPIRiT baseline

```bash
export TOOLBOX_PATH=$PWD/bart
export PYTHONPATH=${TOOLBOX_PATH}/python:${PYTHONPATH}

python experiments/run_espirit_subset.py \
  --fastmri_repo fastMRI \
  --bart_toolbox_path $TOOLBOX_PATH \
  --challenge multicoil \
  --split val \
  --mask_type equispaced \
  --data_path data/multicoil_val \
  --output_path results/espirit_subset \
  --file_list selected_samples.txt \
  --reg_wt 0.01 \
  --num_iters 200 \
  --num_procs 2
```

### C) Pretrained U-Net baseline

```bash
python experiments/run_pretrained_unet.py \
  --fastmri_repo fastMRI \
  --challenge unet_brain_mc \
  --state_dict_file brain_leaderboard_state_dict.pt \
  --data_path data/multicoil_val \
  --file_list selected_samples.txt \
  --output_path results/unet_subset
```

### D) Evaluate all reconstructions

```bash
python experiments/evaluate_reconstructions.py \
  --fastmri_repo fastMRI \
  --challenge multicoil \
  --target_path data/multicoil_val \
  --method zero_filled=results/zero_filled_subset \
  --method espirit=results/espirit_subset/reconstructions \
  --method unet=results/unet_subset/reconstructions \
  --output_csv results/metrics_subset.csv \
  --output_json results/metrics_subset.json
```

### E) Qualitative comparison figures

```bash
python experiments/plot_reconstruction_comparison.py \
  --target_path data/multicoil_val \
  --target_key reconstruction_rss \
  --method zero_filled=results/zero_filled_subset \
  --method espirit=results/espirit_subset/reconstructions \
  --method unet=results/unet_subset/reconstructions \
  --output_dir results/figures \
  --file "$(head -n 1 selected_samples.txt)"
```

---

## Quantitative Results (Fill-in Template)

> Fill this section with your measured values from `results/metrics_subset.csv`.

### Overall Metrics

| Method | NMSE | PSNR (dB) | SSIM |
|---|---:|---:|---:|
| Zero-filled | TODO | TODO | TODO |
| ESPIRiT | TODO | TODO | TODO |
| U-Net | TODO | TODO | TODO |
| VarNet (optional) | TODO | TODO | TODO |
| Domain learning (optional) | TODO | TODO | TODO |

### Relative Improvements (example placeholders)

- U-Net vs Zero-filled:
  - NMSE improvement: **TODO %**
  - PSNR gain: **TODO dB**
  - SSIM gain: **TODO**
- ESPIRiT vs Zero-filled:
  - NMSE improvement: **TODO %**
  - PSNR gain: **TODO dB**
  - SSIM gain: **TODO**

---

## Qualitative Results (Fill-in Template)

Include side-by-side visuals for representative slices:

1. Ground truth
2. Zero-filled
3. ESPIRiT
4. U-Net
5. (Optional) VarNet / domain learning
6. Error maps (optional)

Suggested figure table:

| Case ID | Slice | Figure Path | Main Observation |
|---|---:|---|---|
| TODO | TODO | `results/figures/TODO.png` | TODO |
| TODO | TODO | `results/figures/TODO.png` | TODO |

---

## Discussion Points for Presentation

- Why undersampling creates aliasing and ill-posed inversion
- Why zero-filled is a useful baseline but insufficient clinically
- How ESPIRiT improves reconstruction via sensitivity modeling
- Why U-Net improves perceptual and structural quality
- Failure modes (small lesions, texture smoothing, hallucination risk)
- Trade-off between acceleration factor and image quality

---

## Presentation Outline (Suggested)

1. **Problem & Motivation** (fast MRI, scan-time reduction)
2. **Dataset & Protocol** (brain multicoil, subset/full split)
3. **Methods** (zero-filled, ESPIRiT, U-Net, optional extensions)
4. **Quantitative Results** (NMSE/PSNR/SSIM tables + charts)
5. **Qualitative Results** (visual comparisons + error maps)
6. **Conclusion** (best baseline, limitations, next steps)

---

## Deliverables Checklist

- [ ] Brain data downloaded and organized
- [ ] Baselines re-run (zero-filled, ESPIRiT, U-Net)
- [ ] Metrics computed and inserted
- [ ] Visual comparisons exported
- [ ] Slides prepared with both quantitative and qualitative results
- [ ] Optional advanced models added (VarNet/domain learning)
