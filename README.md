# MRI Reconstruction Scripts

This repository contains my personal scripts and experiment utilities to replicate and evaluate **fastMRI** reconstruction methods, mainly on **brain MRI** data.

## Scope

- Reproduce baseline methods from fastMRI ecosystem:
  - Zero-filled
  - ESPIRiT
  - U-Net
- Evaluate results quantitatively (NMSE, PSNR, SSIM)
- Compare results qualitatively with reconstruction figures
- Optionally test additional models (e.g., VarNet, domain learning)

## Repository Layout

- `experiments/`: runnable experiment/evaluation scripts
- `scripts/`: helper scripts for visualization and analysis
- `results/`: outputs (reconstructions, metrics, figures)
- `docs/`: experiment notes and reporting markdowns
- `fastmri/`: local fastMRI code/resources used by scripts

## Quick Start

1. Prepare data under `data/` (focus on brain multicoil split).
2. Create/select subset file list (`selected_samples.txt`) if qualitative comparison is desired.
3. Run baselines from `experiments/`.
4. Evaluate with shared metrics scripts.
5. Summarize in docs and presentation.

See [commands markdown](commands.md) for command snippets and [reconstruction report markdown](docs/reconstruction.md) for the full report template.
