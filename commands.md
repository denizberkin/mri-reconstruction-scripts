# Commands to use / copy-paste / while replicating experiments.


# List all samples
```bash
find data/multicoil_val -maxdepth 1 -name '*.h5' -printf '%f\n' | sort
```

# Pick subset from first `num_samples`
```bash
find data/multicoil_val -maxdepth 1 -name '*.h5' -printf '%f\n' | sort | head -n $num_samples$ > selected_samples.txt
```

# Pick random subset of `num_samples`
```bash
find data/multicoil_val -maxdepth 1 -name '*.h5' -printf '%f\n' | shuf -n $num_samples$ > selected_samples.txt
```

# Pick stratified subset
```bash
python utils/stratified.py \
    --data_path data/multicoil_val \
    --num_samples $num_samples$ \
    --output selected_samples.txt
```

# check distribution of samples (AXT1, AXT1PRE, AXT2...)
```bash
find data/multicoil_val -maxdepth 1 -name '*.h5' -printf '%f
' | awk -F'_' '{print $3}' | sort | uniq -c | sort -nr
```


# Zero-filled baseline on shared subset
```bash
python experiments/run_zero_filled.py \
    --fastmri_repo fastMRI \
    --challenge multicoil \
    --data_path data/multicoil_val \
    --file_list selected_samples.txt \
    --output_path results/zero_filled_subset
```

# Pretrained U-Net baseline (brain multicoil, shared subset)
```bash
python experiments/run_pretrained_unet.py \
    --fastmri_repo fastMRI \
    --challenge unet_brain_mc \
    --state_dict_file brain_leaderboard_state_dict.pt \
    --data_path data/multicoil_val \
    --file_list selected_samples.txt \
    --output_path results/unet_subset
```

# ESPIRiT baseline on shared subset (slow baseline)
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

# Quantitative evaluation for all methods (saved to disk)
```bash
python experiments/evaluate_reconstructions.py \
    --fastmri_repo fastMRI \
    --challenge multicoil \
    --target_path data/multicoil_val \
    --method zero_filled=results/zero_filled_subset \
    --method unet=results/unet_subset/reconstructions \
    --method espirit=results/espirit_subset/reconstructions \
    --output_csv results/metrics_subset.csv \
    --output_json results/metrics_subset.json
```

# Qualitative side-by-side figures (pick first file from subset list)
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

# Full-dataset evaluation with no saved reconstructions (zero-filled + U-Net only)
```bash
python experiments/evaluate_all.py \
    --fastmri_repo fastMRI \
    --data_path data/multicoil_val \
    --challenge multicoil \
    --mask_type random \
    --accelerations 4 8 \
    --center_fractions 0.08 0.04 \
    --acquisition AXT1 \
    --acquisition AXT1POST \
    --acquisition AXT2 \
    --acquisition AXFLAIR \
    --method zero_filled \
    --method unet \
    --state_dict_file brain_leaderboard_state_dict.pt \
    --output_csv results/metrics_full.csv \
    --output_json results/metrics_full.json \
    --device cuda
```