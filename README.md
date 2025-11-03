# Prostate MRI Self-Supervised Pretraining

This directory hosts a standalone experimental playground for prostate MRI
self-supervised learning that reuses the building blocks from the upstream
`dinov2` package without touching the original source tree.

## Layout

- `configs/` – experiment recipes that are merged on top of
  `dinov2/configs/ssl_default_config.yaml`.
- `data/` – dataset helpers and utilities that keep the BIDS-like layout used by
  MM-DINOv2.
- `engine/` – a lightweight training loop that mirrors the original
  `dinov2/train/train.py` logic, adapted for prostate data.
- `scripts/` – CLI entry points. `run_pretraining.py` is the main launcher.
- `splits/` – placeholder folder for patient ID split files (`train.txt`, `val.txt`,
  `test.txt`).

## Quick Start

1. **Prepare the dataset**
   - Place every patient under `X:/AIS_processed_20251015update/` and keep
     the following files accessible (they can be either `.nii`, `.nii.gz`, or a folder
     containing the single NIfTI volume):
     ```
     ax_t2wi.nii
     ax_adc.nii
     ax_dwi_1500.nii
     roi_Prostate.nii
     ```
     Extra ROIs or modalities are ignored unless the config explicitly lists them.
   - 用脚本快捷收集所有患者并写出 `train.txt`（默认列出全部病例，可通过 `--overwrite` 覆盖旧文件）：
     ```bash
     uv run python pre-ssl-prostate/scripts/create_splits.py \
         --dataset-root X:/AIS_processed_20251015update
     ```
     生成的文件位于 `pre-ssl-prostate/splits/train.txt`，内容是一行一个患者目录
     （相对于数据根目录，例：`patient_0001`）。

2. **Update the config**
   - `configs/prostate_ssl.yaml` already targets `X:/AIS_processed_20251015update`,
     defines `sequences=ax_t2wi,ax_adc,ax_dwi_1500`, and uses
     `segmentation_key=roi_Prostate`. Adjust `train.dataset_path` if filenames differ
     or you want to add/remove modalities.
   - For the 17k-patient cohort on a single 24GB GPU, the default configuration sets:
     - `batch_size_per_gpu=6` — increase if memory allows, decrease if OOM occurs.
     - `auto_epoch_length=true` so the trainer infers `OFFICIAL_EPOCH_LENGTH` from the
       dataset size (`≈ patients ÷ batch`). With `epochs=120` and `warmup_epochs=10`,
       this yields roughly 260k optimisation steps.
     - `num_workers=12`, `prefetch_factor=4`, `persistent_workers=true`, and
       `pin_memory=true` to improve dataloader throughput.
   - Any parameter can be overridden on the CLI, e.g.
     ```bash
     uv run python pre-ssl-prostate/scripts/run_pretraining.py \
         --config pre-ssl-prostate/configs/prostate_ssl.yaml \
         train.batch_size_per_gpu=4 train.num_workers=8
     ```

3. **Install dependencies**
   - Create the Python environment following the repository instructions.
   - Ensure the following extras are available for training:
     - PyTorch with CUDA support.
     - MONAI, nibabel (already declared in the repo requirements).
     - `xformers` (required by `SSLMetaArch`).
     - Optional but recommended: `torchvision`, `omegaconf`.

4. **Launch pretraining**
   ```bash
   uv run python pre-ssl-prostate/scripts/run_pretraining.py \
       --config pre-ssl-prostate/configs/prostate_ssl.yaml
   ```
   The script stores the resolved config, `training_metrics.jsonl`, and checkpoints
   under `out/pre_ssl/prostate_ssl_pretrain/`.

5. **Monitor training**
   - Metrics are appended to `training_metrics.jsonl`. Visualise loss and learning-rate
     curves with:
     ```bash
     uv run python pre-ssl-prostate/scripts/plot_metrics.py \
         --metrics out/pre_ssl/prostate_ssl_pretrain/training_metrics.jsonl \
         --rolling-window 20 --include-lr \
         --output out/pre_ssl/loss_curve.png
     ```

## Extending the Framework

- **Custom datasets:** Implement new builders in `data/` that reuse
  `LoadTumorSliced` or introduce alternative cropping strategies when a mask is
  unavailable.
- **Augmentations:** Adjust `crops` parameters in the config or replace
  `DataAugmentationDINO` with a tailored transform pipeline.
- **Experiment tracking:** The trainer logs JSON lines; integrate your preferred
  logger or experiment tracker by extending `ProstateSSLTrainer`.

## Notes

- The default configuration disables the FSDP gradient scaler if CUDA or the
  fused scaler is not detected. Override `compute_precision.grad_scaler=true`
  only when the environment supports it.
- Checkpoints store both student and teacher weights to simplify downstream
  evaluation and fine-tuning.
