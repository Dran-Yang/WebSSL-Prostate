# Web-DINO MRI Pretraining

This project provides a DINO-style self-supervised pretraining pipeline tailored for large-scale, multi-modal prostate MRI datasets (e.g., 17k+ studies containing T2W, ADC, and DWI volumes). The implementation focuses on throughput, stability, and easy customisation for hospital-scale deployments.

## Data Layout

```
DATA_ROOT/
    patient_0001/
        t2w/volume.nii.gz
        adc/volume.nii.gz
        dwi/volume.nii.gz
    patient_0002/
        ...
```

Each patient folder must provide exactly one volumetric file per modality listed in the config. Supported formats: `.nii`, `.nii.gz`, `.npy`, `.pt`.

## Quick Start

1. Export the MRI root (the loader resolves `${env:WEB_DINO_MRI_ROOT:...}`):

```powershell
setx WEB_DINO_MRI_ROOT D:\datasets\prostate_mri
```

2. Install Hugging Face `transformers` (for loading `facebook/dinov2-large14`) and review `prostate/configs/pretrain.yaml` to adjust batch size, crop scales, or model width/depth.

3. Launch distributed pretraining (example for 4 GPUs):

```bash
torchrun --nproc_per_node=4 prostate/main.py --config prostate/configs/pretrain.yaml
```

4. Checkpoints and logs land in `outputs/pretrain/`.

## Key Components

- `prostate/datasets/mri_slices.py`: fast volumetric indexing, LRU caching, slice selection, percentile normalisation.
- `prostate/utils/augment.py`: MRI-friendly multi-view crops (global/local), rotation, blur, noise.
- `webssl/dinov2/vision_transformer.py`: lightweight ViT backbone with stochastic depth.
- `prostate/model/build_webdino.py`: student/teacher builders with momentum EMA updates and weight-normalised heads.
- `prostate/utils/pretrain.py`: distributed training loop, cosine LR/WD schedules, AMP (fp16 or bf16), checkpointing.

## Customisation Tips

- Increase `dataset.dataloader.num_workers` to saturate storage throughput; ensure workers < available CPU cores.
- Adjust `dataset.slice_selection` to trade coverage vs. throughput (e.g., smaller `stride` for denser supervision).
- Larger batch sizes improve stability; scale `optimization.base_lr` linearly with `batch_size * n_gpus / 256`.
- Tune `augmentations` to match scanner variability (e.g., raise `random_rotate_deg` for more positional diversity).
- For multi-node setups, set `distributed.world_size` and rely on environment variables (`RANK`, `MASTER_ADDR`, `MASTER_PORT`).
