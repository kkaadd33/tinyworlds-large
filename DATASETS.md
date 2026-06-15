# TinyWorlds Datasets — Reference

A consolidated reference for the five gameplay video datasets shipped via the
[`AlmondGod/tinyworlds`](https://huggingface.co/datasets/AlmondGod/tinyworlds) HF
repo and consumed by the TinyWorlds training pipeline.

All numbers below were measured directly against the HDF5 files; "class defaults"
and pipeline knobs are quoted from the code:

- Dataset classes: `datasets/datasets.py`
- Routing / loader helpers: `datasets/data_utils.py`
- Shared training config: `configs/training.yaml`
- Per-stage configs: `configs/{video_tokenizer,latent_actions,dynamics}.yaml`

The same clip tensor (shape `[B, T, C, H, W]`, values in `[-1, 1]`) is fed to all
three models in the pipeline — see [Common Pipeline](#common-pipeline-shared-by-all-five).

---

## Summary table

| Dataset key   | HF file                       | File size | Stored shape `(N, H, W, 3)`      | Class `resize_to` (to model) | `fps` (temporal stride) | Frame skip in clip | Default `preload_ratio` × N | `load_start_index` | Approx. effective frames in RAM | Approx. playback duration (n / fps) |
|---------------|-------------------------------|-----------|----------------------------------|------------------------------|-------------------------|--------------------|-----------------------------|--------------------|---------------------------------|--------------------------------------|
| `PONG`          | `pong_frames.h5`            | 14.3 MB   | (108,034, 64,  64,  3)           | 64×64                        | 30                      | 2                  | 1.0 × 108,034 = 108,034     | 0                  | 108,034                         | ~60 min                              |
| `POLE_POSITION` | `pole_position_frames.h5`   | 17.3 MB   | (  5,385, 64,  64,  3)           | 64×64                        | 30                      | 2                  | 1.0 × 5,385 = 5,385         | 0                  | 5,385                           | ~3 min                               |
| `SONIC`         | `sonic_frames.h5`           | 249.4 MB  | ( 41,242, 64,  64,  3) †         | **128×128 (upscale)** †       | 15                      | 4                  | 1.0 × 41,242 = 41,242       | 100                | 41,142                          | ~46 min                              |
| `PICODOOM`      | `picodoom_frames.h5`        | 662.4 MB  | ( 59,785, 64,  64,  3) †         | **128×128 (upscale)** †       | 30                      | 2                  | 0.3 × 59,785 = 17,935       | 300                | 17,635                          | ~33 min (full file)                  |
| `ZELDA`         | `zelda_frames.h5`           | 1,866.9 MB | ( 72,410,128,128,  3)           | **64×64 (downscale)**        | 15                      | 4                  | 0.2 × 72,410 = 14,482       | 1,000              | 13,482                          | ~80 min (full file)                  |

† The `.h5` was preprocessed at 64×64, but `SonicDataset` / `PicoDoomDataset` declare
`resolution=(128, 128)` as their class default and `VideoHDF5Dataset` re-resizes on
load (`datasets/datasets.py:51-52`). If you only want 64×64 inputs for Sonic /
PicoDoom (matching `frame_size: 64` in `configs/training.yaml`), override the
class default — these datasets will currently be **upscaled** at read time.

### What each model sees per clip (with `context_length=4`)

| Dataset key     | Model-side `H×W` | Clip `T` (= `context_length`) | Frame skip | Approx. number of clips ((preloaded − T·skip) × `fraction_of_dataset`) |
|-----------------|-----------------|-------------------------------|------------|-------------------------------------------------------------------------|
| `PONG`          | 64×64           | 4 (class default is 1)        | 2          | ~108,026                                                                |
| `POLE_POSITION` | 64×64           | 4                             | 2          | ~5,377                                                                  |
| `SONIC`         | 128×128 (†)     | 4                             | 4          | ~41,126                                                                 |
| `PICODOOM`      | 128×128 (†)     | 4                             | 2          | ~17,627                                                                 |
| `ZELDA`         | 64×64           | 4                             | 4          | ~13,466                                                                 |

The default `num_frames` for `PongDataset` is `1`, all others are `4`. In practice
the training scripts pass `num_frames = args.context_length` from `configs/training.yaml`
(default `4`), so `T = 4` is the realistic number for all datasets.

---

## Per-dataset detail

### PONG — `pong_frames.h5`

- **Stored:** `(108,034, 64, 64, 3)` uint8, 14.3 MB.
- **Brightness sample (8 evenly-spaced frames):** min=0, max=243, mean≈1.6, std≈13.5
  → frames are overwhelmingly black with a small bright ball/paddles, as expected.
- **Class:** `PongDataset` (`datasets/datasets.py:148-163`).
- **Defaults:** `num_frames=1`, `resolution=(64, 64)`, `fps=30`, `preload_ratio=1`,
  `preprocess_read_step=10` (every 10th raw frame kept during MP4 → H5 preprocessing).
- **Effective clips fed to models (`T=4`):** ~108,026.

### POLE_POSITION — `pole_position_frames.h5`

- **Stored:** `(5,385, 64, 64, 3)` uint8, 17.3 MB.
- **Brightness sample:** min=0, max=255, mean≈111.7, std≈71.5 (high-contrast, colorful
  racetrack and sky).
- **Class:** `PolePositionDataset` (`datasets/datasets.py:165-181`).
- **Defaults:** `num_frames=4`, `resolution=(64, 64)`, `fps=30`, `preload_ratio=1`,
  `preprocess_read_step=1`, `preprocess_slice=(1/50, 1/4)` — i.e. when generating
  the `.h5` from the source `.mp4`, only frames between the 2% and 25% marks are
  kept. (This slice is already baked into the published `.h5`.)
- **Smallest gameplay dataset.** Only ~3 minutes of stored gameplay at fps=30 →
  ~5,377 training clips with `T=4`.

### SONIC — `sonic_frames.h5`

- **Stored:** `(41,242, 64, 64, 3)` uint8, 249.4 MB.
- **Brightness sample:** min=0, max=254, mean≈40.6, std≈57.7 (dark UI bars with
  saturated platformer colors → high std).
- **Class:** `SonicDataset` (`datasets/datasets.py:183-199`).
- **Defaults:** `num_frames=4`, `resolution=(128, 128)`, `fps=15`, `preload_ratio=1`,
  `load_start_index=100` (skips the first 100 frames — title screen).
- **Mismatch warning:** the `.h5` is 64×64 but the class default is 128×128, so the
  loader upscales every frame on load. If you train Sonic with `configs/training.yaml`
  as-is (`frame_size: 64`), the model will be built for 64×64 but the dataloader will
  feed 128×128 frames — adjust one or the other before training.
- **Effective clips (`T=4`, skip=4):** ~41,126.

### PICODOOM — `picodoom_frames.h5`

- **Stored:** `(59,785, 64, 64, 3)` uint8, 662.4 MB.
- **Brightness sample:** min=0, max=240, mean≈63.2, std≈34.2 (mostly dark corridors).
- **Class:** `PicoDoomDataset` (`datasets/datasets.py:201-217`).
- **Defaults:** `num_frames=4`, `resolution=(128, 128)`, `fps=30`, `preload_ratio=0.3`,
  `load_start_index=300`.
- **Same upscale caveat as Sonic** — `.h5` is 64×64 but the class asks for 128×128.
- **Only 30% of the file is loaded into RAM by default**, starting after the first
  300 frames. With those defaults you get ~17,635 frames in memory → ~17,627
  training clips with `T=4`.

### ZELDA — `zelda_frames.h5`

- **Stored:** `(72,410, 128, 128, 3)` uint8, 1,866.9 MB — the only natively 128×128
  dataset.
- **Brightness sample:** min=0, max=255, mean≈89.1, std≈42.3.
- **Class:** `ZeldaDataset` (`datasets/datasets.py:219-235`).
- **Defaults:** `num_frames=4`, `resolution=(64, 64)`, `fps=15`, `preload_ratio=0.2`,
  `load_start_index=1000` (skips the first 1,000 frames — likely intro / menus).
- **Downscale on load** — stored 128×128 → 64×64, matching `frame_size: 64` in the
  default `configs/training.yaml`. The default `configs/training.yaml` ships with
  `dataset: ZELDA`.
- **Effective clips (`T=4`, skip=4):** ~13,466.

---

## Common pipeline (shared by all five)

Everything below is identical across all datasets and is inherited from the
`VideoHDF5Dataset` base class (`datasets/datasets.py:11-145`):

- **Single HDF5 dataset key:** every `.h5` exposes only a `frames` dataset, shape
  `(N, H, W, 3)`, `dtype=uint8`, channel order **RGB** (`cv2.cvtColor(...,
  cv2.COLOR_BGR2RGB)` during preprocessing).
- **No action / reward labels are stored.** `__getitem__` returns `(frame_sequence,
  0)` — the second element is a dummy zero (`datasets/datasets.py:141`). This is
  by design: the Latent Action Model infers actions unsupervised from frame
  transitions.
- **Clip sampling:** for index `i`, the dataset returns
  `self.data[i : i + num_frames*frame_skip : frame_skip]`, where
  `frame_skip = max(1, sequence_stride or 60 // fps)`
  (`datasets/datasets.py:34, 122-128`). The `fps` knob therefore controls
  *temporal stride within a clip*, not the source video's frame rate.
- **Value range fed to models:** uint8 → float32 in `[0, 1]` via `/255.0`, then
  `transforms.Normalize((0.5,)*3, (0.5,)*3)` → `[-1, 1]`
  (`datasets/datasets.py:130` + `datasets/data_utils.py:20-24`).
- **Output tensor shape from dataloader:** `[B, T, C=3, H, W]`, where `H, W` are
  whatever the dataset class `resize_to` is.
- **Train / val split:** by default `disable_test_split=True`, so train and val
  share the same data; if turned off, a chronological 90/10 split is used
  (`datasets/datasets.py:73-75`). No held-out test set.

### How each model consumes a clip

| Model              | Input from clip            | Trained against                              | Action count / shape                                              |
|--------------------|----------------------------|----------------------------------------------|-------------------------------------------------------------------|
| Video Tokenizer    | `[B, T, 3, H, W]` frames   | smooth-L1 frame reconstruction              | n/a                                                               |
| Latent Action Model| `[B, T, 3, H, W]` frames   | smooth-L1 next-frame recon + variance penalty | `n_actions=4` total tokens (`action_dim=log2(4)=2`, FSQ bins=2)   |
| Dynamics Model     | video tokens `[B, T, P]` + action latents `[B, T-1, A]` | masked cross-entropy on video tokens (MaskGIT) | conditioning vector of shape `[B, T-1, action_dim]`            |

The Latent Action Model never sees real game controls — it learns its own
`n_actions = 4` discrete codes that maximize next-frame reconstruction. At
inference the user maps keyboard keys to integers in `[0, n_actions − 1]`
(`utils/inference_utils.py:127-146`).

---

## Reproducing the measurements above

```bash
# 1. download (skips files already present)
python scripts/download_assets.py datasets \
    --pattern "pong_frames.h5" \
    --pattern "pole_position_frames.h5" \
    --pattern "sonic_frames.h5" \
    --pattern "picodoom_frames.h5" \
    --pattern "zelda_frames.h5"

# 2. inspect with h5py
python - <<'PY'
import h5py, os, numpy as np
for name, path in [
    ("PONG",          "data/pong_frames.h5"),
    ("POLE_POSITION", "data/pole_position_frames.h5"),
    ("SONIC",         "data/sonic_frames.h5"),
    ("PICODOOM",      "data/picodoom_frames.h5"),
    ("ZELDA",         "data/zelda_frames.h5"),
]:
    size_mb = os.path.getsize(path) / 1e6
    with h5py.File(path, "r") as f:
        frames = f["frames"]
        n = frames.shape[0]
        idx = np.linspace(0, n - 1, num=min(8, n), dtype=int)
        sample = frames[list(idx)]
        print(f"{name:14s} {frames.shape} {frames.dtype}  "
              f"size={size_mb:6.1f} MB  "
              f"min={int(sample.min())} max={int(sample.max())} "
              f"mean={sample.mean():.2f} std={sample.std():.2f}")
PY
```
