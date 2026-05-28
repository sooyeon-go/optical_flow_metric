# Optical Flow Magnitude Metric

This folder provides a RAFT-based motion score for input videos.

## What it does

- Accepts either a single video (`--video_path`) or a video folder (`--video_dir`).
- When a folder is provided, processes all supported videos in the folder.
- Uses only the first `81` frames if the video is longer.
- If the video is shorter than `81`, uses all available frames.
- Computes optical flow between consecutive frame pairs with RAFT.
- Returns per-video score as the mean flow magnitude.

## Environment setup (conda)

```bash
conda create -n optical_flow python=3.10 -y
conda activate optical_flow

# GPU (single GPU is enough) - install torch ONLY via conda
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Python deps (numpy/scipy/opencv only; no torch in requirements.txt)
pip install -r /mnt/sy/test_spotting/optical_flow_metric/requirements.txt
```

### If you see `undefined symbol: iJIT_NotifyEvent`

This usually means pip torch and conda torch were mixed.

```bash
conda activate optical_flow
bash /mnt/sy/test_spotting/optical_flow_metric/fix_torch_env.sh
```

Or manually:

```bash
pip uninstall -y torch torchvision torchaudio
rm -rf "$CONDA_PREFIX/lib/python3.10/site-packages/torch"*
conda install -y --force-reinstall pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
unset LD_LIBRARY_PATH
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pip install -r requirements.txt
```

CPU only:

```bash
conda install pytorch torchvision cpuonly -c pytorch
pip install -r /mnt/sy/test_spotting/optical_flow_metric/requirements.txt
```

## Run

```bash
python /mnt/sy/test_spotting/optical_flow_metric/score_video_motion.py \
  --video_path /path/to/input.mp4
```

## Optional arguments

- `--max_frames`: default `81`
- `--weight_dir`: default `/mnt/sy/test_spotting/optical_flow_metric/weight`
- `--device`: default `cuda` if available, otherwise `cpu`
- `--iters`: RAFT update iterations (default `20`)
- `--mixed_precision`: enable AMP in RAFT
- `--save_json`: save result to a JSON file

`--weight_dir` should contain:
- `raft-things.pth`, or
- exactly one `.pth` file (auto-selected)

Example:

```bash
python /mnt/sy/test_spotting/optical_flow_metric/score_video_motion.py \
  --video_dir /path/to/video_folder \
  --max_frames 81 \
  --device cuda \
  --save_json /mnt/sy/test_spotting/optical_flow_metric/result.json
```

## Output JSON shape

`videos` stores each video filename as key and its score details as value.

```json
{
  "num_videos_total": 2,
  "num_videos_scored": 2,
  "max_frames_limit": 81,
  "aggregate_mean_optical_flow_magnitude_score": 2.13,
  "videos": {
    "video_a.mp4": {
      "optical_flow_magnitude_score": 1.95
    },
    "video_b.mp4": {
      "optical_flow_magnitude_score": 2.31
    }
  }
}
```
