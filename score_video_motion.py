#!/usr/bin/env python3
"""Compute optical-flow magnitude score from an input video using RAFT."""

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch


def _add_raft_path(raft_code_dir: Path) -> None:
    if str(raft_code_dir) not in sys.path:
        sys.path.insert(0, str(raft_code_dir))
    parent_dir = str(raft_code_dir.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)


def resolve_raft_model_path(weight_dir: Path) -> Path:
    if not weight_dir.exists():
        raise FileNotFoundError(f"Weight directory not found: {weight_dir}")
    if not weight_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {weight_dir}")

    preferred_path = weight_dir / "raft-things.pth"
    if preferred_path.exists():
        return preferred_path

    pth_candidates = sorted(weight_dir.glob("*.pth"))
    if len(pth_candidates) == 1:
        return pth_candidates[0]
    if len(pth_candidates) > 1:
        raise ValueError(
            "Multiple .pth files found in weight directory. "
            "Keep only one file or include raft-things.pth."
        )
    raise FileNotFoundError(
        "No RAFT weight file found. Expected raft-things.pth or a single .pth file."
    )


def resolve_raft_code_dir(script_dir: Path) -> Path:
    raft_code_dir = script_dir / "RAFT"
    if not raft_code_dir.exists():
        raise FileNotFoundError(f"RAFT code directory not found: {raft_code_dir}")
    if not raft_code_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {raft_code_dir}")
    return raft_code_dir


def resolve_default_weight_dir(script_dir: Path) -> Path:
    return script_dir / "weight"


def initialize_runtime_paths(weight_dir: Path) -> Path:
    script_dir = Path(__file__).resolve().parent
    raft_code_dir = resolve_raft_code_dir(script_dir)
    _add_raft_path(raft_code_dir)
    return resolve_raft_model_path(weight_dir.resolve())


def _ensure_torch_meshgrid_compatibility() -> None:
    meshgrid_function = torch.meshgrid
    if "indexing" in inspect.signature(meshgrid_function).parameters:
        return

    def _meshgrid_compat(*tensors, **kwargs):
        kwargs.pop("indexing", None)
        return meshgrid_function(*tensors, **kwargs)

    torch.meshgrid = _meshgrid_compat


_ensure_torch_meshgrid_compatibility()


def log_progress(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def initialize_raft_model(
    raft_model_path: Path,
    device: torch.device,
    mixed_precision: bool,
):
    # Delayed import so this script can set sys.path first.
    from RAFT import RAFT

    args = argparse.Namespace(
        raft_model=str(raft_model_path),
        small=False,
        mixed_precision=mixed_precision,
        alternate_corr=False,
    )

    model = torch.nn.DataParallel(RAFT(args))
    state_dict = torch.load(str(raft_model_path), map_location=device)
    model.load_state_dict(state_dict)
    model = model.module
    model.to(device)
    model.eval()
    return model


def frame_to_tensor(frame_bgr: Any, device: torch.device) -> torch.Tensor:
    import cv2

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    return frame_tensor


def load_video_frames(video_path: Path, max_frames: int) -> List[Any]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames: List[Any] = []
    while len(frames) < max_frames:
        success, frame = capture.read()
        if not success:
            break
        frames.append(frame)

    capture.release()
    return frames


def score_video_motion(
    raft_model,
    frames: List[Any],
    device: torch.device,
    iters: int,
    video_name: str,
) -> Dict[str, float]:
    from RAFT.utils.utils import InputPadder
    import numpy as np

    if len(frames) < 2:
        raise ValueError("Need at least 2 frames to compute optical flow.")

    pair_mean_magnitudes: List[float] = []

    total_pairs = len(frames) - 1
    with torch.no_grad():
        for idx in range(total_pairs):
            frame1 = frame_to_tensor(frames[idx], device)
            frame2 = frame_to_tensor(frames[idx + 1], device)

            frame1 = frame1 * 2.0 - 1.0
            frame2 = frame2 * 2.0 - 1.0

            padder = InputPadder(frame1.shape)
            frame1_pad, frame2_pad = padder.pad(frame1, frame2)

            _, flow_up = raft_model(frame1_pad, frame2_pad, iters=iters, test_mode=True)
            flow_up = padder.unpad(flow_up)
            flow_xy = flow_up[0].permute(1, 2, 0).cpu().numpy()

            magnitude = np.linalg.norm(flow_xy, axis=2)
            pair_mean_magnitudes.append(float(magnitude.mean()))
            pair_idx = idx + 1
            if pair_idx == 1 or pair_idx == total_pairs or pair_idx % 10 == 0:
                log_progress(
                    f"[{video_name}] frame-pair progress: "
                    f"{pair_idx}/{total_pairs}"
                )

    pair_scores = np.asarray(pair_mean_magnitudes, dtype=np.float64)
    return {
        "num_frame_pairs": int(pair_scores.size),
        "mean_magnitude": float(pair_scores.mean()),
        "std_magnitude": float(pair_scores.std()),
        "max_pair_mean_magnitude": float(pair_scores.max()),
        "min_pair_mean_magnitude": float(pair_scores.min()),
    }


def parse_args() -> argparse.Namespace:
    default_weight_dir = resolve_default_weight_dir(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser(
        description="Compute optical flow magnitude score for a video."
    )
    io_group = parser.add_mutually_exclusive_group(required=True)
    io_group.add_argument(
        "--video_path",
        type=Path,
        default=None,
        help="Path to a single input video file.",
    )
    io_group.add_argument(
        "--video_dir",
        type=Path,
        default=None,
        help="Path to a directory that contains input video files.",
    )
    parser.add_argument(
        "--weight_dir",
        type=Path,
        default=default_weight_dir,
        help="Directory that contains raft-things.pth (or a single .pth file).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=81,
        help="Use only first N frames if video is longer.",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Optional cap on number of videos to process (sorted order).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="RAFT iteration count per frame pair.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Enable mixed precision in RAFT.",
    )
    parser.add_argument(
        "--save_json",
        type=Path,
        default=None,
        help="Optional path to save score result as JSON.",
    )
    return parser.parse_args()


def collect_video_paths(
    video_path: Path,
    video_dir: Path,
    max_videos: int = None,
) -> List[Path]:
    if video_path is not None:
        resolved_video = video_path.resolve()
        if not resolved_video.exists():
            raise FileNotFoundError(f"Video not found: {resolved_video}")
        return [resolved_video]

    resolved_dir = video_dir.resolve()
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {resolved_dir}")
    if not resolved_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {resolved_dir}")

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    if max_videos is not None:
        # Fast path for debug runs: stop scanning once enough videos are found.
        paths: List[Path] = []
        with os.scandir(resolved_dir) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                suffix = Path(entry.name).suffix.lower()
                if suffix not in video_extensions:
                    continue
                paths.append(Path(entry.path))
                if len(paths) >= max_videos:
                    return paths
    else:
        paths = sorted(
            [
                file_path
                for file_path in resolved_dir.iterdir()
                if file_path.is_file() and file_path.suffix.lower() in video_extensions
            ]
        )
    if not paths:
        raise ValueError(f"No supported video files found in: {resolved_dir}")
    return paths


def main() -> None:
    args = parse_args()
    if args.max_videos is not None and args.max_videos <= 0:
        raise ValueError("--max_videos must be a positive integer.")
    device = torch.device(args.device)
    raft_model_path = initialize_runtime_paths(args.weight_dir)

    log_progress(
        f"Start video(s). "
    )
    video_paths = collect_video_paths(
        args.video_path,
        args.video_dir,
        max_videos=args.max_videos,
    )
    log_progress(
        f"Found {len(video_paths)} video(s). "
        f"Device={device.type}, max_frames={args.max_frames}, iters={args.iters}"
    )
    log_progress(f"Loading RAFT weights from: {raft_model_path}")
    raft_model = initialize_raft_model(
        raft_model_path=raft_model_path,
        device=device,
        mixed_precision=args.mixed_precision,
    )
    log_progress("RAFT model loaded.")

    per_video_results: Dict[str, Dict[str, Any]] = {}
    total_videos = len(video_paths)
    for video_idx, video_path in enumerate(video_paths, start=1):
        try:
            log_progress(
                f"[{video_idx}/{total_videos}] Start video: {video_path.name}"
            )
            frames = load_video_frames(video_path=video_path, max_frames=args.max_frames)
            log_progress(
                f"[{video_idx}/{total_videos}] Loaded {len(frames)} frame(s) "
                f"from {video_path.name}"
            )
            score = score_video_motion(
                raft_model=raft_model,
                frames=frames,
                device=device,
                iters=args.iters,
                video_name=video_path.name,
            )
            per_video_results[video_path.name] = {
                "video_path": str(video_path),
                "used_frame_count": len(frames),
                "max_frames_limit": args.max_frames,
                "optical_flow_magnitude_score": score["mean_magnitude"],
                "details": score,
            }
            log_progress(
                f"[{video_idx}/{total_videos}] Done {video_path.name} | "
                f"score={score['mean_magnitude']:.6f}"
            )
        except Exception as error:  # pylint: disable=broad-except
            per_video_results[video_path.name] = {
                "video_path": str(video_path),
                "error": str(error),
            }
            log_progress(
                f"[{video_idx}/{total_videos}] Failed {video_path.name} | "
                f"error={error}"
            )

    valid_scores = [
        item["optical_flow_magnitude_score"]
        for item in per_video_results.values()
        if "optical_flow_magnitude_score" in item
    ]
    aggregate_score = (
        float(sum(valid_scores) / len(valid_scores)) if valid_scores else None
    )

    result = {
        "num_videos_total": len(video_paths),
        "num_videos_scored": len(valid_scores),
        "max_frames_limit": args.max_frames,
        "aggregate_mean_optical_flow_magnitude_score": aggregate_score,
        "videos": per_video_results,
    }

    log_progress(
        f"Completed. Scored {len(valid_scores)}/{len(video_paths)} video(s)."
    )
    print(json.dumps(result, indent=2))

    if args.save_json is not None:
        save_path = args.save_json.resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2)


if __name__ == "__main__":
    main()
