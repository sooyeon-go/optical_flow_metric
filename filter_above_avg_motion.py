#!/usr/bin/env python3
"""Filter videos whose motion score exceeds the mean of source JSON aggregates."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_result_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if "videos" not in payload:
        raise ValueError(f"Missing 'videos' key in: {path}")
    if "aggregate_mean_optical_flow_magnitude_score" not in payload:
        raise ValueError(
            f"Missing 'aggregate_mean_optical_flow_magnitude_score' in: {path}"
        )
    return payload


def merge_videos(payloads: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for payload in payloads:
        for video_name, video_entry in payload["videos"].items():
            if video_name in merged:
                raise ValueError(
                    f"Duplicate video name across inputs: {video_name}"
                )
            merged[video_name] = video_entry
    return merged


def filter_above_threshold(
    videos: Dict[str, Dict[str, Any]],
    threshold: float,
    inclusive: bool,
) -> Dict[str, Dict[str, Any]]:
    filtered: Dict[str, Dict[str, Any]] = {}
    for video_name, video_entry in videos.items():
        score = video_entry.get("optical_flow_magnitude_score")
        if score is None:
            continue
        if inclusive:
            keep = score >= threshold
        else:
            keep = score > threshold
        if keep:
            filtered[video_name] = video_entry
    return filtered


def parse_args() -> argparse.Namespace:
    default_output_dir = Path("/data/project-vilab/sy/optical_flow_metric/output")
    parser = argparse.ArgumentParser(
        description=(
            "Merge two optical-flow result JSON files and keep videos "
            "above the mean of their aggregate scores."
        )
    )
    parser.add_argument(
        "--json_a",
        type=Path,
        default=default_output_dir / "result50k.json",
        help="First result JSON path.",
    )
    parser.add_argument(
        "--json_b",
        type=Path,
        default=default_output_dir / "result_50k_150k.json",
        help="Second result JSON path.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=default_output_dir / "result_above_avg_motion.json",
        help="Output JSON path for filtered videos.",
    )
    parser.add_argument(
        "--inclusive",
        action="store_true",
        help="Use >= instead of > when comparing against threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_paths = [args.json_a.resolve(), args.json_b.resolve()]

    for json_path in json_paths:
        if not json_path.exists():
            raise FileNotFoundError(f"JSON not found: {json_path}")

    payloads = [load_result_json(path) for path in json_paths]
    aggregate_means = [
        float(payload["aggregate_mean_optical_flow_magnitude_score"])
        for payload in payloads
    ]
    threshold = float(sum(aggregate_means) / len(aggregate_means))

    merged_videos = merge_videos(payloads)
    filtered_videos = filter_above_threshold(
        merged_videos,
        threshold=threshold,
        inclusive=args.inclusive,
    )

    filtered_scores = [
        entry["optical_flow_magnitude_score"]
        for entry in filtered_videos.values()
    ]
    filtered_aggregate = (
        float(sum(filtered_scores) / len(filtered_scores))
        if filtered_scores
        else None
    )

    result = {
        "source_files": [str(path) for path in json_paths],
        "source_aggregate_means": aggregate_means,
        "combined_aggregate_mean": threshold,
        "comparison": ">=" if args.inclusive else ">",
        "num_videos_total_scanned": len(merged_videos),
        "num_videos_above_threshold": len(filtered_videos),
        "filtered_aggregate_mean_optical_flow_magnitude_score": filtered_aggregate,
        "videos": filtered_videos,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)

    print(f"Source JSON A: {json_paths[0]}")
    print(f"  aggregate_mean = {aggregate_means[0]:.6f}")
    print(f"Source JSON B: {json_paths[1]}")
    print(f"  aggregate_mean = {aggregate_means[1]:.6f}")
    print(f"Combined threshold ({result['comparison']}): {threshold:.6f}")
    print(f"Total videos scanned: {len(merged_videos)}")
    print(f"Videos above threshold: {len(filtered_videos)}")
    print(f"Saved to: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
