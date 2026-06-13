#!/usr/bin/env python3
"""Filter videos above the mean motion score among transition_count==0 videos."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_result_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if "videos" not in payload:
        raise ValueError(f"Missing 'videos' key in: {path}")
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


def get_transition_count(video_entry: Dict[str, Any]) -> Optional[int]:
    if "transition_count" not in video_entry:
        return None
    return int(video_entry["transition_count"])


def get_motion_score(video_entry: Dict[str, Any]) -> Optional[float]:
    score = video_entry.get("optical_flow_magnitude_score")
    if score is None:
        return None
    return float(score)


def compute_mean_score(videos: Dict[str, Dict[str, Any]], transition_count: int) -> Tuple[float, int]:
    scores: List[float] = []
    for video_entry in videos.values():
        if get_transition_count(video_entry) != transition_count:
            continue
        score = get_motion_score(video_entry)
        if score is not None:
            scores.append(score)
    if not scores:
        raise ValueError(
            f"No videos found with transition_count={transition_count} and a valid score."
        )
    return float(sum(scores) / len(scores)), len(scores)


def filter_videos(
    videos: Dict[str, Dict[str, Any]],
    threshold: float,
    transition_count: int,
    inclusive: bool,
) -> Dict[str, Dict[str, Any]]:
    filtered: Dict[str, Dict[str, Any]] = {}
    for video_name, video_entry in videos.items():
        if get_transition_count(video_entry) != transition_count:
            continue
        score = get_motion_score(video_entry)
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
            "Merge two result JSON files, compute mean optical_flow_magnitude_score "
            "among transition_count==0 videos, and keep those at or above the mean."
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
        "--transition_count",
        type=int,
        default=0,
        help="Only use videos with this transition_count for mean and filtering.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use > instead of >= when comparing against threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_paths = [args.json_a.resolve(), args.json_b.resolve()]

    for json_path in json_paths:
        if not json_path.exists():
            raise FileNotFoundError(f"JSON not found: {json_path}")

    payloads = [load_result_json(path) for path in json_paths]
    merged_videos = merge_videos(payloads)

    threshold, num_for_mean = compute_mean_score(
        merged_videos,
        transition_count=args.transition_count,
    )
    filtered_videos = filter_videos(
        merged_videos,
        threshold=threshold,
        transition_count=args.transition_count,
        inclusive=not args.strict,
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

    num_matching_transition = sum(
        1
        for entry in merged_videos.values()
        if get_transition_count(entry) == args.transition_count
    )

    result = {
        "source_files": [str(path) for path in json_paths],
        "transition_count_filter": args.transition_count,
        "mean_optical_flow_magnitude_score": threshold,
        "comparison": ">" if args.strict else ">=",
        "num_videos_total_scanned": len(merged_videos),
        "num_videos_with_matching_transition_count": num_matching_transition,
        "num_videos_used_for_mean": num_for_mean,
        "num_videos_above_threshold": len(filtered_videos),
        "filtered_aggregate_mean_optical_flow_magnitude_score": filtered_aggregate,
        "videos": filtered_videos,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)

    print(f"Source JSON A: {json_paths[0]}")
    print(f"Source JSON B: {json_paths[1]}")
    print(f"transition_count filter: {args.transition_count}")
    print(f"Videos with transition_count={args.transition_count}: {num_matching_transition}")
    print(f"Mean score (transition_count={args.transition_count}): {threshold:.6f}")
    print(f"Comparison operator: {result['comparison']}")
    print(f"Total videos scanned: {len(merged_videos)}")
    print(f"Videos above threshold: {len(filtered_videos)}")
    print(f"Saved to: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
