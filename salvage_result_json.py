#!/usr/bin/env python3
"""Validate or salvage optical-flow result JSON files."""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def validate_json(path: Path) -> None:
    with path.open("r", encoding="utf-8") as input_file:
        json.load(input_file)


def find_json_error(path: Path) -> Tuple[int, int, str]:
    text = path.read_text(encoding="utf-8")
    try:
        json.loads(text)
        return 0, 0, "valid"
    except json.JSONDecodeError as error:
        return error.lineno, error.colno, error.msg


def show_context(path: Path, line_no: int, radius: int = 8) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = max(1, line_no - radius)
    end = min(len(lines), line_no + radius)
    print(f"\nContext around line {line_no} in {path}:")
    for idx in range(start, end + 1):
        marker = ">>" if idx == line_no else "  "
        print(f"{marker} {idx:6d}: {lines[idx - 1]}")


def salvage_videos(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    videos_start = text.find('"videos"')
    if videos_start < 0:
        raise ValueError(f"No 'videos' key found in {path}")

    brace_start = text.find("{", videos_start)
    if brace_start < 0:
        raise ValueError(f"Could not locate videos object in {path}")

    decoder = json.JSONDecoder()
    videos, end_idx = decoder.raw_decode(text, brace_start)
    if not isinstance(videos, dict):
        raise ValueError(f"'videos' is not an object in {path}")

    header_text = text[:videos_start]
    num_videos_total = None
    max_frames_limit = None
    aggregate_mean = None

    total_match = re.search(r'"num_videos_total"\s*:\s*(\d+)', header_text)
    if total_match:
        num_videos_total = int(total_match.group(1))
    frames_match = re.search(r'"max_frames_limit"\s*:\s*(\d+)', header_text)
    if frames_match:
        max_frames_limit = int(frames_match.group(1))
    aggregate_match = re.search(
        r'"aggregate_mean_optical_flow_magnitude_score"\s*:\s*([0-9.eE+-]+|null)',
        header_text,
    )
    if aggregate_match and aggregate_match.group(1) != "null":
        aggregate_mean = float(aggregate_match.group(1))

    valid_scores = [
        float(entry["optical_flow_magnitude_score"])
        for entry in videos.values()
        if isinstance(entry, dict) and "optical_flow_magnitude_score" in entry
    ]
    salvaged_aggregate = (
        float(sum(valid_scores) / len(valid_scores)) if valid_scores else None
    )

    return {
        "source_file": str(path.resolve()),
        "salvaged": True,
        "parse_end_char_index": end_idx,
        "file_size_chars": len(text),
        "num_videos_total": num_videos_total,
        "num_videos_salvaged": len(videos),
        "max_frames_limit": max_frames_limit,
        "original_aggregate_mean_optical_flow_magnitude_score": aggregate_mean,
        "salvaged_aggregate_mean_optical_flow_magnitude_score": salvaged_aggregate,
        "videos": videos,
    }


def parse_args() -> argparse.Namespace:
    default_output_dir = Path("/data/project-vilab/sy/optical_flow_metric/output")
    parser = argparse.ArgumentParser(description="Validate or salvage result JSON.")
    parser.add_argument("json_path", type=Path, help="Input JSON path.")
    parser.add_argument(
        "--salvage",
        action="store_true",
        help="Extract valid video entries and write a repaired JSON.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Output path for salvaged JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path = args.json_path.resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    try:
        validate_json(json_path)
        print(f"OK: {json_path}")
        return
    except json.JSONDecodeError:
        line_no, col_no, message = find_json_error(json_path)
        print(f"BROKEN: {json_path}")
        print(f"  line={line_no}, col={col_no}, msg={message}")
        show_context(json_path, line_no)

    if not args.salvage:
        print("\nTip: run with --salvage to extract valid video entries.")
        return

    payload = salvage_videos(json_path)
    output_path = args.output_json
    if output_path is None:
        output_path = json_path.with_name(json_path.stem + "_salvaged.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)

    print(f"\nSalvaged videos: {payload['num_videos_salvaged']}")
    print(f"Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
