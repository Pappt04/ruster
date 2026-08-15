#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///
"""render_classification.py — draws the heterogeneous scheduler's tile
classification as a red/green image.

Pair with the Rust side, `examples/classify_dump.rs`, which runs
`scheduler::classifier::partition_frame` for a chosen viewport and writes the
GPU/CPU tile rectangles to JSON — this script never touches the renderer or
the classifier itself, it only draws the rectangles that decision already
produced. Split out of Rust (the user's own call, not a technical
requirement) because a debug visualization like this is a one-off image, not
a render-loop path this codebase's performance work cares about; a small
script that reads JSON and draws rectangles is simpler to iterate on than a
new Rust binary, and PIL's `ImageDraw.rectangle` is a one-liner where the
equivalent in `image`/`egui::ColorImage` is not.

GPU tiles are drawn red, CPU tiles green — coherent, cheap regions the
classifier is confident enough to hand the GPU come out red; divergent
boundary regions it routes to the CPU for exact per-pixel fill come out
green. Each tile gets a thin darker border so adjacent same-color tiles
don't visually merge into one blob, which would hide the actual partition
granularity.

Usage:
    uv run scripts/render_classification.py classification.json
    uv run scripts/render_classification.py classification.json --out tiles.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

GPU_FILL = (200, 40, 40)
GPU_BORDER = (110, 20, 20)
CPU_FILL = (40, 170, 70)
CPU_BORDER = (20, 100, 40)


def draw_tiles(draw: ImageDraw.ImageDraw, tiles: list[list[int]], fill: tuple[int, int, int], border: tuple[int, int, int]) -> None:
    for x0, y0, w, h in tiles:
        draw.rectangle([x0, y0, x0 + w - 1, y0 + h - 1], fill=fill, outline=border, width=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_path", type=Path, help="output of examples/classify_dump.rs")
    ap.add_argument("--out", type=Path, default=None, help="PNG path (default: json_path with .png extension)")
    args = ap.parse_args()

    data = json.loads(args.json_path.read_text())
    width, height = data["width"], data["height"]
    gpu_tiles, cpu_tiles = data["gpu_tiles"], data["cpu_tiles"]

    img = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    # GPU first, CPU second — if a future classifier ever produces adjacent
    # differently-classified tiles with a shared edge pixel (shouldn't
    # happen — partition_frame guarantees disjoint tiles — this ordering is
    # just cheap insurance against that edge ever silently reading GPU).
    draw_tiles(draw, gpu_tiles, GPU_FILL, GPU_BORDER)
    draw_tiles(draw, cpu_tiles, CPU_FILL, CPU_BORDER)

    out_path = args.out or args.json_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

    gpu_px = sum(w * h for _, _, w, h in gpu_tiles)
    cpu_px = sum(w * h for _, _, w, h in cpu_tiles)
    total = width * height
    print(f"{len(gpu_tiles)} gpu tiles ({gpu_px / total:.1%} of pixels, red)")
    print(f"{len(cpu_tiles)} cpu tiles ({cpu_px / total:.1%} of pixels, green)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
