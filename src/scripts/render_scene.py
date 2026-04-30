"""
Render a simulated star tracker scene from the dataset.

Camera defaults (image_size=512, half_fov=6.0) match the training configuration
defined in src/data.py and src/train.py. Change them only if you retrain with
different settings.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt


# Scatter marker area bounds (matplotlib `s` units = points²)
_MIN_AREA = 20
_MAX_AREA = 200
_FIXED_AREA = 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a simulated star tracker scene from the dataset.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "examples:\n"
            "  python render_scene.py --ra 45 --dec 3 --roll 90 --mark-guide\n"
            "  python render_scene.py --random --show-mags --output out.png --no-display"
        ),
    )

    sel = parser.add_argument_group("scene selection")
    sel.add_argument(
        "--data",
        default="data/clean-de-0-6.json",
        metavar="PATH",
        help="Path to scene JSON (default: data/clean-de-0-6.json)",
    )
    sel.add_argument(
        "--ra",
        type=int,
        default=0,
        metavar="INT",
        help="Right ascension in degrees, 0-359 (default: 0)",
    )
    sel.add_argument(
        "--dec",
        type=int,
        default=0,
        metavar="INT",
        help="Declination in degrees (default: 0)",
    )
    sel.add_argument(
        "--roll",
        type=int,
        default=0,
        metavar="INT",
        help="Roll in degrees, 0-355 in multiples of 5 (default: 0)",
    )
    sel.add_argument(
        "--random",
        action="store_true",
        help="Pick a random scene (overrides --ra/--dec/--roll)",
    )

    cam = parser.add_argument_group("camera (must match training)")
    cam.add_argument(
        "--image-size",
        type=int,
        default=512,
        metavar="INT",
        help="Sensor side length in pixels (default: 512)",
    )
    cam.add_argument(
        "--half-fov",
        type=float,
        default=6.0,
        metavar="FLOAT",
        help="Half-FOV in degrees (default: 6.0)",
    )

    rend = parser.add_argument_group("rendering")
    rend.add_argument(
        "--no-scale-mags",
        action="store_true",
        help="Fixed dot size for all stars (default: scale by magnitude)",
    )
    rend.add_argument(
        "--mark-guide",
        action="store_true",
        help="Highlight the guide star (closest to image center) in red",
    )
    rend.add_argument(
        "--show-ids",
        action="store_true",
        help="Annotate each star with its catalog ID",
    )
    rend.add_argument(
        "--show-mags",
        action="store_true",
        help="Annotate each star with its magnitude",
    )

    out = parser.add_argument_group("output")
    out.add_argument(
        "--output",
        metavar="PATH",
        help="Save rendered image to this file (PNG/JPG)",
    )
    out.add_argument(
        "--no-display",
        action="store_true",
        help="Skip the interactive window (useful with --output)",
    )

    return parser.parse_args()


def find_guide_star(centroids: list[list[float]], center: float) -> int:
    """Return index of centroid closest to (center, center)."""
    best_idx = 0
    best_dist_sq = float("inf")
    for i, (x, y) in enumerate(centroids):
        dist_sq = (x - center) ** 2 + (y - center) ** 2
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_idx = i
    return best_idx


def mag_to_area(mag: float, min_mag: float, max_mag: float) -> float:
    """Map magnitude to scatter marker area. Brighter (lower mag) → larger."""
    if max_mag == min_mag:
        return (_MIN_AREA + _MAX_AREA) / 2
    t = (max_mag - mag) / (max_mag - min_mag)  # 1.0 = brightest
    return _MIN_AREA + t * (_MAX_AREA - _MIN_AREA)


def main() -> None:
    args = parse_args()

    if args.no_display and not args.output:
        print("error: --no-display requires --output (nothing would be produced)", file=sys.stderr)
        sys.exit(1)

    # --- Load data ---
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"error: data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    with open(data_path, "r") as f:
        scenes = json.load(f)

    # --- Find scene ---
    if args.random:
        scene_idx = random.randrange(len(scenes))
        scene = scenes[scene_idx]
    else:
        ra, dec, roll = args.ra, args.dec, args.roll

        # Validate inputs against what's actually in the file
        unique_ras = sorted({s["att"][0] for s in scenes})
        unique_decs = sorted({s["att"][1] for s in scenes})
        unique_rolls = sorted({s["att"][2] for s in scenes})

        if ra not in unique_ras:
            print(f"error: --ra {ra} not found in data. Valid range: {unique_ras[0]}–{unique_ras[-1]}", file=sys.stderr)
            sys.exit(1)
        if dec not in unique_decs:
            print(f"error: --dec {dec} not in data. Valid values: {unique_decs}", file=sys.stderr)
            sys.exit(1)
        if roll not in unique_rolls:
            print(f"error: --roll {roll} not found. Valid values are multiples of {unique_rolls[1] - unique_rolls[0]} in {unique_rolls[0]}–{unique_rolls[-1]}", file=sys.stderr)
            sys.exit(1)

        target = [ra, dec, roll]
        scene_idx = next((i for i, s in enumerate(scenes) if s["att"] == target), None)
        if scene_idx is None:
            print(f"error: no scene found for att={target}", file=sys.stderr)
            sys.exit(1)
        scene = scenes[scene_idx]

    att = scene["att"]
    ra, dec, roll = att
    centroids: list[list[float]] = scene["centroids"]
    star_ids: list[int] = scene["stars"]
    mags: list[float] = scene["mags"]
    n_stars = len(centroids)

    # --- Guide star ---
    center = args.image_size / 2.0
    guide_idx = find_guide_star(centroids, center)

    # --- Build marker sizes ---
    if args.no_scale_mags:
        sizes = [_FIXED_AREA] * n_stars
    else:
        min_mag, max_mag = min(mags), max(mags)
        sizes = [mag_to_area(m, min_mag, max_mag) for m in mags]

    xs = [c[0] for c in centroids]
    ys = [c[1] for c in centroids]

    # --- Plot ---
    fig_px = args.image_size
    fig, ax = plt.subplots(figsize=(fig_px / 100, fig_px / 100), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.set_xlim(0, args.image_size)
    ax.set_ylim(0, args.image_size)
    ax.invert_yaxis()  # pixel (0,0) at top-left
    ax.set_aspect("equal")
    ax.axis("off")

    # All stars
    ax.scatter(xs, ys, s=sizes, c="white", zorder=2)

    # Guide star highlight
    if args.mark_guide:
        ax.scatter(
            [xs[guide_idx]], [ys[guide_idx]],
            s=sizes[guide_idx] * 3,
            facecolors="none",
            edgecolors="red",
            linewidths=1.2,
            zorder=3,
        )

    # Annotations
    for i, (x, y) in enumerate(centroids):
        parts = []
        if args.show_ids:
            parts.append(str(star_ids[i]))
        if args.show_mags:
            parts.append(f"{mags[i]:.2f}")
        if parts:
            ax.text(
                x + 6, y - 6,
                " ".join(parts),
                color="white",
                fontsize=5,
                va="bottom",
                zorder=4,
            )

    # Draw image-center crosshair (faint)
    ax.axhline(center, color="#333333", linewidth=0.5, zorder=1)
    ax.axvline(center, color="#333333", linewidth=0.5, zorder=1)

    fig.suptitle(
        f"Scene {scene_idx}  |  RA={ra}°  Dec={dec}°  Roll={roll}°  |  {n_stars} stars"
        f"  |  FOV={args.half_fov * 2:.0f}°  {args.image_size}×{args.image_size}px",
        color="white",
        fontsize=7,
        y=0.98,
    )

    plt.tight_layout(pad=0.3)

    if args.output:
        plt.savefig(args.output, dpi=100, facecolor="black")
        print(f"Saved: {args.output}")

    if not args.no_display:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
