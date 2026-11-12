"""Show saved Isaac camera frames and detector measurements as they arrive.

This lightweight process reads continuous display previews when available,
otherwise the archived control captures. It never drives physics.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulation.isaac.live_stream import request_camera_previews


TILE_W, TILE_H = 640, 360
FONT = cv2.FONT_HERSHEY_SIMPLEX
_images = {}
_depth_insets = {}


def observations(folder: Path, include_live=True):
    """Index complete image/metadata pairs; ignore files mid-write."""
    streams = {name: [] for name in ("global", "wrist", "side", "detector")}
    patterns = (
        ("global", "*-vision.json", lambda p: folder / (p.stem[:-7] + ".png"), "time_s"),
        ("wrist", "*-wrist.json", lambda p: p.with_suffix(".png"), "timestamp"),
        ("side", "fat-side-*.json", lambda p: p.with_suffix(".png"), "time_s"),
    )
    for stream, pattern, image_path, time_key in patterns:
        for metadata in folder.glob(pattern):
            image = image_path(metadata)
            if not image.is_file():
                continue
            try:
                info = json.loads(metadata.read_text())
                stamp = float(info[time_key])
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if np.isfinite(stamp):
                streams[stream].append((stamp, metadata.stem, image))
    for metadata in (folder / "optics").glob("*/measurement.json"):
        image = metadata.parent / "phase-00-detector16.png"
        if not image.is_file():
            continue
        try:
            info = json.loads(metadata.read_text())
            stamp = float(info["time_s"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if np.isfinite(stamp):
            streams["detector"].append((stamp, metadata.parent.name, image, info))
    for entries in streams.values():
        entries.sort(key=lambda row: (row[0], row[1] == "released-1024", row[1]))
    manifest = folder / "live/preview.json"
    if include_live and manifest.is_file():
        try:
            preview = json.loads(manifest.read_text())
            stamp = float(preview["time_s"])
            for stream in ("global", "wrist", "side"):
                path = folder / "live" / preview["frames"][stream]
                if path.is_file():
                    streams[stream] = [(stamp, "live-" + stream, path)]
        except (OSError, KeyError, ValueError, TypeError):
            pass
    return streams


def latest(entries, now):
    return next((row for row in reversed(entries) if row[0] <= now + 1e-6), None)


def detector_preview(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != np.uint16 or raw.ndim != 2:
        return None
    # Fixed logarithmic exposure. A weak fringe must stay visibly weak.
    levels = np.log1p(1000 * raw.astype(np.float32) / 65535) / np.log1p(1000)
    return cv2.applyColorMap(np.uint8(np.clip(levels * 255, 0, 255)), cv2.COLORMAP_INFERNO)


def read_cached(path, stream, detector=False):
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return None
    record = _images.get(stream)
    if record is not None and record[0] == path and record[1] == stamp:
        return record[2]
    image = detector_preview(path) if detector else cv2.imread(str(path))
    _images[stream] = (path, stamp, image)
    return image


def fit(image, width=TILE_W, height=TILE_H):
    canvas = np.full((height, width, 3), (26, 26, 29), dtype=np.uint8)
    if image is None:
        return canvas
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x, y = (width - new_w) // 2, (height - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def tile(row, title, now, detector=False):
    image = read_cached(row[2], title, detector=detector) if row else None
    panel = fit(image)
    cv2.rectangle(panel, (0, 0), (TILE_W, 46), (0, 0, 0), -1)
    status = ("available after five-component assembly" if detector else "awaiting first frame") if row is None else f"{row[1]}  captured t={row[0]:.2f}s  age={max(0, now - row[0]):.1f}s"
    cv2.putText(panel, title, (12, 19), FONT, .52, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(panel, status, (12, 39), FONT, .43, (170, 220, 255), 1, cv2.LINE_AA)
    if row and title.startswith("Wrist"):
        depth_file = row[2].with_name(row[2].stem + "-depth-m.npy")
        if depth_file.is_file():
            try:
                cache_key = (depth_file, depth_file.stat().st_mtime_ns)
                inset = _depth_insets.get(cache_key)
                if inset is None:
                    depth = np.asarray(np.load(depth_file)).squeeze()
                    if depth.ndim != 2:
                        raise ValueError("Wrist depth must be a 2D image")
                    valid = np.isfinite(depth) & (depth > 0)
                    shade = np.zeros_like(depth, dtype=np.uint8)
                    shade[valid] = np.uint8(np.clip((depth[valid] - .05) / .5 * 255, 0, 255))
                    inset = cv2.applyColorMap(cv2.resize(shade, (210, 118)), cv2.COLORMAP_TURBO)
                    _depth_insets.clear()
                    _depth_insets[cache_key] = inset
                if inset is not None:
                    panel[-128:-10, -220:-10] = inset
                    cv2.putText(panel, "metric depth 0.05-0.55m", (TILE_W - 218, TILE_H - 115),
                                FONT, .3, (255, 255, 255), 1, cv2.LINE_AA)
            except (OSError, ValueError):
                pass
    return panel


def compose(streams, now, mode):
    frame = np.full((TILE_H * 2 + 160, TILE_W * 2, 3), (20, 20, 24), dtype=np.uint8)
    panels = (
        ("global", "Global RGB / top camera", 0, 0),
        ("wrist", "Wrist RGB-D / depth inset", 0, TILE_W),
        ("side", "FAT side RGB", TILE_H, 0),
        ("detector", "LightPipes detector / phase 0", TILE_H, TILE_W),
    )
    for key, title, y, x in panels:
        row = latest(streams[key], now)
        frame[y:y + TILE_H, x:x + TILE_W] = tile(row, title, now, detector=key == "detector")
    footer = TILE_H * 2
    live = any(rows and rows[-1][1].startswith("live-") for key, rows in streams.items() if key != "detector")
    description = "continuous preview from recent PhysX poses / display interpolation" if live else "archived control camera captures"
    cv2.putText(frame, f"{mode.upper()} | simulation t={now:.2f}s | {description}",
                (15, footer + 24), FONT, .57, (235, 235, 235), 1, cv2.LINE_AA)
    measurements = [entry for entry in streams["detector"] if entry[0] <= now + 1e-6]
    if measurements:
        last = measurements[-1]
        info = last[3]
        tilt = np.linalg.norm(info.get("relative_tilt_from_pixels_rad", [float("nan")])) * 1e6
        label = (f"Latest: {last[1]} | visibility {100 * info['visibility']:.3f}% | "
                 f"tilt {tilt:.2f} urad | {'valid' if info['valid'] else 'invalid'}")
        cv2.putText(frame, label, (15, footer + 52), FONT, .53, (150, 245, 150), 1, cv2.LINE_AA)
        left, right, top, bottom = 20, TILE_W * 2 - 20, footer + 74, footer + 143
        cv2.rectangle(frame, (left, top), (right, bottom), (60, 60, 65), 1)
        t0 = measurements[0][0]
        t1 = max(measurements[-1][0], t0 + 1)
        points = []
        for entry in measurements:
            x = left + round((entry[0] - t0) / (t1 - t0) * (right - left))
            y = bottom - round(np.clip(entry[3]["visibility"], 0., 1.) * (bottom - top))
            points.append((x, y))
        if len(points) > 1:
            cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, (80, 225, 250), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 3, (80, 225, 250), -1)
        cv2.putText(frame, "100%", (left + 4, top + 14), FONT, .35, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.putText(frame, "0%", (left + 4, bottom - 4), FONT, .35, (190, 190, 190), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Optical feedback starts after five-component assembly", (15, footer + 55),
                    FONT, .55, (150, 180, 220), 1, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True, help="Isaac run directory; may still be active")
    parser.add_argument("--replay", action="store_true", help="Advance captures by saved simulation timestamps")
    parser.add_argument("--speed", type=float, default=2., help="Replay speed only")
    parser.add_argument("--start-seconds", type=float, default=0.)
    parser.add_argument("--snapshot", type=Path, help="Save current dashboard image")
    parser.add_argument("--no-window", action="store_true", help="Write one snapshot and exit")
    parser.add_argument("--max-seconds", type=float, help="Bounded monitor verification")
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be positive")
    if args.no_window and args.snapshot is None:
        parser.error("--no-window requires --snapshot")
    folder = args.folder.resolve()
    if not folder.is_dir():
        parser.error(f"run directory does not exist: {folder}")
    title = "Isaac camera and Michelson optics"
    if not args.no_window:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 1280, 880)
    start_wall = time.monotonic()
    paused_at = None
    selected = None
    preview_lease = None
    last_preview_request = 0.
    monitor_started = time.monotonic()
    channels = {"1": ("global", "Global RGB / top camera"),
                "2": ("wrist", "Wrist RGB-D / depth inset"),
                "3": ("side", "FAT side RGB"),
                "4": ("detector", "LightPipes detector / phase 0")}
    try:
        while True:
            if args.max_seconds is not None and time.monotonic()-monitor_started >= args.max_seconds:
                break
            if not args.replay and not args.no_window and (folder/"live/cameras.json").exists() and time.monotonic()-last_preview_request >= .5:
                preview_lease = request_camera_previews(folder/"live", os.getpid())
                last_preview_request = time.monotonic()
            streams = observations(folder, include_live=not args.replay)
            stamps = [row[0] for rows in streams.values() for row in rows]
            if args.replay:
                now = args.start_seconds if paused_at is not None else args.start_seconds + (time.monotonic() - start_wall) * args.speed
                if stamps:
                    now = min(now, max(stamps))
            else:
                now = max(stamps, default=0.)
            frame = compose(streams, now, "recorded replay" if args.replay else "live file follow")
            if args.snapshot:
                args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.snapshot), frame):
                    raise RuntimeError("Could not write dashboard snapshot")
                print(f"CAMERA_OPTICS_SNAPSHOT {args.snapshot.resolve()}", flush=True)
                args.snapshot = None
            if args.no_window:
                return
            display = frame
            if selected is not None:
                stream, label = channels[selected]
                panel = tile(latest(streams[stream], now), label, now, detector=stream == "detector")
                display = cv2.resize(panel, (1280, 720), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(title, display)
            key = cv2.waitKey(15) & 255
            if key in (27, ord("q")):
                return
            if key in (ord("0"), ord("1"), ord("2"), ord("3"), ord("4")):
                selected = None if key == ord("0") else chr(key)
            if key == ord(" ") and args.replay:
                if paused_at is None:
                    paused_at = now
                else:
                    start_wall = time.monotonic() - (paused_at - args.start_seconds) / args.speed
                    paused_at = None
    finally:
        if preview_lease:
            preview_lease.unlink(missing_ok=True)
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
