"""Check actual frozen Isaac viewport captures for flicker and outline changes.

Generate captures with --freeze --render-probe-dir in the 3D replay viewer.
The check must use a fixed camera and fixed recorded poses, not moving footage.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def check(folder):
    files = sorted(Path(folder).glob("viewport-*.png"))
    if len(files) < 20:
        raise ValueError("At least 20 frozen viewport captures are required")
    frames = [cv2.imread(str(path)) for path in files]
    if any(frame is None or frame.shape != frames[0].shape for frame in frames):
        raise ValueError("Viewport captures must be readable and have equal dimensions")
    height, width = frames[0].shape[:2]
    # Exclude viewport borders and the status overlay. Analyze scene pixels.
    gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[
        round(height*.10):round(height*.95), round(width*.07):round(width*.93)] for frame in frames]
    edges = cv2.Canny(gray[0], 40, 100) > 0
    if not edges.any():
        raise ValueError("Frozen captures contain no scene edges")
    rms, changed, drift = [], [], []
    for a, b in zip(gray, gray[1:]):
        difference = np.abs(a.astype(float)-b.astype(float))
        rms.append(float(np.sqrt(np.mean(difference**2))))
        changed.append(float(np.mean(difference > 5)))
        distance = cv2.distanceTransform((cv2.Canny(b, 40, 100) == 0).astype(np.uint8), cv2.DIST_L2, 5)
        drift.append(float(np.percentile(distance[edges], 95)))
    return {"frames": len(files), "resolution": [width, height],
        "mean_temporal_rms_8bit": float(np.mean(rms)), "max_temporal_rms_8bit": max(rms),
        "changed_pixel_fraction_mean": float(np.mean(changed)),
        "outline_motion_p95_px_max": max(drift),
        "success": max(drift) <= .5 and float(np.mean(changed)) <= .005}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    report = check(args.folder)
    (args.folder / "flicker-check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
