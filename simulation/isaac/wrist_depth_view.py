"""Synchronized wrist RGB + metric depth from the archived scene and mounts.

These are new display renders of recorded poses. Original control depth files
remain in native/*-wrist-depth-m.npy and are never overwritten by replay.
"""
import json
from pathlib import Path
import time

import cv2
import numpy as np


class WristDepthView:
    def __init__(self, camera_path, output=None):
        import omni.replicator.core as rep
        import omni.ui as ui

        self.rep = rep
        self.output = Path(output) if output else None
        self.product = rep.create.render_product(camera_path, (640, 360))
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self.rgb.attach([self.product])
        self.depth.attach([self.product])
        self.product.hydra_texture.set_updates_enabled(True)
        self.rgb_provider, self.depth_provider = ui.ByteImageProvider(), ui.ByteImageProvider()
        self.window = ui.Window("Wrist RGB-D | recorded pose replay", width=1000, height=365)
        with self.window.frame:
            with ui.VStack(spacing=4):
                ui.Label("WRIST RGB + DEPTH | same recorded pose / mount | display re-render", height=24)
                self.status = ui.Label("Waiting for wrist camera...", height=25)
                with ui.HStack(spacing=8):
                    ui.ImageWithProvider(self.rgb_provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                    ui.ImageWithProvider(self.depth_provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                ui.Label("Depth: axial meters | fixed 0.05-1.0 m color range | black = invalid / no return", height=24)
        self.last_update = -1.
        self.saved = False
        self.frames = 0
        self.camera_path = camera_path

    def update(self, source_time_s, force=False):
        wall = time.monotonic()
        if not force and (not self.window.visible or wall - self.last_update < .1):
            return
        self.last_update = wall
        # The owning loop has just rendered the current pose with app.update().
        # Read the active product directly; orchestrator.step() pumps extra app
        # frames and stalls the main viewport for every depth refresh.
        rgba = np.asarray(self.rgb.get_data())
        depth = np.asarray(self.depth.get_data()).squeeze()
        if rgba.ndim != 3 or depth.ndim != 2 or not rgba.size or not depth.size:
            self.status.text = "Waiting for native RGB-D annotators..."
            return
        valid = np.isfinite(depth) & (depth > 0)
        levels = np.zeros(depth.shape, np.uint8)
        levels[valid] = np.uint8(np.clip((depth[valid] - .05) / .95 * 255, 0, 255))
        colored = cv2.applyColorMap(levels, cv2.COLORMAP_TURBO)
        colored[~valid] = 0
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGBA)
        if rgba.shape[2] == 3:
            rgba = cv2.cvtColor(rgba, cv2.COLOR_RGB2RGBA)
        self.rgb_provider.set_data_array(np.ascontiguousarray(rgba), [rgba.shape[1], rgba.shape[0]])
        self.depth_provider.set_data_array(np.ascontiguousarray(colored), [depth.shape[1], depth.shape[0]])
        self.frames += 1
        values = depth[valid]
        stats = ({"min_m": float(values.min()), "median_m": float(np.median(values)),
                  "max_m": float(values.max())} if values.size else {})
        self.status.text = (f"t={source_time_s:.2f}s | {self.camera_path} | valid {100*valid.mean():.1f}% | "
                            + (f"min {stats['min_m']:.3f} m / median {stats['median_m']:.3f} m" if stats else "no return"))
        if self.output:
            self.output.mkdir(parents=True, exist_ok=True)
            report = {"source": "Isaac replay render from recorded body poses; not a new physical validation",
                      "camera": self.camera_path, "source_time_s": source_time_s,
                      "depth_type": "distance_to_image_plane", "unit": "m",
                      "valid_fraction": float(valid.mean()), "frames": self.frames, **stats}
            temp = self.output / "wrist-current.tmp"
            temp.write_text(json.dumps(report, allow_nan=False, indent=2))
            temp.replace(self.output / "wrist-current.json")
            with (self.output / "wrist-depth-frames.jsonl").open("a") as stream:
                stream.write(json.dumps(report, allow_nan=False) + "\n")
            if values.size and not self.saved:
                np.save(self.output / "wrist-replay-depth-m.npy", depth)
                cv2.imwrite(str(self.output / "wrist-replay-rgb.png"), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
                cv2.imwrite(str(self.output / "wrist-replay-depth.png"), cv2.cvtColor(colored, cv2.COLOR_RGBA2BGRA))
                (self.output / "wrist-snapshot.json").write_text(json.dumps(report, indent=2))
                self.saved = True

    def close(self):
        self.rgb.detach([self.product.path])
        self.depth.detach([self.product.path])
        self.product.destroy()
        self.window.visible = False
