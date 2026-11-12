"""Independent native-state display with on-demand continuous camera previews.

The worker owns dynamics, safety and control. This renderer interpolates only
recent measured body transforms and renders preview cameras from that scene.
"""
import argparse
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--folder", type=Path, required=True)
parser.add_argument("--max-frames", type=int, default=0)
parser.add_argument("--camera-fps", type=float, default=15.)
parser.add_argument("--fps", type=float, default=60., help="Display frame rate target")
parser.add_argument("--preview-width", type=int, default=480, help="Width of on-demand RGB/depth previews")
parser.add_argument("--render-probe-dir", type=Path, help="Save fixed-scene render diagnostics")
args = parser.parse_args()
if args.camera_fps <= 0 or args.fps <= 0 or args.preview_width < 160:
    parser.error("--camera-fps and --fps must be positive")
folder = args.folder.resolve()
live = folder / "live"

from isaacsim import SimulationApp
app = SimulationApp({"headless": False, "renderer": "RaytracedLighting", "anti_aliasing": 2,
                     "width": 1440, "height": 810})
signal.signal(signal.SIGINT, signal.default_int_handler)

import json
import cv2
import numpy as np
import omni.usd
import omni.timeline
import omni.ui as ui
import omni.replicator.core as rep
from isaacsim.core.utils.stage import open_stage, is_stage_loading
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, UsdGeom, UsdPhysics
from simulation.isaac.live_stream import DisplayBuffer, write_json, camera_previews_requested
from simulation.isaac.display_rendering import configure, settings_snapshot, PoseWriter, ViewportProbe, add_cameras

open_stage(str(folder / "scene.usda"))
while is_stage_loading():
    app.update()
stage = omni.usd.get_context().get_stage()
omni.timeline.get_timeline_interface().stop()
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
    if prim.IsA(UsdPhysics.Joint):
        UsdPhysics.Joint(prim).GetJointEnabledAttr().Set(False)

view = UsdGeom.Camera.Define(stage, "/World/LiveOverview")
view.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
view.CreateFocalLengthAttr(18.)
viewport = get_active_viewport()
viewport.set_active_camera(str(view.GetPath()))
viewport.resolution = (1440, 810)
viewport.resolution_scale = 1.
set_camera_view(eye=np.array([1.4, -1.1, 1.7]), target=np.array([0., .28, .85]),
                camera_prim_path=str(view.GetPath()))

cameras = json.loads((live / "cameras.json").read_text())
camera_paths = add_cameras(stage, cameras)
products, annotators = {}, {}
for name, path in camera_paths.items():
    product = rep.create.render_product(path, (args.preview_width, round(args.preview_width*9/16)))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([product])
    channels = {"rgb": rgb}
    if name == "wrist":
        depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        depth.attach([product])
        channels["depth"] = depth
    products[name], annotators[name] = product, channels
rep.orchestrator.set_capture_on_play(True)
from simulation.optical_device_visuals import upgrade_isaac_display
device_cameras = upgrade_isaac_display(stage, json.loads((folder/"configuration.json").read_text()))
configure(viewport, stage=stage)
# Products stay continuously active for the whole monitor session, rather than
# toggling at the publication rate. With no monitor they are paused together.
for product in products.values():
    product.hydra_texture.set_updates_enabled(True)
omni.timeline.get_timeline_interface().play()
probe = ViewportProbe(args.render_probe_dir, viewport)

window = ui.Window("Live PhysX experiment", width=540, height=180)
with window.frame:
    with ui.VStack(spacing=5):
        ui.Label("LIVE native state | 100 ms display interpolation")
        status = ui.Label("Starting preview cameras before robot motion...")
        phase_label = ui.Label("")
        ui.Label("Camera previews: camera_optics_monitor.py --folder <run>")
        with ui.HStack(height=26):
            for label, path in {"Overview": str(view.GetPath()), "Global": camera_paths["global"],
                                "Wrist": camera_paths["wrist"], "FAT": camera_paths["side"],
                                **{key.title(): value for key, value in device_cameras.items()}}.items():
                ui.Button(label, clicked_fn=lambda path=path: viewport.set_active_camera(path))

buffer = DisplayBuffer()
last_state_stamp = None
prims = None
pose_writer = None
frames, preview_count = 0, 0
intervals = []
mode_intervals = {"viewport": [], "previews": []}
phase_timings = {name: [] for name in ("state", "pose", "render", "preview")}
last_frame_wall = None
next_preview_wall = 0.
first_frame_time = None
done_at = None
display_time = 0.
preview_active = True
preview_requested = False
last_consumer_check = 0.


def read_state():
    global last_state_stamp, prims, pose_writer
    stamp = (live / "state.npz").stat().st_mtime_ns
    if stamp == last_state_stamp:
        return
    with np.load(live / "state.npz") as data:
        state = {key: data[key].copy() for key in data.files}
    buffer.push(state)
    if prims is None:
        prims = [stage.GetPrimAtPath("/World/Bodies/" + str(name)) for name in state["body_names"]]
        if any(not prim.IsValid() for prim in prims):
            raise RuntimeError("LIVE_NATIVE_BODY_PATH_MISSING")
        pose_writer = PoseWriter(prims)
    last_state_stamp = stamp


def publish_preview(simulation_time, state_age):
    global first_frame_time, preview_count
    images = {}
    for name, channels in annotators.items():
        pixels = np.asarray(channels["rgb"].get_data())
        if pixels.ndim != 3 or pixels.shape[2] < 3 or not pixels.size:
            return False
        images[name] = np.ascontiguousarray(pixels[:, :, :3])
    for name, image in images.items():
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("LIVE_CAMERA_ENCODING_FAILED")
        temporary = live / (name + ".jpg.tmp")
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(live / (name + ".jpg"))
    depth = np.asarray(annotators["wrist"]["depth"].get_data()).squeeze()
    if depth.ndim == 2 and depth.size:
        temporary = live / "wrist-depth-m.npy.tmp"
        with temporary.open("wb") as stream:
            np.save(stream, depth)
        temporary.replace(live / "wrist-depth-m.npy")
    if first_frame_time is None:
        first_frame_time = float(simulation_time)
    preview_count += 1
    write_json(live / "preview.json", {"time_s": float(simulation_time), "wall_time_unix_s": time.time(),
        "first_frame_simulation_time_s": first_frame_time,
        "native_state_wall_age_s": state_age, "frames": {name: name + ".jpg" for name in images},
        "source": "Isaac RGB/depth rendered from interpolated recent native PhysX poses; display only"})
    return True


next_frame = time.monotonic()
try:
    while app.is_running():
        wall = time.monotonic()
        if last_frame_wall is not None:
            intervals.append(wall - last_frame_wall)
            mode_intervals["previews" if preview_active else "viewport"].append(wall-last_frame_wall)
        last_frame_wall = wall
        read_state()
        stamp = time.monotonic()
        phase_timings["state"].append(stamp-wall)
        p, qs, display_time, phase, rate = buffer.at(wall)
        pose_writer.update(p, qs)
        after_pose = time.monotonic()
        phase_timings["pose"].append(after_pose-stamp)
        if wall-last_consumer_check >= .25:
            preview_requested = camera_previews_requested(live, wall)
            last_consumer_check = wall
        active = first_frame_time is None or preview_requested
        if active != preview_active:
            for product in products.values():
                product.hydra_texture.set_updates_enabled(active)
            preview_active = active
        due = first_frame_time is None or (active and wall+.001 >= next_preview_wall)
        app.update()
        after_render = time.monotonic()
        phase_timings["render"].append(after_render-after_pose)
        frames += 1
        if frames == 20:
            observation = settings_snapshot(viewport)
            write_json(live / "render-settings.json", observation)
            probe.save("render-settings.json", observation)
        probe.update(frames)
        source_age = max(0., wall - buffer.samples[-1][0])
        if due and frames >= 8 and publish_preview(display_time, source_age):
            # Advance a continuous publication clock rather than resetting it
            # to the slightly late display frame (which reduced 15 Hz to 10 Hz).
            next_preview_wall = (wall+1/args.camera_fps if next_preview_wall < wall-1/args.camera_fps
                                 else next_preview_wall+1/args.camera_fps)
            if not (live / "ready.json").exists():
                write_json(live / "ready.json", {"first_frame_simulation_time_s": first_frame_time})
                print("LIVE_CAMERAS_READY", first_frame_time, flush=True)
        phase_timings["preview"].append(time.monotonic()-after_render)
        fps = 1 / max(np.median(intervals[-60:]), 1e-6) if intervals else 0.
        status.text = f"Physics t={display_time:.2f}s ({rate:.2f}x wall) | view {fps:.1f} FPS"
        phase_label.text = phase + (" | previews ON" if active else " | single viewport") + (f" | source held {source_age:.1f}s" if source_age > .5 else "")
        if (live / "worker-done.json").exists():
            done_at = done_at or wall
            if wall - done_at >= 5.:
                break
        if args.max_frames and frames >= args.max_frames:
            break
        # Reserve GPU budget when all three preview cameras are being consumed.
        # A uniform 30 Hz is preferable to saturating the GPU with a 60 Hz request.
        display_fps = min(args.fps, 30.) if active else args.fps
        next_frame += 1/display_fps
        now = time.monotonic()
        if now-next_frame > 1/display_fps:
            next_frame = now
        time.sleep(max(0., next_frame-now))
except KeyboardInterrupt:
    pass
finally:
    warm = np.array(intervals[30:] if len(intervals) > 60 else intervals)
    report = {"frames": frames, "camera_updates": preview_count,
              "preview_resolution": [args.preview_width, round(args.preview_width*9/16)],
              "first_frame_simulation_time_s": first_frame_time,
              "median_fps": float(1 / np.median(warm)) if len(warm) else 0.,
              "average_fps": float(1/np.mean(warm)) if len(warm) else 0.,
              "p95_frame_interval_s": float(np.percentile(warm, 95)) if len(warm) else None,
              "modes": {name: {"frames": len(values), "average_fps": float(1/np.mean(values[30:] or values)),
                               "p95_frame_interval_s": float(np.percentile(values[30:] or values, 95))}
                        for name, values in mode_intervals.items() if values},
              "phase_p95_ms": {name: float(np.percentile(values[30:] or values, 95)*1000)
                               for name, values in phase_timings.items() if values}}
    write_json(live / "viewer-performance.json", report)
    probe.save("viewer-performance.json", report)
    write_json(live / "stop-request.json", {"time_s": float(display_time)})
    print("LIVE_VIEW_PERFORMANCE", json.dumps(report), flush=True)
    app.close()
