"""Independent 30 Hz display of a running native PhysX experiment.

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
parser.add_argument("--camera-fps", type=float, default=5.)
parser.add_argument("--render-probe-dir", type=Path, help="Save fixed-scene render diagnostics")
args = parser.parse_args()
if args.camera_fps <= 0:
    parser.error("--camera-fps must be positive")
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
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
import carb.settings
from pxr import Gf, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation
from simulation.isaac.live_stream import DisplayBuffer, write_json

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
products, annotators = {}, {}
for name, descriptor in cameras.items():
    camera = UsdGeom.Camera.Define(stage, descriptor["prim_path"])
    if "local_transform" in descriptor:
        T = np.array(descriptor["local_transform"])
        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, T[:3, 3])))
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        xform.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
    camera.CreateClippingRangeAttr(Gf.Vec2f(.001, 10.))
    camera.CreateHorizontalApertureAttr(descriptor["horizontal_aperture"])
    camera.CreateVerticalApertureAttr(descriptor["horizontal_aperture"] * 360 / 640)
    camera.CreateFocalLengthAttr(descriptor["focal_length"])
    product = rep.create.render_product(str(camera.GetPath()), (640, 360))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([product])
    channels = {"rgb": rgb}
    if name == "wrist":
        depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        depth.attach([product])
        channels["depth"] = depth
    products[name], annotators[name] = product, channels
rep.orchestrator.set_capture_on_play(True)
# Stage/Replicator initialization can replace SimulationApp's AA selection
# with DLSS Performance. The supported helper also disables limited AA modes.
rep.settings.set_render_rtx_realtime(antialiasing="FXAA")
render_settings = carb.settings.get_settings()
for setting in ("/rtx/reflections/enabled", "/rtx/indirectDiffuse/enabled",
                "/rtx/post/motionblur/enabled", "/rtx-transient/dlssg/enabled"):
    render_settings.set_bool(setting, False)
omni.timeline.get_timeline_interface().play()

window = ui.Window("Live PhysX experiment", width=465, height=135)
with window.frame:
    with ui.VStack(spacing=5):
        ui.Label("LIVE native state | 100 ms display interpolation")
        status = ui.Label("Starting preview cameras before robot motion...")
        phase_label = ui.Label("")
        ui.Label("Camera previews: camera_optics_monitor.py --folder <run>")

buffer = DisplayBuffer()
last_state_stamp = None
prims = None
frames, preview_count = 0, 0
intervals = []
last_frame_wall = None
last_preview_wall = 0.
first_frame_time = None
done_at = None
display_time = 0.
captures = []


def read_state():
    global last_state_stamp, prims
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


try:
    while app.is_running():
        wall = time.monotonic()
        if last_frame_wall is not None:
            intervals.append(wall - last_frame_wall)
        last_frame_wall = wall
        read_state()
        p, qs, display_time, phase, rate = buffer.at(wall)
        for prim, position, q in zip(prims, p, qs):
            prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*map(float, position)))
            prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
        due = first_frame_time is None or wall - last_preview_wall >= 1 / args.camera_fps
        for product in products.values():
            product.hydra_texture.set_updates_enabled(due)
        app.update()
        frames += 1
        if frames == 20:
            keys = ("/rtx/post/aa/op", "/rtx/post/dlss/execMode", "/rtx/post/motionblur/enabled",
                    "/persistent/rtx/modes/rt2/enabled", "/rtx-transient/dlssg/enabled",
                    "/rtx/reflections/enabled", "/rtx/indirectDiffuse/enabled")
            observation = {"settings": {key: render_settings.get(key) for key in keys},
                           "viewport_resolution": list(viewport.resolution)}
            write_json(live / "render-settings.json", observation)
            if observation["settings"]["/rtx/post/aa/op"] != 2:
                raise RuntimeError("LIVE_RENDER_AA_WAS_OVERRIDDEN")
            if args.render_probe_dir:
                args.render_probe_dir.mkdir(parents=True, exist_ok=True)
                write_json(args.render_probe_dir / "render-settings.json", observation)
        if args.render_probe_dir and frames in (45, 46, 47, 90):
            captures.append(capture_viewport_to_file(viewport, str(args.render_probe_dir / f"viewport-{frames:03d}.png")))
        source_age = max(0., wall - buffer.samples[-1][0])
        if due and frames >= 8 and publish_preview(display_time, source_age):
            last_preview_wall = wall
            if not (live / "ready.json").exists():
                write_json(live / "ready.json", {"first_frame_simulation_time_s": first_frame_time})
                print("LIVE_CAMERAS_READY", first_frame_time, flush=True)
        fps = 1 / max(np.median(intervals[-60:]), 1e-6) if intervals else 0.
        status.text = f"Physics t={display_time:.2f}s ({rate:.2f}x wall) | view {fps:.1f} FPS"
        phase_label.text = phase + (f" | source held {source_age:.1f}s" if source_age > .5 else "")
        if (live / "worker-done.json").exists():
            done_at = done_at or wall
            if wall - done_at >= 5.:
                break
        if args.max_frames and frames >= args.max_frames:
            break
        time.sleep(max(0., 1 / 30 - (time.monotonic() - wall)))
except KeyboardInterrupt:
    pass
finally:
    warm = np.array(intervals[30:] if len(intervals) > 60 else intervals)
    report = {"frames": frames, "camera_updates": preview_count,
              "first_frame_simulation_time_s": first_frame_time,
              "median_fps": float(1 / np.median(warm)) if len(warm) else 0.,
              "p95_frame_interval_s": float(np.percentile(warm, 95)) if len(warm) else None}
    write_json(live / "viewer-performance.json", report)
    write_json(live / "stop-request.json", {"time_s": float(display_time)})
    print("LIVE_VIEW_PERFORMANCE", json.dumps(report), flush=True)
    app.close()
