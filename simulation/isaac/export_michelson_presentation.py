"""Render a synchronized multi-camera replay movie of a verified experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--folder", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--speed", type=float, default=2.)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--start-seconds", type=float, default=0.)
parser.add_argument("--end-seconds", type=float)
parser.add_argument("--tail-hold", type=float, default=3.)
args = parser.parse_args()
if args.speed <= 0 or args.fps < 1 or args.tail_hold < 0:
    parser.error("speed/fps must be positive and tail-hold must be nonnegative")
args.folder, args.output = args.folder.resolve(), args.output.resolve()
if args.output.exists():
    parser.error("output already exists; choose a new movie path")
ffmpeg_candidates = list((ROOT / ".venv/lib/python3.11/site-packages/imageio_ffmpeg/binaries").glob("ffmpeg-*-v*"))
if not ffmpeg_candidates:
    parser.error("Install requirements-presentation.txt into the project .venv first")
args.output.parent.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "anti_aliasing": 2,
                     "width": 1280, "height": 720})
signal.signal(signal.SIGINT, signal.default_int_handler)
sys.path.append(str(ROOT / ".venv/lib/python3.11/site-packages"))

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import carb.settings
import omni.usd
import omni.timeline
import omni.replicator.core as rep
from isaacsim.core.utils.stage import open_stage, is_stage_loading
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Sdf, UsdGeom, UsdPhysics
from simulation.isaac.presentation_data import RecordedExperiment, PresentationComposer, camera_calibration

source = RecordedExperiment(args.folder)
start = max(0., args.start_seconds)
end = source.end if args.end_seconds is None else min(source.end, args.end_seconds)
if start >= end:
    raise ValueError("Replay interval must have positive duration")
calibration, calibration_error = camera_calibration(args.folder, source.names)
(args.output.parent / "camera-calibration.json").write_text(json.dumps(calibration, indent=2))
open_stage(str(args.folder / "scene.usda"))
while is_stage_loading():
    app.update()
stage = omni.usd.get_context().get_stage()
omni.timeline.get_timeline_interface().stop()
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
    if prim.IsA(UsdPhysics.Joint):
        UsdPhysics.Joint(prim).GetJointEnabledAttr().Set(False)
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)

overview = UsdGeom.Camera.Define(stage, "/World/PresentationOverview")
overview.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
overview.CreateFocalLengthAttr(18.)
set_camera_view(eye=np.array([1.4, -1.1, 1.7]), target=np.array([0., .28, .85]),
                camera_prim_path=str(overview.GetPath()))
calibration["overview"] = {"path": str(overview.GetPath()), "horizontal_aperture": 20.955, "focal_length": 18.}
products, annotators = {}, {}
for name, descriptor in calibration.items():
    camera = UsdGeom.Camera.Define(stage, descriptor["path"])
    if "local_transform" in descriptor:
        T = np.array(descriptor["local_transform"])
        transform = UsdGeom.Xformable(camera.GetPrim())
        transform.ClearXformOpOrder()
        transform.AddTranslateOp().Set(Gf.Vec3d(*map(float, T[:3, 3])))
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        transform.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
    camera.CreateClippingRangeAttr(Gf.Vec2f(.001, 100.))
    camera.CreateHorizontalApertureAttr(descriptor["horizontal_aperture"])
    camera.CreateVerticalApertureAttr(descriptor["horizontal_aperture"] * 9 / 16)
    camera.CreateFocalLengthAttr(descriptor["focal_length"])
    product = rep.create.render_product(descriptor["path"], (1280, 720) if name == "overview" else (640, 360))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([product])
    channels = {"rgb": rgb}
    if name == "wrist":
        depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        depth.attach([product])
        channels["depth"] = depth
    products[name], annotators[name] = product, channels
rep.settings.set_render_rtx_realtime(antialiasing="FXAA")
settings = carb.settings.get_settings()
for key in ("/rtx/reflections/enabled", "/rtx/indirectDiffuse/enabled", "/rtx/post/motionblur/enabled",
            "/rtx-transient/dlssg/enabled"):
    settings.set_bool(key, False)
rep.orchestrator.set_capture_on_play(False)
omni.timeline.get_timeline_interface().play()
prims = [stage.GetPrimAtPath("/World/Bodies/" + name) for name in source.names]
if any(not prim.IsValid() for prim in prims):
    raise ValueError("Replay body path absent from recorded scene")
ray_curves = []
for number, color in enumerate(((1., .03, .015), (1., .48, .02))):
    curves = UsdGeom.BasisCurves.Define(stage, f"/World/PresentationRays/arm_{number}")
    curves.CreateTypeAttr("linear")
    curves.CreateWrapAttr("nonperiodic")
    curves.CreateWidthsAttr([.0007])
    curves.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    curves.CreateVisibilityAttr("invisible")
    ray_curves.append(curves)
last_optical_label = None


def set_recorded_state(time_s):
    global last_optical_label
    def ease(value):
        value = float(np.clip(value, 0., 1.))
        return value * value * (3 - 2 * value)
    close = min(ease((time_s-156.)/18.), 1-ease((time_s-218.)/15.))
    eye = (1-close)*np.array([1.4, -1.1, 1.7]) + close*np.array([.70, -.35, 1.25])
    target = (1-close)*np.array([0., .28, .85]) + close*np.array([.04, .28, .89])
    set_camera_view(eye=eye, target=target, camera_prim_path=str(overview.GetPath()))
    p, qs, _, _ = source.sample(time_s)
    with Sdf.ChangeBlock():
        for prim, position, q in zip(prims, p, qs):
            prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*map(float, position)))
            prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
        measurement = source.measurement(time_s)
        if measurement is not None and measurement[1] != last_optical_label:
            for curve, path in zip(ray_curves, measurement[2]["paths"]):
                points = path["points_m"]
                curve.GetPointsAttr().Set([Gf.Vec3f(*point) for point in points])
                curve.CreateCurveVertexCountsAttr([len(points)])
                curve.GetVisibilityAttr().Set("inherited")
            last_optical_label = measurement[1]


def capture():
    rep.orchestrator.step(delta_time=0., pause_timeline=False, rt_subframes=1)
    images = {}
    for name, channels in annotators.items():
        pixels = np.asarray(channels["rgb"].get_data())
        if pixels.ndim != 3 or pixels.shape[2] < 3:
            raise RuntimeError("Replay camera frame missing: " + name)
        images[name] = cv2.cvtColor(pixels[:, :, :3], cv2.COLOR_RGB2BGR)
    depth = np.asarray(annotators["wrist"]["depth"].get_data()).squeeze()
    return images, depth if depth.ndim == 2 else None


set_recorded_state(start)
for _ in range(8):
    rep.orchestrator.step(delta_time=0., pause_timeline=False, rt_subframes=1)
count = int(np.ceil((end - start) / args.speed * args.fps)) + 1
tail_frames = round(args.tail_hold * args.fps)
temporary = args.output.with_name(args.output.stem + ".partial.mp4")
encoder_log = args.output.with_suffix(".encoder.log").open("w")
encoder = subprocess.Popen([str(ffmpeg_candidates[0]), "-y", "-loglevel", "warning", "-f", "rawvideo",
    "-pix_fmt", "bgr24", "-s", "1920x1080", "-r", str(args.fps), "-i", "-", "-an",
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p",
    "-threads", "3", "-movflags", "+frag_keyframe+empty_moov+default_base_moof", str(temporary)], stdin=subprocess.PIPE,
    stdout=encoder_log, stderr=subprocess.STDOUT)
composer = PresentationComposer(source, args.speed)
started = last_progress = time.monotonic()
samples_shown = set()
success = False

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

try:
    for frame in range(count):
        t = min(start + frame * args.speed / args.fps, end)
        set_recorded_state(t)
        images, depth = capture()
        canvas = composer.compose(images, depth, t)
        encoder.stdin.write(canvas.tobytes())
        measurement = source.measurement(t)
        if measurement:
            samples_shown.add(measurement[1])
        if frame == 0 or frame == count-1 or time.monotonic() - last_progress >= 5:
            elapsed = time.monotonic() - started
            print("PRESENTATION_PROGRESS", json.dumps({"frame": frame+1, "frames": count,
                  "source_time_s": round(t, 3), "render_fps": round((frame+1)/max(elapsed, 1e-6), 2)}), flush=True)
            last_progress = time.monotonic()
        if frame in (0, count-1) or (measurement and measurement[1] == "feedback-008"
                                     and not (args.output.parent / "adjustment-preview.png").exists()):
            name = "first-frame.png" if frame == 0 else "last-frame.png" if frame == count-1 else "adjustment-preview.png"
            cv2.imwrite(str(args.output.parent / name), canvas)
    for _ in range(tail_frames):
        encoder.stdin.write(canvas.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("Video encoder failed: see " + str(args.output.with_suffix(".encoder.log")))
    temporary.replace(args.output)
    success = True
finally:
    if encoder.poll() is None:
        encoder.terminate()
        encoder.wait()
    if not success:
        print("PRESENTATION_INCOMPLETE", str(temporary), flush=True)
    else:
        metadata = {"success": True, "scope": "synchronized replay presentation, not a new physics or optics validation",
            "source_folder": str(args.folder), "source_run_success": source.result["success"],
            "source_time_start_s": start, "source_time_end_s": end, "complete_experiment": start == 0. and end == source.end,
            "source_speed": args.speed, "fps": args.fps, "width": 1920, "height": 1080,
            "rendered_frames": count, "tail_hold_frames": tail_frames, "total_frames": count + tail_frames,
            "duration_s": (count + tail_frames) / args.fps, "render_wall_seconds": time.monotonic() - started,
            "camera_views": list(annotators), "camera_side_calibration_max_error": calibration_error,
            "optical_samples_shown": sorted(samples_shown), "optics_interpolated": False,
            "source_sha256": {name: digest(args.folder / name) for name in ("physx-trajectory.npz", "scene.usda", "result.json")},
            "movie_sha256": digest(args.output)}
        args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        print("PRESENTATION_COMPLETE", str(args.output), flush=True)
    app.close()
