"""View a saved native PhysX trajectory; never counts as a new validation."""
import argparse
import json
import math
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--folder", type=Path, required=True)
parser.add_argument("--speed", type=float, default=4., help="Initial playback speed (display only; default: 4x)")
parser.add_argument("--start-seconds", type=float)
parser.add_argument("--max-frames", type=int, default=0, help="Bounded viewer smoke test")
parser.add_argument("--snapshot", type=Path)
parser.add_argument("--fps", type=float, default=60., help="Display frame rate target")
parser.add_argument("--freeze", action="store_true", help="Hold the recorded start pose for render diagnostics")
parser.add_argument("--render-probe-dir", type=Path, help="Capture 30 frozen viewport frames and timing")
parser.add_argument("--camera", choices=("overview", "global", "wrist", "fat", "laser", "detector"), default="overview")
parser.add_argument("--exit-after", type=float, default=0., help="Exit this many seconds after the end; 0 keeps the viewer open")
args = parser.parse_args()
if not all(math.isfinite(x) and x > 0 for x in (args.speed, args.fps)):
    parser.error("--speed and --fps must be positive")
if not math.isfinite(args.exit_after) or args.exit_after < 0:
    parser.error("--exit-after must be finite and nonnegative")
folder = args.folder.resolve()
from isaacsim import SimulationApp
app = SimulationApp({"headless": False, "renderer": "RaytracedLighting", "anti_aliasing": 2,
                     "width": 1440, "height": 810})
signal.signal(signal.SIGINT, signal.default_int_handler)
import numpy as np
from scipy.spatial.transform import Rotation
import omni.usd
import omni.timeline
import omni.ui as ui
from isaacsim.core.utils.stage import open_stage, is_stage_loading
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from pxr import Gf, UsdGeom, UsdPhysics
from simulation.isaac.display_rendering import configure, settings_snapshot, PoseWriter, ViewportProbe, add_cameras
from simulation.isaac.replay_clock import ReplayClock

trajectory = np.load(folder / "physx-trajectory.npz")
times, poses = trajectory["time_s"], trajectory["body_poses"]
if len(times) < 2 or np.any(np.diff(times) <= 0):
    raise ValueError("Replay requires a monotonic recorded trajectory")
result = json.loads((folder / "result.json").read_text())
phases = trajectory["phase"].tolist()
final_state = folder/"optics/released-1024/native-state.npz"
if result["time_s"] > times[-1] + 1e-9 and final_state.is_file():
    with np.load(final_state) as final:
        if final["body_names"].tolist() != trajectory["body_names"].tolist() or abs(float(final["time_s"])-result["time_s"]) > 1e-9:
            raise ValueError("Final detector state does not match the recorded replay")
        times = np.r_[times, float(final["time_s"])]
        poses = np.concatenate([poses, final["body_poses"][None]], axis=0)
        phases.append("verified after tool return")
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
camera = UsdGeom.Camera.Define(stage, "/World/ReplayCamera")
camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
camera.CreateFocalLengthAttr(18.)
viewport = get_active_viewport()
viewport.set_active_camera(str(camera.GetPath()))
set_camera_view(eye=np.array([1.4, -1.1, 1.7]), target=np.array([0., .28, .85]),
                camera_prim_path=str(camera.GetPath()))
from simulation.optical_device_visuals import upgrade_isaac_display
device_cameras = upgrade_isaac_display(stage, json.loads((folder/"configuration.json").read_text()))
configure(viewport, stage=stage)
probe = ViewportProbe(args.render_probe_dir, viewport)
camera_paths = {"overview": str(camera.GetPath()), "global": "/World/GlobalCamera", **device_cameras}
if (folder/"fat-side-000.json").is_file():
    sys.path.append(str(ROOT/".venv/lib/python3.11/site-packages"))
    from simulation.isaac.presentation_data import camera_calibration
    descriptors, _ = camera_calibration(folder, trajectory["body_names"].tolist())
    calibrated = add_cameras(stage, descriptors)
    camera_paths.update(wrist=calibrated["wrist"], fat=calibrated["side"])
if args.camera not in camera_paths:
    raise ValueError("This run has no recorded calibration for camera " + args.camera)
viewport.set_active_camera(camera_paths[args.camera])
start = float(times[0]) if args.start_seconds is None else max(float(times[0]), args.start_seconds)
start = min(start, float(times[-1]))
clock = ReplayClock(start, float(times[-1]), args.speed, paused=args.freeze)
window = ui.Window("Recorded PhysX trajectory", width=590, height=245)
with window.frame:
    with ui.VStack(spacing=5):
        ui.Label("REPLAY ONLY - schematic device visuals; recorded physics")
        ui.Label("Source: " + ("empty-area execution" if result["started_from_empty_area"] else "explicit development resume"))
        status = ui.Label("")
        with ui.HStack(height=26):
            for label, path in camera_paths.items():
                ui.Button(label.title(), clicked_fn=lambda path=path: viewport.set_active_camera(path))
        with ui.HStack(height=26):
            pause = ui.Button("Resume" if clock.paused else "Pause")
            def toggle_pause():
                clock.set_paused(not clock.paused)
            pause.set_clicked_fn(toggle_pause)
            ui.Button("Restart", clicked_fn=clock.restart)
            ui.Button("-10 s", clicked_fn=lambda: clock.seek(clock.advance()-10.))
            ui.Button("+10 s", clicked_fn=lambda: clock.seek(clock.advance()+10.))
        with ui.HStack(height=26):
            ui.Label("Playback speed", width=115)
            for speed in (1., 2., 4., 8.):
                ui.Button(f"{speed:g}x", clicked_fn=lambda speed=speed: clock.set_speed(speed))
        ui.Label("End: holds the final scene. Restart to play again.", height=22)
prims = [stage.GetPrimAtPath("/World/Bodies/" + name) for name in trajectory["body_names"]]
pose_writer = PoseWriter(prims)
quaternions = Rotation.from_matrix(poses[:, :, :3, :3].reshape(-1, 3, 3)).as_quat().reshape(len(times), len(prims), 4)
frames = 0
capture = None
intervals = []
last_frame_wall = None
visited_phases = set()


def show_time(t):
    i = min(max(int(np.searchsorted(times, t, side="right") - 1), 0), len(times)-2)
    alpha = float(np.clip((t-times[i])/(times[i+1]-times[i]), 0., 1.))
    position = poses[i, :, :3, 3]*(1-alpha) + poses[i+1, :, :3, 3]*alpha
    q0, q1 = quaternions[i], quaternions[i+1].copy()
    q1[np.sum(q0*q1, axis=1) < 0] *= -1
    qs = q0*(1-alpha) + q1*alpha
    qs /= np.linalg.norm(qs, axis=1)[:, None]
    pose_writer.update(position, qs)
    return phases[-1] if t >= times[-1] else phases[i]


# Warm the real viewport at the starting pose before starting the replay clock.
show_time(start)
for _ in range(12):
    app.update()
wall_start = next_frame = time.monotonic()
clock.seek(start, wall_start)
ended_wall = None
t = start
try:
    while app.is_running():
        wall = time.monotonic()
        if last_frame_wall is not None:
            intervals.append(wall-last_frame_wall)
        last_frame_wall = wall
        t = clock.advance(wall)
        phase = show_time(t)
        visited_phases.add(phase)
        fps = 1/max(np.median(intervals[-60:]), 1e-6) if intervals else 0.
        status.text = f"t={t:.2f} / {times[-1]:.2f} s | {phase} | {clock.speed:g}x | {fps:.1f} FPS"
        pause.text = "Resume" if clock.paused else "Pause"
        app.update()
        frames += 1
        if frames == 20:
            probe.save("render-settings.json", settings_snapshot(viewport))
        probe.update(frames)
        if args.snapshot and frames == 5:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            capture = capture_viewport_to_file(viewport, str(args.snapshot.resolve()))
        if args.max_frames and frames >= args.max_frames:
            break
        ended_wall = (wall if ended_wall is None else ended_wall) if t >= times[-1] and not clock.paused else None
        if args.exit_after and ended_wall is not None and wall-ended_wall >= args.exit_after:
            break
        next_frame += 1/args.fps
        now = time.monotonic()
        if now-next_frame > 1/args.fps:
            next_frame = now
        time.sleep(max(0., next_frame-now))
    for _ in range(5):
        app.update()
except KeyboardInterrupt:
    pass
finally:
    warm = np.array(intervals[30:] if len(intervals) > 60 else intervals)
    report = {"frames": frames, "frozen": args.freeze, "source_time_s": float(t), "playback_speed": clock.speed,
        "playback_wall_s": time.monotonic()-wall_start,
        "source_start_s": start, "source_end_s": float(times[-1]), "reached_end": bool(t >= times[-1]),
        "visited_phases": sorted(visited_phases),
        "median_fps": float(1/np.median(warm)) if len(warm) else 0.,
        "average_fps": float(1/np.mean(warm)) if len(warm) else 0.,
        "p95_frame_interval_s": float(np.percentile(warm, 95)) if len(warm) else None}
    probe.save("viewer-performance.json", report)
    print("REPLAY_PERFORMANCE", json.dumps(report), flush=True)
    print(f"REPLAY_ONLY frames={frames} source={folder}", flush=True)
    app.close()
