"""View a saved native PhysX trajectory; never counts as a new validation."""
import argparse
import json
from pathlib import Path
import signal
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--folder", type=Path, required=True)
parser.add_argument("--speed", type=float, default=1.)
parser.add_argument("--start-seconds", type=float)
parser.add_argument("--max-frames", type=int, default=0, help="Bounded viewer smoke test")
parser.add_argument("--snapshot", type=Path)
args = parser.parse_args()
if args.speed <= 0:
    parser.error("--speed must be positive")
folder = args.folder.resolve()
from isaacsim import SimulationApp
app = SimulationApp({"headless": False, "renderer": "RaytracedLighting", "anti_aliasing": 1})
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

trajectory = np.load(folder / "physx-trajectory.npz")
times, poses = trajectory["time_s"], trajectory["body_poses"]
if len(times) < 2 or np.any(np.diff(times) <= 0):
    raise ValueError("Replay requires a monotonic recorded trajectory")
result = json.loads((folder / "result.json").read_text())
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
window = ui.Window("Recorded PhysX trajectory", width=380, height=115)
with window.frame:
    with ui.VStack(spacing=5):
        ui.Label("REPLAY ONLY - no control or physics validation")
        ui.Label("Source: " + ("empty-area execution" if result["started_from_empty_area"] else "explicit development resume"))
        status = ui.Label("")
prims = [stage.GetPrimAtPath("/World/Bodies/" + name) for name in trajectory["body_names"]]
quaternions = Rotation.from_matrix(poses[:, :, :3, :3].reshape(-1, 3, 3)).as_quat().reshape(len(times), len(prims), 4)
start = float(times[0]) if args.start_seconds is None else max(float(times[0]), args.start_seconds)
start = min(start, float(times[-1]))
wall_start = time.monotonic()
frames = 0
capture = None
try:
    while app.is_running():
        wall = time.monotonic()
        t = min(start + (wall - wall_start) * args.speed, float(times[-1]))
        i = min(max(int(np.searchsorted(times, t, side="right") - 1), 0), len(times) - 2)
        alpha = float(np.clip((t - times[i]) / (times[i + 1] - times[i]), 0., 1.))
        position = poses[i, :, :3, 3] * (1 - alpha) + poses[i + 1, :, :3, 3] * alpha
        q0, q1 = quaternions[i], quaternions[i + 1].copy()
        q1[np.sum(q0 * q1, axis=1) < 0] *= -1
        qs = q0 * (1 - alpha) + q1 * alpha
        qs /= np.linalg.norm(qs, axis=1)[:, None]
        for prim, p, q in zip(prims, position, qs):
            prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*map(float, p)))
            prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
        status.text = f"t={t:.2f} s | {trajectory['phase'][i]} | {args.speed:g}x"
        app.update()
        frames += 1
        if args.snapshot and frames == 5:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            capture = capture_viewport_to_file(viewport, str(args.snapshot.resolve()))
        if args.max_frames and frames >= args.max_frames:
            break
        if t >= times[-1] and wall - wall_start > (times[-1] - start) / args.speed + 15:
            break
        time.sleep(max(0., 1/30 - (time.monotonic() - wall)))
    for _ in range(5):
        app.update()
except KeyboardInterrupt:
    pass
finally:
    print(f"REPLAY_ONLY frames={frames} source={folder}", flush=True)
    app.close()
