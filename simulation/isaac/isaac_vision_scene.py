"""Single STL housing and 720p camera; run with Isaac Sim 5.1 python.sh."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import signal
import struct
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=240)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--capture-dir", type=Path)
parser.add_argument("--with-arm", action="store_true")
parser.add_argument("--view", action="store_true",
                    help="Open the Isaac Sim window; use --frames 0 for continuous run")
args = parser.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": not args.view, "renderer": "RaytracedLighting"})

import cv2
import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import Articulation
from pxr import Gf, UsdGeom, UsdLux

enable_extension("isaacsim.ros2.bridge")
app.update()
import rclpy
from geometry_msgs.msg import TransformStamped
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster

stl = Path(__file__).resolve().parents[2] / "assets/optical_housing/highousing.stl"
assert hashlib.sha256(stl.read_bytes()).hexdigest() == (
    "815981fe57a1e078a52855fc3ce756ff0e036d3fc6d6e695a603d51cf7c2ef96")
world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 30)
stage = omni.usd.get_context().get_stage()
world.scene.add_default_ground_plane()
if args.with_arm:
    arm_asset = Path(__file__).resolve().parents[2] / "assets/xarm7_isaac/xarm7.usd"
    add_reference_to_stage(usd_path=str(arm_asset), prim_path="/World/XArm7")
    UsdGeom.XformCommonAPI(stage.GetPrimAtPath("/World/XArm7")).SetTranslate(
        Gf.Vec3d(0, -.35, 0))
    app.update()


def cube(path, xyz, scale, color):
    prim = UsdGeom.Cube.Define(stage, path)
    prim.GetSizeAttr().Set(1.0)
    xform = UsdGeom.XformCommonAPI(prim)
    xform.SetTranslate(Gf.Vec3d(*xyz))
    xform.SetScale(Gf.Vec3f(*scale))
    prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])


cube("/World/Table", (0, 0, -.015), (.8, .8, .03), (.22, .25, .29))
housing = UsdGeom.Xform.Define(stage, "/World/Housing")
housing_xform = UsdGeom.XformCommonAPI(housing)
housing_y = .25 if args.with_arm else 0.
housing_xform.SetTranslate(Gf.Vec3d(-.08, housing_y, .114))

# Preserve the selected STL's shape, including its optical opening. Source is mm.
raw = stl.read_bytes()
count = struct.unpack_from("<I", raw, 80)[0]
assert len(raw) == 84 + count * 50
points = []
for i in range(count):
    triangle = struct.unpack_from("<12fH", raw, 84 + i * 50)
    for j in range(3):
        points.append(Gf.Vec3f(triangle[3 + 3*j] * .001,
                               triangle[4 + 3*j] * .001,
                               triangle[5 + 3*j] * .001 - .050))
mesh = UsdGeom.Mesh.Define(stage, "/World/Housing/SourceSTL")
mesh.CreatePointsAttr(points)
mesh.CreateFaceVertexCountsAttr([3] * count)
mesh.CreateFaceVertexIndicesAttr(list(range(3 * count)))
mesh.CreateDoubleSidedAttr(True)
mesh.CreateDisplayColorAttr([Gf.Vec3f(.23, .36, .46)])
for name, radius, height, z, color in (
    ("Base", .027, .012, -.108, (.28, .30, .32)),
    ("Post", .0063, .048, -.078, (.72, .74, .77)),
):
    prim = UsdGeom.Cylinder.Define(stage, f"/World/Housing/{name}")
    prim.GetRadiusAttr().Set(radius)
    prim.GetHeightAttr().Set(height)
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(0, 0, z))
    prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])

# Match global_camera.py: DICT_4X4_50 ID 0, black outer edge 24 mm.
pattern = cv2.aruco.generateImageMarker(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 6)
cube("/World/Housing/MarkerWhite", (0, 0, .05240),
     (.032, .032, .0001), (1, 1, 1))
for row in range(6):
    for col in range(6):
        if pattern[row, col] == 0:
            cube(f"/World/Housing/Marker_{row}_{col}",
                 ((col-2.5)*.004, (2.5-row)*.004, .05252),
                 (.004, .004, .0001), (0, 0, 0))

# Side marker ID 10 on the flat -Y panel, matching stl_preview.py.
side = UsdGeom.Xform.Define(stage, "/World/Housing/SideMarker")
side_xform = UsdGeom.XformCommonAPI(side)
side_xform.SetTranslate(Gf.Vec3d(0, -.03235, .0381))
side_xform.SetRotate(Gf.Vec3f(90, 0, 0))
side_pattern = cv2.aruco.generateImageMarker(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 10, 6)
cube("/World/Housing/SideMarker/White", (0, 0, .00005),
     (.016, .016, .0001), (1, 1, 1))
for row in range(6):
    for col in range(6):
        if side_pattern[row, col] == 0:
            cube(f"/World/Housing/SideMarker/Cell_{row}_{col}",
                 ((col-2.5)*.002, (2.5-row)*.002, .00017),
                 (.002, .002, .0001), (0, 0, 0))

light = UsdLux.DistantLight.Define(stage, "/World/Light")
light.CreateIntensityAttr(1500.)
UsdGeom.XformCommonAPI(light).SetRotate(Gf.Vec3f(30, 30, 0))

# A +90-degree world-Y camera aims down; its optical X/Y map to world -Y/-X.
world.reset()
if args.with_arm:
    arm = Articulation("/World/XArm7/link_base/link_base")
    arm.initialize()
    arm_names = [f"joint{i}" for i in range(1, 8)]
    # Turn the Menagerie home pose away from the inspection area for a clear view.
    home = np.array([[np.pi, -.247, 0., .909, 0., 1.15644, 0.]])
    arm.set_joint_positions(home, joint_names=arm_names)
    arm.set_gains(kps=np.full((1, 7), 1000.), kds=np.full((1, 7), 50.),
                  joint_names=arm_names)
    arm.set_joint_position_targets(home, joint_names=arm_names)
camera_z = 1.20 if args.with_arm else .60
camera_y = .20 if args.with_arm else 0.
camera = Camera(prim_path="/World/Camera", position=np.array([0., camera_y, camera_z]),
                orientation=np.array([np.sqrt(.5), 0., np.sqrt(.5), 0.]),
                resolution=(args.width, args.height), frequency=10)
camera.initialize()
camera.set_clipping_range(.01, 100.)
print("camera_pose_usd", camera.get_world_pose(camera_axes="usd"), flush=True)
print("stage_up_axis", UsdGeom.GetStageUpAxis(stage), flush=True)
camera.set_focal_length(3.0 if args.with_arm else 1.8)
camera.set_horizontal_aperture(2.0955)
camera.set_vertical_aperture(2.0955 * args.height / args.width)
K = np.asarray(camera.get_intrinsics_matrix(), dtype=float)
if args.view:
    from isaacsim.core.utils.viewports import set_camera_view
    set_camera_view(eye=np.array([1.1, -1.3, 1.1]),
                    target=np.array([0., 0., .1]))

rclpy.init()
node = rclpy.create_node("isaac_single_housing_camera")
image_pub = node.create_publisher(Image, "/optics/camera/image_raw", 10)
info_pub = node.create_publisher(CameraInfo, "/optics/camera/camera_info", 10)
clock_pub = node.create_publisher(Clock, "/clock", 10)
tf_pub = StaticTransformBroadcaster(node)
tf = TransformStamped()
tf.header.frame_id = "world"
tf.child_frame_id = "optics_camera_optical_frame"
tf.transform.translation.z = camera_z
tf.transform.translation.y = camera_y
tf.transform.rotation.x = np.sqrt(.5)
tf.transform.rotation.y = -np.sqrt(.5)
tf_pub.sendTransform(tf)

world.play()
stop_requested = False


def request_stop(_signum, _frame):
    global stop_requested
    stop_requested = True


signal.signal(signal.SIGINT, request_stop)
move_frame = args.frames // 2 if args.frames > 0 else 120
frames = range(args.frames) if args.frames > 0 else itertools.count()
for frame in frames:
    if stop_requested:
        break
    if frame == move_frame:
        housing_xform.SetTranslate(Gf.Vec3d(.08, housing_y + .03, .114))
        housing_xform.SetRotate(Gf.Vec3f(0, 0, 25))
    world.step(render=True)
    t = float(world.current_time)
    clock = Clock()
    clock.clock.sec = int(t)
    clock.clock.nanosec = int((t - int(t)) * 1e9)
    clock_pub.publish(clock)
    if frame % 3:
        continue
    rgba = camera.get_rgba()
    if rgba is None or getattr(rgba, "ndim", 0) != 3 or rgba.shape[0] == 0:
        continue
    rgb = np.ascontiguousarray(np.asarray(rgba[:, :, :3], dtype=np.uint8))
    if args.capture_dir and frame in (6, move_frame + 6):
        args.capture_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.capture_dir / f"frame_{frame}.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    image = Image()
    image.header.frame_id = "optics_camera_optical_frame"
    image.header.stamp = clock.clock
    image.height, image.width = rgb.shape[:2]
    image.encoding = "rgb8"
    image.step = image.width * 3
    image.data = rgb.tobytes()
    image_pub.publish(image)
    info = CameraInfo()
    info.header = image.header
    info.height, info.width = image.height, image.width
    info.distortion_model = "plumb_bob"
    info.d = [0.] * 5
    info.k = K.reshape(-1).tolist()
    info.r = np.eye(3).reshape(-1).tolist()
    info.p = [K[0, 0], 0., K[0, 2], 0., 0., K[1, 1], K[1, 2], 0., 0., 0., 1., 0.]
    info_pub.publish(info)
    if frame % 30 == 0:
        print(f"published frame={frame} t={t:.2f} {image.width}x{image.height}", flush=True)

world.stop()
node.destroy_node()
rclpy.shutdown()
app.close()
