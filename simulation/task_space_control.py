"""Online Cartesian velocity control for the unchanged five-object MuJoCo scene.

This controller uses the current tool pose and MuJoCo Jacobian at each physics
step. It does not call the waypoint IK solver or play back a joint trajectory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from . import line_assembly as scene
except ImportError:
    import line_assembly as scene


CONTROL_RATE = 0.8  # maximum commanded arm joint speed, rad/s
POSITION_GAIN = 4.0
ORIENTATION_GAIN = 3.0
DAMPING = 0.08
MAX_LINEAR_SPEED = 0.16  # m/s
MAX_ANGULAR_SPEED = 0.8  # rad/s
MAX_COMMAND_LEAD = 0.07  # rad between target and measured joint angle
HOLD_FEEDBACK_GAIN = 0.35  # keep the arm supported after the task completes
POST_RUN_HOLD_SECONDS = 12.0  # catch delayed sag/collisions in headless validation
PARK_MIN_TOOL_CLEARANCE = 0.20  # pinch center must stay this far above the table
POSITION_TOLERANCE = 0.005  # m
ORIENTATION_TOLERANCE = np.radians(3.0)
GIF_WIDTH = 480
GIF_HEIGHT = 360
GIF_SAMPLE_PERIOD = 0.4  # simulation seconds per source frame
GIF_OUTPUT_FPS = 12


class GifRecorder:
    """Stream MuJoCo offscreen frames to ffmpeg and encode an optimized GIF."""

    def __init__(self, model: mujoco.MjModel, path: Path):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("GIF recording requires ffmpeg (install with `brew install ffmpeg`)")
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.renderer = mujoco.Renderer(model, height=GIF_HEIGHT, width=GIF_WIDTH)
        except Exception as error:
            raise RuntimeError(
                "MuJoCo could not create its offscreen OpenGL context. Run GIF recording "
                "from an active macOS desktop session."
            ) from error
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self.camera)
        self.camera.lookat[:] = [0.0, 0.13, 0.83]
        self.camera.distance = 1.75
        self.camera.azimuth = 135
        self.camera.elevation = -28
        self.next_capture = 0.0
        filter_graph = (
            f"fps={GIF_OUTPUT_FPS},split[s0][s1];"
            "[s0]palettegen=max_colors=192[p];"
            "[s1][p]paletteuse=dither=bayer:bayer_scale=3"
        )
        self.process = subprocess.Popen(
            [
                ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
                "-pixel_format", "rgb24", "-video_size", f"{GIF_WIDTH}x{GIF_HEIGHT}",
                "-framerate", str(1 / GIF_SAMPLE_PERIOD), "-i", "pipe:0",
                "-filter_complex", filter_graph, "-loop", "0", str(self.path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def capture(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if data.time + 1e-9 < self.next_capture:
            return
        self.renderer.update_scene(data, camera=self.camera)
        frame = self.renderer.render()
        assert self.process.stdin is not None
        self.process.stdin.write(frame.tobytes())
        self.next_capture += GIF_SAMPLE_PERIOD

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        error = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
        code = self.process.wait()
        self.renderer.close()
        if code:
            raise RuntimeError(f"ffmpeg failed to create {self.path}: {error.strip()}")
        print(f"GIF saved: {self.path}")


@dataclass(frozen=True)
class Phase:
    label: str
    position: np.ndarray
    orientation: Rotation
    duration: float
    grip: float
    wait_for_pose: bool = True


def clamp_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    return vector if length <= maximum else vector * (maximum / length)


def progress(elapsed: float, duration: float) -> tuple[float, float]:
    u = np.clip(elapsed / duration, 0.0, 1.0)
    position = u**3 * (10.0 + u * (-15.0 + 6.0 * u))
    rate = 30.0 * u**2 * (1.0 - u)**2 / duration
    return float(position), float(rate)


def phases_for_scene(
    model: mujoco.MjModel, data: mujoco.MjData,
    target_xs: tuple[float, ...] = scene.TARGET_XS,
    object_poses: dict[str, np.ndarray] | None = None,
) -> list[Phase]:
    site = model.site("pinch_center").id
    home_position = data.site_xpos[site].copy()
    home_orientation = Rotation.from_matrix(data.site_xmat[site].reshape(3, 3))
    place_orientation = Rotation.from_euler("z", scene.GRIP_YAW) * home_orientation
    phases: list[Phase] = []

    for (name, _), target_x in zip(scene.COMPONENTS, target_xs):
        if object_poses is None:
            body = model.body(name).id
            position = data.xpos[body].copy()
            rotation = data.xmat[body].reshape(3, 3)
        else:
            if name not in object_poses:
                continue
            pose = np.asarray(object_poses[name])
            position, rotation = pose[:3, 3], pose[:3, :3]
        yaw = Rotation.from_matrix(rotation).as_euler("xyz")[2]
        pickup_yaw = yaw + scene.GRIP_YAW
        pickup_orientation = Rotation.from_euler("z", pickup_yaw) * home_orientation
        grasp_height = position[2] + scene.GRASP_Z_OFFSET
        pickup = np.array([position[0], position[1], grasp_height + scene.PICKUP_CLEARANCE])
        grasp = np.array([position[0], position[1], grasp_height])
        lifted = np.array([position[0], position[1], grasp_height + scene.CARRY_CLEARANCE])
        carried = np.array([
            target_x, scene.TARGET_Y,
            scene.BLOCK_Z + scene.GRASP_Z_OFFSET + scene.CARRY_CLEARANCE,
        ])
        placed = np.array([
            target_x, scene.TARGET_Y,
            scene.BLOCK_Z + scene.GRASP_Z_OFFSET + scene.PLACE_SETTLE,
        ])
        retreated = np.array([target_x, scene.TARGET_Y, scene.BLOCK_Z + 0.25])
        phases.extend((
            Phase(f"{name}: pregrasp", pickup, pickup_orientation, 3.0, 0.0),
            Phase(f"{name}: approach", grasp, pickup_orientation, 2.0, 0.0),
            Phase(f"{name}: close", grasp, pickup_orientation, 1.6, 255.0, False),
            Phase(f"{name}: lift", lifted, pickup_orientation, 2.0, 255.0),
            Phase(f"{name}: transfer", carried, place_orientation, 4.5, 255.0),
            Phase(f"{name}: lower", placed, place_orientation, 2.5, 255.0),
            Phase(f"{name}: release", placed, place_orientation, 1.0, 0.0, False),
            Phase(f"{name}: retreat", retreated, place_orientation, 2.5, 0.0),
        ))
    phases.append(Phase("park above table", home_position, home_orientation, 3.0, 0.0))
    return phases


class CartesianController:
    def __init__(
        self, model: mujoco.MjModel, data: mujoco.MjData,
        target_xs: tuple[float, ...] = scene.TARGET_XS,
        object_poses: dict[str, np.ndarray] | None = None,
    ):
        self.model = model
        self.site = model.site("pinch_center").id
        self.qpos_ids, self.actuator_ids = scene.arm_indices(model)
        self.dof_ids = np.array([
            model.jnt_dofadr[model.joint(f"joint{i}").id] for i in range(1, 8)
        ])
        self.gripper_actuator = model.actuator("gripper").id
        self.phases = phases_for_scene(model, data, target_xs, object_poses)
        self.phase_index = 0
        self.phase_start_time = float(data.time)
        self.start_position = data.site_xpos[self.site].copy()
        self.start_orientation = Rotation.from_matrix(data.site_xmat[self.site].reshape(3, 3))
        self.start_grip = 0.0
        self.command_q = data.qpos[self.qpos_ids].copy()
        self.hold_q = self.command_q.copy()
        self.home_q = self.command_q.copy()
        self.last_time = float(data.time)
        self.failed: str | None = None
        self.done = False
        self.max_position_error = 0.0
        self.jac_position = np.zeros((3, model.nv))
        self.jac_rotation = np.zeros((3, model.nv))
        self.joint_limits = np.array([
            model.jnt_range[model.joint(f"joint{i}").id] for i in range(1, 8)
        ])
        for index in (0, 2, 4, 6):
            self.joint_limits[index] = [-np.pi, np.pi]
        self.object_geoms = {
            model.geom(f"{name}_collision").id: name for name, _ in scene.COMPONENTS
        }
        self.table_geom = model.geom("workbench").id
        self.pad_geoms = {
            model.geom(f"{side}_finger_pad_{number}").id: side
            for side in ("left", "right") for number in (1, 2)
        }
        self.gripper_geoms: set[int] = set()
        self.arm_geoms: set[int] = set()
        arm_base = model.body("link_base").id
        gripper_base = model.body("xarm_gripper_base_link").id
        for geom in range(model.ngeom):
            body = int(model.geom_bodyid[geom])
            ancestor = body
            while ancestor > 0 and ancestor != arm_base:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == arm_base and body != arm_base:
                self.arm_geoms.add(geom)
            ancestor = body
            while ancestor > 0 and ancestor != gripper_base:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == gripper_base:
                self.gripper_geoms.add(geom)
        self.pad_contacts = {name: set() for name, _ in scene.COMPONENTS}
        self.lifted = {name: False for name, _ in scene.COMPONENTS}
        self.transfer_min_height = {name: float("inf") for name, _ in scene.COMPONENTS}
        self.transfer_max_grip_gap = {name: 0.0 for name, _ in scene.COMPONENTS}
        self.unsafe_contacts: list[tuple[float, str, str]] = []
        self.park_min_tool_clearance = float("inf")
        print(f"{data.time:6.1f}s  {self.phases[0].label}")

    @property
    def phase(self) -> Phase:
        return self.phases[self.phase_index]

    def advance(self, state: mujoco.MjData) -> None:
        if self.phase_index == len(self.phases) - 1:
            self.hold_q = self.command_q.copy()
            self.done = True
            print(f"{state.time:6.1f}s  parked; holding arm above the table")
            return
        previous = self.phase
        self.phase_index += 1
        self.phase_start_time = float(state.time)
        self.start_position = state.site_xpos[self.site].copy()
        self.start_orientation = Rotation.from_matrix(state.site_xmat[self.site].reshape(3, 3))
        self.start_grip = previous.grip
        print(f"{state.time:6.1f}s  {self.phase.label}")

    def observe(self, state: mujoco.MjData) -> None:
        phase = self.phase
        if self.done:
            self.park_min_tool_clearance = min(
                self.park_min_tool_clearance,
                float(state.site_xpos[self.site, 2] - scene.TABLE_Z),
            )
        active_name = phase.label.split(":", 1)[0]
        transferring = phase.label.endswith(": transfer")
        if transferring:
            cube = state.xpos[self.model.body(active_name).id]
            self.transfer_min_height[active_name] = min(
                self.transfer_min_height[active_name],
                float(cube[2] - scene.BLOCK_HALF_HEIGHT - scene.TABLE_Z),
            )
            self.transfer_max_grip_gap[active_name] = max(
                self.transfer_max_grip_gap[active_name],
                float(np.linalg.norm(cube - state.site_xpos[self.site])),
            )
        for name, _ in scene.COMPONENTS:
            if state.xpos[self.model.body(name).id, 2] > scene.TABLE_Z + 0.085:
                self.lifted[name] = True
        for contact in state.contact[: state.ncon]:
            a, b = contact.geom1, contact.geom2
            for object_geom, pad_geom in ((a, b), (b, a)):
                if object_geom in self.object_geoms and pad_geom in self.pad_geoms:
                    self.pad_contacts[self.object_geoms[object_geom]].add(
                        self.pad_geoms[pad_geom]
                    )
            if len(self.unsafe_contacts) >= 10:
                continue
            other = b if a == self.table_geom else a if b == self.table_geom else None
            if other in self.arm_geoms:
                self.unsafe_contacts.append((state.time, phase.label, "arm/table"))
            if transferring and self.object_geoms.get(other) == active_name:
                self.unsafe_contacts.append((state.time, phase.label, "carried cube/table"))
            for arm_geom, object_geom in ((a, b), (b, a)):
                other_name = self.object_geoms.get(object_geom)
                if (arm_geom in self.arm_geoms and other_name is not None
                        and other_name != active_name):
                    self.unsafe_contacts.append((
                        state.time, phase.label, f"arm/{other_name}"
                    ))
            if transferring and a in self.object_geoms and b in self.object_geoms:
                names = {self.object_geoms[a], self.object_geoms[b]}
                if active_name in names and len(names) > 1:
                    self.unsafe_contacts.append((
                        state.time, phase.label, "carried cube/other cube"
                    ))

    def __call__(self, model: mujoco.MjModel, state: mujoco.MjData) -> None:
        self.observe(state)
        measured_q = state.qpos[self.qpos_ids]
        dt = max(float(state.time - self.last_time), model.opt.timestep)
        self.last_time = float(state.time)
        if self.failed or self.done:
            state.ctrl[self.actuator_ids] = (
                self.hold_q + HOLD_FEEDBACK_GAIN * (self.hold_q - measured_q)
            )
            state.ctrl[self.gripper_actuator] = self.phase.grip
            return

        phase = self.phase
        elapsed = float(state.time - self.phase_start_time)
        s, ds = progress(elapsed, phase.duration)
        desired_position = self.start_position + s * (phase.position - self.start_position)
        desired_velocity = ds * (phase.position - self.start_position)
        rotation_delta = (phase.orientation * self.start_orientation.inv()).as_rotvec()
        desired_orientation = Rotation.from_rotvec(s * rotation_delta) * self.start_orientation
        desired_angular_velocity = ds * rotation_delta

        actual_position = state.site_xpos[self.site]
        actual_orientation = Rotation.from_matrix(state.site_xmat[self.site].reshape(3, 3))
        position_error = desired_position - actual_position
        rotation_error = (desired_orientation * actual_orientation.inv()).as_rotvec()
        linear_velocity = clamp_norm(
            desired_velocity + POSITION_GAIN * position_error, MAX_LINEAR_SPEED
        )
        angular_velocity = clamp_norm(
            desired_angular_velocity + ORIENTATION_GAIN * rotation_error,
            MAX_ANGULAR_SPEED,
        )

        mujoco.mj_jacSite(model, state, self.jac_position, self.jac_rotation, self.site)
        jacobian = np.vstack((
            self.jac_position[:, self.dof_ids], self.jac_rotation[:, self.dof_ids]
        ))
        inverse = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + DAMPING**2 * np.eye(6), np.eye(6)
        )
        joint_velocity = inverse @ np.r_[linear_velocity, angular_velocity]
        nullspace = np.eye(7) - inverse @ jacobian
        joint_velocity += nullspace @ (0.08 * (self.home_q - measured_q))
        maximum = float(np.max(np.abs(joint_velocity)))
        if maximum > CONTROL_RATE:
            joint_velocity *= CONTROL_RATE / maximum
        self.command_q += joint_velocity * dt
        self.command_q = np.clip(
            self.command_q,
            np.maximum(self.joint_limits[:, 0] + 0.005, measured_q - MAX_COMMAND_LEAD),
            np.minimum(self.joint_limits[:, 1] - 0.005, measured_q + MAX_COMMAND_LEAD),
        )
        state.ctrl[self.actuator_ids] = self.command_q
        grip_s, _ = progress(elapsed, phase.duration)
        state.ctrl[self.gripper_actuator] = self.start_grip + grip_s * (
            phase.grip - self.start_grip
        )

        if elapsed < phase.duration:
            return
        final_position_error = float(np.linalg.norm(phase.position - actual_position))
        final_orientation_error = float(np.linalg.norm(
            (phase.orientation * actual_orientation.inv()).as_rotvec()
        ))
        self.max_position_error = max(self.max_position_error, final_position_error)
        if (not phase.wait_for_pose or (
            final_position_error < POSITION_TOLERANCE
            and final_orientation_error < ORIENTATION_TOLERANCE
        )):
            self.advance(state)
        elif elapsed > phase.duration + 8.0:
            self.hold_q = self.command_q.copy()
            self.failed = (
                f"{phase.label}: pose error {final_position_error:.3f} m, "
                f"{np.degrees(final_orientation_error):.1f} deg"
            )


def run(
    headless: bool, randomize: bool = True, seed: int | None = None,
    yaw_range_deg: float = 30, record_gif: Path | None = None,
) -> None:
    if randomize and seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    model = scene.make_scene(
        randomize=randomize, seed=seed, yaw_range_deg=yaw_range_deg,
    )
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    print(f"Cartesian Jacobian controller; seed={seed if randomize else 'fixed'}")
    controller = CartesianController(model, data)
    recorder = GifRecorder(model, record_gif) if record_gif is not None else None
    mujoco.set_mjcb_control(controller)
    try:
        if headless or recorder is not None:
            while not controller.done and controller.failed is None and data.time < 300:
                mujoco.mj_step(model, data)
                if recorder is not None:
                    recorder.capture(model, data)
            if controller.done:
                stop_time = data.time + POST_RUN_HOLD_SECONDS
                while data.time < stop_time:
                    mujoco.mj_step(model, data)
        else:
            from mujoco import viewer
            viewer.launch(model, data)
    finally:
        mujoco.set_mjcb_control(None)
        if recorder is not None:
            recorder.close()
    if controller.failed:
        raise RuntimeError(controller.failed)
    if not controller.done:
        raise RuntimeError(f"Controller stopped before completion at {data.time:.1f}s")
    print("Final XY errors:")
    passed = 0
    for (name, _), x in zip(scene.COMPONENTS, scene.TARGET_XS):
        body = model.body(name).id
        actual = data.xpos[body]
        error = float(np.linalg.norm(actual[:2] - [x, scene.TARGET_Y]))
        rotation = data.xmat[body].reshape(3, 3)
        yaw = Rotation.from_matrix(rotation).as_euler("xyz")[2]
        yaw_error = abs(float(np.degrees(np.arctan2(np.sin(yaw), np.cos(yaw)))))
        tilt_error = float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))
        ok = (
            controller.lifted[name]
            and controller.pad_contacts[name] == {"left", "right"}
            and controller.transfer_min_height[name] > 0.05
            and controller.transfer_max_grip_gap[name] < 0.05
            and error < 0.010 and abs(actual[2] - scene.BLOCK_Z) < 0.020
            and yaw_error < 5.0 and tilt_error < 5.0
        )
        passed += ok
        print(
            f"  {name:14s} XY={error:.3f} m, yaw={yaw_error:.1f} deg, "
            f"tilt={tilt_error:.1f} deg, both pads="
            f"{controller.pad_contacts[name] == {'left', 'right'}}, "
            f"carry clearance={controller.transfer_min_height[name]:.3f} m, "
            f"grip gap={controller.transfer_max_grip_gap[name]:.3f} m, placed={ok}"
        )
    print(f"{passed}/5 placed; simulation time {data.time:.1f}s")
    print(f"Unsafe contacts: {controller.unsafe_contacts[:5]}")
    if headless or recorder is not None:
        print(f"Parked tool clearance: at least "
              f"{controller.park_min_tool_clearance:.3f} m above table during hold")
    if (passed != len(scene.COMPONENTS) or controller.unsafe_contacts
            or ((headless or recorder is not None)
                and controller.park_min_tool_clearance < PARK_MIN_TOOL_CLEARANCE)):
        raise RuntimeError("Cartesian controller failed physical validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--randomize", dest="randomize", action="store_true", default=True)
    mode.add_argument("--fixed", dest="randomize", action="store_false")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--yaw-range-deg", type=float, default=30)
    parser.add_argument(
        "--record-gif", type=Path, metavar="PATH",
        help="record a time-lapse GIF (runs headlessly; requires ffmpeg and an OpenGL display)",
    )
    args = parser.parse_args()
    if not 0 <= args.yaw_range_deg <= 45:
        parser.error("yaw range must be 0–45 degrees")
    run(args.headless, args.randomize, args.seed, args.yaw_range_deg, args.record_gif)
