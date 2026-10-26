"""Scene setup and historical joint-trajectory controller.

The cubes and top labels stand in for tagged component housings. Object poses
come from the simulator; no camera or ArUco detection is claimed here. Direct
execution starts the newer Cartesian feedback controller by default.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco_menagerie as menagerie
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation


TABLE_Z = 0.76
BLOCK_HALF_WIDTH = 0.025
BLOCK_HALF_HEIGHT = 0.040
BLOCK_Z = TABLE_Z + BLOCK_HALF_HEIGHT
PICKUP_Y = 0.55
TARGET_Y = 0.35
SOURCE_SLOT_XS = (-0.32, -0.16, 0.00, 0.16, 0.32)
TARGET_XS = (-0.18, -0.09, 0.00, 0.09, 0.18)
SPAWN_X_RANGE = (-0.35, 0.35)
SPAWN_Y_RANGE = (0.08, 0.56)
MIN_SOURCE_SPACING = 0.14
MIN_TARGET_SPACING = 0.14
MAX_COMMAND_SPEED = 0.8  # rad/s, nominal simulation command speed
HOLE_PITCH = 0.025  # optical breadboard hole spacing, metres
GRASP_Z_OFFSET = 0.010  # pinch near the housing centre, clear of the table
PICKUP_CLEARANCE = 0.14
CARRY_CLEARANCE = 0.16
PLACE_SETTLE = 0.003  # release just above the surface; avoid a tall housing tipping
GRIP_YAW = np.pi / 2  # open along Y, perpendicular to the target row
PLANNER_CLEARANCE = 0.002  # minimum gripper clearance from the table and other housings
PLANNER_JOINT_STEP = 0.15  # max joint increment between collision samples, radians
IK_BRANCH_CHANGE_PENALTY = 0.5  # preserve continuity unless another branch is clearly better
TRACKING_COMPENSATION = 0.35  # increase position-servo correction on bounded paths
PLANNER_NAMES = ("bounded", "simple", "current")
# Each entry is (component name, RGBA color). Coordinates are set below.
COMPONENTS = (
    ("lens_1", (0.23, 0.58, 0.90, 1.0)),
    ("pinhole", (0.95, 0.68, 0.20, 1.0)),
    ("lens_2", (0.25, 0.80, 0.45, 1.0)),
    ("beam_splitter", (0.72, 0.40, 0.83, 1.0)),
    ("mirror", (0.90, 0.38, 0.42, 1.0)),
)


@dataclass(frozen=True)
class Segment:
    name: str
    duration: float
    q0: np.ndarray
    q1: np.ndarray
    grip0: float
    grip1: float
    q_path: np.ndarray | None = None


def add_optical_table_surface(spec: mujoco.MjSpec) -> None:
    """Draw a dark, regularly perforated top without adding collision holes."""
    width = 64
    yy, xx = np.mgrid[:width, :width]
    radius = np.hypot(xx - (width - 1) / 2, yy - (width - 1) / 2)
    pixels = np.empty((width, width, 3), dtype=np.uint8)
    pixels[:] = (215, 215, 215)  # metal tone comes from the material
    pixels[(radius >= 7) & (radius < 10)] = (255, 255, 255)  # machined rim
    pixels[radius < 7] = (38, 38, 38)  # recessed threaded hole
    texture = "optical_table_holes"
    material = "optical_table_top"
    spec.add_texture(
        name=texture, type=mujoco.mjtTexture.mjTEXTURE_2D,
        width=width, height=width, nchannel=3, data=pixels.ravel(),
    )
    spec.add_material(
        name=material, textures=["", texture],  # slot 1 is the RGB texture role
        texrepeat=[1.16 / HOLE_PITCH, 1.04 / HOLE_PITCH],
        rgba=[0.38, 0.42, 0.46, 1], specular=0.75, shininess=0.7,
    )
    spec.worldbody.add_geom(
        name="optical_table_pattern", type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0, 0.13, TABLE_Z + 0.0002], size=[0.58, 0.52, 0.001],
        material=material, rgba=[0.38, 0.42, 0.46, 1], contype=0, conaffinity=0,
    )


def motion_duration(q_path: np.ndarray, requested: float) -> float:
    steps = np.max(np.abs(np.diff(q_path, axis=0)), axis=1)
    knots = np.r_[0.0, np.cumsum(np.maximum(steps, 1e-6))]
    curve = PchipInterpolator(knots / knots[-1], q_path, axis=0)
    u = np.linspace(0.0, 1.0, 101)
    eased = u**3 * (10.0 + u * (-15.0 + 6.0 * u))
    ease_rate = 30.0 * u**2 * (1.0 - u)**2
    peak = np.max(np.abs(curve.derivative()(eased) * ease_rate[:, None]))
    return max(requested, float(peak / MAX_COMMAND_SPEED))


def random_start_poses(seed: int | None, yaw_range_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample scattered, separated tabletop poses in a nominal arm work area."""
    rng = np.random.default_rng(seed)
    targets = np.column_stack((TARGET_XS, np.full(len(COMPONENTS), TARGET_Y)))
    positions: list[np.ndarray] = []
    for _ in COMPONENTS:
        for _attempt in range(10000):
            point = np.array([
                rng.uniform(*SPAWN_X_RANGE), rng.uniform(*SPAWN_Y_RANGE),
            ])
            reach = np.linalg.norm(point)
            if not 0.28 <= reach <= 0.68:
                continue
            if np.min(np.linalg.norm(targets - point, axis=1)) < MIN_TARGET_SPACING:
                continue
            if any(np.linalg.norm(point - other) < MIN_SOURCE_SPACING for other in positions):
                continue
            positions.append(point)
            break
        else:
            raise RuntimeError("Could not fit five separated cubes in the spawn area")
    yaws = rng.uniform(-np.radians(yaw_range_deg), np.radians(yaw_range_deg), len(COMPONENTS))
    return np.array(positions), yaws


def make_scene(
    randomize: bool = True, seed: int | None = None,
    yaw_range_deg: float = 30,
) -> mujoco.MjModel:
    source = menagerie.get("ufactory_xarm7").path() / "scene.xml"
    spec = mujoco.MjSpec.from_file(str(source))
    source_xy = np.column_stack((SOURCE_SLOT_XS, np.full(len(COMPONENTS), PICKUP_Y)))
    source_yaw = np.zeros(len(COMPONENTS))
    if randomize:
        source_xy, source_yaw = random_start_poses(seed, yaw_range_deg)
    # Mount the arm on the table centreline; the target row shares x=0.
    spec.body("link_base").pos = [0, 0, TABLE_Z]
    spec.body("link_base").quat = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    for geom in list(spec.worldbody.geoms):
        if not geom.name and geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            spec.delete(geom)  # The Menagerie floor stand is below our workbench.
    spec.body("xarm_gripper_base_link").add_site(
        name="pinch_center", pos=[0, 0, 0.14], size=[0.006], rgba=[1, 0.8, 0, 1]
    )
    for side in ("left", "right"):
        for number in (1, 2):
            spec.geom(f"{side}_finger_pad_{number}").friction = [2.0, 0.01, 0.0001]
    world = spec.worldbody
    world.add_geom(
        name="workbench", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0.13, TABLE_Z - 0.045], size=[0.58, 0.52, 0.045],
        rgba=[0.24, 0.27, 0.30, 1], friction=[1.0, 0.005, 0.0001],
    )
    add_optical_table_surface(spec)
    for x in (-0.52, 0.52):
        for y in (-0.33, 0.59):
            world.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x, y, (TABLE_Z - 0.09) / 2],
                size=[0.025, 0.025, (TABLE_Z - 0.09) / 2],
                rgba=[0.33, 0.35, 0.38, 1],
            )

    for index, ((name, color), x) in enumerate(zip(COMPONENTS, TARGET_XS)):
        world.add_geom(
            name=f"target_{name}", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x, TARGET_Y, TABLE_Z + 0.001], size=[0.03, 0.03, 0.001],
            rgba=[*color[:3], 0.6], contype=0, conaffinity=0,
        )
        px, py = source_xy[index]
        yaw = source_yaw[index]
        cube = world.add_body(
            name=name, pos=[px, py, BLOCK_Z],
            quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
        )
        cube.add_freejoint(name=f"{name}_free")
        cube.add_geom(
            name=f"{name}_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[BLOCK_HALF_WIDTH, BLOCK_HALF_WIDTH, BLOCK_HALF_HEIGHT],
            mass=0.045, rgba=color,
            friction=[1.5, 0.01, 0.0001],
        )
        # A visible fiducial *placeholder*. It is not a dictionary-valid ArUco marker.
        cube.add_geom(
            name=f"{name}_tag", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0, 0, BLOCK_HALF_HEIGHT + 0.0005],
            size=[0.016, 0.016, 0.0004],
            mass=0, contype=0, conaffinity=0, rgba=[0.96, 0.96, 0.96, 1],
        )
        for bit in range(4):
            if (index + bit) % 3 == 0:
                continue
            cube.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[(-1 if bit % 2 == 0 else 1) * 0.006,
                     (-1 if bit < 2 else 1) * 0.006, BLOCK_HALF_HEIGHT + 0.001],
                size=[0.003, 0.003, 0.0005], mass=0,
                contype=0, conaffinity=0, rgba=[0.02, 0.02, 0.02, 1],
            )
    return spec.compile()


def arm_indices(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    joints = [model.joint(f"joint{i}").id for i in range(1, 8)]
    qpos = np.array([model.jnt_qposadr[j] for j in joints])
    actuators = np.array([model.actuator(f"act{i}").id for i in range(1, 8)])
    return qpos, actuators


def solve_ik(
    model: mujoco.MjModel,
    state: mujoco.MjData,
    qpos_ids: np.ndarray,
    seed: np.ndarray,
    point: np.ndarray,
    orientation: Rotation,
    wrist_preference: float | None = None,
    winding_limit: float | None = None,
) -> np.ndarray:
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = state.qpos
    site = model.site("pinch_center").id
    limits = np.array([model.jnt_range[model.joint(f"joint{i}").id] for i in range(1, 8)])
    if winding_limit is not None:
        for index in (0, 2, 4):
            limits[index] = [-winding_limit, winding_limit]
    # A full revolution of joint 7 produces the same tool pose, but a long and
    # visibly awkward move. This task never needs more than half a revolution.
    limits[6] = np.array([-np.pi, np.pi])
    wrist_target = seed[6] if wrist_preference is None else wrist_preference

    def residual(q: np.ndarray) -> np.ndarray:
        scratch.qpos[qpos_ids] = q
        mujoco.mj_forward(model, scratch)
        actual = Rotation.from_matrix(scratch.site_xmat[site].reshape(3, 3))
        angle_error = (orientation * actual.inv()).as_rotvec()
        return np.r_[
            scratch.site_xpos[site] - point,
            0.12 * angle_error,
            0.004 * (q - seed),
            0.020 * (q[6] - wrist_target),
        ]

    result = least_squares(
        residual, np.clip(seed, limits[:, 0] + 1e-5, limits[:, 1] - 1e-5),
        bounds=(limits[:, 0] + 1e-6, limits[:, 1] - 1e-6),
        max_nfev=180, ftol=1e-7, xtol=1e-7,
    )
    error = residual(result.x)
    if np.linalg.norm(error[:3]) > 0.006 or np.linalg.norm(error[3:6]) > 0.008:
        raise RuntimeError(
            f"IK failed at {point}: position error {np.linalg.norm(error[:3]):.3f} m, "
            f"orientation error {np.degrees(np.linalg.norm(error[3:6]) / 0.12):.1f} deg"
        )
    return result.x


def _schedule_once(
    model: mujoco.MjModel, data: mujoco.MjData,
    planner_stats: dict[str, int | float] | None = None,
    planner: str = "bounded",
    winding_limit: float | None = None,
) -> list[Segment]:
    if planner not in PLANNER_NAMES:
        raise ValueError(f"Unknown planner {planner!r}")
    qpos_ids, _ = arm_indices(model)
    home = data.qpos[qpos_ids].copy()
    site = model.site("pinch_center").id
    orientation = Rotation.from_matrix(data.site_xmat[site].reshape(3, 3))
    place_orientation = Rotation.from_euler("z", GRIP_YAW) * orientation
    park_point = data.site_xpos[site].copy()
    q = home
    grip = 0.0
    segments: list[Segment] = []
    planning_state = mujoco.MjData(model)
    planning_state.qpos[:] = data.qpos
    mujoco.mj_forward(model, planning_state)
    collision_state = mujoco.MjData(model)
    table_geom = model.geom("workbench").id
    object_geoms = {model.geom(f"{name}_collision").id: name for name, _ in COMPONENTS}
    arm_geoms: set[int] = set()
    gripper_geoms: set[int] = set()
    arm_base = model.body("link_base").id
    gripper_base = model.body("xarm_gripper_base_link").id
    for geom in range(model.ngeom):
        if model.geom_contype[geom] == 0 and model.geom_conaffinity[geom] == 0:
            continue
        body = int(model.geom_bodyid[geom])
        descendant = body
        while descendant > 0 and descendant != arm_base:
            descendant = int(model.body_parentid[descendant])
        if descendant == arm_base and body != arm_base:
            arm_geoms.add(geom)
        descendant = body
        while descendant > 0 and descendant != gripper_base:
            descendant = int(model.body_parentid[descendant])
        if descendant == gripper_base:
            gripper_geoms.add(geom)
    previous_delta = np.zeros(len(qpos_ids))
    candidate_count = 0
    safe_count = 0
    alternative_count = 0

    def transition_clearance(start_q: np.ndarray, end_q: np.ndarray,
                             active_name: str) -> tuple[float | None, str | None]:
        obstacles = {table_geom, *(geom for geom, name in object_geoms.items()
                                   if name != active_name)}
        steps = max(2, int(np.ceil(np.max(np.abs(end_q - start_q)) /
                                   PLANNER_JOINT_STEP)))
        smallest = 0.04
        for fraction in np.linspace(0.0, 1.0, steps + 1):
            collision_state.qpos[:] = planning_state.qpos
            collision_state.qpos[qpos_ids] = start_q + fraction * (end_q - start_q)
            mujoco.mj_forward(model, collision_state)
            for contact in collision_state.contact[: collision_state.ncon]:
                if ((contact.geom1 in arm_geoms and contact.geom2 in obstacles)
                        or (contact.geom2 in arm_geoms and contact.geom1 in obstacles)):
                    return None, (
                        f"contact {model.geom(contact.geom1).name or contact.geom1}/"
                        f"{model.geom(contact.geom2).name or contact.geom2}"
                    )
            for moving in gripper_geoms:
                for obstacle in obstacles:
                    gap = mujoco.mj_geomDistance(
                        model, collision_state, moving, obstacle, 0.04, None)
                    if gap < PLANNER_CLEARANCE:
                        return None, (
                            f"gap {model.geom(moving).name or moving}/"
                            f"{model.geom(obstacle).name or obstacle}={gap:.4f} m"
                        )
                    smallest = min(smallest, gap)
        return smallest, None

    def choose_ik(point: np.ndarray, tool_orientation: Rotation,
                  wrist_preference: float | None, active_name: str) -> np.ndarray:
        nonlocal q, previous_delta, candidate_count, safe_count, alternative_count
        if planner == "simple":
            # One warm-started IK solve per waypoint, with the same collision gate.
            candidate_count += 1
            candidate = solve_ik(
                model, planning_state, qpos_ids, q, point,
                tool_orientation, wrist_preference,
            )
            clearance, reason = transition_clearance(q, candidate, active_name)
            if clearance is None:
                raise RuntimeError(
                    f"No collision-free IK transition for {active_name} at {point}: {reason}"
                )
            safe_count += 1
            previous_delta = candidate - q
            return candidate
        starts = [q.copy()]
        for elbow_offset in (0.7, -0.7):
            alternative = q.copy()
            alternative[2] += elbow_offset
            starts.append(alternative)
        fallback_starts = [home.copy()]
        for elbow_offset in (1.4, -1.4):
            alternative = q.copy()
            alternative[2] += elbow_offset
            fallback_starts.append(alternative)
        for shoulder_offset in (0.6, -0.6):
            alternative = q.copy()
            alternative[1] += shoulder_offset
            alternative[3] -= shoulder_offset
            fallback_starts.append(alternative)
        solutions: list[np.ndarray] = []
        best: tuple[float, np.ndarray, int] | None = None
        rejections: list[str] = []
        solve_failures = 0
        for index, start in enumerate(starts + fallback_starts):
            if index >= len(starts) and best is not None:
                break
            candidate_count += 1
            try:
                candidate = solve_ik(
                    model, planning_state, qpos_ids, start, point,
                    tool_orientation, wrist_preference, winding_limit,
                )
            except RuntimeError:
                solve_failures += 1
                continue
            if any(np.linalg.norm(candidate - other) < 0.03 for other in solutions):
                continue
            solutions.append(candidate)
            clearance, reason = transition_clearance(q, candidate, active_name)
            if clearance is None:
                if reason is not None:
                    rejections.append(reason)
                continue
            safe_count += 1
            delta = candidate - q
            score = (
                np.max(np.abs(delta)) / MAX_COMMAND_SPEED
                + 0.15 * np.linalg.norm(delta)
                + 0.06 * np.linalg.norm(delta - previous_delta)
                + 5.0 * max(0.0, 0.025 - clearance)
                + (IK_BRANCH_CHANGE_PENALTY if index != 0 else 0.0)
            )
            if best is None or score < best[0]:
                best = (float(score), candidate, index)
        if best is None:
            raise RuntimeError(
                f"No collision-free IK transition for {active_name} at {point}: "
                f"{solve_failures} IK failures; {'; '.join(rejections)}"
            )
        previous_delta = best[1] - q
        if best[2] != 0:
            alternative_count += 1
        return best[1]

    def move(
        label: str, x: float, y: float, z: float, seconds: float,
        tool_orientation: Rotation = orientation,
        wrist_preference: float | None = None,
    ) -> None:
        nonlocal q
        active_name = label.split(":", 1)[0]
        next_q = choose_ik(
            np.array([x, y, z]), tool_orientation, wrist_preference, active_name,
        )
        path = np.vstack((q, next_q))
        segments.append(Segment(label, motion_duration(path, seconds), q, next_q, grip, grip))
        q = next_q

    def gripper(label: str, command: float, seconds: float) -> None:
        nonlocal grip
        segments.append(Segment(label, seconds, q, q, grip, command))
        grip = command

    def line(
        label: str, start: tuple[float, float, float], end: tuple[float, float, float],
        seconds: float, tool_orientation: Rotation, steps: int = 8,
        wrist_preference: float | None = None,
        yaw_start: float | None = None, yaw_end: float | None = None,
    ) -> None:
        nonlocal q
        path = [q]
        for fraction in np.linspace(1 / steps, 1, steps):
            point = np.asarray(start) + fraction * (np.asarray(end) - start)
            if yaw_start is not None and yaw_end is not None:
                current_yaw = yaw_start + fraction * (yaw_end - yaw_start)
                waypoint_orientation = Rotation.from_euler("z", current_yaw) * orientation
                waypoint_wrist = -current_yaw
            else:
                waypoint_orientation = tool_orientation
                waypoint_wrist = wrist_preference
            q = path[-1]
            path.append(choose_ik(
                point, waypoint_orientation, waypoint_wrist,
                label.split(":", 1)[0],
            ))
        path_array = np.array(path)
        segments.append(Segment(
            label, motion_duration(path_array, seconds), path[0], path[-1],
            grip, grip, path_array,
        ))
        q = path[-1]

    previous_target_x: float | None = None
    for (name, _), x in zip(COMPONENTS, TARGET_XS):
        body = model.body(name).id
        px, py, pz = data.xpos[body]
        grasp_z = pz + GRASP_Z_OFFSET
        pickup_z = grasp_z + PICKUP_CLEARANCE
        carry_z = grasp_z + CARRY_CLEARANCE
        target_carry_z = BLOCK_Z + GRASP_Z_OFFSET + CARRY_CLEARANCE
        target_place_z = BLOCK_Z + GRASP_Z_OFFSET + PLACE_SETTLE
        yaw = Rotation.from_matrix(data.xmat[body].reshape(3, 3)).as_euler("xyz")[2]
        pickup_tool_yaw = yaw + GRIP_YAW
        pickup_orientation = Rotation.from_euler("z", pickup_tool_yaw) * orientation
        if previous_target_x is None:
            move(f"{name}: pregrasp", px, py, pickup_z, 2.0,
                 pickup_orientation, -pickup_tool_yaw)
        else:
            line(
                f"{name}: transit to pickup",
                (previous_target_x, TARGET_Y, BLOCK_Z + 0.25),
                (px, py, pickup_z), 3.0, orientation,
                yaw_start=GRIP_YAW, yaw_end=pickup_tool_yaw,
            )
        line(f"{name}: approach", (px, py, pickup_z), (px, py, grasp_z), 1.5,
             pickup_orientation, wrist_preference=-pickup_tool_yaw)
        gripper(f"{name}: close", 255, 1.6)
        line(f"{name}: lift", (px, py, grasp_z), (px, py, carry_z), 1.7,
             pickup_orientation, wrist_preference=-pickup_tool_yaw)
        line(
            f"{name}: transfer", (px, py, carry_z), (x, TARGET_Y, target_carry_z),
            4.5, pickup_orientation, yaw_start=pickup_tool_yaw, yaw_end=GRIP_YAW,
        )
        line(
            f"{name}: lower", (x, TARGET_Y, target_carry_z),
            (x, TARGET_Y, target_place_z), 2.0,
            place_orientation, wrist_preference=-GRIP_YAW,
        )
        gripper(f"{name}: release", 0, 1.0)
        line(
            f"{name}: retreat", (x, TARGET_Y, target_place_z),
            (x, TARGET_Y, BLOCK_Z + 0.25), 2.0,
            place_orientation, wrist_preference=-GRIP_YAW,
        )
        previous_target_x = x
        # Subsequent picks must plan around housings already placed on the line.
        freejoint = model.joint(f"{name}_free").id
        address = int(model.jnt_qposadr[freejoint])
        planning_state.qpos[address:address + 7] = [
            x, TARGET_Y, BLOCK_Z, 1, 0, 0, 0,
        ]
        mujoco.mj_forward(model, planning_state)
    line(
        "park above table", (previous_target_x, TARGET_Y, BLOCK_Z + 0.25),
        tuple(park_point), 3.0, orientation,
        yaw_start=GRIP_YAW, yaw_end=0.0,
    )
    if planner_stats is not None:
        planner_stats.update(
            candidates=candidate_count, safe_candidates=safe_count,
            alternate_choices=alternative_count,
        )
    return segments


def schedule(
    model: mujoco.MjModel, data: mujoco.MjData,
    planner_stats: dict[str, int | float] | None = None,
    planner: str = "bounded",
) -> list[Segment]:
    """Retry a blocked bounded route with the smallest useful winding range."""
    if planner not in PLANNER_NAMES:
        raise ValueError(f"Unknown planner {planner!r}")
    if planner != "bounded":
        segments = _schedule_once(model, data, planner_stats, planner)
        if planner_stats is not None:
            planner_stats.update(planning_attempts=1, winding_limit_deg=360.0)
        return segments
    limits = (np.pi, 4.0, 4.7, 2 * np.pi)
    last_error: RuntimeError | None = None
    for attempt, limit in enumerate(limits, start=1):
        try:
            segments = _schedule_once(model, data, planner_stats, planner, limit)
        except RuntimeError as error:
            if not str(error).startswith("No collision-free IK transition"):
                raise
            last_error = error
            continue
        if planner_stats is not None:
            planner_stats.update(
                planning_attempts=attempt,
                winding_limit_deg=float(np.degrees(limit)) if limit is not None else 360.0,
            )
        return segments
    raise RuntimeError(
        f"No collision-free route after {len(limits)} winding limits: {last_error}"
    ) from last_error


def smoothstep(value: float) -> float:
    value = np.clip(value, 0.0, 1.0)
    return float(value**3 * (10.0 + value * (-15.0 + 6.0 * value)))


def run(
    headless: bool, randomize: bool = True, seed: int | None = None,
    yaw_range_deg: float = 30,
    planner: str = "bounded",
) -> None:
    if randomize and seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    model = make_scene(randomize, seed, yaw_range_deg)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    if randomize:
        print(f"Randomized initial positions and yaw (seed={seed}):")
        for name, _ in COMPONENTS:
            body = model.body(name).id
            yaw = Rotation.from_matrix(data.xmat[body].reshape(3, 3)).as_euler("xyz")[2]
            print(f"  {name:14s} xyz={np.round(data.xpos[body], 3)}  yaw={np.degrees(yaw):+.1f} deg")
    qpos_ids, actuator_ids = arm_indices(model)
    gripper_id = model.actuator("gripper").id
    planner_stats: dict[str, int | float] = {}
    try:
        segments = schedule(model, data, planner_stats, planner)
    except RuntimeError as error:
        if planner == "simple":
            raise RuntimeError(
                f"Simple planner stopped at an unreachable or unsafe waypoint: {error}. "
                "Try --planner bounded for multi-start IK and route retries."
            ) from error
        raise
    print(f"Planner profile: {planner}")
    if planner_stats["planning_attempts"] > 1:
        print("Winding limit relaxed to {winding_limit_deg:.0f} deg after "
              "{planning_attempts} planning attempts".format(**planner_stats))
    print("Planner: {candidates} IK attempts, {safe_candidates} safe candidates, "
          "{alternate_choices} alternate choices".format(**planner_stats))
    trajectories = []
    for segment in segments:
        if segment.q_path is None:
            trajectories.append(None)
            continue
        steps = np.max(np.abs(np.diff(segment.q_path, axis=0)), axis=1)
        knots = np.r_[0.0, np.cumsum(np.maximum(steps, 1e-6))]
        trajectories.append(PchipInterpolator(knots / knots[-1], segment.q_path, axis=0))
    ends = np.cumsum([segment.duration for segment in segments])
    total = float(ends[-1])
    last_segment = [-1]
    lifted = {name: False for name, _ in COMPONENTS}
    pad_contacts = {name: set() for name, _ in COMPONENTS}
    transfer_min_height = {name: float("inf") for name, _ in COMPONENTS}
    transfer_max_grip_gap = {name: 0.0 for name, _ in COMPONENTS}
    table_gripper_contacts: list[tuple[float, str]] = []
    transfer_table_contacts: list[tuple[float, str]] = []
    neighbor_gripper_contacts: list[tuple[float, str, str]] = []
    wrist_range = [float("inf"), float("-inf")]
    object_geoms = {model.geom(f"{name}_collision").id: name for name, _ in COMPONENTS}
    pad_geoms = {
        model.geom(f"{side}_finger_pad_{number}").id: side
        for side in ("left", "right") for number in (1, 2)
    }
    table_geom = model.geom("workbench").id
    gripper_body = model.body("xarm_gripper_base_link").id
    gripper_geoms: set[int] = set()
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        while body > 0 and body != gripper_body:
            body = int(model.body_parentid[body])
        if body == gripper_body:
            gripper_geoms.add(geom)

    def control(_model: mujoco.MjModel, state: mujoco.MjData) -> None:
        index = min(int(np.searchsorted(ends, state.time, side="right")), len(segments) - 1)
        segment = segments[index]
        start = 0.0 if index == 0 else ends[index - 1]
        fraction = smoothstep((state.time - start) / segment.duration)
        target_q = (
            trajectories[index](fraction) if trajectories[index] is not None
            else segment.q0 + fraction * (segment.q1 - segment.q0)
        )
        if planner in {"bounded", "simple"}:
            target_q = target_q + TRACKING_COMPENSATION * (target_q - state.qpos[qpos_ids])
        state.ctrl[actuator_ids] = target_q
        state.ctrl[gripper_id] = segment.grip0 + fraction * (segment.grip1 - segment.grip0)
        wrist_angle = float(state.qpos[qpos_ids[6]])
        wrist_range[0] = min(wrist_range[0], wrist_angle)
        wrist_range[1] = max(wrist_range[1], wrist_angle)
        if index != last_segment[0]:
            if last_segment[0] < 0 or segment.name != segments[last_segment[0]].name:
                print(f"{state.time:6.1f}s  {segment.name}")
            last_segment[0] = index
        active_name = segment.name.split(":", 1)[0]
        transferring = segment.name.endswith(": transfer")
        if transferring:
            cube = state.xpos[model.body(active_name).id]
            transfer_min_height[active_name] = min(
                transfer_min_height[active_name],
                float(cube[2] - BLOCK_HALF_HEIGHT - TABLE_Z),
            )
            transfer_max_grip_gap[active_name] = max(
                transfer_max_grip_gap[active_name],
                float(np.linalg.norm(cube - state.site_xpos[model.site("pinch_center").id])),
            )
        for name, _ in COMPONENTS:
            if state.xpos[model.body(name).id, 2] > TABLE_Z + 0.085:
                lifted[name] = True
        for contact in state.contact[: state.ncon]:
            if table_geom in (contact.geom1, contact.geom2):
                other = contact.geom2 if contact.geom1 == table_geom else contact.geom1
                if other in gripper_geoms and len(table_gripper_contacts) < 5:
                    table_gripper_contacts.append((state.time, segment.name))
                if (transferring and object_geoms.get(other) == active_name
                        and len(transfer_table_contacts) < 5):
                    transfer_table_contacts.append((state.time, active_name))
            for gripper_geom, object_geom in (
                (contact.geom1, contact.geom2), (contact.geom2, contact.geom1)
            ):
                other_name = object_geoms.get(object_geom)
                if (gripper_geom in gripper_geoms and other_name is not None
                        and other_name != active_name
                        and len(neighbor_gripper_contacts) < 5):
                    neighbor_gripper_contacts.append(
                        (state.time, segment.name, other_name)
                    )
            for object_id, pad_id in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                if object_id in object_geoms and pad_id in pad_geoms:
                    pad_contacts[object_geoms[object_id]].add(pad_geoms[pad_id])

    mujoco.set_mjcb_control(control)
    try:
        if headless:
            while data.time < total + 0.5:
                mujoco.mj_step(model, data)
        else:
            from mujoco import viewer

            print("The managed viewer runs the sequence; close it to exit.")
            viewer.launch(model, data)
    finally:
        mujoco.set_mjcb_control(None)

    print("\nResult (XY error relative to the straight target line):")
    successful = 0
    for (name, _), x in zip(COMPONENTS, TARGET_XS):
        body = model.body(name).id
        xyz = data.xpos[body]
        error = float(np.linalg.norm(xyz[:2] - [x, TARGET_Y]))
        rotation = data.xmat[body].reshape(3, 3)
        yaw = Rotation.from_matrix(rotation).as_euler("xyz")[2]
        yaw_error = abs(float(np.degrees(np.arctan2(np.sin(yaw), np.cos(yaw)))))
        tilt_error = float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))
        settled = abs(xyz[2] - BLOCK_Z) < 0.02
        both_pads = pad_contacts[name] == {"left", "right"}
        carried_clear = transfer_min_height[name] > 0.05
        held_through_transfer = transfer_max_grip_gap[name] < 0.05
        ok = (lifted[name] and both_pads and settled and error < 0.010
              and yaw_error < 5.0 and tilt_error < 5.0
              and carried_clear and held_through_transfer)
        successful += ok
        print(
            f"  {name:14s} lifted={lifted[name]}  both-pads={both_pads}  "
            f"XY error={error:.3f} m  yaw error={yaw_error:.1f} deg  "
            f"tilt error={tilt_error:.1f} deg  "
            f"carry clearance={transfer_min_height[name]:.3f} m  "
            f"grip gap={transfer_max_grip_gap[name]:.3f} m  placed={ok}"
        )
    print(f"{successful}/{len(COMPONENTS)} cubes placed; simulation time {data.time:.1f}s")
    print(f"Gripper/table contact detected: {bool(table_gripper_contacts)}; "
          f"active cube/table contact during transfer: {bool(transfer_table_contacts)}; "
          f"gripper/neighbor contact detected: {bool(neighbor_gripper_contacts)}")
    if table_gripper_contacts:
        print(f"  first gripper/table contact: {table_gripper_contacts[0]}")
    if transfer_table_contacts:
        print(f"  first active cube/table transfer contact: {transfer_table_contacts[0]}")
    if neighbor_gripper_contacts:
        print(f"  first gripper/neighbor contact: {neighbor_gripper_contacts[0]}")
    print(f"Joint 7 moved through {np.degrees(wrist_range[1] - wrist_range[0]):.1f} deg")
    if headless and (successful != len(COMPONENTS) or table_gripper_contacts
                     or transfer_table_contacts or neighbor_gripper_contacts):
        raise RuntimeError("Unsafe or incomplete physical pick-and-place")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run without opening the viewer")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--randomize", dest="randomize", action="store_true", default=True,
                      help="Scatter start poses across the table (default)")
    mode.add_argument("--fixed", dest="randomize", action="store_false",
                      help="Use the original staging row for comparison")
    parser.add_argument("--seed", type=int, help="Repeatable random scene seed")
    parser.add_argument("--yaw-range-deg", type=float, default=30, help="Random yaw half-range")
    parser.add_argument(
        "--record-gif", type=Path, metavar="PATH",
        help="record the online Cartesian run as a GIF (requires ffmpeg and an OpenGL display)",
    )
    parser.add_argument("--legacy-planner", "--planner", dest="legacy_planner",
                        choices=PLANNER_NAMES,
                        help="Run the historical IK trajectory controller for comparison")
    args = parser.parse_args()
    if not 0 <= args.yaw_range_deg <= 45:
        parser.error("yaw range must be 0–45 degrees")
    if args.legacy_planner:
        if args.record_gif is not None:
            parser.error("--record-gif is only available with the online Cartesian controller")
        run(args.headless, args.randomize, args.seed,
            args.yaw_range_deg, args.legacy_planner)
    else:
        try:
            from .task_space_control import run as run_cartesian
        except ImportError:
            from task_space_control import run as run_cartesian
        run_cartesian(
            args.headless, args.randomize, args.seed, args.yaw_range_deg,
            args.record_gif,
        )
