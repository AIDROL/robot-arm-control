"""Coarse assembly followed by xArm7 deployment of a visual fine-alignment tool.

The FAT is a flat, kinematically attached proxy. Its two key tips approach
the marker and adjustment targets on the same side of each housing; no motor torque, camera
measurement, screw rotation, or optical feedback is simulated here.
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from . import line_assembly as scene
    from .task_space_control import (
        CartesianController, Phase, PARK_MIN_TOOL_CLEARANCE,
        POST_RUN_HOLD_SECONDS,
    )
except ImportError:
    import line_assembly as scene
    from task_space_control import (
        CartesianController, Phase, PARK_MIN_TOOL_CLEARANCE,
        POST_RUN_HOLD_SECONDS,
    )


FAT_TIP_LOCAL = np.array([0.0, scene.FAT_KEY_TIP_Y, 0.0])
INSERT_STANDOFF = 0.0015  # pin tips stop just short of the visual screw faces
PREALIGN_DISTANCE = 0.025  # retract within the widened gap before lifting out
TRAVEL_HEIGHT = 0.16
TIP_TOLERANCE = 0.010  # proxy alignment criterion, not micrometer accuracy
ANGLE_TOLERANCE_DEG = 5.0
FAT_GRIP_COMMAND = 150.0  # small clearance around the 36 mm side-to-side width


def rotation_from_mujoco_quat(quaternion: np.ndarray) -> Rotation:
    return Rotation.from_quat(quaternion[[1, 2, 3, 0]])


def mujoco_quat(rotation: Rotation) -> np.ndarray:
    x, y, z, w = rotation.as_quat()
    return np.array([w, x, y, z])


class FatAlignmentController(CartesianController):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        super().__init__(model, data, scene.FAT_TARGET_XS)
        self.fat_body = model.body("fat").id
        self.fat_geoms = [
            model.geom("fat_housing").id,
            *(model.geom(f"fat_{label}_key").id
              for label, *_ in scene.FAT_KEY_OFFSETS),
        ]
        self.mount_geoms = {
            name: model.geom(f"{name}_collision").id
            for name, _ in scene.COMPONENTS
        }
        self.table_geom = model.geom("workbench").id
        self.mocap_id = int(model.body_mocapid[self.fat_body])
        self.fat_start_position = data.mocap_pos[self.mocap_id].copy()
        self.fat_start_rotation = rotation_from_mujoco_quat(
            data.mocap_quat[self.mocap_id]
        )
        self.home_position = data.site_xpos[self.site].copy()
        self.home_orientation = Rotation.from_matrix(
            data.site_xmat[self.site].reshape(3, 3)
        )
        self.attached = False
        self.settle_start_position: np.ndarray | None = None
        self.settle_start_time = 0.0
        self.relative_body_position = np.zeros(3)
        self.relative_tip_position = np.zeros(3)
        self.relative_rotation = Rotation.identity()
        self.targets: dict[str, tuple[np.ndarray, Rotation]] = {}
        self.best_tip_error = {name: float("inf") for name, _ in scene.COMPONENTS}
        self.best_key_error = {name: float("inf") for name, _ in scene.COMPONENTS}
        self.best_angle_error = {name: float("inf") for name, _ in scene.COMPONENTS}
        self.min_fat_table_clearance = float("inf")
        self.min_fat_mount_clearance = float("inf")
        self.min_fat_neighbor_clearance = float("inf")
        self.min_fat_gripper_clearance = float("inf")
        self.min_clearance_phase = ""

    def _gripper_pose_for_tip(
        self, tip: np.ndarray, fat_rotation: Rotation,
    ) -> tuple[np.ndarray, Rotation]:
        gripper_rotation = fat_rotation * self.relative_rotation.inv()
        gripper_position = tip - gripper_rotation.apply(self.relative_tip_position)
        return gripper_position, gripper_rotation

    def _pickup_phases(self, state: mujoco.MjData) -> list[Phase]:
        grasp = state.site_xpos[self.model.site("fat_grasp").id].copy()
        above = grasp + [0.0, 0.0, 0.14]
        # The home jaw axis spans the FAT's 36 mm left/right side faces.
        orientation = self.home_orientation
        return [
            Phase("fat: pregrasp", above, orientation, 3.0, 0.0),
            Phase("fat: approach", grasp, orientation, 2.5, 0.0),
            Phase("fat: close", grasp, orientation, 1.4, FAT_GRIP_COMMAND, False),
            Phase("fat: lift", above, orientation, 2.5, FAT_GRIP_COMMAND),
        ]

    def _attach(self, state: mujoco.MjData) -> None:
        grip_position = state.site_xpos[self.site].copy()
        grip_rotation = Rotation.from_matrix(
            state.site_xmat[self.site].reshape(3, 3)
        )
        fat_position = state.mocap_pos[self.mocap_id].copy()
        fat_rotation = rotation_from_mujoco_quat(state.mocap_quat[self.mocap_id])
        self.relative_body_position = grip_rotation.inv().apply(
            fat_position - grip_position
        )
        self.relative_tip_position = grip_rotation.inv().apply(
            fat_position + fat_rotation.apply(FAT_TIP_LOCAL) - grip_position
        )
        self.relative_rotation = grip_rotation.inv() * fat_rotation
        self.attached = True

    def _alignment_phases(self, state: mujoco.MjData) -> list[Phase]:
        phases: list[Phase] = []
        for name, _ in scene.COMPONENTS:
            marker_id = self.model.site(f"{name}_side_marker").id
            marker_position = state.site_xpos[marker_id]
            marker_rotation = Rotation.from_matrix(
                state.site_xmat[marker_id].reshape(3, 3)
            )
            # This simulator-ground-truth marker pose stands in for a future
            # camera estimate; the offset is the fixed marker-to-screw calibration.
            marker_local_rotation = Rotation.from_euler(
                "xyz", scene.SIDE_MARKER_EULER
            )
            mount_rotation = marker_rotation * marker_local_rotation.inv()
            tool_rotation = mount_rotation * Rotation.from_euler(
                "z", scene.FAT_INSERT_YAW
            )
            insertion_axis = mount_rotation.apply([1.0, 0.0, 0.0])
            screw_midpoint = marker_position + mount_rotation.apply(
                scene.SCREW_MIDPOINT_LOCAL - scene.SIDE_MARKER_LOCAL
            )
            tip_goal = screw_midpoint - INSERT_STANDOFF * insertion_axis
            self.targets[name] = (tip_goal.copy(), tool_rotation)
            prealign_tip = tip_goal - PREALIGN_DISTANCE * insertion_axis
            travel_tip = prealign_tip + [0.0, 0.0, TRAVEL_HEIGHT]
            near_tip = tip_goal - 0.010 * insertion_axis
            def add(label: str, tip: np.ndarray, duration: float,
                    wait: bool = True) -> None:
                position, orientation = self._gripper_pose_for_tip(
                    tip, tool_rotation
                )
                phases.append(Phase(
                    f"fat {name}: {label}", position, orientation,
                    duration, FAT_GRIP_COMMAND, wait,
                ))
            add("travel", travel_tip, 3.5)
            add("prealign", prealign_tip, 2.2)
            add("near", near_tip, 1.5)
            add("insert", tip_goal, 2.0)
            add("hold", tip_goal, 0.8, False)
            add("withdraw", prealign_tip, 2.0)
            add("clear", travel_tip, 2.0)

        return_orientation = self.fat_start_rotation * self.relative_rotation.inv()
        return_position = self.fat_start_position - return_orientation.apply(
            self.relative_body_position
        )
        return_position[2] += 0.004  # leave room for position-servo settling at the tabletop
        phases.extend([
            Phase("fat: return high", return_position + [0, 0, 0.14],
                  return_orientation, 3.5, FAT_GRIP_COMMAND),
            Phase("fat: return lower", return_position,
                  return_orientation, 2.5, FAT_GRIP_COMMAND),
            Phase("fat: release", return_position,
                  return_orientation, 1.2, 0.0, False),
            Phase("fat: return clear", return_position + [0, 0, 0.14],
                  return_orientation, 2.3, 0.0),
            Phase("fat: park", self.home_position,
                  self.home_orientation, 3.5, 0.0),
        ])
        return phases

    def advance(self, state: mujoco.MjData) -> None:
        finished = self.phase.label
        if finished == "park above table":
            self.phases.extend(self._pickup_phases(state))
        elif finished == "fat: close":
            self._attach(state)
        elif finished == "fat: lift":
            self.phases.extend(self._alignment_phases(state))
        elif finished == "fat: release":
            self.attached = False
            self.settle_start_position = state.mocap_pos[self.mocap_id].copy()
            self.settle_start_time = float(state.time)
        super().advance(state)

    def observe(self, state: mujoco.MjData) -> None:
        super().observe(state)
        if self.phase.label.startswith("fat"):
            for gripper_geom in self.gripper_geoms:
                for fat_geom in self.fat_geoms:
                    clearance = mujoco.mj_geomDistance(
                        self.model, state, gripper_geom, fat_geom, 0.2, None
                    )
                    self.min_fat_gripper_clearance = min(
                        self.min_fat_gripper_clearance, clearance
                    )
                    if clearance < -0.0005 and self.failed is None:
                        self.hold_q = self.command_q.copy()
                        self.failed = (
                            f"gripper/FAT penetration {-clearance:.4f} m "
                            f"during {self.phase.label}"
                        )
        if self.attached:
            clearance = min(
                mujoco.mj_geomDistance(
                    self.model, state, fat_geom, self.table_geom, 0.2, None
                )
                for fat_geom in self.fat_geoms
            )
            on_table = self.phase.label in (
                "fat: lift", "fat: return lower", "fat: release"
            )
            if not on_table and clearance < self.min_fat_table_clearance:
                self.min_fat_table_clearance = clearance
                self.min_clearance_phase = self.phase.label
            minimum = -0.0005 if on_table else 0.005
            if clearance < minimum and self.failed is None:
                self.hold_q = self.command_q.copy()
                self.failed = (
                    f"FAT/table clearance {clearance:.3f} m "
                    f"during {self.phase.label}"
                )
            active_name = (
                self.phase.label[4:].split(":", 1)[0]
                if self.phase.label.startswith("fat ") else None
            )
            for name, mount_geom in self.mount_geoms.items():
                mount_clearance = min(
                    mujoco.mj_geomDistance(
                        self.model, state, fat_geom, mount_geom, 0.2, None
                    )
                    for fat_geom in self.fat_geoms
                )
                self.min_fat_mount_clearance = min(
                    self.min_fat_mount_clearance, mount_clearance
                )
                if name != active_name:
                    self.min_fat_neighbor_clearance = min(
                        self.min_fat_neighbor_clearance, mount_clearance
                    )
                if mount_clearance < 0.002 and self.failed is None:
                    self.hold_q = self.command_q.copy()
                    self.failed = (
                        f"FAT/{name} clearance {mount_clearance:.3f} m "
                        f"during {self.phase.label}"
                    )
        label = self.phase.label
        if not label.startswith("fat ") or not label.endswith(": hold"):
            return
        name = label[4:].split(":", 1)[0]
        target_tip, target_rotation = self.targets[name]
        fat_rotation = rotation_from_mujoco_quat(
            state.mocap_quat[self.mocap_id]
        )
        actual_tip = state.mocap_pos[self.mocap_id] + fat_rotation.apply(
            FAT_TIP_LOCAL
        )
        error = float(np.linalg.norm(actual_tip - target_tip))
        angle = float(np.degrees(np.linalg.norm(
            (target_rotation * fat_rotation.inv()).as_rotvec()
        )))
        self.best_tip_error[name] = min(self.best_tip_error[name], error)
        self.best_angle_error[name] = min(self.best_angle_error[name], angle)
        key_errors = []
        for key_label, x_offset, z_offset in scene.FAT_KEY_OFFSETS:
            key_tip = state.mocap_pos[self.mocap_id] + fat_rotation.apply(
                [x_offset, scene.FAT_KEY_TIP_Y, z_offset]
            )
            screw = state.site_xpos[
                self.model.site(f"{name}_{key_label}_screw_center").id
            ]
            key_errors.append(float(np.linalg.norm(key_tip - screw)))
        self.best_key_error[name] = min(self.best_key_error[name], max(key_errors))

    def __call__(self, model: mujoco.MjModel, state: mujoco.MjData) -> None:
        if self.attached:
            grip_position = state.site_xpos[self.site]
            grip_rotation = Rotation.from_matrix(
                state.site_xmat[self.site].reshape(3, 3)
            )
            fat_rotation = grip_rotation * self.relative_rotation
            state.mocap_pos[self.mocap_id] = (
                grip_position + grip_rotation.apply(self.relative_body_position)
            )
            state.mocap_quat[self.mocap_id] = mujoco_quat(fat_rotation)
            mujoco.mj_kinematics(model, state)
        elif self.settle_start_position is not None:
            u = float(np.clip((state.time - self.settle_start_time) / 0.6, 0, 1))
            state.mocap_pos[self.mocap_id] = (
                self.settle_start_position
                + u * (self.fat_start_position - self.settle_start_position)
            )
            state.mocap_quat[self.mocap_id] = mujoco_quat(self.fat_start_rotation)
            mujoco.mj_kinematics(model, state)
            if u >= 1:
                self.settle_start_position = None
        super().__call__(model, state)


def run(headless: bool, randomize: bool, seed: int | None,
        yaw_range_deg: float) -> None:
    if randomize and seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    model = scene.make_scene(
        randomize=randomize, seed=seed, yaw_range_deg=yaw_range_deg,
        include_fat=True, target_xs=scene.FAT_TARGET_XS,
    )
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    print(f"Coarse assembly + FAT alignment; seed={seed if randomize else 'fixed'}")
    controller = FatAlignmentController(model, data)
    mujoco.set_mjcb_control(controller)
    try:
        if headless:
            while not controller.done and controller.failed is None and data.time < 450:
                mujoco.mj_step(model, data)
            if controller.done:
                stop_time = data.time + POST_RUN_HOLD_SECONDS
                while data.time < stop_time:
                    mujoco.mj_step(model, data)
        else:
            from mujoco import viewer
            viewer.launch(model, data)
    finally:
        mujoco.set_mjcb_control(None)
    if controller.failed:
        raise RuntimeError(controller.failed)
    if not controller.done:
        raise RuntimeError(f"Controller stopped before completion at {data.time:.1f}s")

    passed = 0
    for (name, _), x in zip(scene.COMPONENTS, scene.FAT_TARGET_XS):
        actual = data.xpos[model.body(name).id]
        xy_error = float(np.linalg.norm(actual[:2] - [x, scene.TARGET_Y]))
        tip_error = controller.best_tip_error[name]
        key_error = controller.best_key_error[name]
        angle_error = controller.best_angle_error[name]
        ok = (
            controller.lifted[name]
            and controller.pad_contacts[name] == {"left", "right"}
            and xy_error < 0.010
            and tip_error < TIP_TOLERANCE
            and key_error < TIP_TOLERANCE
            and angle_error < ANGLE_TOLERANCE_DEG
        )
        passed += ok
        print(
            f"  {name:14s} placed XY={xy_error:.3f} m; "
            f"FAT tip={tip_error:.3f} m, keys={key_error:.3f} m, "
            f"angle={angle_error:.1f} deg; ok={ok}"
        )
    fat_return_error = float(np.linalg.norm(
        data.mocap_pos[controller.mocap_id] - controller.fat_start_position
    ))
    print(f"{passed}/5 FAT targets aligned; FAT return error={fat_return_error:.3f} m")
    print(
        f"Minimum carried FAT/table clearance: "
        f"{controller.min_fat_table_clearance:.3f} m "
        f"during {controller.min_clearance_phase}"
    )
    print(f"Minimum FAT/mount clearance: "
          f"{controller.min_fat_mount_clearance:.3f} m")
    print(f"Minimum FAT/other-mount clearance: "
          f"{controller.min_fat_neighbor_clearance:.3f} m")
    print(f"Minimum gripper/FAT clearance: "
          f"{controller.min_fat_gripper_clearance:.4f} m")
    print(f"Unsafe contacts: {controller.unsafe_contacts[:5]}")
    if headless:
        print(f"Parked tool clearance: {controller.park_min_tool_clearance:.3f} m")
    if (passed != len(scene.COMPONENTS) or fat_return_error > 0.015
            or controller.min_fat_table_clearance < 0.005
            or controller.min_fat_mount_clearance < 0.002
            or controller.min_fat_neighbor_clearance < 0.005
            or controller.min_fat_gripper_clearance < -0.0005
            or controller.unsafe_contacts
            or (headless and controller.park_min_tool_clearance < PARK_MIN_TOOL_CLEARANCE)):
        raise RuntimeError("FAT alignment controller failed validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--randomize", dest="randomize", action="store_true", default=True)
    mode.add_argument("--fixed", dest="randomize", action="store_false")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--yaw-range-deg", type=float, default=30)
    args = parser.parse_args()
    if not 0 <= args.yaw_range_deg <= 45:
        parser.error("yaw range must be 0–45 degrees")
    run(args.headless, args.randomize, args.seed, args.yaw_range_deg)
