"""One object: dual 4K RGB localization -> physical grasp/place -> evaluation."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from itertools import count
import os
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import global_camera as vision
from . import line_assembly as scene
from . import wrist_camera
from .task_space_control import CartesianController, GifRecorder, Phase
from .placement_diagnostics import PlacementDiagnostics, markdown_report
from . import housing_profile as profiles
from .timing import SimulationPacer,positive_speed


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def reserve_output_directory(requested: Path) -> Path:
    """Atomically reserve a fresh directory while preserving earlier runs."""
    requested = requested.expanduser().absolute()
    for number in count():
        candidate = requested if number == 0 else requested.with_name(
            f'{requested.name}-{number:03d}')
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate


def require_pose(observation: dict, now: float) -> np.ndarray:
    """Validate a perception message without any simulator state access."""
    if not observation.get("valid"):
        raise ValueError(observation.get("reason", "INVALID_OBSERVATION"))
    if observation.get("frame_id") != "world":
        raise ValueError("WRONG_POSE_FRAME")
    if not 0 <= now - observation.get("timestamp", -1e9) <= 0.1:
        raise ValueError("STALE_OBSERVATION")
    T = np.array(observation.get("T_world_object"), dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("INVALID_TRANSFORM")
    R = T[:3, :3]
    if (not np.allclose(T[3], [0, 0, 0, 1])
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(R), 1, atol=1e-5)):
        raise ValueError("INVALID_TRANSFORM")
    x, y, z = T[:3, 3]
    geometry = profiles.from_id(observation.get('housing_geometry', profiles.PROXY.id))
    if not (-0.55 <= x <= 0.55 and -0.35 <= y <= 0.65
            and abs(z - scene.TABLE_Z - geometry.body_height) < 0.005 and R[2, 2] > 0.999):
        raise ValueError("POSE_OUTSIDE_TABLETOP_PRIOR")
    return T


class Evaluator:
    """Truth-based monitoring only; never generates or changes motion targets."""
    def __init__(self, model, name, target, target_yaw=None):
        self.name, self.target = name, np.array(target)
        self.geometry = profiles.for_model(model)
        self.target_yaw = self.geometry.placement_yaw if target_yaw is None else target_yaw
        self.object_geoms = set(scene.component_collision_geoms(model, name))
        self.body = model.body(name).id
        joint = model.joint(f"{name}_free").id
        self.velocity_start = int(model.jnt_dofadr[joint])
        self.driver_qpos = [int(model.jnt_qposadr[model.joint(f"{s}_driver_joint").id])
                            for s in ("left", "right")]
        self.min_bottom_clearance = float("inf")
        self.min_clearance_phase = None
        self.transfer_offsets = []
        self.transfer_samples = 0
        self.transfer_two_pad_samples = 0
        self.stable_since = None
        self.last_stable_duration = 0.
        self.open_pad_gap_m = None

    def observe(self, model, data, controller):
        p = data.xpos[self.body]
        R = data.xmat[self.body].reshape(3, 3)
        half_extent_z = np.abs(R[2]) @ np.array([
            *self.geometry.half_extents])
        clearance = float((p + R @ np.array(self.geometry.bounds_center))[2] - half_extent_z - scene.TABLE_Z)
        if clearance < self.min_bottom_clearance:
            self.min_bottom_clearance = clearance
            self.min_clearance_phase = {"phase": controller.phase.label,
                                        "time": float(data.time)}
        both_pads = set()
        for contact in data.contact[:data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            other = b if a in self.object_geoms else a if b in self.object_geoms else None
            if other in controller.pad_geoms:
                both_pads.add(controller.pad_geoms[other])
        if controller.phase.label == f"{self.name}: transfer":
            self.transfer_samples += 1
            self.transfer_two_pad_samples += both_pads == {"left", "right"}
            tool_rotation = data.site_xmat[controller.site].reshape(3, 3)
            self.transfer_offsets.append(tool_rotation.T @ (
                p - data.site_xpos[controller.site]))
        velocity = data.qvel[self.velocity_start:self.velocity_start + 6]
        driver_open = bool(np.max(np.abs(data.qpos[self.driver_qpos])) < 0.03)
        if self.geometry.is_selected:
            gaps=[]
            for index in (1,2):
                a=model.geom(f'left_finger_pad_{index}').id;b=model.geom(f'right_finger_pad_{index}').id
                vector=data.geom_xpos[b]-data.geom_xpos[a];axis=vector/np.linalg.norm(vector)
                gaps.append(float(np.linalg.norm(vector)
                    -np.abs(axis@data.geom_xmat[a].reshape(3,3))@model.geom_size[a]
                    -np.abs(axis@data.geom_xmat[b].reshape(3,3))@model.geom_size[b]))
            self.open_pad_gap_m=min(gaps)
            driver_open=self.open_pad_gap_m>2*self.geometry.half_extents[0]+.004
        good = (controller.done and driver_open and not both_pads
                and np.linalg.norm(p[:2] - self.target[:2]) < 0.010
                and abs(p[2] - scene.TABLE_Z - self.geometry.body_height) < 0.005
                and np.linalg.norm(velocity[:3]) < 0.005
                and np.linalg.norm(velocity[3:]) < 0.05
                and R[2, 2] > np.cos(np.deg2rad(5)))
        if good:
            if self.stable_since is None:
                self.stable_since = float(data.time)
        else:
            self.stable_since = None
        self.last_stable_duration = (float(data.time - self.stable_since)
                                     if self.stable_since is not None else 0.)

    def result(self, data, controller):
        p = data.xpos[self.body]
        R = data.xmat[self.body].reshape(3, 3)
        yaw = Rotation.from_matrix(R).as_euler("xyz")[2]
        yaw_error = float(abs(np.arctan2(np.sin(yaw-self.target_yaw),
                                         np.cos(yaw-self.target_yaw))))
        xy = float(np.linalg.norm(p[:2] - self.target[:2]))
        drift = (float(np.max(np.linalg.norm(
            np.array(self.transfer_offsets) - self.transfer_offsets[0], axis=1)))
            if self.transfer_offsets else None)
        contact_fraction = self.transfer_two_pad_samples / max(self.transfer_samples, 1)
        carry_height = controller.transfer_min_height[self.name]
        checks = {
            "controller_completed": controller.done and controller.failed is None,
            "lifted": controller.lifted[self.name],
            "both_pads_contacted": controller.pad_contacts[self.name] == {"left", "right"},
            "sustained_two_pad_contact": contact_fraction > 0.8,
            "followed_gripper": drift is not None and drift < 0.010,
            "carry_clearance": carry_height > 0.05 and np.isfinite(carry_height),
            "placed_xy": xy < 0.010,
            "placed_height": abs(p[2] - scene.TABLE_Z - self.geometry.body_height) < 0.005,
            "placed_yaw": yaw_error < np.deg2rad(5),
            "placed_tilt": R[2, 2] > np.cos(np.deg2rad(5)),
            "open_and_stable_0_5_seconds": self.last_stable_duration >= 0.5,
            "no_table_penetration": self.min_bottom_clearance > -0.001,
            "no_unsafe_contacts": not controller.unsafe_contacts,
        }
        return {"success": bool(all(checks.values())),
                "checks": {k: bool(v) for k, v in checks.items()},
                "placement_xy_error_m": xy, "placement_yaw_error_rad": yaw_error,
                "target_yaw_rad": self.target_yaw, "actual_yaw_rad": float(yaw),
                "transfer_relative_drift_m": drift,
                "transfer_two_pad_contact_fraction": contact_fraction,
                "transfer_min_clearance_m": float(carry_height) if np.isfinite(carry_height) else None,
                "min_object_table_clearance_m": self.min_bottom_clearance,
                "min_clearance_phase": self.min_clearance_phase,
                "stable_duration_seconds": self.last_stable_duration,
                "final_free_velocity": data.qvel[self.velocity_start:self.velocity_start+6].tolist(),
                "final_driver_angles_rad": data.qpos[self.driver_qpos].tolist(),
                "open_pad_gap_m": self.open_pad_gap_m,
                "unsafe_contacts": controller.unsafe_contacts}


class PlacementController(CartesianController):
    def __init__(self, *args, placement_strategy='baseline', precision=False, precision_position_tolerance=.0003, grip_command=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.placement_strategy = placement_strategy
        self.seating_samples = []
        self.seated_since = None
        self.seating_confirmed = False
        self.seating_confirmation = None
        self.bilateral_since = None
        self.contact_geoms={name:{g for g,n in self.object_geoms.items() if n==name}
                            for name in set(self.object_geoms.values())}
        self.support_hold = None
        if placement_strategy in ('settled','contact_supported'):
            phases = []
            for phase in self.phases:
                phases.append(replace(phase, duration=3.) if phase.label.endswith(': release') else phase)
                if phase.label.endswith(': lower'):
                    phases.append(Phase(phase.label.replace(': lower', ': settle'),
                                        phase.position.copy(), phase.orientation, .8, self.geometry.grip_command, False))
            self.phases = phases
        if precision:
            critical_suffixes = (': approach', ': close', ': lower', ': settle', ': release')
            if self.geometry.is_selected:
                critical_suffixes += (': transfer',)
            self.phases = [replace(p, position_tolerance=precision_position_tolerance,
                                   orientation_tolerance=np.deg2rad(.5), wait_for_pose=True)
                           if p.label.endswith(critical_suffixes) else p
                           for p in self.phases]
        if grip_command is not None:
            if not 0<grip_command<=255:raise ValueError('Gripper command must be in (0,255]')
            self.phases=[replace(p,grip=grip_command) if p.grip>0 else p for p in self.phases]
        if placement_strategy=='contact_supported':
            self.phases=[replace(p,wait_for_pose=False) if p.label.endswith(': close') else p for p in self.phases]

    def advance(self, state):
        geometry = getattr(self, 'geometry', profiles.PROXY)
        if geometry.is_selected and self.phase.label.endswith(': close'):
            if self.bilateral_since is None or state.time-self.bilateral_since < .2:
                if state.time-self.phase_start_time > 4.:
                    self.failed = 'NO_BILATERAL_GRASP'
                    self.hold_q = self.command_q.copy()
                return
        if getattr(self, 'placement_strategy', 'baseline') in ('settled','contact_supported'):
            if self.phase.label.endswith(': lower'):
                # Hold the attained pose instead of continuing to pull toward
                # the old lower target while the fingers open. FK only.
                position = state.site_xpos[self.site].copy()
                orientation = Rotation.from_matrix(state.site_xmat[self.site].reshape(3, 3))
                for index in ((self.phase_index+1,self.phase_index+2) if self.placement_strategy=='contact_supported' or not geometry.is_selected else ()):
                    self.phases[index] = replace(self.phases[index], position=position.copy(), orientation=orientation)
            elif self.phase.label.endswith(': settle'):
                name = self.phase.label.split(':')[0]
                object_geoms = (set(scene.component_collision_geoms(self.model, name)) if geometry.is_selected
                                else {self.model.geom(f'{name}_collision').id})
                normal_force = 0.
                force = np.zeros(6)
                for index, contact in enumerate(state.contact[:state.ncon]):
                    if self.table_geom in (int(contact.geom1), int(contact.geom2)) and object_geoms.intersection((int(contact.geom1), int(contact.geom2))):
                        mujoco.mj_contactForce(self.model, state, index, force)
                        normal_force += max(0., float(force[0]))
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(self.model, state, mujoco.mjtObj.mjOBJ_SITE, self.site, velocity, 0)
                speed = float(np.linalg.norm(velocity[3:]))
                stable = normal_force > .02 and speed < .002 and np.linalg.norm(velocity[:3]) < .03
                self.seated_since = (float(state.time) if self.seated_since is None else self.seated_since) if stable else None
                duration = 0. if self.seated_since is None else float(state.time - self.seated_since)
                if not self.seating_samples or state.time-self.seating_samples[-1]['time'] >= .05:
                    self.seating_samples.append({'time': float(state.time), 'table_normal_force_N': normal_force,
                                                 'tcp_speed_m_s': speed, 'stable_duration_s': duration})
                if duration < .3:
                    elapsed = state.time-self.phase_start_time
                    if geometry.is_selected and self.placement_strategy!='contact_supported':
                        position = self.phase.position.copy()
                        position[2] = scene.TABLE_Z+geometry.body_height+geometry.grasp_z-min(.004, max(0., elapsed-.8)*.001)
                        self.phases[self.phase_index] = replace(self.phase, position=position)
                    if elapsed > (5. if geometry.is_selected else 3.):
                        self.failed = 'SEATING_TIMEOUT: no sustained support contact and stationary TCP'
                        self.hold_q = self.command_q.copy()
                    return
                self.seating_confirmed = True
                self.seating_confirmation = {'time': float(state.time), 'stable_duration_s': duration,
                                             'table_normal_force_N': normal_force, 'tcp_speed_m_s': speed}
                if geometry.is_selected:
                    self.phases[self.phase_index+1] = replace(self.phases[self.phase_index+1], position=state.site_xpos[self.site].copy())
        super().advance(state)

    def __call__(self,model,state):
        if self.geometry.is_selected and self.phase.label.endswith(': close'):
            name=self.phase.label.split(':')[0];geoms=self.contact_geoms[name]
            force=np.zeros(6);pads=set()
            for i,c in enumerate(state.contact[:state.ncon]):
                a,b=int(c.geom1),int(c.geom2);other=b if a in geoms else a if b in geoms else None
                if other in self.pad_geoms:
                    mujoco.mj_contactForce(model,state,i,force)
                    if force[0]>.01:pads.add(self.pad_geoms[other])
            self.bilateral_since=(float(state.time) if self.bilateral_since is None else self.bilateral_since) if pads=={'left','right'} else None
        if self.placement_strategy=='contact_supported' and self.phase.label.endswith(': lower'):
            name=self.phase.label.split(':')[0]
            geoms=self.contact_geoms[name];force=np.zeros(6);normal=0.
            for i,c in enumerate(state.contact[:state.ncon]):
                if self.table_geom in (int(c.geom1),int(c.geom2)) and geoms.intersection((int(c.geom1),int(c.geom2))):
                    mujoco.mj_contactForce(model,state,i,force);normal+=max(0.,float(force[0]))
            if normal>.15 and self.support_hold is None:
                self.support_hold=(state.site_xpos[self.site].copy(),Rotation.from_matrix(state.site_xmat[self.site].reshape(3,3)))
            if self.support_hold is not None:
                self.phases[self.phase_index]=replace(self.phase,position=self.support_hold[0],orientation=self.support_hold[1])
        super().__call__(model,state)


class WristController(PlacementController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending_refinement = False
        self.refined = False
        # Set the initial observation distance from the configured hand-eye
        # geometry, not from an object ground-truth query.
        pose = np.array(next(iter(kwargs['object_poses'].values())))
        marker = pose[:3, 3] + pose[:3, :3] @ np.array([0, 0, self.geometry.top_z])
        phase = self.phases[0]
        R = phase.orientation.as_matrix()
        camera_position = phase.position + R @ (wrist_camera.POSITION - np.array([0, 0, .14]))
        forward = R @ wrist_camera.E_FROM_D[:3, 2]
        depth = float(forward @ (marker - camera_position))
        offset_z = (wrist_camera.PREGRASP_DEPTH_M - depth) / -forward[2]
        self.phases[0] = replace(phase, position=phase.position + [0, 0, offset_z])

    def advance(self, state):
        if self.phase.label.endswith(': pregrasp') and not self.refined:
            self.pending_refinement = True
            self.hold_q = self.command_q.copy()
        else:
            super().advance(state)

    def apply_refinement(self, state, correction):
        for index, phase in enumerate(self.phases):
            if phase.label.endswith((': approach', ': close', ': lift')):
                self.phases[index] = replace(phase, position=phase.position + correction)
        self.refined = True
        self.pending_refinement = False
        super().advance(state)


def run(seed: int, name: str, output: Path, hide_marker=False, record_gif=False,
        coarse_only=False, invalid_wrist_depth=False, coarse_offset_mm=(0., 0.), view=False,
        camera_monitor=True, camera_fps=5, camera_display='native', placement_strategy='baseline', precision=False,
        model=None, data=None, shared_viewer=None, shared_monitor=None, require_precision_goal=True,
        relative_shift_m=None,target_pose=None,precision_position_tolerance=.0003,pickup_yaw_prior=None,release_hook=None,grip_command=None,state_observer=None,support_hook=None,speed=1.):
    speed=positive_speed(speed)
    if relative_shift_m is not None:
        shift=np.asarray(relative_shift_m,dtype=float)
        if shift.shape!=(3,) or not np.isfinite(shift).all() or np.linalg.norm(shift)>.0002 or shift[2]!=0:
            raise ValueError('Relative re-placement requires a finite XY shift <=0.2 mm')
        if coarse_only:raise ValueError('Relative re-placement requires wrist refinement')
    output = reserve_output_directory(output)
    print(f'Run output directory: {output}', flush=True)
    report = {"seed": seed, "object": name, "success": False,
              "output_directory": str(output),
              "failure_code": None, "control_started": False,
              "scope": "single global + wrist RGB-D pick/place" if not coarse_only else "global-only pick/place",
              "coarse_only": coarse_only, "invalid_wrist_depth": invalid_wrist_depth,
              "injected_coarse_offset_mm": list(coarse_offset_mm),
              "resolution": [vision.WIDTH, vision.HEIGHT],
              "localization_prior": "known-height upright housings on fixed table",
              "hide_target_marker": hide_marker}
    report['placement_strategy'] = placement_strategy
    report['precision'] = precision
    report['requested_view_speed']=speed
    report['precision_xy_goal_m'] = .0005 if precision else None
    report['require_precision_goal'] = require_precision_goal
    controller, recorder, live_viewer = None, None, shared_viewer
    monitor = shared_monitor
    owns_ui = shared_viewer is None
    trace = []
    if model is None:
        if data is not None:
            raise ValueError('data requires its model')
        model = scene.make_scene(seed=seed, include_fat=True, vision=True,
                                 target_xs=scene.FAT_TARGET_XS)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
    elif data is None:
        raise ValueError('shared model requires shared data; refusing to reset scene')
    start_time = float(data.time)
    geometry = profiles.for_model(model)
    report['housing_geometry'] = geometry.id
    report['target_layout'] = scene.target_layout_id(model)
    report['target_yaw_rad'] = geometry.placement_yaw
    report['scene_start_time'] = start_time
    if hide_marker:
        for geom in range(model.ngeom):
            if model.geom(geom).name.startswith(f"{name}_top_aruco_"):
                model.geom_rgba[geom, 3] = 0
    try:
        cameras = vision.capture_observations(model, data, output)
        objects = vision.select_objects(cameras)
        write_json(output / "observations.json", {"cameras": cameras, "objects": objects})
        estimate = objects[name]
        try:
            T = require_pose(estimate, float(data.time))
        except ValueError as error:
            report["failure_code"] = str(error)
            raise
        report["control_observation"] = estimate
        if pickup_yaw_prior is not None:
            if not np.isfinite(pickup_yaw_prior):raise ValueError('Invalid staging orientation')
            T[:3,:3]=Rotation.from_euler('z',pickup_yaw_prior).as_matrix()
            report['pickup_yaw_prior_rad']=float(pickup_yaw_prior)
            report['pickup_orientation_source']='configured staging-slot orientation; image position and wrist depth retained'
        T[:2, 3] += np.array(coarse_offset_mm) / 1000
        T = require_pose({**estimate, 'T_world_object': T.tolist()}, float(data.time))
        report["coarse_control_pose"] = T.tolist()
        # Isolated diagnostic reads never feed back into the controller.
        report["diagnostics"] = {"coarse_position_error_m": float(np.linalg.norm(
            T[:3, 3] - data.body(name).xpos))}
        index = [n for n, _ in scene.components_for_model(model)].index(name)
        target_xs = scene.target_xs_for_model(model)
        target = [target_xs[index], scene.TARGET_Y, scene.TABLE_Z+geometry.body_height]
        target_yaw=geometry.placement_yaw
        if target_pose is not None:
            target_pose=np.asarray(target_pose,dtype=float)
            if (target_pose.shape!=(4,4) or not np.isfinite(target_pose).all()
                    or not np.allclose(target_pose[3],[0,0,0,1])
                    or not np.allclose(target_pose[:3,:3].T@target_pose[:3,:3],np.eye(3),atol=1e-6)
                    or np.linalg.det(target_pose[:3,:3])<.999
                    or target_pose[2,2]<.999 or abs(target_pose[2,3]-scene.TABLE_Z-geometry.body_height)>.0001):
                raise ValueError('INVALID_TABLETOP_TARGET_POSE')
            target=target_pose[:3,3].tolist()
            target_yaw=float(Rotation.from_matrix(target_pose[:3,:3]).as_euler('xyz')[2])
        report['target_yaw_rad']=target_yaw
        report["target_world_m"] = target
        controller_type = PlacementController if coarse_only else WristController
        controller = controller_type(model, data, target_xs,
                                          object_poses={name: T}, placement_strategy=placement_strategy, precision=precision,
                                          target_poses=None if target_pose is None else {name:target_pose},
                                          precision_position_tolerance=precision_position_tolerance,grip_command=grip_command)
        evaluator = Evaluator(model, name, target,target_yaw=target_yaw)
        placement_diagnostics = PlacementDiagnostics(model, name, target)
        report["plan"] = [{"phase": p.label, "position_world_m": p.position.tolist(),
                           "orientation_xyzw": p.orientation.as_quat().tolist(),
                           "gripper_command": p.grip, "duration_seconds": p.duration}
                          for p in controller.phases]
        report['phase_tolerances'] = {p.label: {'position_m': p.position_tolerance,
                                                'orientation_rad': float(p.orientation_tolerance)}
                                      for p in controller.phases}
        report["control_started"] = True
        if view and owns_ui:
            from mujoco import viewer
            live_viewer = viewer.launch_passive(model, data, show_right_ui=False)
            overview = vision.overview_camera(model)
            live_viewer.cam.lookat[:] = overview.lookat
            live_viewer.cam.distance = overview.distance
            live_viewer.cam.azimuth = overview.azimuth
            live_viewer.cam.elevation = overview.elevation
            live_viewer.set_texts((None, None, 'Visual pick/place', 'Global RGB -> wrist RGB-D -> grasp'))
            live_viewer.sync()
            if camera_monitor:
                from .camera_monitor import CameraMonitor
                monitor = CameraMonitor(model, fps=camera_fps/max(1.,speed), open_browser=camera_display == 'browser')
                report['camera_monitor_url'] = monitor.url
                monitor.update(model, data, '开始抓放', force=True)
                if camera_display == 'native':
                    monitor.start_player(output)
            time.sleep(2.)
        if record_gif:
            recorder = GifRecorder(model, output / "pick-place.gif")
        transitions = []
        last_phase = None
        finished_time = None
        next_trace = 0.
        next_view_sync = 0.
        release_hook_called=False
        support_hook_called=False
        with mujoco.Renderer(model, height=vision.HEIGHT, width=vision.WIDTH) as renderer:
            pacer=SimulationPacer(data.time,speed,enabled=live_viewer is not None)
            while data.time - start_time < 120:
                if live_viewer is not None and not live_viewer.is_running():
                    report['failure_code'] = 'VIEWER_CLOSED'
                    break
                controller(model, data)
                if (support_hook is not None and not support_hook_called
                        and controller.phase.label==name+': release' and controller.seating_confirmed):
                    support_hook(model,data,name);support_hook_called=True
                if state_observer is not None:state_observer(model,data,controller.phase.label)
                if (release_hook is not None and not release_hook_called
                        and controller.phase.label==name+': retreat' and controller.seating_confirmed):
                    release_hook(model,data,name)
                    release_hook_called=True
                    report['release_hook_called']=True
                if isinstance(controller, WristController) and controller.pending_refinement:
                    if live_viewer is not None:
                        live_viewer.set_texts((None, None, 'Wrist RGB-D', 'Capturing target and refining grasp'))
                        live_viewer.sync()
                    fine = wrist_camera.capture(model, data, output, T, index, invalid_wrist_depth)
                    report['fine_observation'] = fine
                    write_json(output / 'fine_observation.json', fine)
                    if not fine['valid']:
                        report['failure_code'] = fine['reason']
                        break
                    try:
                        refined_pose = require_pose(fine, float(data.time))
                    except ValueError as error:
                        report['failure_code'] = str(error)
                        break
                    report['diagnostics']['fine_position_error_m'] = float(np.linalg.norm(
                        refined_pose[:3, 3] - data.body(name).xpos))
                    if relative_shift_m is not None:
                        # New placement target comes solely from the fresh wrist
                        # observation plus a bounded command, never object truth.
                        new_target=refined_pose[:3,3]+shift
                        new_target[2]=scene.TABLE_Z+geometry.body_height
                        delta=new_target-np.array(target)
                        controller.phases=[replace(p,position=p.position+delta)
                            if p.label.endswith((': transfer',': lower',': settle',': release',': retreat')) else p
                            for p in controller.phases]
                        target=new_target.tolist();evaluator.target=new_target.copy()
                        placement_diagnostics.target=new_target.copy()
                        report['target_world_m']=target
                        report['relative_shift_m']=shift.tolist()
                        report['target_source']='fresh wrist RGB-D pose plus bounded relative command'
                    controller.apply_refinement(data, refined_pose[:3, 3] - T[:3, 3])
                    report['refined_plan'] = [{'phase': p.label, 'position_world_m': p.position.tolist()}
                                               for p in controller.phases]
                placement_diagnostics.observe(data, controller)
                if controller.failed:
                    report["failure_code"] = 'SEATING_TIMEOUT' if controller.failed.startswith('SEATING_TIMEOUT') else "MOTION_TIMEOUT"
                    break
                # Evaluation is separate from the source of control targets.
                evaluator.observe(model, data, controller)
                if controller.unsafe_contacts:
                    report["failure_code"] = "UNSAFE_CONTACT"
                    break
                label = controller.phase.label
                if label != last_phase:
                    transitions.append({"phase": label, "time": float(data.time)})
                    last_phase = label
                    if label.endswith((": transfer", ": release", ": retreat")):
                        renderer.update_scene(data, camera="global_a")
                        cv2.imwrite(str(output / f"{label.split(': ')[1]}.png"),
                                    cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
                if data.time >= next_trace:
                    trace.append({"time": float(data.time), "phase": label,
                                  "tcp_world_m": data.site_xpos[controller.site].tolist(),
                                  "target_world_m": controller.phase.position.tolist(),
                                  "gripper_command": float(data.ctrl[controller.gripper_actuator])})
                    next_trace = float(data.time) + 0.1
                if recorder is not None:
                    recorder.capture(model, data)
                if monitor is not None:
                    monitor.update(model, data, controller.phase.label)
                if live_viewer is not None and time.monotonic() >= next_view_sync:
                    live_viewer.set_texts((None, None, f'Phase | requested {speed:g}x', controller.phase.label))
                    live_viewer.sync()
                    next_view_sync = time.monotonic() + 1. / 30.
                if controller.done:
                    if finished_time is None:
                        finished_time = float(data.time)
                    if data.time - finished_time >= 1.0:
                        break
                mujoco.mj_step(model, data)
                pacer.wait(data.time)
            renderer.update_scene(data, camera="global_a")
            cv2.imwrite(str(output / "final.png"),
                        cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
        report["phases"] = transitions
        report["evaluation"] = evaluator.result(data, controller)
        if precision:
            met = report['evaluation']['placement_xy_error_m'] <= report['precision_xy_goal_m']
            report['precision_goal_met'] = met
            if require_precision_goal:
                report['evaluation']['checks']['precision_xy_goal'] = met
                report['evaluation']['success'] = report['evaluation']['success'] and met
        breakdown = placement_diagnostics.result(data, controller)
        report['error_breakdown'] = breakdown
        write_json(output / 'error-breakdown.json', breakdown)
        write_json(output / 'diagnostic-trace.json', placement_diagnostics.rows)
        (output / 'error-breakdown.md').write_text(markdown_report(breakdown))
        report["success"] = report["evaluation"]["success"] and report["failure_code"] is None
        if not report["success"] and report["failure_code"] is None:
            report["failure_code"] = "VERIFICATION_FAILED" if controller.done else "RUN_TIMEOUT"
        report["controller_failure"] = controller.failed
        report['seating'] = {'confirmed': controller.seating_confirmed, 'samples': controller.seating_samples,
                             'confirmation': controller.seating_confirmation,
                             'source': 'simulated table contact normal force and robot TCP velocity; no object pose targeting'}
        print(json.dumps(report["evaluation"], indent=2))
    except Exception as error:
        report["failure_code"] = report["failure_code"] or "PIPELINE_ERROR"
        report["error"] = str(error)
    finally:
        if recorder is not None:
            try:
                recorder.close()
            except Exception as error:
                report["success"] = False
                report["failure_code"] = report["failure_code"] or "RECORDING_ERROR"
                report["recording_error"] = str(error)
        report["simulation_seconds"] = float(data.time)
        np.savez_compressed(output/'final-state.npz',qpos=data.qpos,qvel=data.qvel,ctrl=data.ctrl,time=data.time)
        report['elapsed_simulation_seconds'] = float(data.time-start_time)
        if monitor is not None:
            report['camera_preview'] = {'resolution': [640, 360],
                                        'requested_simulation_fps': camera_fps,
                                        'frame_count': monitor.status['sequence']}
        write_json(output / "trace.json", trace)
        write_json(output / "result.json", report)
    print(f"success={report['success']} code={report['failure_code']}; {output.resolve()}")
    if live_viewer is not None and owns_ui:
        status = 'SUCCESS' if report['success'] else report['failure_code']
        if monitor is not None:
            monitor.update(model, data, status, force=True)
            monitor.finish(status + ' · 任务结束（画面停在最终状态）')
        live_viewer.set_texts((None, None, status, 'Run finished. Close window to exit.'))
        while live_viewer.is_running():
            live_viewer.sync()
            time.sleep(.05)
        live_viewer.close()
    if monitor is not None and owns_ui:
        monitor.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--object", choices=[n for n, _ in scene.COMPONENTS], default="lens_1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hide-target-marker", action="store_true",
                        help="Negative test: remove the selected object's top marker")
    parser.add_argument("--record-gif", action="store_true", help="Requires ffmpeg")
    parser.add_argument('--coarse-only', action='store_true', help='Run the previous global-only baseline')
    parser.add_argument('--view', action='store_true', help='Live viewer; on macOS launch with mjpython')
    parser.add_argument('--speed',type=positive_speed,default=1.,help='Live-view speed; physics timestep is unchanged')
    parser.add_argument('--no-camera-monitor', action='store_true', help='Disable the live camera player/dashboard')
    parser.add_argument('--camera-display', choices=['native', 'browser'], default='native',
                        help='Live camera player type (default native desktop window)')
    parser.add_argument('--placement-strategy', choices=['baseline', 'settled'], default='baseline',
                        help='Compare original placement with support-confirmed settling and slower opening')
    parser.add_argument('--precision', action='store_true', help='Require 0.3 mm / 0.5 degree at grasp and placement endpoints')
    parser.add_argument('--camera-fps', type=int, choices=range(1, 11), default=5,
                        help='Camera preview sampling rate in simulation time (default 5)')
    parser.add_argument('--invalid-wrist-depth', action='store_true', help='Negative test: invalidate wrist depth')
    parser.add_argument('--coarse-offset-mm', type=float, nargs=2, default=(0., 0.), metavar=('DX', 'DY'),
                        help='Explicit test injection into coarse XY estimate')
    args = parser.parse_args()
    if args.view and sys.platform == 'darwin' and not os.environ.get('MJPYTHON_BIN'):
        # Invoke explicitly with Python to support virtualenv paths with spaces.
        launcher = Path(sys.executable).with_name('mjpython')
        os.execv(sys.executable, [sys.executable, str(launcher), '-m',
                                 'simulation.vision_pick_place', *sys.argv[1:]])
    result = run(args.seed, args.object, args.output, args.hide_target_marker, args.record_gif,
                 args.coarse_only, args.invalid_wrist_depth, args.coarse_offset_mm, args.view,
                 not args.no_camera_monitor, args.camera_fps, args.camera_display, args.placement_strategy, args.precision,speed=args.speed)
    raise SystemExit(0 if result["success"] else 1)
