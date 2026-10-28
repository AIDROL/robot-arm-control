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
from .task_space_control import CartesianController, GifRecorder


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
    if not (-0.55 <= x <= 0.55 and -0.35 <= y <= 0.65
            and abs(z - scene.BLOCK_Z) < 0.005 and R[2, 2] > 0.999):
        raise ValueError("POSE_OUTSIDE_TABLETOP_PRIOR")
    return T


class Evaluator:
    """Truth-based monitoring only; never generates or changes motion targets."""
    def __init__(self, model, name, target):
        self.name, self.target = name, np.array(target)
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

    def observe(self, model, data, controller):
        p = data.xpos[self.body]
        R = data.xmat[self.body].reshape(3, 3)
        half_extent_z = np.abs(R[2]) @ np.array([
            scene.BLOCK_HALF_WIDTH, scene.BLOCK_HALF_WIDTH, scene.BLOCK_HALF_HEIGHT])
        clearance = float(p[2] - half_extent_z - scene.TABLE_Z)
        if clearance < self.min_bottom_clearance:
            self.min_bottom_clearance = clearance
            self.min_clearance_phase = {"phase": controller.phase.label,
                                        "time": float(data.time)}
        both_pads = set()
        object_geom = model.geom(f"{self.name}_collision").id
        for contact in data.contact[:data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            other = b if a == object_geom else a if b == object_geom else None
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
        good = (controller.done and driver_open and not both_pads
                and np.linalg.norm(p[:2] - self.target[:2]) < 0.010
                and abs(p[2] - scene.BLOCK_Z) < 0.005
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
            "placed_height": abs(p[2] - scene.BLOCK_Z) < 0.005,
            "placed_yaw": abs(yaw) < np.deg2rad(5),
            "placed_tilt": R[2, 2] > np.cos(np.deg2rad(5)),
            "open_and_stable_0_5_seconds": self.last_stable_duration >= 0.5,
            "no_table_penetration": self.min_bottom_clearance > -0.001,
            "no_unsafe_contacts": not controller.unsafe_contacts,
        }
        return {"success": bool(all(checks.values())),
                "checks": {k: bool(v) for k, v in checks.items()},
                "placement_xy_error_m": xy, "placement_yaw_error_rad": float(abs(yaw)),
                "transfer_relative_drift_m": drift,
                "transfer_two_pad_contact_fraction": contact_fraction,
                "transfer_min_clearance_m": float(carry_height) if np.isfinite(carry_height) else None,
                "min_object_table_clearance_m": self.min_bottom_clearance,
                "min_clearance_phase": self.min_clearance_phase,
                "stable_duration_seconds": self.last_stable_duration,
                "unsafe_contacts": controller.unsafe_contacts}


class WristController(CartesianController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending_refinement = False
        self.refined = False

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
        camera_monitor=True, camera_fps=5, camera_display='native'):
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
    controller, recorder, live_viewer = None, None, None
    monitor = None
    trace = []
    model = scene.make_scene(seed=seed, include_fat=True, vision=True,
                             target_xs=scene.FAT_TARGET_XS)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
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
        T[:2, 3] += np.array(coarse_offset_mm) / 1000
        T = require_pose({**estimate, 'T_world_object': T.tolist()}, float(data.time))
        report["coarse_control_pose"] = T.tolist()
        # Isolated diagnostic reads never feed back into the controller.
        report["diagnostics"] = {"coarse_position_error_m": float(np.linalg.norm(
            T[:3, 3] - data.body(name).xpos))}
        index = [n for n, _ in scene.COMPONENTS].index(name)
        target = [scene.FAT_TARGET_XS[index], scene.TARGET_Y, scene.BLOCK_Z]
        report["target_world_m"] = target
        controller_type = CartesianController if coarse_only else WristController
        controller = controller_type(model, data, scene.FAT_TARGET_XS,
                                          object_poses={name: T})
        evaluator = Evaluator(model, name, target)
        report["plan"] = [{"phase": p.label, "position_world_m": p.position.tolist(),
                           "orientation_xyzw": p.orientation.as_quat().tolist(),
                           "gripper_command": p.grip, "duration_seconds": p.duration}
                          for p in controller.phases]
        report["control_started"] = True
        if view:
            from mujoco import viewer
            live_viewer = viewer.launch_passive(model, data, show_right_ui=False)
            live_viewer.cam.lookat[:] = [0., .22, .88]
            live_viewer.cam.distance = 1.65
            live_viewer.cam.azimuth = 135
            live_viewer.cam.elevation = -35
            live_viewer.set_texts((None, None, 'Visual pick/place', 'Global RGB -> wrist RGB-D -> grasp'))
            live_viewer.sync()
            if camera_monitor:
                from .camera_monitor import CameraMonitor
                monitor = CameraMonitor(model, fps=camera_fps, open_browser=camera_display == 'browser')
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
        with mujoco.Renderer(model, height=vision.HEIGHT, width=vision.WIDTH) as renderer:
            while data.time < 120:
                wall_start = time.perf_counter()
                if live_viewer is not None and not live_viewer.is_running():
                    report['failure_code'] = 'VIEWER_CLOSED'
                    break
                controller(model, data)
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
                    controller.apply_refinement(data, refined_pose[:3, 3] - T[:3, 3])
                    if live_viewer is not None and monitor is None:
                        image = cv2.imread(str(output / 'wrist_detections.png'))
                        thumbnail = cv2.cvtColor(cv2.resize(image, (384, 216)), cv2.COLOR_BGR2RGB)
                        live_viewer.set_images((mujoco.MjrRect(20, 20, 384, 216), thumbnail))
                    report['refined_plan'] = [{'phase': p.label, 'position_world_m': p.position.tolist()}
                                               for p in controller.phases]
                if controller.failed:
                    report["failure_code"] = "MOTION_TIMEOUT"
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
                    preview = monitor.update(model, data, controller.phase.label)
                    if preview is not None:
                        live_viewer.set_images((mujoco.MjrRect(20, 20, 320, 180),
                                                cv2.resize(preview, (320, 180))))
                if live_viewer is not None and data.time >= next_view_sync:
                    live_viewer.set_texts((None, None, 'Phase', controller.phase.label))
                    live_viewer.sync()
                    next_view_sync = float(data.time) + 1. / 30.
                if controller.done:
                    if finished_time is None:
                        finished_time = float(data.time)
                    if data.time - finished_time >= 1.0:
                        break
                mujoco.mj_step(model, data)
                if live_viewer is not None:
                    time.sleep(max(0., model.opt.timestep - (time.perf_counter() - wall_start)))
            renderer.update_scene(data, camera="global_a")
            cv2.imwrite(str(output / "final.png"),
                        cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
        report["phases"] = transitions
        report["evaluation"] = evaluator.result(data, controller)
        report["success"] = report["evaluation"]["success"] and report["failure_code"] is None
        if not report["success"] and report["failure_code"] is None:
            report["failure_code"] = "VERIFICATION_FAILED" if controller.done else "RUN_TIMEOUT"
        report["controller_failure"] = controller.failed
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
        if monitor is not None:
            report['camera_preview'] = {'resolution': [640, 360],
                                        'requested_simulation_fps': camera_fps,
                                        'frame_count': monitor.status['sequence']}
        write_json(output / "trace.json", trace)
        write_json(output / "result.json", report)
    print(f"success={report['success']} code={report['failure_code']}; {output.resolve()}")
    if live_viewer is not None:
        status = 'SUCCESS' if report['success'] else report['failure_code']
        if monitor is not None:
            monitor.update(model, data, status, force=True)
            monitor.finish(status + ' · 任务结束（画面停在最终状态）')
        live_viewer.set_texts((None, None, status, 'Run finished. Close window to exit.'))
        while live_viewer.is_running():
            live_viewer.sync()
            time.sleep(.05)
        live_viewer.close()
    if monitor is not None:
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
    parser.add_argument('--no-camera-monitor', action='store_true', help='Disable the live camera player/dashboard')
    parser.add_argument('--camera-display', choices=['native', 'browser'], default='native',
                        help='Live camera player type (default native desktop window)')
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
                 not args.no_camera_monitor, args.camera_fps, args.camera_display)
    raise SystemExit(0 if result["success"] else 1)
