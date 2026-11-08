"""Physical grasp/carry/place of the selected STL at a declared known pose.

Object motion comes only from gravity and contacts: no mocap, weld, attachment
or object pose updates during the run. This is a mechanics test, not vision.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

import cv2
import mujoco
import mujoco_menagerie as menagerie
import numpy as np
from scipy.spatial.transform import Rotation

from . import line_assembly as base
from . import selected_housing as housing
from .task_space_control import CartesianController, Phase, GifRecorder
from .vision_pick_place import reserve_output_directory

SOURCE_XY = np.array([.18, .42])
TARGET_XY = np.array([-.16, .35])
GRIP_COMMAND = 100.  # below full closure; bilateral contact is checked before lift


def make_scene():
    spec = mujoco.MjSpec.from_file(str(menagerie.get('ufactory_xarm7').path()/'scene.xml'))
    spec.modelname = 'selected_STL_physical_grasp'
    spec.body('link_base').pos = [0, 0, base.TABLE_Z]
    spec.body('link_base').quat = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    for geom in list(spec.worldbody.geoms):
        if not geom.name and geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            spec.delete(geom)
    spec.body('xarm_gripper_base_link').add_site(name='pinch_center', pos=[0, 0, .14], size=[.003])
    for side in ['left', 'right']:
        for number in [1, 2]:
            spec.geom(f'{side}_finger_pad_{number}').friction = [2., .01, .0001]
    spec.worldbody.add_geom(name='workbench', type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=[0, .13, base.TABLE_Z-.045], size=[.58, .52, .045],
                            friction=[1., .005, .0001], solref=[.005, 1], rgba=[.25, .28, .31, 1])
    base.add_optical_table_surface(spec)
    spec.worldbody.add_geom(name='housing_target', type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=[*TARGET_XY, base.TABLE_Z+.0003], size=[.038, .034, .0002],
                            rgba=[.3, .8, .4, .5], contype=0, conaffinity=0, mass=0)
    body = spec.worldbody.add_body(name='housing', pos=[*SOURCE_XY, base.TABLE_Z-housing.BOTTOM])
    body.add_freejoint(name='housing_free')
    housing.add_housing(spec, body)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
    spec.visual.quality.offsamples = 2
    return spec.compile()


def phases(model, data):
    home = data.site('pinch_center').xpos.copy()
    rotation = Rotation.from_matrix(data.site('pinch_center').xmat.reshape(3, 3))
    height = base.TABLE_Z-housing.BOTTOM+housing.GRIP_Z
    pick = np.r_[SOURCE_XY, height]
    place = np.r_[TARGET_XY, height]
    result = []
    for label, position, duration, grip, wait in [
        ('pregrasp', pick+[0, 0, .14], 3., 0., True),
        ('approach', pick, 3., 0., True),
        ('close', pick, 2., GRIP_COMMAND, False),
        ('lift', pick+[0, 0, .16], 3., GRIP_COMMAND, True),
        ('transfer', place+[0, 0, .16], 4., GRIP_COMMAND, True),
        ('lower', place, 3., GRIP_COMMAND, True),
        ('settle', place, .8, GRIP_COMMAND, False),
        ('release', place, 2.5, 0., False),
        ('retreat', place+[0, 0, .15], 3., 0., True),
        ('park', home, 3., 0., True),
    ]:
        result.append(Phase(f'housing: {label}', np.array(position), rotation, duration, grip, wait,
                            position_tolerance=.0005, orientation_tolerance=np.deg2rad(.5)))
    return result


class ContactAudit:
    def __init__(self, model):
        self.body = model.body('housing').id
        self.geoms = {i for i in range(model.ngeom)
                      if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or '').startswith('housing_contact_')}
        self.pads = {model.geom(f'{side}_finger_pad_{n}').id: side
                     for side in ['left', 'right'] for n in [1, 2]}
        self.table = model.geom('workbench').id
        self.arm = set()
        robot_root = model.body('link_base').id
        for i in range(model.ngeom):
            b = int(model.geom_bodyid[i])
            while b and b != robot_root:
                b = int(model.body_parentid[b])
            if b == robot_root:
                self.arm.add(i)
        self.pad_sides = set()
        self.groove_sides = set()
        self.support_force = 0.
        self.bilateral_since = None
        self.transfer_total = self.transfer_bilateral = 0
        self.transfer_groove_bilateral = 0
        self.offsets = []
        self.transfer_clearances = []
        self.max_lift = 0.
        self.min_clearance = float('inf')
        self.max_penetration = 0.
        self.max_penetration_event = None
        self.unsafe = []
        self.stable_since = None
        self.released_duration = 0.

    def observe(self, model, data, controller):
        p = data.xpos[self.body]
        R = data.xmat[self.body].reshape(3, 3)
        # Exact outer bounds for an upright housing, conservative under tilt.
        centre = p+R@np.array([0, 0, .05])
        bottom = centre[2]-abs(R[2])@np.array([.036, .032, .054])
        clearance = float(bottom-base.TABLE_Z)
        self.min_clearance = min(self.min_clearance, clearance)
        self.max_lift = max(self.max_lift, clearance)
        sides, groove = set(), set()
        self.support_force = 0.
        force = np.zeros(6)
        for index, contact in enumerate(data.contact[:data.ncon]):
            a, b = map(int, contact.geom)
            pair = {a, b}
            if self.table in pair and pair & self.arm:
                if len(self.unsafe) < 10:
                    self.unsafe.append([float(data.time), controller.phase.label, 'arm/table'])
            if not pair & self.geoms:
                continue
            if -float(contact.dist) > self.max_penetration:
                self.max_penetration = -float(contact.dist)
                self.max_penetration_event = {'time': float(data.time), 'phase': controller.phase.label,
                                             'geoms': [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                                                       for g in [a, b]]}
            mujoco.mj_contactForce(model, data, index, force)
            if self.table in pair:
                self.support_force += max(0., float(force[0]))
            other = b if a in self.geoms else a
            if other in self.arm and other not in self.pads and force[0] > .01:
                if len(self.unsafe) < 10:
                    self.unsafe.append([float(data.time), controller.phase.label,
                                        f'non-pad robot/housing: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)}'])
            if other in self.pads and force[0] > .01:
                side = self.pads[other]
                sides.add(side)
                local = R.T@(contact.pos-p)
                if (abs(abs(local[0])-.033) < .001 and abs(local[1]) < .016
                        and .072 < local[2] < .1045):
                    groove.add(side)
        self.pad_sides.update(sides)
        self.groove_sides.update(groove)
        self.bilateral_since = (float(data.time) if self.bilateral_since is None else self.bilateral_since) if sides == {'left', 'right'} else None
        if controller.phase.label.endswith(': transfer'):
            self.transfer_total += 1
            self.transfer_bilateral += sides == {'left', 'right'}
            self.transfer_groove_bilateral += groove == {'left', 'right'}
            tcp = data.site_xpos[controller.site]
            tool_R = data.site_xmat[controller.site].reshape(3, 3)
            self.offsets.append(tool_R.T@(p-tcp))
            self.transfer_clearances.append(clearance)
        joint = model.joint('housing_free').id
        start = model.jnt_dofadr[joint]
        velocity = data.qvel[start:start+6]
        stable = (controller.done and not sides and self.support_force > .02
                  and np.linalg.norm(velocity[:3]) < .005 and np.linalg.norm(velocity[3:]) < .05)
        self.stable_since = (float(data.time) if self.stable_since is None else self.stable_since) if stable else None
        self.released_duration = 0. if self.stable_since is None else float(data.time-self.stable_since)

    def result(self, model, data, controller):
        p = data.body('housing').xpos
        R = data.body('housing').xmat.reshape(3, 3)
        drift = max((float(np.linalg.norm(v-self.offsets[0])) for v in self.offsets), default=float('inf'))
        fraction = self.transfer_bilateral/max(1, self.transfer_total)
        groove_fraction = self.transfer_groove_bilateral/max(1, self.transfer_total)
        xy = float(np.linalg.norm(p[:2]-TARGET_XY))
        checks = {
            'controller_completed': controller.done and controller.failed is None,
            'both_finger_sides_contacted': self.pad_sides == {'left', 'right'},
            'both_recesses_contacted': self.groove_sides == {'left', 'right'},
            'lifted_above_50_mm': self.max_lift > .05,
            'transfer_bilateral_contact_above_80_percent': fraction > .8,
            'transfer_both_recess_contacts_above_80_percent': groove_fraction > .8,
            'transfer_clearance_above_50_mm': bool(self.transfer_clearances) and min(self.transfer_clearances) > .05,
            'transfer_slip_below_5_mm': drift < .005,
            'placement_xy_below_10_mm': xy < .010,
            'placement_height_below_2_mm': abs(p[2]-(base.TABLE_Z-housing.BOTTOM)) < .002,
            'placement_tilt_below_5_deg': R[2, 2] > np.cos(np.deg2rad(5)),
            'released_and_supported_0_5_s': self.released_duration >= .5,
            'no_unsafe_robot_contacts': not self.unsafe,
            'bottom_penetration_below_1_mm': self.min_clearance > -.001,
            'contact_penetration_below_1_mm': self.max_penetration < .001,
        }
        return {'success': bool(all(checks.values())), 'checks': {k: bool(v) for k, v in checks.items()},
                'failure': controller.failed, 'placement_xy_error_mm': xy*1000,
                'max_lift_mm': self.max_lift*1000, 'transfer_bilateral_fraction': fraction,
                'transfer_both_recess_fraction': groove_fraction,
                'transfer_slip_mm': drift*1000 if np.isfinite(drift) else None,
                'min_bottom_clearance_mm': self.min_clearance*1000,
                'max_contact_penetration_mm': self.max_penetration*1000,
                'max_contact_penetration_event': self.max_penetration_event,
                'released_supported_s': self.released_duration, 'unsafe_contacts': self.unsafe}


class HousingController(CartesianController):
    def __init__(self, model, data):
        self.audit = ContactAudit(model)
        self.seated_since = None
        self.seating_confirmed = False
        super().__init__(model, data, phases=phases(model, data),
                         object_geoms={i: 'housing' for i in self.audit.geoms})

    def observe(self, state):
        self.audit.observe(self.model, state, self)
        if self.audit.unsafe and self.failed is None:
            self.failed = 'UNSAFE_ROBOT_CONTACT'
            self.hold_q = self.command_q.copy()

    def advance(self, state):
        if self.phase.label.endswith(': close'):
            if self.audit.bilateral_since is None or state.time-self.audit.bilateral_since < .2:
                if state.time-self.phase_start_time > 4.:
                    self.failed = 'NO_SUSTAINED_BILATERAL_GRASP'
                    self.hold_q = self.command_q.copy()
                return
        if self.phase.label.endswith(': settle'):
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, state, mujoco.mjtObj.mjOBJ_SITE, self.site, velocity, 0)
            stable = self.audit.support_force > .02 and np.linalg.norm(velocity[3:]) < .002
            self.seated_since = (float(state.time) if self.seated_since is None else self.seated_since) if stable else None
            if self.seated_since is None or state.time-self.seated_since < .3:
                elapsed = float(state.time-self.phase_start_time)
                if elapsed > 5.:
                    self.failed = 'NO_STABLE_TABLE_SUPPORT'
                    self.hold_q = self.command_q.copy()
                    return
                position = self.phase.position.copy()
                nominal_z = base.TABLE_Z-housing.BOTTOM+housing.GRIP_Z
                position[2] = nominal_z-min(.004, max(0., elapsed-.8)*.001)
                self.phases[self.phase_index] = replace(self.phase, position=position)
                return
            self.seating_confirmed = True
            self.phases[self.phase_index+1] = replace(self.phases[self.phase_index+1],
                                                      position=state.site_xpos[self.site].copy())
        super().advance(state)


def run(output, view=False, record=False):
    output = reserve_output_directory(output)
    model = make_scene()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    controller = HousingController(model, data)
    window = None
    recorder = None
    if view:
        from mujoco import viewer
        window = viewer.launch_passive(model, data, show_right_ui=False)
        window.cam.lookat[:] = [0, .30, .92]
        window.cam.distance, window.cam.azimuth, window.cam.elevation = 1.35, 135, -30
    if record:
        recorder = GifRecorder(model, output/'grasp.gif')
        recorder.camera.lookat[:] = [0, .34, .91]
        recorder.camera.distance = 1.25
    trace, events = [], []
    last_phase = None
    next_trace = next_sync = 0.
    finished = None
    while data.time < 90:
        started = time.perf_counter()
        if window is not None and not window.is_running():
            controller.failed = 'VIEWER_CLOSED'
            break
        controller(model, data)
        if controller.phase.label != last_phase:
            events.append({'time': float(data.time), 'phase': controller.phase.label})
            last_phase = controller.phase.label
            if recorder is not None and last_phase.endswith((': lift', ': transfer', ': lower', ': retreat')):
                closeup = mujoco.MjvCamera()
                mujoco.mjv_defaultFreeCamera(model, closeup)
                closeup.lookat[:] = data.body('housing').xpos+[0, 0, .07]
                closeup.distance, closeup.azimuth, closeup.elevation = .35, 65, -15
                recorder.renderer.update_scene(data, camera=closeup)
                cv2.imwrite(str(output/f"{last_phase.split(': ')[1]}-detail.png"),
                            cv2.cvtColor(recorder.renderer.render(), cv2.COLOR_RGB2BGR))
        if data.time >= next_trace:
            trace.append({'time': float(data.time), 'phase': controller.phase.label,
                          'housing_position': data.body('housing').xpos.tolist(),
                          'tcp_position': data.site('pinch_center').xpos.tolist(),
                          'support_force_N': controller.audit.support_force})
            next_trace = float(data.time)+.1
        if recorder is not None:
            recorder.capture(model, data)
        if window is not None and data.time >= next_sync:
            window.set_texts((None, None, 'Physical STL grasp', controller.phase.label))
            window.sync()
            next_sync = float(data.time)+1/30
        if controller.failed:
            break
        if controller.done:
            if finished is None:
                finished = float(data.time)
            if data.time-finished > 2.:
                break
        mujoco.mj_step(model, data)
        if window is not None:
            time.sleep(max(0., model.opt.timestep-(time.perf_counter()-started)))
    result = controller.audit.result(model, data, controller)
    result.update({'simulation_time_s': float(data.time), 'output': str(output),
                   'nominal_mass_kg': housing.NOMINAL_MASS,
                   'grip_command': GRIP_COMMAND,
                   'source_pose': 'declared known initial pose; no vision localization in this mechanics test',
                   'object_motion': 'free joint, gravity and contacts only; no object pose writes during run',
                   'seating_confirmed': controller.seating_confirmed,
                   'collision_parts': len(controller.audit.geoms)})
    (output/'result.json').write_text(json.dumps(result, indent=2))
    (output/'events.json').write_text(json.dumps(events, indent=2))
    (output/'trace.json').write_text(json.dumps(trace, indent=2))
    if recorder is not None:
        recorder.close()
    print(json.dumps(result, indent=2), flush=True)
    if window is not None:
        window.set_texts((None, None, 'PASS' if result['success'] else 'FAILED', 'Close window to exit'))
        while window.is_running():
            window.sync()
            time.sleep(1/30)
        window.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--record-gif', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('runs/housing-grasp'))
    args = parser.parse_args()
    if args.view and sys.platform == 'darwin' and not os.environ.get('MJPYTHON_BIN'):
        os.execv(sys.executable, [sys.executable, str(Path(sys.executable).with_name('mjpython')),
                                 '-m', 'simulation.housing_grasp', *sys.argv[1:]])
    result = run(args.output, args.view, args.record_gif)
    raise SystemExit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()
