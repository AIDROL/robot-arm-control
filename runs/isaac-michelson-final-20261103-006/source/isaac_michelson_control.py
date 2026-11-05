"""Image-targeted Cartesian control; all motion/contacts are native PhysX."""
import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from simulation.task_space_control import progress, clamp_norm


DOWN = Rotation.from_matrix([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])


def upright_marker_rotation(points):
    """Least-squares in-plane orientation from all four ordered square corners."""
    points = np.asarray(points)
    along_x = points[1] - points[0] + points[2] - points[3]
    along_y = points[0] - points[3] + points[1] - points[2]
    axis = along_x[:2] + np.array([along_y[1], -along_y[0]])
    return Rotation.from_euler("z", np.arctan2(axis[1], axis[0])).as_matrix()


def detect_side_markers(rgb, diagnostics=None):
    """Decode current camera pixels even under the gripper's near-field shadow."""
    import cv2
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshConstant = 2.
    params.minOtsuStdDev = 2.
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    result = detector.detectMarkers(gray)
    processing = "raw_gray"
    if result[1] is None or 14 not in result[1].flatten():
        normalized = cv2.createCLAHE(clipLimit=10., tileGridSize=(8, 8)).apply(gray)
        result = detector.detectMarkers(normalized)
        processing = "local_contrast_CLAHE_10_8x8"
    if diagnostics is not None:
        diagnostics["processing"] = processing
    return result


class Experiment:
    def __init__(self, mechanics, feedback_action_limit=60):
        self.sim = mechanics
        self.m, self.d = mechanics.model, mechanics.data
        self.names = [f"joint{i}" for i in range(1, 8)]
        self.qids = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in self.names])
        self.dofids = np.array([self.m.jnt_dofadr[self.m.joint(n).id] for n in self.names])
        self.home = self.d.qpos[self.qids].copy()
        self.command = self.home.copy()
        self.grip_value = 0.
        self.active = None
        self.phase = "settle"
        self.events, self.trajectory, self.placements = [], [], []
        self.contact_telemetry = []
        self.socket_telemetry = []
        self.monitor_socket = False
        self.metrics = None
        self.root_names = {self.m.body(n).id: n for n in ("link_base", "fat", "lens_1", "pinhole", "lens_2", "beam_splitter", "mirror")}
        self.component_names = {"lens_1", "pinhole", "lens_2", "beam_splitter", "mirror"}
        self.pads = {self.m.geom(f"{s}_finger_pad_{i}").id: s for s in ("left", "right") for i in (1, 2)}
        self.table = self.m.geom("workbench").id
        self.unsafe = []
        self.jacp, self.jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        self.tcp = self.m.site("pinch_center").id
        self.max_fk_error = 0.
        self.max_transmission_error = 0.
        self.feedback_action_limit = feedback_action_limit
        self.trajectory.append((self.sim.time, self.phase, self.d.qpos.copy(),
                                np.array(list(self.sim.body_poses.values())), self.command.copy()))

    def event(self, label, **details):
        self.phase = label
        row = {"time_s": self.sim.time, "stage": label, **details}
        self.events.append(row)
        with (self.sim.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print("ISAAC_EVENT", json.dumps(row), flush=True)
        print("ISAAC_TIMING", self.sim.time, self.sim.profile, flush=True)

    def root(self, geom):
        body = int(self.m.geom_bodyid[geom])
        while body and body not in self.root_names:
            body = int(self.m.body_parentid[body])
        return self.root_names.get(body)

    def force(self, name, other=None):
        return sum(f for a, b, f, _ in self.sim.contacts
                   if (self.root(a) == name and (other is None or b == other))
                   or (self.root(b) == name and (other is None or a == other)))

    def pads_on(self, name):
        pads = set()
        for a, b, force, _ in self.sim.contacts:
            if force > .001:
                if self.root(a) == name and b in self.pads:
                    pads.add(self.pads[b])
                if self.root(b) == name and a in self.pads:
                    pads.add(self.pads[a])
        return pads

    def tick(self):
        self.sim.step(render=self.sim.view and self.sim.steps % 33 == 0)
        if self.monitor_socket:
            seating = self.socket_seating()
            self.socket_telemetry.append((self.sim.time, *seating))
            if any(not (.0035 <= seating[k] <= .0095 and seating[k + 1] < .0015) for k in (0, 2)):
                raise RuntimeError("FAT_SHAFT_DISLODGED_DURING_OPTICAL_FEEDBACK:" + str(seating))
        for a, b, force, _ in self.sim.contacts:
            if force < .001:
                continue
            root_a, root_b = self.root(a), self.root(b)
            if root_a in self.component_names and root_b in self.component_names and root_a != root_b:
                self.unsafe.append([self.sim.time, self.phase, "component/component", root_a, root_b])
            if self.active != "fat" and "fat" in (root_a, root_b) and (
                    root_a in self.component_names or root_b in self.component_names):
                self.unsafe.append([self.sim.time, self.phase, "component/staged tool", root_a, root_b])
            if self.root(a) == self.root(b) == "link_base":
                self.unsafe.append([self.sim.time, self.phase, "arm self-contact", self.m.geom(a).name, self.m.geom(b).name])
            for robot, other in ((a, b), (b, a)):
                if self.root(robot) != "link_base" or self.m.geom_bodyid[robot] == self.m.body("link_base").id:
                    continue
                if other == self.table:
                    self.unsafe.append([self.sim.time, self.phase, "arm/table"])
                elif self.root(other) in self.root_names.values() and self.root(other) != "link_base":
                    if self.root(other) != self.active or robot not in self.pads:
                        self.unsafe.append([self.sim.time, self.phase, "arm/" + str(self.root(other)), self.m.geom(robot).name])
            if self.active == "fat":
                for tool, fixture in ((a, b), (b, a)):
                    if self.root(tool) != "fat" or self.root(fixture) not in ("lens_1", "pinhole", "lens_2", "beam_splitter", "mirror"):
                        continue
                    tn, fn = self.m.geom(tool).name or "", self.m.geom(fixture).name or ""
                    intended = any(tn == f"fat_{side}_key" and fn.startswith(side)
                                   and any(tag in fn for tag in ("socket", "lead")) for side in ("lower", "upper"))
                    if not intended:
                        self.unsafe.append([self.sim.time, self.phase, "tool/" + str(self.root(fixture)), tn, fn])
        if self.unsafe:
            raise RuntimeError("UNEXPECTED_PHYSX_CONTACT:" + str(self.unsafe[-1]))
        if self.active:
            pad_forces = {"left": 0., "right": 0.}
            for a, b, force, _ in self.sim.contacts:
                for pad, obj in ((a, b), (b, a)):
                    if pad in self.pads and self.root(obj) == self.active:
                        pad_forces[self.pads[pad]] += force
            engagement = self.engagement() if self.active == "fat" else set()
            z = self.sim.body_poses[self.m.body(self.active).id][2, 3]
            link = self.sim.body_poses[self.m.body("link7").id]
            tcp_position = link[:3, 3] + link[:3, :3] @ np.array([0., 0., .14])
            gap = np.linalg.norm(self.sim.body_poses[self.m.body(self.active).id][:3, 3] - tcp_position)
            self.contact_telemetry.append((self.sim.time, self.phase, self.active,
                pad_forces["left"], pad_forces["right"], self.force(self.active, self.table),
                "lower" in engagement, "upper" in engagement, z, gap))
        if self.metrics is not None and self.active:
            position = self.sim.body_poses[self.m.body(self.active).id][:3, 3]
            self.metrics["max_lift_m"] = max(self.metrics["max_lift_m"], float(position[2] - self.metrics["initial_z_m"]))
            carrying = (": lift", ": transfer", ": prealign", ": insert", ": seat", ": drive") if self.active == "fat" else (": lift", ": transfer")
            if self.phase.endswith(carrying):
                self.metrics["carrying_samples"] += 1
                self.metrics["bilateral_samples"] += self.pads_on(self.active) == {"left", "right"}
            if self.phase.endswith(": transfer") and self.force(self.active, self.table) > .02:
                raise RuntimeError("CARRIED_OBJECT_TABLE_CONTACT")
        if self.sim.steps % 25 == 0:
            native = self.sim.site_pose("pinch_center")
            error = np.linalg.norm(native[:3, 3] - self.d.site_xpos[self.tcp])
            self.max_fk_error = max(self.max_fk_error, float(error))
            if error > .0001:
                raise RuntimeError(f"PHYSX_KINEMATIC_CALIBRATION_MISMATCH:{error}")
            for axis, knob in (("pitch", "lower"), ("yaw", "upper")):
                mirror = float(self.d.joint("mirror_" + axis).qpos[0])
                screw = float(self.d.joint(knob + "_knob_joint").qpos[0])
                residual = abs(mirror - .0016 * screw)
                self.max_transmission_error = max(self.max_transmission_error, residual)
                # Transport/contact impulses are a dynamic diagnostic. Optical
                # measurements separately require a settled <=2 urad residual.
                if residual > .002:
                    raise RuntimeError("PHYSX_SCREW_LEVER_CONSTRAINT_MISMATCH:" + str(residual))
            self.trajectory.append((self.sim.time, self.phase, self.d.qpos.copy(),
                                    np.array(list(self.sim.body_poses.values())), self.command.copy()))

    def servo(self, goal, orientation, dt):
        actual = self.sim.site_pose("pinch_center")
        rotation = Rotation.from_matrix(actual[:3, :3])
        linear = clamp_norm(5. * (goal - actual[:3, 3]), .12)
        angular = clamp_norm(4. * (orientation * rotation.inv()).as_rotvec(), .8)
        mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.tcp)
        jac = np.vstack([self.jacp[:, self.dofids], self.jacr[:, self.dofids]])
        inverse = jac.T @ np.linalg.solve(jac @ jac.T + .04**2 * np.eye(6), np.eye(6))
        measured = self.d.qpos[self.qids]
        velocity = inverse @ np.r_[linear, angular]
        velocity += (np.eye(7) - np.linalg.pinv(jac, rcond=1e-6) @ jac) @ (.08 * (self.home - measured))
        velocity *= min(1., .8 / max(abs(velocity).max(), 1e-12))
        self.command = np.clip(self.command + velocity * dt, measured - .07, measured + .07)
        bounds = np.array([self.m.jnt_range[self.m.joint(n).id] for n in self.names])
        self.command = np.clip(self.command, bounds[:, 0] + .005, bounds[:, 1] - .005)
        self.sim.command(self.names, self.command)

    def move(self, label, goal, orientation, duration=2., grip=None, wait=True, support_stop=False,
             position_tolerance=.0005, orientation_tolerance=np.deg2rad(.5)):
        self.event(label, commanded_position_m=np.asarray(goal).tolist())
        start = self.sim.site_pose("pinch_center")
        start_R = Rotation.from_matrix(start[:3, :3])
        delta = (orientation * start_R.inv()).as_rotvec()
        initial_grip = self.grip_value
        target_grip = initial_grip if grip is None else grip
        started = self.sim.time
        for i in range(round((duration + 8.) / self.sim.dt)):
            elapsed = self.sim.time - started
            fraction, _ = progress(elapsed, duration)
            if i % 5 == 0:
                self.servo(start[:3, 3] + fraction * (goal - start[:3, 3]),
                           Rotation.from_rotvec(fraction * delta) * start_R, self.sim.dt * 5)
                self.grip_value = initial_grip + fraction * (target_grip - initial_grip)
                self.sim.grip(self.grip_value)
            self.tick()
            if support_stop and elapsed > duration * .7 and self.force(self.active, self.table) > .05:
                self.command = self.d.qpos[self.qids].copy()
                self.sim.command(self.names, self.command)
                self.event(label + " supported", table_force_N=self.force(self.active, self.table))
                return
            if elapsed >= duration:
                actual = self.sim.site_pose("pinch_center")
                error = float(np.linalg.norm(goal - actual[:3, 3]))
                angle = float((orientation * Rotation.from_matrix(actual[:3, :3]).inv()).magnitude())
                if not wait or (error < position_tolerance and angle < orientation_tolerance):
                    return
        raise RuntimeError(f"PHYSX_POSE_TIMEOUT:{label}:{error:.6f}m:{angle:.6f}rad")

    def hold(self, seconds):
        target = self.sim.site_pose("pinch_center")
        rotation = Rotation.from_matrix(target[:3, :3])
        for i in range(round(seconds / self.sim.dt)):
            if i % 5 == 0:
                self.servo(target[:3, 3], rotation, self.sim.dt * 5)
            self.tick()

    def seat(self, name, orientation):
        """Bounded contact-guided seating; no object pose enters the target."""
        goal = self.sim.site_pose("pinch_center")[:3, 3].copy()
        initial_z = goal[2]
        stable = 0
        self.event(name + ": settle")
        for i in range(round(4. / self.sim.dt)):
            force = self.force(name, self.table)
            stable = stable + 1 if force > .05 else 0
            if stable * self.sim.dt >= .25:
                self.event(name + ": support confirmed", table_force_N=force,
                           commanded_extra_descent_m=initial_z - goal[2])
                return
            if i % 5 == 0:
                if force < .05:
                    goal[2] -= .0005 * self.sim.dt * 5
                if initial_z - goal[2] > .003:
                    break
                self.servo(goal, orientation, self.sim.dt * 5)
            self.tick()
        raise RuntimeError("STABLE_TABLE_SUPPORT_NOT_ESTABLISHED:" + name)

    def placement(self, name, marker, grip_only=False):
        self.active = name
        self.metrics = {"max_lift_m": 0., "carrying_samples": 0, "bilateral_samples": 0,
                        "initial_z_m": float(self.sim.body_poses[self.m.body(name).id][2, 3])}
        observed = self.sim.capture("before-" + name)
        if marker not in observed:
            raise RuntimeError("FRESH_PICKUP_MARKER_MISSING:" + name)
        # Staged modules have the documented 0 rad slot orientation. Position
        # comes from image PnP; the known supporting plane defines resting Z.
        pickup = observed[marker][:3, 3].copy()
        pickup[2] = .874
        grasp = pickup + [0., 0., .0381]
        target = self.sim.targets[name]
        yaw = Rotation.from_matrix(target[:3, :3]).as_euler("xyz")[2]
        placed_R = Rotation.from_euler("z", yaw) * DOWN
        from simulation import wrist_camera
        from simulation.housing_profile import STL
        coarse = observed[marker].copy()
        coarse[:3, :3] = np.eye(3)
        pregrasp = grasp + [0., 0., .13]
        marker_world = pickup + [0., 0., STL.top_z]
        camera_position = pregrasp + DOWN.apply(wrist_camera.POSITION - [0., 0., .14])
        forward = DOWN.apply(wrist_camera.E_FROM_D[:3, 2])
        distance = float(forward @ (marker_world - camera_position))
        pregrasp[2] += (wrist_camera.PREGRASP_DEPTH_M - distance) / -forward[2]
        self.move(name + ": pregrasp", pregrasp, DOWN, 3., grip=0.)
        rgb, depth, K, W_C = self.sim.capture_sensor("wrist_rgbd", name + "-wrist", depth=True)
        fine, overlay = wrist_camera.refine(rgb, depth, K, W_C, coarse, marker, self.sim.time, STL)
        import cv2
        cv2.imwrite(str(self.sim.output / (name + "-wrist-detection.png")), overlay)
        (self.sim.output / (name + "-wrist.json")).write_text(json.dumps(fine, indent=2))
        if not fine["valid"]:
            raise RuntimeError("ISAAC_WRIST_REFINEMENT_FAILED:" + fine["reason"])
        grasp += np.asarray(fine["T_world_object"])[:3, 3] - pickup
        self.move(name + ": approach", grasp, DOWN, 2.)
        self.move(name + ": close", grasp, DOWN, 1.5, grip=120., wait=False)
        self.move(name + ": lift", grasp + [0., 0., .24], DOWN, 3.)
        if grip_only:
            self.hold(.5)
            settled = self.check_transmission()
            fraction = self.metrics["bilateral_samples"] / max(1, self.metrics["carrying_samples"])
            report = {"id": name, "scope": "physical grip/lift gate only", **self.metrics,
                      "bilateral_fraction": fraction, "settled_screw_lever_error_rad": settled,
                      "success": bool(self.metrics["max_lift_m"] > .1 and fraction > .8)}
            self.placements.append(report)
            return report
        self.move(name + ": transfer", target[:3, 3] + [0., 0., .2781], placed_R, 4.5)
        self.move(name + ": lower", target[:3, 3] + [0., 0., .0361], placed_R, 3., support_stop=True)
        self.seat(name, placed_R)
        self.sim.magnets.add(name)
        self.event("MAGNET_FORCE_ENABLED", component=name, downward_force_N=30.,
                   scope="contact-supported release attraction proxy, not a magnetic field model")
        # Remove actuator squeeze preload before the opening ramp. The target
        # uses measured robot joint angles, never object truth.
        measured_grip = np.mean([float(self.d.joint(side + "_driver_joint").qpos[0])
                                 for side in ("left", "right")])
        self.grip_value = float(np.clip(measured_grip / .00333, 0., 255.))
        self.sim.grip(self.grip_value)
        self.move(name + ": release", self.sim.site_pose("pinch_center")[:3, 3], placed_R,
                  1.5, grip=0., wait=False)
        self.move(name + ": retreat", target[:3, 3] + [0., 0., .25], placed_R, 2.5)
        self.hold(.3)
        actual = self.sim.body_poses[self.m.body(name).id]
        xy = float(np.linalg.norm(actual[:2, 3] - target[:2, 3]))
        tilt = float(np.arccos(np.clip(actual[2, 2], -1., 1.)))
        orientation_error = Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec()
        bilateral = self.metrics["bilateral_samples"] / max(1, self.metrics["carrying_samples"])
        record = {"id": name, **self.metrics, "bilateral_fraction": bilateral,
                  "xy_error_m": xy, "tilt_rad": tilt, "actual_pose": actual.tolist(),
                  "orientation_error_rotvec_rad": orientation_error.tolist(),
                  "success": bool(xy < .010 and tilt < np.deg2rad(5.) and self.metrics["max_lift_m"] > .1 and bilateral > .8)}
        self.placements.append(record)
        self.event("PLACEMENT_EVALUATED", **record)
        if not record["success"]:
            raise RuntimeError("PHYSX_PLACEMENT_ACCEPTANCE_FAILED:" + name)
        if self.force(name, self.table) < .02 or self.pads_on(name):
            raise RuntimeError("BASE_LATCH_REQUIRES_RELEASED_TABLE_SUPPORT:" + name)
        attained = self.sim.latch(name)
        self.sim.magnets.discard(name)
        self.hold(.2)
        if np.linalg.norm(self.sim.body_poses[self.m.body(name).id][:3, 3] - attained[:3, 3]) > .0001:
            raise RuntimeError("BASE_LATCH_MOVED_THE_ACHIEVED_POSE:" + name)
        self.event("BASE_LATCHED_AT_ACHIEVED_POSE", component=name, pose=attained.tolist())
        self.sim.capture("after-" + name)
        self.metrics = None
        return record

    def save(self):
        if self.socket_telemetry:
            values = np.array(self.socket_telemetry)
            np.savez_compressed(self.sim.output / "physx-socket-seating.npz", time_s=values[:, 0],
                lower_depth_m=values[:, 1], lower_radial_m=values[:, 2],
                upper_depth_m=values[:, 3], upper_radial_m=values[:, 4])
        if not self.trajectory:
            return
        np.savez_compressed(self.sim.output / "physx-trajectory.npz",
                            time_s=np.array([r[0] for r in self.trajectory]),
                            phase=np.array([r[1] for r in self.trajectory]),
                            measured_qpos=np.array([r[2] for r in self.trajectory]),
                            body_poses=np.array([r[3] for r in self.trajectory]),
                            body_names=np.array([self.m.body(i).name for i in self.sim.rigid_ids]),
                            arm_commands=np.array([r[4] for r in self.trajectory]),
                            native_joint_names=np.array(list(self.sim.joint_handles)),
                            native_joint_qpos_indices=np.array([self.m.jnt_qposadr[self.m.joint(n).id]
                                                               for n in self.sim.joint_handles]),
                            physics_dt_s=self.sim.dt)
        if self.contact_telemetry:
            rows = self.contact_telemetry
            np.savez_compressed(self.sim.output / "physx-contacts.npz",
                time_s=np.array([r[0] for r in rows]), phase=np.array([r[1] for r in rows]),
                active=np.array([r[2] for r in rows]),
                left_normal_N=np.array([r[3] for r in rows]), right_normal_N=np.array([r[4] for r in rows]),
                table_normal_N=np.array([r[5] for r in rows]),
                lower_socket_contact=np.array([r[6] for r in rows]), upper_socket_contact=np.array([r[7] for r in rows]),
                active_body_z_m=np.array([r[8] for r in rows]), active_body_to_tcp_m=np.array([r[9] for r in rows]))

    def optical_sample(self, label, n=1024):
        from simulation.michelson_assembly_optics import ROLES, from_poses
        self.await_optical_settling(label)
        poses = {role: self.sim.site_pose(name + "_optical_axis") for role, name in ROLES.items()}
        folder = self.sim.output / "optics" / label
        folder.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(folder / "native-state.npz", time_s=self.sim.time,
            qpos=self.d.qpos.copy(), body_names=np.array([self.m.body(i).name for i in self.sim.rigid_ids]),
            body_poses=np.array(list(self.sim.body_poses.values())),
            joint_names=np.array(list(self.sim.joint_handles)),
            joint_qpos_indices=np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in self.sim.joint_handles]),
            command_names=np.array(list(self.sim.last_commands)), commands=np.array(list(self.sim.last_commands.values())))
        observation, intensity = from_poses(poses, n=n, capture_folder=folder, config=self.sim.config)
        observation.update(time_s=self.sim.time, geometry_source="native PhysX rigid-body poses + fixed optical-plane calibration")
        (folder / "measurement.json").write_text(json.dumps(observation, indent=2))
        (folder / "actual-optical-planes.json").write_text(json.dumps({r: T.tolist() for r, T in poses.items()}, indent=2))
        np.save(folder / "intensity-W-m2.npy", intensity)
        print("ISAAC_OPTICS", label, observation["valid"], observation["visibility"], flush=True)
        return observation

    def await_optical_settling(self, label):
        """Wait for a contact transient; never measure outside the 2 urad gate."""
        try:
            self.check_transmission()
            return
        except RuntimeError as initial:
            reason = str(initial)
        started = self.sim.time
        stable = 0
        for _ in range(100):
            self.hold(.02)
            try:
                residuals = self.check_transmission()
                stable += 1
            except RuntimeError:
                stable = 0
                continue
            if stable >= 3:
                record = {"label": label, "initial_failure": reason,
                          "wait_s": self.sim.time - started, "settled_residuals_rad": residuals,
                          "threshold_rad": 2e-6}
                (self.sim.output / (label + "-settling.json")).write_text(json.dumps(record, indent=2))
                print("ISAAC_OPTICAL_SETTLED", json.dumps(record), flush=True)
                return
        raise RuntimeError("OPTICAL_TRANSMISSION_SETTLING_TIMEOUT:" + reason)

    def check_transmission(self):
        residuals = {}
        for axis, knob in (("pitch", "lower"), ("yaw", "upper")):
            mirror = float(self.d.joint("mirror_" + axis).qpos[0])
            screw = float(self.d.joint(knob + "_knob_joint").qpos[0])
            residuals[axis] = abs(mirror - .0016 * screw)
        if max(residuals.values()) > 2e-6:
            raise RuntimeError("UNSETTLED_SCREW_LEVER_FOR_MEASUREMENT:" + str(residuals))
        return residuals

    def engagement(self):
        engaged = set()
        for a, b, force, _ in self.sim.contacts:
            if force < .005:
                continue
            for tool, socket in ((a, b), (b, a)):
                tool_name = self.m.geom(tool).name or ""
                socket_name = self.m.geom(socket).name or ""
                for label in ("lower", "upper"):
                    # Entry chamfers guide insertion but do not establish a
                    # torque-bearing hex engagement. Require one of six walls.
                    prefix = label + "_socket_"
                    wall = socket_name.startswith(prefix) and socket_name[len(prefix):].isdigit()
                    if tool_name == "fat_" + label + "_key" and wall:
                        engaged.add(label)
        return engaged

    def socket_seating(self):
        """Independent native-geometry safety readback, never a servo target.

        A shaft remains inserted while its torque-bearing wall unloads through
        backlash. Positive normal force is not an insertion-depth sensor.
        """
        values = []
        for label in ("lower", "upper"):
            driver = self.sim.body_poses[self.m.body(label + "_driver").id]
            knob = self.sim.body_poses[self.m.body(label + "_knob").id]
            tip = driver[:3, 3] + driver[:3, :3] @ np.array([0., self.sim.config["driver_half_length_m"], 0.])
            local = knob[:3, :3].T @ (tip - knob[:3, 3])
            values.extend([float(local[1]), float(np.linalg.norm(local[[0, 2]]))])
        return values

    def tool_observation(self, fixture, index):
        import cv2
        from simulation.housing_profile import STL
        rgb, _, K, _ = self.sim.capture_sensor("fat_rgb", f"fat-side-{index:03d}")
        processing = {}
        corners, ids, _ = detect_side_markers(rgb, processing)
        if ids is None or 14 not in ids.flatten():
            raise RuntimeError("ISAAC_FAT_SIDE_MARKER_MISSING")
        points = corners[list(ids.flatten()).index(14)].reshape(4, 2).astype(float)
        half = .012 / 2
        square = np.array([[-half, half, 0.], [half, half, 0.], [half, -half, 0.], [-half, -half, 0.]])
        ok, rv, tv = cv2.solvePnP(square, points, K, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or tv[2, 0] <= 0:
            raise RuntimeError("ISAAC_FAT_SIDE_POSE_INVALID")
        C_M = np.eye(4)
        C_M[:3, :3], C_M[:3, 3] = cv2.Rodrigues(rv)[0], tv.ravel()
        O_M = np.eye(4)
        O_M[:3, :3], O_M[:3, 3] = STL.side_rotation.as_matrix(), STL.side_center
        _, _, F_Cusd = self.sim.sensors["fat_rgb"]
        F_C = F_Cusd @ np.diag([1., -1., -1., 1.])
        W_F = fixture @ O_M @ np.linalg.inv(C_M) @ np.linalg.inv(F_C)
        G_F = np.linalg.inv(self.sim.site_pose("pinch_center")) @ W_F
        (self.sim.output / f"fat-side-{index:03d}.json").write_text(json.dumps({
            "time_s": self.sim.time, "source": "fresh Isaac side-marker pixels + global marker pose + robot FK",
            "adaptive_threshold_constant": 2., "minimum_otsu_stddev": 2.,
            **processing,
            "K": K.tolist(), "corners_px": points.tolist(), "T_camera_marker": C_M.tolist(),
            "T_world_fixture_from_top_image": fixture.tolist(), "T_tool_camera_optical": F_C.tolist(),
            "T_gripper_tool_from_image": G_F.tolist()}, indent=2))
        self.tool_grasp = G_F
        return G_F

    def tool_goal(self, phase):
        if not hasattr(self, "tool_grasp") or phase.label == "fat: park":
            return phase.position, phase.orientation
        W_F = np.eye(4)
        W_F[:3, :3] = (phase.orientation * DOWN.inv()).as_matrix()
        W_F[:3, 3] = phase.position - W_F[:3, :3] @ np.array([0., 0., .018])
        W_G = W_F @ np.linalg.inv(self.tool_grasp)
        if np.linalg.norm(W_G[:3, 3] - phase.position) > .010:
            raise RuntimeError("FAT_VISUAL_CORRECTION_TOO_LARGE")
        return W_G[:3, 3], Rotation.from_matrix(W_G[:3, :3])

    def insert_tool(self, phase, fixture):
        self.event("fat: insert")
        start = self.sim.site_pose("pinch_center")
        start_R = Rotation.from_matrix(start[:3, :3])
        started = self.sim.time
        last_vision = -1.
        both = 0
        for i in range(round(13. / self.sim.dt)):
            elapsed = self.sim.time - started
            if elapsed - last_vision >= .5:
                self.tool_observation(fixture, self.side_index)
                self.side_index += 1
                last_vision = elapsed
            goal, orientation = self.tool_goal(phase)
            s, _ = progress(elapsed, 5.)
            if i % 5 == 0:
                delta = (orientation * start_R.inv()).as_rotvec()
                self.servo(start[:3, 3] + s * (goal - start[:3, 3]),
                           Rotation.from_rotvec(s * delta) * start_R, self.sim.dt * 5)
            self.tick()
            both = both + 1 if self.engagement() == {"lower", "upper"} else 0
            if both * self.sim.dt > .15:
                self.event("FAT_BILATERAL_ENGAGEMENT_CONFIRMED")
                return
        raise RuntimeError("FAT_PHYSX_BILATERAL_ENGAGEMENT_TIMEOUT")

    def fat(self):
        from simulation.fat_mechanics import plan
        from simulation.michelson_assembly_optics import PhaseGradientSearch
        self.event("FRESH_POST_ASSEMBLY_VISION")
        observed = self.sim.capture("post-assembly", high_resolution=True)
        if 4 not in observed:
            raise RuntimeError("POST_ASSEMBLY_FIXTURE_MARKER_MISSING")
        fixture = observed[4]
        phases = plan(self.d, fixture_pose=fixture, staging_xy=self.sim.config["fat_staging_xy_m"],
                      shaft_extension=self.sim.config["driver_half_length_m"] - .016)
        initial = self.sim.body_poses[self.m.body("fat").id].copy()
        self.active = "fat"
        self.metrics = {"max_lift_m": 0., "carrying_samples": 0, "bilateral_samples": 0,
                        "initial_z_m": float(initial[2, 3])}
        self.side_index = 0
        history = [self.optical_sample("initial")]
        result = {"success": False, "history": history, "contact_source": "native PhysX contact-report impulses"}
        self.fat_result = result
        for phase in phases:
            if phase.label == "fat: insert":
                self.insert_tool(phase, fixture)
            elif phase.label == "fat: seat":
                actual = self.sim.site_pose("pinch_center")
                goal = actual[:3, 3] + fixture[:3, :3] @ np.array([0., .003, 0.])
                self.move(phase.label, goal, Rotation.from_matrix(actual[:3, :3]), 2., wait=False)
                # Fixed, bounded physical seating motion. Passive shaft travel
                # can compress at the chamfer even after wall contact begins.
                # Rock the finite-force drives while holding the attained TCP;
                # native socket truth only vetoes the subsequent drive phase.
                baseline = np.array([float(self.d.joint(label + "_motor_joint").qpos[0])
                                     for label in ("lower", "upper")])
                held = self.sim.site_pose("pinch_center")
                held_R = Rotation.from_matrix(held[:3, :3])
                for i in range(round(4. / self.sim.dt)):
                    if i % 5 == 0:
                        self.sim.command(["lower_motor_joint", "upper_motor_joint"],
                            baseline + .18 * np.sin(2 * np.pi * .5 * i * self.sim.dt))
                        self.servo(held[:3, 3], held_R, self.sim.dt * 5)
                    self.tick()
                self.sim.command(["lower_motor_joint", "upper_motor_joint"], baseline)
                self.hold(.5)
                self.event("FAT_BOUNDED_SEATING_COMPLETED", rocking_amplitude_rad=.18,
                           rocking_duration_s=4., safety_readback=self.socket_seating())
            elif phase.label == "fat: drive":
                self.event(phase.label)
                policy = PhaseGradientSearch()
                policy.commands = np.array([float(self.d.joint(label + "_motor_joint").qpos[0])
                                            for label in ("lower", "upper")])
                hold = self.sim.site_pose("pinch_center")
                orientation = Rotation.from_matrix(hold[:3, :3])
                start = self.sim.time
                last_sample = -10.
                accepted = False
                contact_samples = 0
                self.monitor_socket = True
                for i in range(round(190. / self.sim.dt)):
                    seating = self.socket_seating()
                    if any(not (.0035 <= seating[k] <= .0095 and seating[k + 1] < .0015) for k in (0, 2)):
                        raise RuntimeError("FAT_SHAFT_DISLODGED_DURING_OPTICAL_FEEDBACK:" + str(seating))
                    if self.engagement() == {"lower", "upper"}:
                        contact_samples += 1
                    if self.sim.time - last_sample >= 3.:
                        observation = self.optical_sample(f"feedback-{len(history):03d}")
                        if self.feedback_action_limit < 60 and policy.actions >= self.feedback_action_limit:
                            history.append({**observation, "decision": "development_probe_limit", "motor_commands_rad": policy.commands.tolist()})
                            raise RuntimeError("DEVELOPMENT_FEEDBACK_PROBE_LIMIT")
                        status, commands = policy.decide(observation)
                        history.append({**observation, "decision": status, "motor_commands_rad": commands.tolist()})
                        if status == "accept":
                            accepted = True
                            break
                        if status != "move":
                            raise RuntimeError("OPTICAL_FEEDBACK_" + status.upper())
                        self.sim.command(["lower_motor_joint", "upper_motor_joint"], commands)
                        last_sample = self.sim.time
                    if i % 5 == 0:
                        self.servo(hold[:3, 3], orientation, self.sim.dt * 5)
                    self.tick()
                if not accepted:
                    raise RuntimeError("OPTICAL_FEEDBACK_TIME_LIMIT")
                self.monitor_socket = False
                result.update(bilateral_socket_samples=contact_samples, drive_seconds=self.sim.time - start)
            elif phase.label == "fat: withdraw":
                # Retain the attained robot orientation. A stale grasp estimate
                # must not introduce a rotation while the hex shafts are loaded.
                actual = self.sim.site_pose("pinch_center")
                self.move(phase.label, actual[:3, 3] - .040 * fixture[:3, 1],
                          Rotation.from_matrix(actual[:3, :3]), phase.duration)
            else:
                if phase.label in ("fat: prealign", "fat: return lift"):
                    if phase.label == "fat: return lift":
                        self.tool_observation(fixture, self.side_index)
                        self.side_index += 1
                goal, orientation = self.tool_goal(phase)
                self.move(phase.label, goal, orientation, phase.duration, grip=phase.grip,
                          wait=phase.wait_for_pose, support_stop=phase.label == "fat: return lower",
                          position_tolerance=phase.position_tolerance,
                          orientation_tolerance=phase.orientation_tolerance)
                if phase.label == "fat: prealign":
                    # The hex vertices start on socket-wall normals. Rotate
                    # 30 degrees to present flats before axial insertion.
                    # Use finite native drives in free space, never a pose snap.
                    self.sim.command(["lower_motor_joint", "upper_motor_joint"], [np.pi / 6] * 2)
                    self.hold(.7)
                    encoder = [float(self.d.joint(label + "_motor_joint").qpos[0])
                               for label in ("lower", "upper")]
                    self.event("FAT_HEX_PHASE_PREALIGNED", motor_encoder_rad=encoder)
                    if np.max(np.abs(np.array(encoder) - np.pi / 6)) > .02:
                        raise RuntimeError("FAT_HEX_PHASE_PREALIGN_FAILED")
                if phase.label == "fat: return lower":
                    self.seat("fat", orientation)
        self.event("POST_RETURN_SETTLING")
        self.hold(2.)
        final = self.optical_sample("released-512", 512)
        fine = self.optical_sample("released-1024", 1024)
        error = float(np.linalg.norm(self.sim.body_poses[self.m.body("fat").id][:3, 3] - initial[:3, 3]))
        fraction = self.metrics["bilateral_samples"] / max(1, self.metrics["carrying_samples"])
        result.update(after_release=final, released_grid_check=fine, fat_return_error_m=error,
                      bilateral_grip_fraction=fraction, grid_visibility_difference=abs(final["visibility"] - fine["visibility"]))
        result["success"] = bool(error < .01 and fraction > .8 and final["valid"] and fine["valid"]
                                  and min(final["visibility"], fine["visibility"]) >= self.sim.config["accept_visibility"]
                                  and result["grid_visibility_difference"] < .005 and not self.unsafe)
        (self.sim.output / "fat-result.json").write_text(json.dumps(result, indent=2))
        if not result["success"]:
            raise RuntimeError("POST_RETURN_PHYSX_MICHELSON_ACCEPTANCE_FAILED")
        return result
