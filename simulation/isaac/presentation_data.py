"""One recorded time axis for replay poses, cameras and measured optics."""
from bisect import bisect_right
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


class RecordedExperiment:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.result = json.loads((self.folder / "result.json").read_text())
        if not (self.result["success"] and self.result["stage"] == "full"
                and self.result["started_from_empty_area"]):
            raise ValueError("Presentation requires a successful full empty-area execution")
        with np.load(self.folder / "physx-trajectory.npz") as trace:
            self.times = trace["time_s"].copy()
            self.poses = trace["body_poses"].copy()
            self.names = trace["body_names"].tolist()
            self.phases = trace["phase"].tolist()
            self.qpos = trace["measured_qpos"].copy()
            self.joint_indices = dict(zip(trace["native_joint_names"].tolist(),
                                          trace["native_joint_qpos_indices"].tolist()))
        if len(self.times) < 2 or np.any(np.diff(self.times) <= 0):
            raise ValueError("Replay requires a strictly increasing recorded timeline")
        self.end = float(self.result["time_s"])
        if abs(self.end - self.times[-1]) > .1:
            raise ValueError("Recorded trajectory does not cover the experiment end")
        final_snapshot = self.folder / "optics/released-1024/native-state.npz"
        if self.end > self.times[-1] + 1e-9 and final_snapshot.is_file():
            with np.load(final_snapshot) as final:
                if final["body_names"].tolist() != self.names or abs(float(final["time_s"]) - self.end) > 1e-9:
                    raise ValueError("Final recorded detector state does not match native replay")
                self.times = np.r_[self.times, self.end]
                self.poses = np.concatenate([self.poses, final["body_poses"][None]], axis=0)
                self.qpos = np.concatenate([self.qpos, final["qpos"][None]], axis=0)
                self.phases.append("verified after tool return")
        self.quaternions = Rotation.from_matrix(self.poses[:, :, :3, :3].reshape(-1, 3, 3)).as_quat().reshape(
            len(self.times), len(self.names), 4)
        self.optics = []
        for path in (self.folder / "optics").glob("*/measurement.json"):
            info = json.loads(path.read_text())
            self.optics.append((float(info["time_s"]), path.parent.name, info, path.parent))
        self.optics.sort(key=lambda row: (row[0], row[1] == "released-1024", row[1]))
        self.optical_times = [row[0] for row in self.optics]

    def sample(self, time_s):
        time_s = float(np.clip(time_s, self.times[0], self.end))
        i = min(max(int(np.searchsorted(self.times, time_s, side="right") - 1), 0), len(self.times) - 2)
        alpha = float(np.clip((time_s - self.times[i]) / (self.times[i + 1] - self.times[i]), 0., 1.))
        p = (1 - alpha) * self.poses[i, :, :3, 3] + alpha * self.poses[i + 1, :, :3, 3]
        q0, q1 = self.quaternions[i], self.quaternions[i + 1].copy()
        q1[np.sum(q0 * q1, axis=1) < 0] *= -1
        q = q0 * (1 - alpha) + q1 * alpha
        q /= np.linalg.norm(q, axis=1)[:, None]
        joints = (1 - alpha) * self.qpos[i] + alpha * self.qpos[i + 1]
        return p, q, self.phases[-1] if time_s >= self.end else self.phases[i], joints

    def measurement(self, time_s):
        index = bisect_right(self.optical_times, time_s + 1e-8) - 1
        return self.optics[index] if index >= 0 else None

    def history(self, time_s):
        return self.optics[:bisect_right(self.optical_times, time_s + 1e-8)]


def camera_calibration(folder, body_names):
    """Compile source camera mounts only; no MuJoCo dynamics or rendering."""
    import mujoco
    from simulation.mujoco.michelson_experiment import load_layout, make_scene
    folder = Path(folder)
    config, targets = load_layout(folder / "configuration.json")
    model = make_scene(config, targets)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)

    def matrix(position, quaternion):
        result = np.eye(4)
        result[:3, :3] = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
        result[:3, 3] = position
        return result

    descriptors = {"global": {"path": "/World/GlobalCamera", "horizontal_aperture": 2.0955,
                              "focal_length": 2.}}
    for label, name in (("wrist", "wrist_rgbd"), ("side", "fat_rgb")):
        camera = model.camera(name).id
        body = owner = int(model.cam_bodyid[camera])
        while model.body(owner).name not in body_names:
            owner = int(model.body_parentid[owner])
            if not owner:
                raise ValueError("Camera owner is absent from native trajectory: " + name)
        world = matrix(data.xpos[body], data.xquat[body]) @ matrix(model.cam_pos[camera], model.cam_quat[camera])
        local = np.linalg.inv(matrix(data.xpos[owner], data.xquat[owner])) @ world
        descriptors[label] = {"path": "/World/Bodies/" + model.body(owner).name + "/Presentation/" + name,
            "local_transform": local.tolist(), "horizontal_aperture": 2.0955,
            "focal_length": float(2.0955 * 720 / 1280 / (2 * np.tan(np.deg2rad(model.cam_fovy[camera]) / 2)))}
    reference = json.loads((folder / "fat-side-000.json").read_text())["T_tool_camera_optical"]
    error = float(np.max(np.abs(np.array(descriptors["side"]["local_transform"]) @ np.diag([1., -1., -1., 1.]) - reference)))
    if error > 1e-9:
        raise ValueError("Replay side-camera mount differs from recorded control calibration")
    return descriptors, error


FONT = cv2.FONT_HERSHEY_SIMPLEX


class PresentationComposer:
    def __init__(self, experiment, speed):
        self.experiment, self.speed = experiment, speed
        self._detector_cache = {}

    @staticmethod
    def text(image, line, xy, scale=.55, color=(230, 230, 230)):
        cv2.putText(image, line, xy, FONT, scale, color, 1, cv2.LINE_AA)

    def camera(self, image, title, subtitle=""):
        panel = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA).copy()
        cv2.rectangle(panel, (0, 0), (640, 46), (18, 21, 25), -1)
        self.text(panel, title, (14, 20), .55)
        self.text(panel, subtitle, (14, 39), .38, (160, 178, 190))
        return panel

    def optics_panel(self, time_s, measurement):
        panel = np.full((360, 640, 3), (18, 21, 25), np.uint8)
        self.text(panel, "OPTICAL DETECTOR", (14, 24), .60)
        if measurement is None:
            self.text(panel, "Assembling the optical path", (25, 135), .7)
            self.text(panel, "Measurements begin after all five modules are placed.", (25, 170), .46)
            return panel
        stamp, label, info, folder = measurement
        if label not in self._detector_cache:
            raw = cv2.imread(str(folder / "phase-00-detector16.png"), cv2.IMREAD_UNCHANGED)
            if raw is None or raw.dtype != np.uint16:
                raise ValueError("Missing recorded 16-bit detector frame: " + label)
            levels = np.log1p(1000 * raw.astype(float) / 65535) / np.log1p(1000)
            colored = cv2.applyColorMap(np.uint8(np.clip(levels * 255, 0, 255)), cv2.COLORMAP_INFERNO)
            self._detector_cache[label] = cv2.resize(colored, (300, 300), interpolation=cv2.INTER_AREA)
        panel[45:345, 10:310] = self._detector_cache[label]
        self.text(panel, label, (325, 70), .52, (125, 225, 245))
        self.text(panel, f"Measured t = {stamp:.2f} s", (325, 104), .47)
        self.text(panel, f"Held for {max(0, time_s-stamp):.2f} s", (325, 133), .47)
        self.text(panel, "Recorded phase 0 / 8", (325, 184), .46)
        self.text(panel, "Fixed logarithmic display", (325, 213), .43)
        self.text(panel, "Ideal scalar optics", (325, 265), .46, (165, 180, 190))
        self.text(panel, "Quantized detector pixels", (325, 294), .43, (165, 180, 190))
        return panel

    def metrics_panel(self, time_s, measurement, joints):
        panel = np.full((360, 640, 3), (18, 21, 25), np.uint8)
        self.text(panel, "OPTICAL ADJUSTMENT / RECORDED MEASUREMENTS", (14, 24), .53)
        if measurement:
            info = measurement[2]
            self.text(panel, f"Visibility  {100*info['visibility']:7.3f}%", (20, 63), .85, (125, 240, 165))
            tilt = np.linalg.norm(info["relative_tilt_from_pixels_rad"]) * 1e6
            self.text(panel, f"Pixel wavefront tilt  {tilt:.2f} urad", (20, 94), .52)
            status = "POST-RETURN CHECK / 512 + 1024 grids" if measurement[1].startswith("released") else "FAT feedback measurements"
            self.text(panel, status, (20, 122), .48, (125, 225, 245))
        else:
            self.text(panel, "Waiting for assembled interferometer", (20, 67), .6)
        angles = [float(joints[self.experiment.joint_indices[name + "_knob_joint"]]) for name in ("lower", "upper")]
        self.text(panel, f"Measured knobs  lower {angles[0]:+.3f} / upper {angles[1]:+.3f} rad", (20, 153), .46)
        left, right, top, bottom = 48, 612, 185, 315
        cv2.rectangle(panel, (left, top), (right, bottom), (65, 72, 80), 1)
        threshold_y = bottom - round(.95 * (bottom - top))
        cv2.line(panel, (left, threshold_y), (right, threshold_y), (95, 100, 105), 1)
        self.text(panel, "95%", (left+5, threshold_y+16), .35, (150, 155, 160))
        start = self.experiment.optical_times[0]
        end = self.experiment.end
        points = []
        for stamp, _, info, _ in self.experiment.history(time_s):
            x = left + round(np.clip((stamp - start) / (end - start), 0, 1) * (right-left))
            y = bottom - round(np.clip(info["visibility"], 0., 1.) * (bottom-top))
            points.append((x, y))
        if len(points) > 1:
            cv2.polylines(panel, [np.array(points, np.int32)], False, (90, 220, 250), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(panel, point, 3, (90, 220, 250), -1)
        self.text(panel, "0", (20, bottom+3), .38)
        self.text(panel, "100", (10, top+5), .38)
        self.text(panel, f"{start:.0f}s", (left, 339), .40)
        self.text(panel, "Measurement time", (235, 339), .40)
        self.text(panel, f"{end:.0f}s", (right-40, 339), .40)
        return panel

    def compose(self, frames, depth, time_s):
        _, _, phase, joints = self.experiment.sample(time_s)
        measurement = self.experiment.measurement(time_s)
        canvas = np.full((1080, 1920, 3), 18, np.uint8)
        canvas[:720, :1280] = cv2.resize(frames["overview"], (1280, 720), interpolation=cv2.INTER_AREA)
        cv2.rectangle(canvas, (0, 0), (1280, 62), (18, 21, 25), -1)
        self.text(canvas, "MICHELSON INTERFEROMETER EXPERIMENT", (20, 25), .72)
        self.text(canvas, f"REPLAY  t={time_s:6.2f} / {self.experiment.end:.2f}s  |  {self.speed:g}x  |  {phase}", (20, 51), .54, (155, 210, 240))
        cv2.rectangle(canvas, (0, 687), (1280, 720), (18, 21, 25), -1)
        self.text(canvas, "Recorded successful execution | camera views re-rendered from saved poses | detector measurements recorded", (17, 709), .43, (170, 180, 190))
        canvas[:360, 1280:] = self.camera(frames["global"], "GLOBAL CAMERA", "Continuous replay render / fixed overhead view")
        wrist = self.camera(frames["wrist"], "WRIST RGB-D CAMERA", "Continuous replay render / robot-mounted view")
        if depth is not None:
            valid = np.isfinite(depth) & (depth > 0)
            levels = np.zeros(depth.shape, np.uint8)
            levels[valid] = np.uint8(np.clip((depth[valid]-.05)/.5*255, 0, 255))
            inset = cv2.applyColorMap(cv2.resize(levels, (180, 101)), cv2.COLORMAP_TURBO)
            wrist[-112:-11, -191:-11] = inset
            self.text(wrist, "Depth 0.05-0.55m", (452, 271), .30)
        canvas[360:720, 1280:] = wrist
        canvas[720:, 1280:] = self.camera(frames["side"], "FAT SIDE CAMERA", "Continuous replay render / tool-mounted view")
        canvas[720:, :640] = self.optics_panel(time_s, measurement)
        canvas[720:, 640:1280] = self.metrics_panel(time_s, measurement, joints)
        return canvas
