"""Two 4K global RGB cameras: rendered ArUco -> PnP -> world pose.

The detector only accepts images and calibration. Simulator object state is
read separately by the diagnostic evaluator, never used as a pose fallback.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from . import line_assembly as scene
except ImportError:
    import line_assembly as scene

WIDTH, HEIGHT = 3840, 2160
TOP_EDGE = 0.024  # black outer square, excludes white margin
SIDE_EDGE = 0.012
TOP_Z = scene.BLOCK_HALF_HEIGHT + 0.0008
SIDE_CENTER = np.array([-scene.BLOCK_HALF_WIDTH - 0.0008, 0, 0.024])
SCREW_CENTER_Z = -0.010  # retain tool spacing, move the pair below the tag
CAMERAS = {
    "global_a": np.array([-0.38, 0.25, 2.15]),
    "global_b": np.array([0.38, 0.25, 2.15]),
}
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def add_marker(body, name: str, marker_id: int, edge: float,
               center: np.ndarray, rotation: np.ndarray) -> None:
    """Exact dictionary pattern as flat visual cells; local +Z points out.

    Image row zero maps to marker +Y (OpenCV's top-left corner convention).
    Using cells avoids mesh UV interpolation, flips and texture bleeding.
    """
    pattern = cv2.aruco.generateImageMarker(DICTIONARY, marker_id, 6)
    quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]

    def tile(label, xy, half, rgb, depth):
        body.add_geom(
            name=f"{name}_{label}", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=center + rotation @ np.array([*xy, depth]), quat=quat,
            size=[half, half, 0.00005], rgba=[rgb, rgb, rgb, 1],
            mass=0, contype=0, conaffinity=0,
        )

    tile("white", [0, 0], edge * 4 / 6, 1, -0.0001)
    for row in range(6):
        for col in range(6):
            if pattern[row, col] == 0:
                tile(f"{row}_{col}", [(col - 2.5) * edge / 6,
                                      (2.5 - row) * edge / 6], edge / 12,
                     0, -0.00005)


def configure_vision_scene(spec: mujoco.MjSpec) -> None:
    try:
        from .wrist_camera import add_camera
    except ImportError:
        from wrist_camera import add_camera
    add_camera(spec)
    spec.visual.global_.offwidth = WIDTH
    spec.visual.global_.offheight = HEIGHT
    # 4K multisampling can be expensive; detection uses the native output pixels.
    spec.visual.quality.offsamples = 0
    # Keep the 45 g proxies from visibly sinking into the table while pinched.
    # Both interacting geoms use this value (MuJoCo mixes contact parameters).
    spec.geom("workbench").solref = [0.005, 1]
    for index, (name, _) in enumerate(scene.COMPONENTS):
        body = spec.body(name)
        spec.geom(f"{name}_collision").solref = [0.005, 1]
        for geom in list(body.geoms):
            if geom.name != f"{name}_collision" and not geom.name.endswith("_screw"):
                spec.delete(geom)
        add_marker(body, f"{name}_top_aruco", index, TOP_EDGE,
                   np.array([0, 0, TOP_Z]), np.eye(3))
        side_rotation = Rotation.from_euler("xyz", scene.SIDE_MARKER_EULER).as_matrix()
        add_marker(body, f"{name}_side_aruco", index + 10, SIDE_EDGE,
                   SIDE_CENTER, side_rotation)
        # Only this vision scene uses the new schematic layout; the historical
        # FAT controller and its tuning geometry remain unchanged.
        spec.site(f"{name}_side_marker").pos = SIDE_CENTER
        for label, _, dz in scene.FAT_KEY_OFFSETS:
            spec.geom(f"{name}_{label}_screw").pos[2] = SCREW_CENTER_Z + dz
            spec.site(f"{name}_{label}_screw_center").pos[2] = SCREW_CENTER_Z + dz
    lookat = np.array([0.0, 0.27, scene.BLOCK_Z])
    for name, position in CAMERAS.items():
        zaxis = position - lookat  # MuJoCo looks along camera -Z
        zaxis /= np.linalg.norm(zaxis)
        xaxis = np.cross([0, 1, 0], zaxis)
        xaxis /= np.linalg.norm(xaxis)
        yaxis = np.cross(zaxis, xaxis)
        spec.worldbody.add_camera(name=name, pos=position,
                                  xyaxes=[*xaxis, *yaxis], fovy=45)


def calibration(model, data, camera: str, width: int, height: int):
    camera_id = model.camera(camera).id
    focal = height / (2 * np.tan(np.deg2rad(model.cam_fovy[camera_id]) / 2))
    K = np.array([[focal, 0, (width - 1) / 2],
                  [0, focal, (height - 1) / 2], [0, 0, 1]])
    T = np.eye(4)
    # MuJoCo camera: right, up, backward. OpenCV: right, down, forward.
    T[:3, :3] = data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
    T[:3, 3] = data.cam_xpos[camera_id]
    return K, T


def locate(rgb: np.ndarray, K: np.ndarray, world_from_camera: np.ndarray,
           marker_plane_z: float):
    """No access to MuJoCo model/data or object ground truth."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 4
    corners, ids, rejected = cv2.aruco.ArucoDetector(DICTIONARY, params).detectMarkers(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
    estimates = []
    for points, marker_id in zip(corners, [] if ids is None else ids.flatten()):
        if not 0 <= marker_id < 5:  # side markers belong to the future FAT pipeline
            continue
        h = TOP_EDGE / 2
        object_points = np.array([[-h, h, 0], [h, h, 0],
                                  [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
        image_points = points.reshape(4, 2).astype(np.float64)
        _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            object_points, image_points, K, np.zeros(5),
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        candidates = []
        for rvec, tvec in zip(rvecs, tvecs):
            camera_from_marker = np.eye(4)
            camera_from_marker[:3, :3] = cv2.Rodrigues(rvec)[0]
            camera_from_marker[:3, 3] = tvec.ravel()
            world_from_marker = world_from_camera @ camera_from_marker
            if tvec[2, 0] <= 0 or world_from_marker[2, 2] < 0.8:
                continue
            projected = cv2.projectPoints(object_points, rvec, tvec, K, np.zeros(5))[0]
            error = float(np.sqrt(np.mean(np.sum(
                (projected.reshape(4, 2) - image_points) ** 2, axis=1))))
            candidates.append((error, world_from_marker, rvec, tvec))
        if not candidates:
            continue
        error, T, rvec, tvec = min(candidates, key=lambda item: item[0])
        cv2.drawFrameAxes(overlay, K, np.zeros(5), rvec, tvec, TOP_EDGE)
        raw_pnp = T.copy()
        # Explicit prior: known-height upright housings on a fixed workbench.
        # Intersect observed corner rays with that plane; do not read object state.
        rays = (world_from_camera[:3, :3] @ np.linalg.inv(K)
                @ np.column_stack((image_points, np.ones(4))).T).T
        origin = world_from_camera[:3, 3]
        if np.any(np.abs(rays[:, 2]) < 1e-8):
            continue
        distances = (marker_plane_z - origin[2]) / rays[:, 2]
        if np.any(distances <= 0):
            continue
        world_corners = origin + distances[:, None] * rays
        axis = world_corners[1] - world_corners[0] + world_corners[2] - world_corners[3]
        yaw = np.arctan2(axis[1], axis[0])
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
        T[:3, 3] = world_corners.mean(axis=0)
        estimated_edge = float(np.mean(np.linalg.norm(
            np.roll(world_corners, 1, axis=0) - world_corners, axis=1)))
        scale_ok = abs(estimated_edge - TOP_EDGE) / TOP_EDGE <= 0.1
        valid = error <= 2.0 and scale_ok
        estimates.append({"id": int(marker_id), "T_world_marker": T.tolist(),
                          "raw_pnp_T_world_marker": raw_pnp.tolist(),
                          "method": "known_height_plane_and_upright_yaw",
                          "estimated_edge_m": estimated_edge,
                          "reprojection_error_px": error,
                          "corners_px": image_points.tolist(),
                          "valid": valid,
                          "reason": "OK" if valid else "REPROJECTION_OR_SCALE_ERROR"})
    return estimates, overlay, len(rejected)


def capture_observations(model, data, output: Path, width=WIDTH, height=HEIGHT):
    """Capture images and fixed camera calibration, without object state reads."""
    marker_plane_z = scene.TABLE_Z + scene.BLOCK_HALF_HEIGHT + TOP_Z
    cameras = {}
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for camera in CAMERAS:
            K, T = calibration(model, data, camera, width, height)
            start = perf_counter()
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render().copy()
            render_time = perf_counter() - start
            start = perf_counter()
            estimates, overlay, rejected = locate(rgb, K, T, marker_plane_z)
            detection_time = perf_counter() - start
            cv2.imwrite(str(output / f"{camera}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(output / f"{camera}_detections.png"), overlay)
            cameras[camera] = {
                "K": K.tolist(), "distortion": [0] * 5, "T_world_camera": T.tolist(),
                "simulation_time": float(data.time), "render_seconds": render_time,
                "detection_seconds": detection_time, "estimates": estimates,
                "rejected_candidates": rejected,
            }
    return cameras


def select_objects(cameras: dict) -> dict:
    """Camera selection and marker-to-object transform; no simulator inputs."""
    observations = {i: [] for i in range(5)}
    for camera, frame in cameras.items():
        for estimate in frame["estimates"]:
            if estimate["valid"] and estimate["id"] in observations:
                observations[estimate["id"]].append((camera, estimate))
    objects = {}
    for index, (name, _) in enumerate(scene.COMPONENTS):
        candidates = observations[index]
        result = {"valid": False, "reason": "MARKER_NOT_DETECTED", "frame_id": "world"}
        if candidates:
            conflict = 0.0
            angle_conflict = 0.0
            if len(candidates) == 2:
                a, b = [np.array(e["T_world_marker"]) for _, e in candidates]
                conflict = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
                angle_conflict = float(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())
            camera, estimate = min(candidates, key=lambda x: x[1]["reprojection_error_px"])
            T_marker = np.array(estimate["T_world_marker"])
            T_body = T_marker.copy()
            T_body[:3, 3] -= T_marker[:3, :3] @ np.array([0, 0, TOP_Z])
            valid = bool(conflict <= 0.020 and angle_conflict <= np.deg2rad(15))
            result = {"valid": valid, "reason": "OK" if valid else "CAMERA_CONFLICT",
                      "source_camera": camera, "frame_id": "world",
                      "T_world_object": T_body.tolist(),
                      "camera_position_disagreement_m": conflict,
                      "camera_rotation_disagreement_rad": angle_conflict,
                      "timestamp": cameras[camera]["simulation_time"]}
        objects[name] = result
    return objects


def run(seed: int, output: Path, width: int, height: int):
    output.mkdir(parents=True, exist_ok=False)
    model = scene.make_scene(seed=seed, include_fat=True, vision=True,
                             target_xs=scene.FAT_TARGET_XS)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    cameras = capture_observations(model, data, output, width, height)
    report = {"seed": seed, "resolution": [width, height], "units": "metres/radians",
              "scope": "static global RGB localization; no robot control",
              "dictionary": "DICT_4X4_50", "top_edge_m": TOP_EDGE,
              "side_edge_m": SIDE_EDGE, "cameras": cameras,
              "objects": select_objects(cameras),
              "localization_prior": {"upright": True,
                  "marker_plane_z_m": scene.TABLE_Z + scene.BLOCK_HALF_HEIGHT + TOP_Z,
                  "source": "configured table and housing dimensions"}}
    # Truth access is isolated here, after all image-derived estimates exist.
    for name, result in report["objects"].items():
        if "T_world_object" in result:
            T_body = np.array(result["T_world_object"])
            result["diagnostic_position_error_m"] = float(np.linalg.norm(
                T_body[:3, 3] - data.body(name).xpos))
            result["diagnostic_rotation_error_rad"] = float(Rotation.from_matrix(
                T_body[:3, :3].T @ data.body(name).xmat.reshape(3, 3)).magnitude())
    report["success"] = all(x["valid"] and x["diagnostic_position_error_m"] <= 0.020
                            for x in report["objects"].values())
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, item in report["objects"].items():
        print(name, item["reason"],
              f'{item.get("diagnostic_position_error_m", float("nan")) * 1000:.2f} mm')
    print(f"Images and result: {output.resolve()}")
    if not report["success"]:
        raise RuntimeError("Global localization failed; inspect result.json and images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    args = parser.parse_args()
    if not 0 < args.width <= WIDTH or not 0 < args.height <= HEIGHT:
        parser.error("Resolution must be positive and at most 3840 x 2160")
    run(args.seed, args.output, args.width, args.height)
