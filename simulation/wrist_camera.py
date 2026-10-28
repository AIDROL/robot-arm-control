"""Ideal wrist RGB-D proxy; not a calibrated RealSense D435 sensor model.

Mount/FOV are provisional simulation choices. In particular, this model does
not implement the D435's resolution-dependent minimum depth or stereo noise.
Image/depth estimator cannot access object state.
"""
from pathlib import Path

import cv2
import mujoco
import numpy as np

WIDTH, HEIGHT = 1280, 720
POSITION = np.array([0.065, 0., 0.040])
back = POSITION - np.array([0., 0., 0.250])
back /= np.linalg.norm(back)
right = np.cross([0., 1., 0.], back)
right /= np.linalg.norm(right)
up = np.cross(back, right)
E_FROM_D = np.eye(4)
E_FROM_D[:3, :3] = np.column_stack((right, up, back)) @ np.diag([1, -1, -1])
E_FROM_D[:3, 3] = POSITION


def add_camera(spec):
    spec.body('xarm_gripper_base_link').add_camera(
        name='wrist_rgbd', pos=POSITION, xyaxes=[*right, *up], fovy=65)


def refine(rgb, depth, K, world_from_camera, coarse, marker_id, timestamp):
    from .global_camera import DICTIONARY, TOP_Z
    failure = {"valid": False, "frame_id": "world", "timestamp": timestamp,
               "source_camera": "wrist_rgbd"}
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(DICTIONARY, params).detectMarkers(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
    if ids is None or marker_id not in ids.flatten():
        return {**failure, 'reason': 'FINE_MARKER_NOT_DETECTED'}, overlay
    quad = corners[list(ids.flatten()).index(marker_id)].reshape(4, 2)
    if depth.shape != rgb.shape[:2]:
        return {**failure, 'reason': 'DEPTH_RGB_SIZE_MISMATCH'}, overlay
    centre = cv2.perspectiveTransform(np.array([[[.5, .5]]], np.float32),
        cv2.getPerspectiveTransform(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32), quad))[0, 0]
    mask = np.zeros(depth.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.rint(quad.mean(axis=0) + .6 * (quad - quad.mean(axis=0))).astype(np.int32), 1)
    yy, xx = np.nonzero(mask)
    z = depth[yy, xx]
    valid = np.isfinite(z) & (z > .03) & (z < 1.)
    count = int(np.sum(valid))
    if count < 50 or count < .6 * len(z):
        return {**failure, 'reason': 'INVALID_DEPTH', 'valid_depth_pixels': count}, overlay
    rays = (np.linalg.inv(K) @ np.column_stack((xx[valid], yy[valid], np.ones(count))).T).T
    points = rays * z[valid, None]
    # Fit the observed top surface in camera coordinates; no tabletop height prior.
    centroid = np.median(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    residuals = np.abs((points - centroid) @ normal)
    inliers = points[residuals < .0005]
    if len(inliers) < .8 * count:
        return {**failure, 'reason': 'INCONSISTENT_DEPTH_PLANE'}, overlay
    centroid = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vh[-1]
    ray = np.linalg.inv(K) @ np.r_[centre, 1.]
    denominator = normal @ ray
    if abs(denominator) < 1e-6:
        return {**failure, 'reason': 'INVALID_DEPTH_PLANE'}, overlay
    zcentre = float(normal @ centroid / denominator)
    if not .03 < zcentre < 1.:
        return {**failure, 'reason': 'INVALID_DEPTH'}, overlay
    marker_world = world_from_camera[:3, :3] @ (ray * zcentre) + world_from_camera[:3, 3]
    T = np.array(coarse, copy=True)
    T[:3, 3] = marker_world - T[:3, :3] @ np.array([0, 0, TOP_Z])
    delta = float(np.linalg.norm(T[:3, 3] - coarse[:3, 3]))
    cv2.circle(overlay, tuple(np.rint(centre).astype(int)), 5, (0, 0, 255), 2)
    return {**failure, 'valid': delta <= .020,
            'reason': 'OK' if delta <= .020 else 'FINE_COARSE_CONFLICT',
            'T_world_object': T.tolist(), 'valid_depth_pixels': count,
            'plane_inliers': len(inliers), 'centre_depth_m': zcentre,
            'correction_m': delta, 'position_source': 'rendered_metric_depth',
            'orientation_source': 'retained_global_yaw'}, overlay


def capture(model, data, output: Path, coarse, marker_id, invalid_depth=False):
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        renderer.update_scene(data, camera='wrist_rgbd')
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera='wrist_rgbd')
        depth = renderer.render().copy()
    if invalid_depth:
        depth[:] = np.nan
    focal = HEIGHT / (2 * np.tan(np.deg2rad(model.cam_fovy[model.camera('wrist_rgbd').id]) / 2))
    K = np.array([[focal, 0, (WIDTH-1)/2], [0, focal, (HEIGHT-1)/2], [0, 0, 1]])
    tool = data.body('xarm_gripper_base_link')
    W_FROM_E = np.eye(4)
    W_FROM_E[:3, :3], W_FROM_E[:3, 3] = tool.xmat.reshape(3, 3), tool.xpos
    W_FROM_D = W_FROM_E @ E_FROM_D
    result, overlay = refine(rgb, depth, K, W_FROM_D, coarse, marker_id, float(data.time))
    cv2.imwrite(str(output / 'wrist_rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output / 'wrist_detections.png'), overlay)
    np.save(output / 'wrist_depth_m.npy', depth)
    preview = np.uint8(np.clip(np.nan_to_num(depth, nan=1.) / .6, 0, 1) * 255)
    cv2.imwrite(str(output / 'wrist_depth.png'), cv2.applyColorMap(preview, cv2.COLORMAP_TURBO))
    result.update({'K': K.tolist(), 'T_tool_camera': E_FROM_D.tolist(),
                   'T_world_tool': W_FROM_E.tolist(), 'T_world_camera': W_FROM_D.tolist(),
                   'depth_units': 'metres, optical z', 'distortion': [0.] * 5,
                   'sensor_model': 'ideal_rgbd_proxy_not_D435',
                   'hand_eye_source': 'configured_exact_simulation_transform_not_measured'})
    return result


def check_depth(output: Path):
    """Independent known-distance planes verify renderer units and optical Z."""
    import json
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for distance in [.2, .4]:
        model = mujoco.MjModel.from_xml_string(f'''<mujoco>
          <visual><global offwidth="1280" offheight="720"/></visual>
          <worldbody><geom type="plane" size="2 2 .01"/>
          <camera name="test" pos="0 0 {distance}" fovy="65"/></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera='test')
            depth = renderer.render().copy()
        error = float(np.max(np.abs(depth[100:-100, 100:-100] - distance)))
        results.append({'plane_distance_m': distance, 'max_depth_error_m': error})
    report = {'success': all(x['max_depth_error_m'] < .0001 for x in results),
              'units': 'optical Z metres', 'planes': results}
    (output / 'depth_check.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return report['success']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Validate rendered metric depth against known planes')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if check_depth(args.output) else 1)
