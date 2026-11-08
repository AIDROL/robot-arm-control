"""FAT side-marker RGB pose estimation; no marker/object truth input."""
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

from . import global_camera as vision
from . import line_assembly as scene
from .housing_profile import for_model


def locate(rgb, K, world_from_camera, marker_id, coarse_marker, tabletop=False):
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(vision.DICTIONARY, params).detectMarkers(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
    result = {'valid': False, 'reason': 'FAT_MARKER_NOT_DETECTED', 'source_camera': 'fat_rgb', 'frame_id': 'world'}
    if ids is None or marker_id not in ids.flatten():
        return result, overlay
    points = corners[list(ids.flatten()).index(marker_id)].reshape(4, 2).astype(float)
    h = vision.SIDE_EDGE / 2
    square = np.array([[-h,h,0], [h,h,0], [h,-h,0], [-h,-h,0]])
    _, rvecs, tvecs, _ = cv2.solvePnPGeneric(square, points, K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    candidates = []
    def consider(rvec,tvec):
        if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            return
        T = np.eye(4)
        T[:3,:3], T[:3,3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
        T = world_from_camera @ T
        angle = Rotation.from_matrix(coarse_marker[:3,:3].T @ T[:3,:3]).magnitude()
        delta = float(np.linalg.norm(T[:3,3]-coarse_marker[:3,3]))
        if tvec[2,0] <= 0 or angle > np.deg2rad(25) or delta > .020:
            return
        projected = cv2.projectPoints(square,rvec,tvec,K,np.zeros(5))[0].reshape(4,2)
        error = float(np.sqrt(np.mean(np.sum((projected-points)**2,axis=1))))
        candidates.append((error,T,rvec,tvec))
    for rvec,tvec in zip(rvecs,tvecs):
        consider(rvec,tvec)
    if not candidates:
        # Frontal planar squares can be degenerate for IPPE. Still solve from
        # observed corners, initializing with the image-derived coarse pose.
        initial=np.linalg.inv(world_from_camera)@coarse_marker
        ok,rvec,tvec=cv2.solvePnP(square,points,K,np.zeros(5),
            rvec=cv2.Rodrigues(initial[:3,:3])[0],tvec=initial[:3,3].reshape(3,1),
            useExtrinsicGuess=True,flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:consider(rvec,tvec)
    if not candidates:
        return {**result,'reason':'FAT_POSE_CONFLICT'}, overlay
    error,T,rvec,tvec = min(candidates,key=lambda x:x[0])
    raw_pnp = T.copy()
    method = 'planar_pnp'
    if tabletop:
        # A 12 mm frontal square weakly constrains tilt. The assembly is upright
        # at a configured height; retain its image-derived global yaw and fit XY
        # from the actual observed side corners. No target truth enters this fit.
        camera_from_world = np.linalg.inv(world_from_camera)
        def projected_at(delta):
            pose = coarse_marker.copy()
            pose[:2,3] += delta
            camera_pose = camera_from_world@pose
            rv = cv2.Rodrigues(camera_pose[:3,:3])[0]
            tv = camera_pose[:3,3].reshape(3,1)
            pixels = cv2.projectPoints(square,rv,tv,K,np.zeros(5))[0].reshape(4,2)
            return pose,rv,tv,pixels
        fit = least_squares(lambda delta:(projected_at(delta)[3]-points).ravel(),
                            [0.,0.],bounds=(-.020,.020),max_nfev=60,
                            ftol=1e-12,xtol=1e-12,gtol=1e-12)
        T,rvec,tvec,projected = projected_at(fit.x)
        error = float(np.sqrt(np.mean(np.sum((projected-points)**2,axis=1))))
        if not fit.success or np.linalg.norm(fit.x) > .020 or tvec[2,0] <= 0:
            return {**result,'reason':'FAT_POSE_CONFLICT'}, overlay
        method = 'known_height_upright_xy_fit_retained_global_yaw'
    cv2.drawFrameAxes(overlay,K,np.zeros(5),rvec,tvec,.012)
    return {**result,'valid':error <= 2.,'reason':'OK' if error <= 2. else 'FAT_REPROJECTION_ERROR',
            'T_world_marker':T.tolist(),'raw_pnp_T_world_marker':raw_pnp.tolist(),
            'method':method,'reprojection_error_px':error,'marker_id':marker_id},overlay


def capture(model,data,controller,name,output,hide_marker=False):
    width,height=1280,720
    with mujoco.Renderer(model,width=width,height=height) as renderer:
        renderer.update_scene(data,camera='fat_rgb')
        rgb=renderer.render().copy()
    if hide_marker:
        rgb[:]=255
    focal=height/(2*np.tan(np.deg2rad(model.cam_fovy[model.camera('fat_rgb').id])/2))
    K=np.array([[focal,0,(width-1)/2],[0,focal,(height-1)/2],[0,0,1.]])
    Rg=Rotation.from_matrix(data.site_xmat[controller.site].reshape(3,3))
    Rf=(Rg*controller.relative_rotation).as_matrix()
    pf=data.site_xpos[controller.site]+Rg.apply(controller.relative_body_position)
    camera=model.camera('fat_rgb')
    W_C=np.eye(4)
    W_C[:3,:3]=Rf@Rotation.from_quat(camera.quat[[1,2,3,0]]).as_matrix()@np.diag([1,-1,-1])
    W_C[:3,3]=pf+Rf@camera.pos
    body=controller.object_poses[name]
    coarse=np.eye(4)
    geometry=for_model(model)
    coarse[:3,:3]=body[:3,:3]@geometry.side_rotation.as_matrix()
    coarse[:3,3]=body[:3,3]+body[:3,:3]@geometry.side_center
    index=[n for n,_ in scene.COMPONENTS].index(name)
    result,overlay=locate(rgb,K,W_C,index+10,coarse,tabletop=geometry.is_selected)
    result.update(timestamp=float(data.time),K=K.tolist(),T_world_camera=W_C.tolist())
    cv2.imwrite(str(output/f'{name}_fat_rgb.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output/f'{name}_fat_detection.png'),overlay)
    return result
