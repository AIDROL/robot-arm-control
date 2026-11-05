"""Inspect an original STL in MuJoCo, with millimetres converted to metres.

Visual mesh only: no contact hull, moving joints, or optical physics claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .global_camera import add_marker, TOP_EDGE, SIDE_EDGE


SELECTED_STL_SHA256 = '815981fe57a1e078a52855fc3ce756ff0e036d3fc6d6e695a603d51cf7c2ef96'
SOURCE_URL = 'https://github.com/ShrishChou/CAD_Engraver/blob/adad25ca4a27f4480518f8484316d1ef3706c73f/highousing.stl'


def add_selected_interfaces(body, prefix='housing', marker_id=0, z_offset=0.):
    """Coordinates in the unchanged STL's millimetre-to-metre source frame.

    The flat -Y upper panel carries the side tag. Recessed +/-X faces are
    reserved for fingers; site normals point out of each contact surface.
    """
    side_rotation = Rotation.from_euler('x', np.pi/2)
    offset = np.array([0, 0, z_offset])
    add_marker(body, f'{prefix}_top_aruco', marker_id, TOP_EDGE,
               np.array([0, 0, .10235])+offset, np.eye(3))
    add_marker(body, f'{prefix}_side_aruco', marker_id+10, SIDE_EDGE,
               np.array([0, -.03235, .0881])+offset, side_rotation.as_matrix())
    for geom in body.geoms:
        if geom.name.startswith((f'{prefix}_top_aruco', f'{prefix}_side_aruco')):
            geom.group = 1
    for name, pos, rotation in [
        ('housing_top_marker', [0, 0, .10235], Rotation.identity()),
        ('housing_side_marker', [0, -.03235, .0881], side_rotation),
        ('housing_grip_left', [-.033, 0, .0881], Rotation.from_euler('y', -np.pi/2)),
        ('housing_grip_right', [.033, 0, .0881], Rotation.from_euler('y', np.pi/2)),
        ('housing_grasp_center', [0, 0, .0881], Rotation.identity()),
    ]:
        body.add_site(name=name.replace('housing_', prefix+'_'), pos=np.array(pos)+offset, quat=rotation.as_quat()[[3, 0, 1, 2]],
                      size=[.001], rgba=[0, 0, 0, 0])


def verify_selected_interfaces(model, data):
    """Query imported triangle surfaces to check the chosen planes/recesses."""
    shift = data.body('original_housing').xpos
    group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    geom_id = np.array([-1], dtype=np.int32)
    surfaces = {}
    for name, origin, direction, axis, plane in [
        ('flat_marker_face', [0, -.1, .0881], [0, 1, 0], 1, -.032),
        ('left_grip_recess', [-.1, 0, .0881], [1, 0, 0], 0, -.033),
        ('right_grip_recess', [.1, 0, .0881], [-1, 0, 0], 0, .033),
        ('top_marker_face', [0, 0, .2], [0, 0, -1], 2, .102),
    ]:
        direction = np.array(direction, dtype=float)
        origin = np.array(origin)+shift
        distance = mujoco.mj_ray(model, data, origin, direction, group, True, -1, geom_id)
        hit = origin+distance*direction-shift
        if distance < 0 or not np.isclose(hit[axis], plane, atol=1e-6):
            raise RuntimeError(f'Selected STL interface does not match surface: {name}')
        surfaces[name] = {'surface_point_stl_mm': (hit*1000).tolist()}
    surfaces['grip_recess_depth_mm'] = 3.0
    surfaces['grip_contact_spacing_mm'] = 66.0
    surfaces['scope'] = 'geometric finger contact references only; no physical grasp test'
    return surfaces


def load_stl(path):
    """Read triangle bounds for framing; MuJoCo itself imports the STL file."""
    raw = path.read_bytes()
    count = int.from_bytes(raw[80:84], 'little')
    if len(raw) != 84+count*50:
        raise ValueError('Expected a binary STL')
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertex', '<f4', (3, 3)), ('attr', '<u2')])
    points = np.frombuffer(raw[84:], dtype=dtype)['vertex'].reshape(-1, 3)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError('STL has no finite geometry')
    return points.min(axis=0).astype(float)*.001, points.max(axis=0).astype(float)*.001


def make_scene(path, interfaces=True):
    low, high = load_stl(path)
    extent = high-low
    spec = mujoco.MjSpec()
    spec.modelname = 'original_STL_housing'
    spec.stat.extent = max(extent)*3
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1100, 900
    spec.visual.quality.offsamples = 4
    spec.add_texture(name='background', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[.84, .89, .93], rgb2=[.64, .72, .79], width=512, height=512)
    spec.worldbody.add_light(pos=[-.15, -.2, .3], dir=[.4, .5, -1],
                             diffuse=[.8, .8, .8], ambient=[.3, .3, .3])
    spec.worldbody.add_light(pos=[.2, .1, .2], dir=[-.7, -.3, -.6],
                             diffuse=[.5, .5, .5], castshadow=False)
    spec.worldbody.add_geom(name='bench', type=mujoco.mjtGeom.mjGEOM_BOX,
                           pos=[0, 0, -.006], size=[.18, .15, .006],
                           rgba=[.65, .68, .70, 1], group=2)
    spec.add_mesh(name='original_stl', file=str(path.resolve()), scale=[.001]*3,
                  smoothnormal=False)
    # Translation only: centre XY and rest the raw STL's lowest vertex on the bench.
    shift = np.r_[-(low[:2]+high[:2])/2, -low[2]]
    body = spec.worldbody.add_body(name='original_housing', pos=shift)
    body.add_geom(name='original_housing_visual', type=mujoco.mjtGeom.mjGEOM_MESH,
                  meshname='original_stl', rgba=[.23, .36, .46, 1], contype=0, conaffinity=0)
    if interfaces:
        if hashlib.sha256(path.read_bytes()).hexdigest() != SELECTED_STL_SHA256:
            raise ValueError('Interface locations are for the selected highousing.stl; use --bare for other meshes')
        add_selected_interfaces(body)
    lookat = np.array([0, 0, extent[2]/2])
    return spec, lookat, extent, shift


def camera_for(model, lookat, extent, azimuth, elevation):
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = lookat
    camera.distance = float(max(extent)*2.5)
    camera.azimuth, camera.elevation = azimuth, elevation
    return camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stl', type=Path,
                        default=Path(__file__).resolve().parents[2]/'assets/optical_housing/highousing.stl')
    parser.add_argument('--output', type=Path, default=Path('runs/selected-housing-preview'))
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--bare', action='store_true', help='Only show the original mesh, without marker/grasp interfaces')
    args = parser.parse_args()
    if args.view and sys.platform == 'darwin' and not os.environ.get('MJPYTHON_BIN'):
        os.execv(sys.executable, [sys.executable, str(Path(sys.executable).with_name('mjpython')),
                                 '-m', 'simulation.stl_preview', *sys.argv[1:]])
    # Copy an unchanged inspection asset next to the exported XML for portability.
    asset = args.output/'source'/args.stl.name
    asset.parent.mkdir(parents=True, exist_ok=True)
    if args.stl.resolve() != asset.resolve():
        shutil.copyfile(args.stl, asset)
    spec, lookat, extent, shift = make_scene(asset, interfaces=not args.bare)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    report = {'source_url': SOURCE_URL if hashlib.sha256(asset.read_bytes()).hexdigest() == SELECTED_STL_SHA256 else None,
              'source_stl': str(asset.resolve()), 'sha256': hashlib.sha256(asset.read_bytes()).hexdigest(),
              'extent_mm': (extent*1000).tolist(), 'translation_m': shift.tolist(),
              'uniform_scale': .001, 'mesh_vertices': model.nmeshvert, 'mesh_triangles': model.nmeshface,
              'scope': 'original visual mesh; contact disabled; no internal optical mount added'}
    report['markers_added'] = not args.bare
    if not args.bare:
        report['interfaces'] = verify_selected_interfaces(model, data)
        report['marker_ids'] = {'top': 0, 'flat_front': 10}
    (args.output/'import-report.json').write_text(json.dumps(report, indent=2))
    # Keep XML references relative to the output folder, not to /tmp or this computer.
    xml = ET.fromstring(spec.to_xml())
    xml.find("./asset/mesh[@name='original_stl']").set('file', f'source/{asset.name}')
    ET.ElementTree(xml).write(args.output/'housing.xml', encoding='unicode')
    with mujoco.Renderer(model, height=900, width=1100) as renderer:
        for name, azimuth, elevation in [('overview', 65, -25), ('front', 90, -5),
                                          ('side', 0, -5), ('top', 45, -65)]:
            renderer.update_scene(data, camera=camera_for(model, lookat, extent, azimuth, elevation))
            cv2.imwrite(str(args.output/f'{name}.png'),
                        cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    print(json.dumps(report, indent=2), flush=True)
    if args.view:
        from mujoco import viewer
        with viewer.launch_passive(model, data, show_right_ui=False) as window:
            camera = camera_for(model, lookat, extent, 65, -25)
            window.cam.lookat[:] = camera.lookat
            window.cam.distance, window.cam.azimuth, window.cam.elevation = camera.distance, 65, -25
            window.set_texts((None, None, 'Selected STL housing',
                             'Flat -Y face: ArUco | recessed +/-X sides: gripper | visual prototype'
                             if not args.bare else f'{asset.name} | original visual mesh'))
            while window.is_running():
                window.sync()
                time.sleep(1/30)


if __name__ == '__main__':
    main()
