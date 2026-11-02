"""Standalone parametric housing prototype; dimensions in metres.

Original primitive geometry, inspired by the open housing in CAD_Engraver.
No upstream meshes are copied. No screw kinematics or optical propagation.
The housing, internal kinematic mount and optic carrier are separate bodies.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
import time

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import global_camera as vision
from . import line_assembly as scene


@dataclass(frozen=True)
class HousingDimensions:
    width: float = .050
    depth: float = .050
    height: float = .080
    wall: float = .005
    opening_top: float = .016  # expose the internal mount below the label panel
    socket_radius: float = .0023  # new provisional clearance for 2 mm FAT shafts
    socket_outer_radius: float = .0045  # existing target radius
    socket_recess: float = .007  # inside the front opening, not on a housing wall
    socket_depth: float = .004  # blind tool engagement pocket
    frame_half_width: float = .019
    frame_aperture_radius: float = .0115
    optic_radius: float = .011

    def validate(self):
        values = np.array(list(asdict(self).values()))
        if not np.isfinite(values).all():
            raise ValueError('Housing dimensions must be finite')
        if min(self.width, self.depth, self.height, self.wall) <= 0:
            raise ValueError('Housing dimensions must be positive')
        if 2*self.wall >= min(self.width, self.depth, self.height):
            raise ValueError('Wall thickness leaves no cavity')
        if not -self.height/2+self.wall < self.opening_top < self.height/2-self.wall:
            raise ValueError('Opening top must lie between floor and cap')
        if not 0 < self.socket_radius < self.socket_outer_radius or min(self.socket_recess, self.socket_depth) <= 0:
            raise ValueError('Invalid socket dimensions')
        if not 0 < self.optic_radius < self.frame_aperture_radius < self.frame_half_width < self.width/2-self.wall:
            raise ValueError('Internal mount must fit inside the housing')
        if self.socket_recess+self.socket_depth+.009 >= self.depth/2:
            raise ValueError('Socket pocket is too deep')
        if (vision.SCREW_CENTER_Z-self.frame_half_width <= -self.height/2+self.wall or
                vision.SCREW_CENTER_Z+self.frame_half_width >= self.opening_top):
            raise ValueError('Internal mount must fit inside the opening')


def socket_centres(dim: HousingDimensions):
    # Rz(+90 degrees) maps the old -X-facing layout onto the -Y opening.
    # Lateral spacing/heights are retained; depth now follows the internal mount.
    return {label: np.array([dx, -dim.depth/2+dim.socket_recess,
                             vision.SCREW_CENTER_Z+dz])
            for label, dx, dz in scene.FAT_KEY_OFFSETS}


def annulus(spec, body, name, centre, ymin, ymax, inner, outer, square=False,
            color=(.16, .17, .19, 1), group=3):
    """An XZ ring extruded along Y, decomposed into convex wedges."""
    faces = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
             0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5,
             2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7]
    # XZ section coordinates reverse the handedness of an XY extrusion.
    faces = np.array(faces).reshape(-1, 3)[:, ::-1].ravel()
    for k in range(32):
        angles = np.array([k, k+1])*2*np.pi/32
        directions = np.column_stack([np.cos(angles), np.sin(angles)])
        inside = directions*inner
        outside = directions*(outer/np.max(abs(directions), axis=1)[:, None] if square else outer)
        section = np.array([inside[0], outside[0], outside[1], inside[1]])
        verts = [[centre[0]+xx, yy, centre[1]+zz]
                 for yy in (ymin, ymax) for xx, zz in section]
        mesh_name = f'{name}_{k}'
        spec.add_mesh(name=mesh_name, uservert=np.array(verts).ravel(), userface=faces,
                      smoothnormal=False)
        body.add_geom(name=mesh_name, type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=mesh_name, rgba=color, density=600, group=group,
                      friction=[1.5, .01, .0001], solref=[.005, 1])


def add_internal_mount(spec, housing, name, dim, optic):
    """Static representation of a mount's fixed frame, knobs and optic carrier."""
    frame = housing.add_body(name=f'{name}_kinematic_frame')
    frame_z = vision.SCREW_CENTER_Z
    annulus(spec, frame, f'{name}_frame', [0, frame_z], .0025, .0055,
            dim.frame_aperture_radius, dim.frame_half_width, square=True)
    floor_top = -dim.height/2+dim.wall
    frame_bottom = frame_z-dim.frame_half_width
    frame.add_geom(name=f'{name}_frame_support', type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[0, .004, (floor_top+frame_bottom)/2],
                   size=[.005, .004, (frame_bottom-floor_top)/2],
                   rgba=[.22, .23, .25, 1], group=3)
    for label, centre in socket_centres(dim).items():
        cx, entry, cz = centre
        annulus(spec, frame, f'{name}_{label}_knob', [cx, cz], entry,
                entry+dim.socket_depth, dim.socket_radius, dim.socket_outer_radius,
                color=(.48, .50, .53, 1))
        # A blind pocket, backed by the knob head; not a hole through the housing.
        for part, ya, yb, radius in [
            ('back', entry+dim.socket_depth, -.009, dim.socket_outer_radius),
            ('shaft', -.009, .004, .0025),
        ]:
            frame.add_geom(name=f'{name}_{label}_{part}', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           pos=[cx, (ya+yb)/2, cz], quat=[np.sqrt(.5), np.sqrt(.5), 0, 0],
                           size=[radius, (yb-ya)/2], rgba=[.36, .38, .41, 1], group=3)
        frame.add_site(name=f'{name}_{label}_screw_center', pos=centre,
                       size=[.0005], rgba=[0, 0, 0, 0])
    carrier = frame.add_body(name=f'{name}_optic_carrier')
    annulus(spec, carrier, f'{name}_carrier', [0, frame_z], .007, .010,
            dim.frame_aperture_radius, .014, color=(.26, .27, .29, 1))
    if optic:
        carrier.add_geom(name=f'{name}_mirror', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                         pos=[0, .0085, frame_z], quat=[np.sqrt(.5), np.sqrt(.5), 0, 0],
                         size=[dim.optic_radius, .0006], rgba=[.72, .81, .87, 1],
                         group=4)
    housing.add_site(name=f'{name}_optical_axis', pos=[0, 0, frame_z],
                     size=[.0005], rgba=[0, 0, 0, 0])


def add_housing(spec, body, name='mount', dim=HousingDimensions(), optic=True):
    """Add convex wall pieces with actual empty front/back and tool bores.

    Group 0: housing; 1: non-colliding tags; 3: internal mount; 4: optic.
    The frame uses the old proxy's centre and axes; clear propagation is along Y.
    """
    dim.validate()
    x, y, z = dim.width/2, dim.depth/2, dim.height/2
    w = dim.wall
    color = [.18, .30, .40, 1]

    def box(label, pos, size):
        body.add_geom(name=f'{name}_{label}', type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=pos, size=size, rgba=color, density=600,
                      friction=[1.5, .01, .0001], solref=[.005, 1])

    box('floor', [0, 0, -z+w/2], [x, y, w/2])
    box('cap', [0, 0, z-w/2], [x, y, w/2])
    for sign, label in [(-1, 'left_wall'), (1, 'right_wall')]:
        box(label, [sign*(x-w/2), 0, 0], [w/2, y, z-w])
    # The label/grasp panel sits above the internal mount and exposed knobs.

    low, high = dim.opening_top, z-w
    for sign, label in [(-1, 'front_grip_band'), (1, 'back_grip_band')]:
        box(label, [0, sign*(y-w/2), (low+high)/2], [x-w, w/2, (high-low)/2])

    add_internal_mount(spec, body, name, dim, optic)

    before = {g.name for g in body.geoms}
    vision.add_marker(body, f'{name}_top_aruco', 0, vision.TOP_EDGE,
                      np.array([0, 0, z+.0008]), np.eye(3))
    side = np.array([0, -y-.0008, .024])
    rotation = (Rotation.from_euler('z', np.pi/2) *
                Rotation.from_euler('xyz', scene.SIDE_MARKER_EULER)).as_matrix()
    vision.add_marker(body, f'{name}_side_aruco', 10, vision.SIDE_EDGE, side, rotation)
    for geom in body.geoms:
        if geom.name not in before:
            geom.group = 1
    body.add_site(name=f'{name}_side_marker', pos=side,
                  quat=Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]],
                  size=[.0005], rgba=[0, 0, 0, 0])


def make_preview(dim=HousingDimensions(), optic=True):
    spec = mujoco.MjSpec()
    spec.modelname = 'hollow_optical_housing_prototype'
    spec.visual.global_.offwidth = 1100
    spec.visual.global_.offheight = 900
    spec.visual.quality.offsamples = 4
    spec.stat.extent = .2
    spec.add_texture(name='background', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[.84, .89, .93], rgb2=[.64, .72, .79], width=512, height=512)
    spec.worldbody.add_light(pos=[-.15, -.2, .3], dir=[.4, .5, -1],
                             diffuse=[.8, .8, .8], ambient=[.3, .3, .3])
    spec.worldbody.add_light(pos=[.2, .1, .2], dir=[-.7, -.3, -.6],
                             diffuse=[.45, .45, .45], castshadow=False)
    spec.worldbody.add_geom(name='bench', type=mujoco.mjtGeom.mjGEOM_BOX,
                           pos=[0, 0, -.006], size=[.14, .12, .006],
                           rgba=[.65, .68, .70, 1], group=2)
    body = spec.worldbody.add_body(name='mount', pos=[0, 0, dim.height/2])
    add_housing(spec, body, dim=dim, optic=optic)
    return spec


def verify(model, data, dim):
    """Check the same convex geometry used for contacts, with ray queries."""
    group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    gid = np.array([-1], dtype=np.int32)

    def query(origin, direction, groups=None):
        distance = mujoco.mj_ray(model, data, np.array(origin, dtype=float),
                                 np.array(direction, dtype=float),
                                 np.array(groups, dtype=np.uint8) if groups is not None else group,
                                 True, -1, gid)
        return {'clear': bool(distance < 0), 'distance_m': float(distance),
                'first_geom': None if distance < 0 else
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid[0]))}

    # Interior grid: rays pass the complete front and rear opening.
    clear = []
    for xx in np.linspace(-dim.width/2+dim.wall+.001, dim.width/2-dim.wall-.001, 7):
        for zz in np.linspace(dim.wall+.001, dim.height/2+dim.opening_top-.001, 7):
            clear.append(query([xx, -.1, zz], [0, 1, 0])['clear'])
    sockets = {}
    for label, centre in socket_centres(dim).items():
        sockets[label] = query([centre[0], -.1, centre[2]+dim.height/2], [0, 1, 0],
                               [1, 0, 0, 1, 0, 0])
        sockets[label]['accessible_blind_pocket'] = bool(
            sockets[label]['first_geom'] == f'mount_{label}_back' and
            np.isclose(-.1+sockets[label]['distance_m'], centre[1]+dim.socket_depth))
    blocked = query([0, -.1, dim.height/2+.024], [0, 1, 0])
    axis_origin = [0, -.1, dim.height/2+vision.SCREW_CENTER_Z]
    frame_clear = query(axis_origin, [0, 1, 0], [1, 0, 0, 1, 0, 0])
    with_optic = query(axis_origin, [0, 1, 0], [1, 0, 0, 1, 1, 0])
    mirror_present = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'mount_mirror') >= 0
    optic_ok = with_optic['first_geom'] == 'mount_mirror' if mirror_present else with_optic['clear']
    passed = (all(clear) and all(s['accessible_blind_pocket'] for s in sockets.values())
              and not blocked['clear'] and frame_clear['clear'] and optic_ok)
    if not passed:
        raise RuntimeError('Housing geometry verification failed')
    return {'passed': passed, 'opening_clear_rays': sum(clear), 'opening_total_rays': len(clear),
            'grasp_band_blocks_ray': blocked, 'sockets': sockets,
            'mount_aperture_clear_without_optic': frame_clear['clear'],
            'axis_with_optic': with_optic, 'mirror_present': mirror_present,
            'interface_face_normal': [0, -1, 0],
            'dimensions_m': asdict(dim),
            'opening_size_mm': [(dim.width-2*dim.wall)*1000,
                                (dim.height/2-dim.wall+dim.opening_top)*1000],
            'optical_axis_height_mm': (dim.height/2+vision.SCREW_CENTER_Z)*1000,
            'scope': 'static housing and internal mirror mount; no optical propagation or screw kinematics'}


def export_preview(output, spec, model, data):
    options = mujoco.MjvOption()
    options.geomgroup[:5] = 1
    with mujoco.Renderer(model, height=900, width=1100) as renderer:
        for title, azimuth, elevation in [('overview', 65, -20), ('front', 90, 0),
                                           ('fat_side', 90, -8), ('rear', 270, -10)]:
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, camera)
            camera.lookat[:] = [0, 0, .04]
            camera.distance, camera.azimuth, camera.elevation = .185, azimuth, elevation
            renderer.update_scene(data, camera=camera, scene_option=options)
            rgb = renderer.render().copy()
            cv2.imwrite(str(output/f'{title}.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    (output/'housing.xml').write_text(spec.to_xml())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--empty', action='store_true', help='Omit the mirror to inspect the clear mount aperture')
    parser.add_argument('--output', type=Path, default=Path('runs/hollow-mount-preview'))
    args = parser.parse_args()
    if args.view and sys.platform == 'darwin' and not os.environ.get('MJPYTHON_BIN'):
        os.execv(sys.executable, [sys.executable, str(Path(sys.executable).with_name('mjpython')),
                                 '-m', 'simulation.hollow_mount', *sys.argv[1:]])
    args.output.mkdir(parents=True, exist_ok=True)
    dim = HousingDimensions()
    spec = make_preview(dim, optic=not args.empty)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    report = verify(model, data, dim)
    (args.output/'geometry-check.json').write_text(json.dumps(report, indent=2))
    export_preview(args.output, spec, model, data)
    print(json.dumps(report, indent=2), flush=True)
    if args.view:
        from mujoco import viewer
        with viewer.launch_passive(model, data, show_right_ui=False) as window:
            window.opt.geomgroup[:5] = 1
            window.cam.lookat[:] = [0, 0, .04]
            window.cam.distance, window.cam.azimuth, window.cam.elevation = .185, 65, -20
            window.set_texts((None, None, 'Hollow housing prototype',
                             'Internal mirror mount | ArUco and FAT face opening | drag to rotate'))
            while window.is_running():
                window.sync()
                time.sleep(1/30)


if __name__ == '__main__':
    main()
