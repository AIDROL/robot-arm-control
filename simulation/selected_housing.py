"""Selected original STL visual plus convex contact geometry in its raw frame."""
from pathlib import Path
import hashlib

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from .stl_preview import SELECTED_STL_SHA256, add_selected_interfaces

ASSET = Path(__file__).resolve().parents[1]/'assets/optical_housing/highousing.stl'
BOTTOM = -.004
GRIP_Z = .0881
NOMINAL_MASS = .120  # provisional complete housing mass, not measured hardware


def add_post_base(body, name, geometry):
    """Rigid post/base transported with the housing; magnetic force is not modeled.

    The 60 mm stack comprises a 12 mm base and a 48 mm post. Dimensions and
    masses are provisional, separate from the unchanged source STL.
    """
    if not np.isclose(geometry.support_height, .060):
        raise ValueError('Unsupported post/base height')
    body.add_geom(name=f'{name}_magnetic_base', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  pos=[0, 0, -.108], size=[.027, .006], mass=.120,
                  rgba=[.28, .30, .32, 1], friction=[1.5, .01, .0001], solref=[.005, 1])
    body.add_geom(name=f'{name}_support_post', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  pos=[0, 0, -.078], size=[.0063, .024], mass=.025,
                  rgba=[.72, .74, .77, 1], friction=[1.5, .01, .0001], solref=[.005, 1])
    body.add_site(name=f'{name}_support_bottom', pos=[0, 0, -geometry.body_height],
                  size=[.001], rgba=[0, 0, 0, 0])


def area(poly):
    return abs(np.sum(poly[:, 0]*np.roll(poly[:, 1], -1)-poly[:, 1]*np.roll(poly[:, 0], -1)))/2


def floor_polygons(path):
    """Merge bottom-face triangles only when their union stays convex.

    The STL's floor slots remain empty; no global convex hull is introduced.
    """
    raw = path.read_bytes()
    dtype = np.dtype([('n', '<f4', (3,)), ('v', '<f4', (3, 3)), ('a', '<u2')])
    triangles = np.frombuffer(raw[84:], dtype=dtype)['v'].astype(float)
    polys = [t[:, :2][ConvexHull(t[:, :2]).vertices]
             for t in triangles if np.all(abs(t[:, 2]+4)<1e-5)]
    changed = True
    while changed:
        changed = False
        for i in range(len(polys)):
            for j in range(i+1, len(polys)):
                shared = set(map(tuple, np.round(polys[i], 5))) & set(map(tuple, np.round(polys[j], 5)))
                if len(shared) < 2:
                    continue
                points = np.unique(np.vstack([polys[i], polys[j]]), axis=0)
                hull = points[ConvexHull(points).vertices]
                if abs(area(hull)-area(polys[i])-area(polys[j])) < 1e-5:
                    polys[i] = hull
                    polys.pop(j)
                    changed = True
                    break
            if changed:
                break
    return polys


def collision_parts(path=ASSET):
    """Convex XY prisms, expressed in original STL mm for easy inspection.

    Four 8 mm corner posts preserve the side windows; the upper block has
    3 mm recesses on +/-X with 45 degree shoulders. Roof tray stays open.
    Tiny raised mounting tabs/engraving are visual-only approximations.
    """
    parts = []

    def rectangle(name, xa, xb, ya, yb, za, zb):
        parts.append((name, np.array([[xa, ya], [xb, ya], [xb, yb], [xa, yb]]), za, zb))

    for i, poly in enumerate(floor_polygons(path)):
        parts.append((f'floor_{i}', poly, -4., 0.))
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            xa, xb = sorted([sx*28., sx*36.])
            ya, yb = sorted([sy*24., sy*32.])
            rectangle(f'post_{sx}_{sy}', xa, xb, ya, yb, 0., 72.2)
            ya, yb = sorted([sy*18.9, sy*32.])
            xa, xb = sorted([sx*33., sx*36.])
            rectangle(f'outer_shoulder_{sx}_{sy}', xa, xb, ya, yb, 72.2, 104.)
            points = np.array([[sx*33., sy*15.9], [sx*36., sy*18.9], [sx*33., sy*18.9]])
            points = points[ConvexHull(points).vertices]
            parts.append((f'ramp_{sx}_{sy}', points, 72.2, 104.))
    rectangle('grip_core', -33., 33., -32., 32., 72.2, 102.)
    for i, (xa, xb, ya, yb) in enumerate([
        (-33., 33., -32., -30.97), (-33., 33., 31., 32.),
        (-33., -31., -30.97, 31.), (30.97, 33., -30.97, 31.),
    ]):
        rectangle(f'top_rim_{i}', xa, xb, ya, yb, 102., 104.)
    return parts


def add_housing(spec, body, path=ASSET, mass=NOMINAL_MASS, name='housing',
                marker_id=0, centered=False, color=(.23, .36, .46, 1)):
    if hashlib.sha256(path.read_bytes()).hexdigest() != SELECTED_STL_SHA256:
        raise ValueError('Collision decomposition requires the selected original STL')
    z_offset = -.050 if centered else 0.
    spec.add_mesh(name=f'{name}_selected_visual', file=str(path), scale=[.001]*3, smoothnormal=False)
    body.add_geom(name=f'{name}_visual', type=mujoco.mjtGeom.mjGEOM_MESH,
                  meshname=f'{name}_selected_visual', pos=[0, 0, z_offset], rgba=color,
                  contype=0, conaffinity=0, mass=0, group=2)
    parts = collision_parts(path)
    total_volume = sum(area(p)*(hi-lo) for _, p, lo, hi in parts)
    for label, poly, lo, hi in parts:
        poly = poly*.001
        n = len(poly)
        vertices = [[x, y, z*.001+z_offset] for z in (lo, hi) for x, y in poly]
        faces = []
        for i in range(1, n-1):
            faces.extend([[0, i+1, i], [n, n+i, n+i+1]])
        for i in range(n):
            j = (i+1) % n
            faces.extend([[i, j, n+j], [i, n+j, n+i]])
        mesh_name = f'{name}_contact_{label}'
        spec.add_mesh(name=mesh_name, uservert=np.array(vertices).ravel(),
                      userface=np.array(faces).ravel(), smoothnormal=False)
        body.add_geom(name=mesh_name, type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=mesh_name, group=3, rgba=[0, .9, .3, .35],
                      mass=mass*(area(poly)*1e6)*(hi-lo)/total_volume,
                      friction=[1.5, .01, .0001], solref=[.005, 1])
    add_selected_interfaces(body, prefix=name, marker_id=marker_id, z_offset=z_offset)
    return [f'{name}_contact_{label}' for label, *_ in parts]


def add_mount_targets(spec, body, name):
    """Fixed internal mount/knob proxies; no optical or turning model."""
    from .hollow_mount import annulus
    from .housing_profile import STL
    frame = body.add_body(name=f'{name}_internal_frame')
    mid = np.array(STL.screw_midpoint)
    # Site +Z is the actual internal aperture axis (local +Y).
    frame.add_site(name=f'{name}_optical_axis', pos=[0, .004, mid[2]],
                   quat=[np.sqrt(.5), -np.sqrt(.5), 0, 0], size=[.0005], rgba=[0, 0, 0, 0])
    annulus(spec, frame, f'{name}_frame', [0, mid[2]], .0025, .0055,
            .0115, .019, square=True, group=2)
    for geom in frame.geoms:
        geom.mass = 0
    # Raise the internal mount from the floor to the chosen optical axis height.
    frame.add_geom(name=f'{name}_frame_support', type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[0, .004, -.0395], size=[.005, .004, .0105],
                   mass=0, group=2, rgba=[.18, .18, .20, 1])
    for label, dx, dz in [('lower', .012, -.012), ('upper', -.012, .012)]:
        centre = mid+np.array([dx, 0, dz])
        before = len(list(frame.geoms))
        annulus(spec, frame, f'{name}_{label}_knob', [centre[0], centre[2]],
                -.018, -.014, .0023, .0045, color=(.45, .47, .50, 1), group=2)
        for geom in list(frame.geoms)[before:]:
            geom.mass = 0
        frame.add_geom(name=f'{name}_{label}_knob_back', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       pos=[centre[0], -.0115, centre[2]], quat=[np.sqrt(.5), np.sqrt(.5), 0, 0],
                       size=[.0045, .0025], mass=0, group=2, rgba=[.35, .37, .40, 1])
        frame.add_geom(name=f'{name}_{label}_shaft', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       pos=[centre[0], -.0025, centre[2]], quat=[np.sqrt(.5), np.sqrt(.5), 0, 0],
                       size=[.0025, .0065], mass=0, group=2, rgba=[.35, .37, .40, 1])
        frame.add_site(name=f'{name}_{label}_screw_center', pos=centre, size=[.0005], rgba=[0, 0, 0, 0])
