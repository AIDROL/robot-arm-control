"""Data-only adapter for the first supported three-component assembly layout.

No generated Python is executed. Optical centers are converted into housing
origins using the configured insert frame; this is not an optical validator.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .housing_profile import STL


@dataclass(frozen=True)
class AssemblyLayout:
    sha256: str
    names: tuple
    target_xs: tuple
    report: dict


def _vector(value):
    if (not isinstance(value, list) or len(value) != 3 or
            any(type(x) not in (int, float) or not np.isfinite(x) for x in value)):
        raise ValueError('Expected three finite coordinates')
    return np.array(value, dtype=float)


def load(path):
    raw = Path(path).read_bytes()
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate layout key: '+key)
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError('Nonfinite JSON value: '+value)
    doc = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
    if not isinstance(doc, dict):
        raise ValueError('Layout must be a JSON object')
    if (doc.get('schema_version') != 'optical-assembly-v1' or
            doc.get('length_unit') != 'm' or doc.get('frame_id') != 'optical_table'):
        raise ValueError('Unsupported layout schema, unit or frame')
    # This first adapter deliberately supports the calibrated straight row only.
    origin = _vector(doc.get('world_origin_m'))
    if not np.allclose(origin, [0, 0, .760], atol=1e-9, rtol=0):
        raise ValueError('Uncalibrated optical-table to world transform')
    components = doc.get('components')
    if (not isinstance(components, list) or not all(isinstance(c, dict) for c in components) or
            tuple(c.get('id') for c in components) != ('lens_1', 'pinhole', 'lens_2')):
        raise ValueError('First adapter requires lens_1, pinhole, lens_2 in inventory order')
    if doc.get('connections') != [['lens_1', 'pinhole'], ['pinhole', 'lens_2']]:
        raise ValueError('First adapter requires the straight three-component path')
    rotation = Rotation.from_euler('z', STL.placement_yaw).as_matrix()
    optical_local = np.array([0, .004, STL.screw_midpoint[2]])
    half_xy = (np.abs(rotation) @ np.array(STL.half_extents))[:2]
    targets, rows = [], []
    for c, kind in zip(components, ('lens', 'aperture', 'lens')):
        if c.get('type') != kind or c.get('inventory_id') != STL.id:
            raise ValueError('Unsupported component inventory: '+c['id'])
        direction = _vector(c.get('direction_xyz'))
        if c.get('direction_role') != 'optical_axis' or not np.allclose(direction, [-1, 0, 0], atol=1e-9, rtol=0):
            raise ValueError('First adapter requires the configured local +Y axis at yaw +90 degrees')
        optical = _vector(c.get('optical_center_m')) + origin
        body = optical - rotation @ optical_local
        if abs(body[1]-.350) > 1e-9 or abs(body[2]-(.760+STL.body_height)) > 1e-9:
            raise ValueError('Optical center incompatible with row or post/base height')
        if np.any(body[:2]-half_xy < [-.58, -.39]) or np.any(body[:2]+half_xy > [.58, .65]):
            raise ValueError('Housing footprint outside the table')
        if not .25 <= np.linalg.norm(body[:2]) <= .68:
            raise ValueError('Target outside nominal radial workspace')
        if any(abs(body[0]-p[0]) < max(.14, 2*half_xy[0]+.02) for p in targets):
            raise ValueError('Insufficient housing/gripper spacing')
        targets.append(body)
        tcp = body + rotation @ np.array([0, 0, STL.grasp_z])
        rows.append({'id': c['id'], 'type': kind,
                     'optical_center_world_m': optical.tolist(),
                     'body_origin_world_m': body.tolist(), 'target_yaw_rad': STL.placement_yaw,
                     'grasp_center_world_m': tcp.tolist()})
    digest = hashlib.sha256(raw).hexdigest()
    return AssemblyLayout(digest, tuple(c['id'] for c in components),
                          tuple(float(p[0]) for p in targets),
                          {'success': True, 'source_layout_sha256': digest,
                           'housing_geometry': STL.id, 'components': rows,
                           'optical_axis_height_m': .100, 'support_stack_height_m': STL.support_height,
                           'scope': 'assembly adaptation only; no optical propagation or full swept-path IK certification',
                           'optical_validation': 'not_run',
                           'recovery': 'not_implemented; existing mechanical failures stop execution'})
