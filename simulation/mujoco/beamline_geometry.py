"""Independent geometry check of the straight X-row apertures; not optics."""
import mujoco
import numpy as np

from . import line_assembly as scene
from .housing_profile import for_model


def inspect(model, data):
    geometry = for_model(model)
    axis = np.array([1., 0., 0.])
    height = scene.TABLE_Z+geometry.body_height+geometry.screw_midpoint[2]
    xs = scene.target_xs_for_model(model)
    origin = np.array([min(xs)-.10, scene.TARGET_Y, height])
    length = max(xs)-min(xs)+.20
    mounts = {}
    for name, _ in scene.components_for_model(model):
        site = data.site(f'{name}_optical_axis')
        direction = site.xmat.reshape(3, 3)[:, 2]
        angle = float(np.arccos(np.clip(abs(direction@axis), 0., 1.)))
        transverse = site.xpos-origin
        transverse -= axis*(transverse@axis)
        mounts[name] = {'axis_world': direction.tolist(), 'centre_world_m': site.xpos.tolist(),
                        'axis_error_deg': float(np.degrees(angle)),
                        'centre_distance_to_line_m': float(np.linalg.norm(transverse))}
    geom = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(model, data, origin, axis, None, True, -1, geom)
    blocked = 0 <= distance <= length
    checks = {'axes_parallel_to_row': all(m['axis_error_deg'] < 2 for m in mounts.values()),
              'centres_within_5_mm_of_line': all(m['centre_distance_to_line_m'] < .005 for m in mounts.values()),
              'finite_line_unobstructed': not blocked}
    return {'success': all(checks.values()), 'checks': checks, 'mounts': mounts,
            'line_start_world_m': origin.tolist(), 'line_end_world_m': (origin+axis*length).tolist(),
            'first_blocking_geom': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom[0])) if blocked else None,
            'scope': 'straight-ray aperture geometry only; no laser intensity, lenses, reflection or interference'}
