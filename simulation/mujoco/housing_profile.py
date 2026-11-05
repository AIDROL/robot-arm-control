"""Versioned geometry calibration shared by perception and motion planning.

These are configured dimensions, not object pose reads. Historical non-vision
scenes keep their original proxy profile; vision scenes default to the STL.
"""
from dataclasses import dataclass, replace
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class HousingProfile:
    id: str
    half_extents: tuple
    body_height: float
    top_z: float
    side_center: tuple
    side_euler: tuple
    grasp_z: float
    grip_yaw: float
    grip_command: float
    place_offset: float
    screw_midpoint: tuple
    insert_axis: tuple
    tool_yaw: float
    placement_yaw: float = 0.
    support_height: float = 0.
    bounds_center: tuple = (0., 0., 0.)

    @property
    def is_selected(self):
        return self.id.startswith('selected_stl_')

    @property
    def side_rotation(self):
        return Rotation.from_euler('xyz', self.side_euler)

    @property
    def layout_id(self):
        if self.support_height:
            return 'x_row_yaw90_post_v2'
        return 'x_row_yaw90_v1' if self.is_selected else 'proxy_x_row_yaw0'

    @property
    def gripper_reference(self):
        # Configured world-down TCP frame for the mounted xArm: jaws close on X.
        # Do not redefine this calibration from an imperfect achieved park pose.
        return Rotation.from_matrix([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])


PROXY = HousingProfile('proxy', (.025, .025, .040), .040, .0408,
                       (-.0258, 0, .024), (0, -np.pi/2, 0), .010, np.pi/2,
                       255., .003, (-.029, 0, -.010), (1, 0, 0), -np.pi/2)
STL_FRAME = HousingProfile('selected_stl_v1', (.036, .032, .054), .054, .05235,
                     (0, -.03235, .0381), (np.pi/2, 0, 0), .0381, 0.,
                     100., 0., (0, -.018, -.014), (0, 1, 0), 0., np.pi/2)
STL = replace(STL_FRAME, id='selected_stl_post_v2', body_height=.114,
              half_extents=(.036, .032, .084), bounds_center=(0., 0., -.030),
              support_height=.060)


def from_id(name):
    if name == PROXY.id:
        return PROXY
    if name == STL.id:
        return STL
    if name == STL_FRAME.id:
        return STL_FRAME
    raise ValueError(f'Unknown housing geometry calibration: {name}')


def for_model(model):
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, 'selected_stl_profile')
    if index < 0:
        return PROXY
    version = model.numeric_data[int(model.numeric_adr[index])]
    return STL if version == 2 else STL_FRAME
