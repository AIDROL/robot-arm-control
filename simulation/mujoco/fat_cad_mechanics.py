"""Matching simulation fixture for the supplied upright FAT CAD.

Dimensions are extracted from the versioned tessellation. Mass, motor limits,
friction and screw/lever gain are provisional simulation parameters. This is
not a model calibrated to hardware. Contact prisms retain the open frame;
small fastener holes and electronics shell recesses are approximated.
"""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ASSETS = Path(__file__).resolve().parents[2] / 'assets/fat'
GRASP = np.array([.005057254, -.007274999, .095])
SHAFT_HALF = .0145  # CAD's terminal hex section is 29 mm long
SHAFT_Y = .0346749985
SHAFT_TIP_Y = SHAFT_Y + SHAFT_HALF
SHAFT_XZ = {'lower': (-.013442746, -.0085), 'upper': (.023557254, .0285)}
SOCKET_XZ = {'lower': (-.0185, -.0325), 'upper': (.0185, .0045)}
ROTATING_PARTS = {'lower': (0, 5, 6, 12, 13), 'upper': (2, 3, 9, 10, 11)}


def geometry():
    profile = json.loads((ASSETS/'fat_mobile_v1.json').read_text())
    raw = json.loads((ASSETS/'fat_mobile_v1_geometry.json').read_text())
    if raw['source_usd_sha256'] != profile['converted_sha256']:
        raise ValueError('FAT_GEOMETRY_SOURCE_MISMATCH')
    if hashlib.sha256((ASSETS/profile['converted']).read_bytes()).hexdigest() != raw['source_usd_sha256']:
        raise ValueError('FAT_GEOMETRY_ASSET_MISMATCH')
    return raw['parts']


def configure(spec):
    parts = geometry()
    spec.delete(spec.geom('fat_housing'))
    fat = spec.body('fat')
    spec.site('fat_grasp').pos = GRASP
    # Explicit CAD-aligned contact decomposition; avoid a global hull that
    # would fill the frame windows and create fictitious contact surfaces.
    boxes = []
    def box(label, lo, hi, mass):
        lo, hi = np.array(lo), np.array(hi)
        boxes.append(label)
        fat.add_geom(name='fat_cad_'+label, type=mujoco.mjtGeom.mjGEOM_BOX,
                     pos=(lo+hi)/2, size=(hi-lo)/2, mass=mass, group=3,
                     rgba=[.2,.7,.3,.3], friction=[2,.005,.0001], condim=6,
                     solref=[.002,1])
    x0, x1 = -.044942746, .055057254
    ya, yb = -.029775, .015225
    box('left_wall', [x0,ya,-.045], [x0+.002,yb,.065], .015)
    box('right_wall', [x1-.002,ya,-.045], [x1,yb,.065], .015)
    box('floor', [x0+.002,ya,-.045], [x1-.002,yb,-.040], .020)
    box('roof', [x0+.002,ya,.060], [x1-.002,yb,.065], .020)
    # Motor mounting plates occupy opposite diagonal corners in the CAD.
    for end, yl, yh in [('rear',ya,-.023775), ('front',.010225,yb)]:
        box(end+'_lower', [x0+.002,yl,-.040], [-.004942746,yh,0.], .010)
        box(end+'_upper', [.015057254,yl,.020], [x1-.002,yh,.060], .010)
    sx0, sx1 = -.017442746, .027557254
    box('grasp_left', [sx0,ya,.065], [sx0+.002,yb,.105], .010)
    box('grasp_right', [sx1-.002,ya,.065], [sx1,yb,.105], .010)
    box('grasp_rear', [sx0+.002,ya,.065], [sx1-.002,ya+.002,.105], .010)
    box('grasp_front', [sx0+.002,yb-.002,.065], [sx1-.002,yb,.105], .010)
    box('grasp_cap', [sx0+.002,ya+.002,.103], [sx1-.002,yb-.002,.105], .005)
    # Individual motor and electronics envelopes; no collision hull spans
    # several disjoint parts. The transparent window remains open.
    for index in (1,4,7,8):
        v = np.array(parts[index]['vertices_m'])
        box('part_'+str(index), v.min(0), v.max(0), .025 if index in (1,4) else .008)
    for label in ('lower','upper'):
        dx,dz = SHAFT_XZ[label]
        spec.body(label+'_driver').pos = [dx,SHAFT_Y,dz]
        spec.body(label+'_knob').pos = [*SOCKET_XZ[label][:1],-.018,SOCKET_XZ[label][1]]
        # Exact CAD terminal hex prism, in the motor's local frame. It starts
        # at y=.020175, so shoulder geometry cannot enter the 8 mm socket.
        p = np.array(parts[13 if label=='lower' else 10]['vertices_m'])
        tip = np.unique(np.round(p[abs(p[:,1]-p[:,1].max())<1e-6][:,[0,2]],9),axis=0)
        tip -= [dx,dz]
        vertices = np.array([[x,y,z] for y in (-SHAFT_HALF,SHAFT_HALF) for x,z in tip])
        spec.mesh(label+'_hex').uservert = vertices.ravel()
        spec.geom('fat_'+label+'_key').mass = .001
        # Back section of shaft: two cylindrical/hex diameters from CAD.
        driver = spec.body(label+'_driver')
        for suffix,yl,yh,radius in [('shank',-.015825,.004175,.00231),('neck',.004175,.020175,.00135)]:
            driver.add_geom(name=f'fat_{label}_{suffix}', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[0,(yl+yh)/2-SHAFT_Y,0], quat=[2**-.5,2**-.5,0,0],
                size=[radius,(yh-yl)/2], mass=.001, group=3, rgba=[0,0,0,0])
        # Matching hex socket: 2.1 mm across flats, 0.05 mm radial
        # clearance from the 2.0 mm CAD tip. Keep six independent walls.
        for i,theta in enumerate(np.arange(6)*np.pi/3):
            g=spec.geom(f'{label}_socket_{i}')
            g.pos=[.00165*np.cos(theta),.0055,-.00165*np.sin(theta)]
            g.size=[.0006,.0025,.0013]
            # Chamfer from 3.4 mm entry across flats to 2.1 mm socket.
            poly=[(.0017,0),(.0027,0),(.0027,.003),(.00105,.003)]
            spec.mesh(f'{label}_lead_{i}').uservert=np.array([[x,y,z] for z in (-.0013,.0013) for x,y in poly]).ravel()
        spec.geom(label+'_socket_back').size=[.0027,.0005,0]
    # Camera sits just in front of the protruding stem and views the existing
    # M2 side marker. Calibrated virtual sensor; no physical camera CAD supplied.
    cam=spec.camera('fat_rgb');cam.pos=[.005057254,.017,.078];cam.fovy=100.
    forward=np.array([0.,.018,-.016]);forward/=np.linalg.norm(forward)
    back=-forward;right=np.cross([0,0,1.],back);right/=np.linalg.norm(right)
    cam.alt.xyaxes=[*right,*np.cross(back,right)]


def hex_prealign():
    p=np.array(geometry()[10]['vertices_m'])
    x,z=SHAFT_XZ['upper']
    tip=p[abs(p[:,1]-p[:,1].max())<1e-6]
    angle=np.arctan2(-(tip[:,2]-z),tip[:,0]-x)[0] % (np.pi/3)
    return float(np.pi/6-angle)


def plan(data, fixture_pose, staging_xy, pickup_pose=None, grip=160.):
    from .fat_mechanics import ROOT_CENTER
    from .housing_profile import STL
    from .task_space_control import Phase
    home=data.site('pinch_center').xpos.copy()
    pickup=np.eye(4) if pickup_pose is None else np.array(pickup_pose)
    if pickup_pose is None:pickup[:3,3]=[*staging_xy,.805]
    pick=pickup[:3,3]+pickup[:3,:3]@GRASP
    pick_R=Rotation.from_matrix(pickup[:3,:3])*STL.gripper_reference
    R=fixture_pose[:3,:3];origin=fixture_pose[:3,3]
    # Known tool/fixture interface geometry; the fixture's world pose comes
    # from fresh pixels. Final insertion is observed, never teleported.
    body_goal=np.array([-.005057254,-.018+.006-SHAFT_TIP_Y,-.024])
    engage=origin+R@(body_goal+GRASP)
    orientation=Rotation.from_matrix(R)*STL.gripper_reference
    seq=[('pregrasp',pick+[0,0,.15],3,0,True,pick_R),('approach',pick,3,0,True,pick_R),
         ('close',pick,2,grip,False,pick_R),('lift',pick+[0,0,.18],3,grip,True,pick_R),
         ('transfer',engage+R@np.array([0,-.020,.18]),4,grip,True,orientation),
         ('prealign',engage-R@np.array([0,.020,0]),3,grip,True,orientation),
         ('insert',engage,5,grip,True,orientation),('seat',engage,2,grip,False,orientation),
         ('drive',engage,3,grip,False,orientation),('withdraw',engage-R@np.array([0,.040,0]),3,grip,True,orientation),
         ('return lift',engage-R@np.array([0,.040,0])+[0,0,.18],3,grip,True,orientation),
         ('return high',pick+[0,0,.24],4,grip,True,pick_R),('return lower',pick,3,grip,True,pick_R),
         ('release',pick,2,0,False,pick_R),('retreat',pick+[0,0,.15],3,0,True,pick_R),('park',home,3,0,True,STL.gripper_reference)]
    return [Phase('fat: '+name,np.array(pos),rot,seconds,command,wait,
                  position_tolerance=.0003,orientation_tolerance=np.deg2rad(.4))
            for name,pos,seconds,command,wait,rot in seq]
