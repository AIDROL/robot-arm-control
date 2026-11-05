"""Contact-gripped FAT with keyed sockets and a coupled two-axis mirror mount.

Provisional mechanical parameters; fixture-known-pose bench, not vision assembly.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from . import line_assembly as scene
from .task_space_control import CartesianController,Phase
from .housing_profile import STL
from .selected_housing import add_mount_targets
from .timing import SimulationPacer,positive_speed

BENCH=json.loads((Path(__file__).resolve().parents[1]/'configs/optical_setups/michelson_bench.json').read_text())
RATIO=BENCH['mirror_rad_per_knob_rad']  # provisional screw pitch / lever arm
ROOT_CENTER=np.array(BENCH['adjustable_mirror_center_m'])-[0,.004,-.014]
FAT_XY=np.array([.25,.45]);MOUNT_XY=ROOT_CENTER[:2]


def hex_mesh(spec,name,radius,half):
    vertices=[[radius*np.cos(t),y,radius*np.sin(t)] for y in [-half,half] for t in np.arange(6)*np.pi/3]
    spec.add_mesh(name=name,uservert=np.array(vertices).ravel())


def modify(spec):
    spec.option.timestep=.001
    spec.option.iterations=100
    spec.option.cone=mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio=10
    spec.option.noslip_iterations=20
    cam=spec.camera('fat_rgb');cam.fovy=110;cam.pos=[0,.012,.005]
    back=-np.array([0,.030,.022]);back/=np.linalg.norm(back)
    right=np.cross([0,0,1.],back);right/=np.linalg.norm(right)
    up=np.cross(back,right)
    cam.alt.xyaxes=[*right,*up]
    spec.visual.map.znear=.001
    root=spec.body('mirror');root.pos=ROOT_CENTER;root.quat=[1,0,0,0]
    # Fixed base in this mechanism bench; module assembly is tested separately.
    # Mirror has two driven angular freedoms; no ideal translational stage.
    mount=root.add_body(name='mirror_tilt',pos=[0,.004,-.014])
    for axis,vector in [('pitch',[1,0,0]),('yaw',[0,0,1])]:
        mount.add_joint(name='mirror_'+axis,type=mujoco.mjtJoint.mjJNT_HINGE,axis=vector,
                        limited=True,range=[-.02,.02],damping=.00002)
    mount.add_geom(name='mirror_disc',type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[.011,.0015],quat=[2**-.5,2**-.5,0,0],mass=.015,rgba=[.75,.85,.9,1],contype=0,conaffinity=0)
    mount.add_site(name='mirror_optical_axis',quat=[2**-.5,-2**-.5,0,0],size=[.001])
    for label,dx,dz,axis in [('lower',.012,-.026,'pitch'),('upper',-.012,-.002,'yaw')]:
        knob=root.add_body(name=label+'_knob',pos=[dx,-.018,dz])
        knob.add_joint(name=label+'_knob_joint',type=mujoco.mjtJoint.mjJNT_HINGE,
                       axis=[0,1,0],damping=.00002,frictionloss=.00001,armature=.0000001)
        for i,theta in enumerate(np.arange(6)*np.pi/3):
            knob.add_geom(name=f'{label}_socket_{i}',type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[.0036*np.cos(theta),.0055,-.0036*np.sin(theta)],
                quat=Rotation.from_rotvec([0,theta,0]).as_quat()[[3,0,1,2]],
                size=[.0012,.0025,.0022],mass=.0003,friction=[.8,.005,.0001],solref=[.002,1],rgba=[.5,.5,.5,1])
        for i,theta in enumerate(np.arange(6)*np.pi/3):
            mesh=f'{label}_lead_{i}'
            poly=[(.0037,0),(.0048,0),(.0048,.003),(.0024,.003)]
            spec.add_mesh(name=mesh,uservert=np.array([[x,y,z] for z in [-.0022,.0022] for x,y in poly]).ravel())
            knob.add_geom(name=mesh,type=mujoco.mjtGeom.mjGEOM_MESH,meshname=mesh,
                quat=Rotation.from_rotvec([0,theta,0]).as_quat()[[3,0,1,2]],mass=.0001,
                friction=[.4,.005,.0001],solref=[.002,1],rgba=[.5,.5,.5,1])
        knob.add_geom(name=label+'_socket_back',type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[0,.009,0],size=[.004,.0005],quat=[2**-.5,2**-.5,0,0],mass=.001)
        knob.add_site(name=label+'_socket_face',size=[.0004])
        spec.add_equality(name=label+'_screw_lever',type=mujoco.mjtEq.mjEQ_JOINT,
            name1='mirror_'+axis,name2=label+'_knob_joint',objtype=mujoco.mjtObj.mjOBJ_JOINT,
            data=[0,RATIO]+[0.]*9,solref=[.002,1])
    fat=spec.body('fat');fat.mocap=False;fat.pos=[*FAT_XY,.805]
    fat.add_freejoint(name='fat_free')
    spec.site('fat_grasp').pos=[0,0,.018]
    housing=spec.geom('fat_housing');housing.contype=1;housing.conaffinity=1;housing.mass=.070
    housing.friction=[2,.005,.0001];housing.solref=[.002,1];housing.condim=6
    for side in ['left','right']:
        for i in [1,2]:
            pad=spec.geom(f'{side}_finger_pad_{i}');pad.condim=6;pad.friction=[2,.005,.0001]

    for label,dx,dz in scene.FAT_KEY_OFFSETS:
        bit=fat.add_body(name=label+'_driver',pos=[dx,.024,dz-.025])
        for suffix,axis in [('float_x',[1,0,0]),('float_y',[0,1,0]),('float_z',[0,0,1])]:
            bit.add_joint(name=label+'_'+suffix,type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=axis,limited=True,range=([-.008,.008] if suffix=='float_y' else [-.0015,.0015]),stiffness=(50. if suffix=='float_y' else 200.),damping=.05)
        joint=bit.add_joint(name=label+'_motor_joint',type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0,1,0],limited=True,range=[-2,2],damping=.00001,armature=.000001)
        hex_mesh(spec,label+'_hex',.0026,.016)
        bit.add_geom(name='fat_'+label+'_key',type=mujoco.mjtGeom.mjGEOM_MESH,meshname=label+'_hex',
                     mass=.003,friction=[1,.005,.0001],solref=[.002,1],rgba=[.7,.7,.75,1])
        spec.add_actuator(name=label+'_motor',target=joint.name,trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,gainprm=[.03]+[0.]*9,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,biasprm=[0,-.03,-.0001]+[0.]*7,
            ctrllimited=True,ctrlrange=[-2,2],forcelimited=True,forcerange=[-.002,.002])


def add_michelson(spec):
    from .selected_housing import add_housing,add_post_base
    # Optical bench is preassembled for validating the adjustment mechanism.
    for name,center,yaw,marker,color in [
        ('splitter',BENCH['splitter_center_m'],-3*np.pi/4,0,[.1,.8,.3,1]),
        ('reference_mirror',BENCH['reference_mirror_center_m'],np.pi/2,1,[.1,.5,.8,1]),
        ('optical_detector',BENCH['detector_center_m'],0,2,[.8,.7,.1,1])]:
        rotation=Rotation.from_euler('z',yaw)
        root_center=np.array(center)-rotation.apply([0,.004,-.014])
        body=spec.worldbody.add_body(name=name,pos=root_center,quat=rotation.as_quat()[[3,0,1,2]])
        add_housing(spec,body,name=name,marker_id=marker,centered=True,color=color)
        add_post_base(body,name,STL);add_mount_targets(spec,body,name)
    source=spec.worldbody.add_body(name='laser_source',pos=np.array(BENCH['splitter_center_m'])-[.25,0,0])
    source.add_geom(name='laser_body',type=mujoco.mjtGeom.mjGEOM_CYLINDER,size=[.012,.035],
        quat=Rotation.from_euler('y',np.pi/2).as_quat()[[3,0,1,2]],rgba=[.15,.18,.22,1],mass=.05)
    bias=Rotation.from_euler('xz',[.00035,-.0003])
    axis=spec.site('mirror_optical_axis')
    axis.quat=(bias*Rotation.from_euler('x',-np.pi/2)).as_quat()[[3,0,1,2]]
    spec.geom('mirror_disc').quat=(bias*Rotation.from_euler('x',np.pi/2)).as_quat()[[3,0,1,2]]
    for i,(a,b) in enumerate([([-.25,.29,.86],[.204,.29,.86]),([0,.15,.86],[0,.494,.86])]):
        a,b=np.array(a),np.array(b)
        direction=b-a;direction/=np.linalg.norm(direction)
        rotation=Rotation.align_vectors(direction[None,:],np.array([[0,0,1.]]))[0]
        spec.worldbody.add_geom(name=f'beam_guide_{i}',type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=(a+b)/2,size=[.0005,np.linalg.norm(b-a)/2],quat=rotation.as_quat()[[3,0,1,2]],
            contype=0,conaffinity=0,rgba=[1,.2,.1,.6])


def make_scene(michelson=False):
    def modifier(spec):
        modify(spec)
        if michelson:add_michelson(spec)
    return scene.make_scene(randomize=False,include_fat=True,vision=True,
        component_names=('mirror',),target_xs=(0.,),spec_modifier=modifier,
        fixed_components=('mirror',),mount_proxies=False,fat_key_proxies=False)


def plan(data,grip=255.,fixture_pose=None,staging_xy=None,shaft_extension=0.):
    rotation=STL.gripper_reference
    pick=np.r_[FAT_XY if staging_xy is None else staging_xy,.823]
    # Tool +Y shaft ends at y=.040; its two tips straddle the optical height.
    engage=np.array([MOUNT_XY[0],MOUNT_XY[1]-.018+.012-.040-shaft_extension,.903])
    home=data.site('pinch_center').xpos.copy()
    sequence=[('pregrasp',pick+[0,0,.15],3,0,True),('approach',pick,3,0,True),
              ('close',pick,2,grip,False),('lift',pick+[0,0,.15],3,grip,True),
              ('transfer',engage+[0,-.06,.12],4,grip,True),
              ('prealign',engage+[0,-.02,0],3,grip,True),
              ('insert',engage,5,grip,True),('seat',engage,2,grip,True),('drive',engage,3,grip,False),
              ('withdraw',engage+[0,-.04,0],3,grip,True),
              ('return high',pick+[0,0,.15],4,grip,True),('return lower',pick,3,grip,True),
              ('release',pick,2,0,False),('retreat',pick+[0,0,.15],3,0,True),('park',home,3,0,True)]
    phases=[Phase('fat: '+label,np.array(pos),rotation,duration,command,wait,
                  position_tolerance=.0003,orientation_tolerance=np.deg2rad(.4))
            for label,pos,duration,command,wait in sequence]
    if fixture_pose is not None:
        from dataclasses import replace
        R=fixture_pose[:3,:3];origin=fixture_pose[:3,3]
        for i,p in enumerate(phases):
            if p.label in ('fat: transfer','fat: prealign','fat: insert','fat: seat','fat: drive','fat: withdraw'):
                phases[i]=replace(p,position=origin+R@(p.position-ROOT_CENTER),orientation=Rotation.from_matrix(R)*p.orientation)
        index=next(i for i,p in enumerate(phases) if p.label=='fat: withdraw')+1
        withdrawal=phases[index-1]
        phases.insert(index,replace(withdrawal,label='fat: return lift',position=withdrawal.position+[0,0,.18],duration=3.))
        index=next(i for i,p in enumerate(phases) if p.label=='fat: return high')
        phases[index]=replace(phases[index],position=phases[index].position+[0,0,.11])
    return phases


def refine_tool(model,data,controller,output,fixture_pose=None):
    import cv2
    from .global_camera import DICTIONARY,SIDE_EDGE
    with mujoco.Renderer(model,width=1280,height=720) as renderer:
        renderer.update_scene(data,camera='fat_rgb');rgb=renderer.render().copy()
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    corners,ids,_=cv2.aruco.ArucoDetector(DICTIONARY).detectMarkers(gray)
    cv2.imwrite(str(output/'fat-alignment-rgb.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    if ids is None or 14 not in ids.flatten():raise RuntimeError('FAT_SIDE_MARKER_MISSING')
    points=corners[list(ids.flatten()).index(14)].reshape(4,2).astype(float)
    h=SIDE_EDGE/2;square=np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]],float)
    f=720/(2*np.tan(np.deg2rad(model.cam_fovy[model.camera('fat_rgb').id])/2))
    K=np.array([[f,0,639.5],[0,f,359.5],[0,0,1.]])
    ok,rv,tv=cv2.solvePnP(square,points,K,np.zeros(5),flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or tv[2,0]<=0:raise RuntimeError('FAT_SIDE_POSE_INVALID')
    W_M=np.eye(4);W_M[:3,:3]=STL.side_rotation.as_matrix()
    W_M[:3,3]=ROOT_CENTER+STL.side_center
    if fixture_pose is not None:
        object_marker=np.eye(4);object_marker[:3,:3]=STL.side_rotation.as_matrix();object_marker[:3,3]=STL.side_center
        W_M=fixture_pose@object_marker
    C_M=np.eye(4);C_M[:3,:3]=cv2.Rodrigues(rv)[0];C_M[:3,3]=tv.ravel()
    F_C=np.eye(4);cam=model.camera('fat_rgb')
    F_C[:3,:3]=Rotation.from_quat(cam.quat[[1,2,3,0]]).as_matrix()@np.diag([1,-1,-1]);F_C[:3,3]=cam.pos
    W_F=W_M@np.linalg.inv(C_M)@np.linalg.inv(F_C)
    W_G=np.eye(4);W_G[:3,:3]=data.site('pinch_center').xmat.reshape(3,3);W_G[:3,3]=data.site('pinch_center').xpos
    G_F=np.linalg.inv(W_G)@W_F
    from dataclasses import replace
    if not hasattr(controller,'nominal_tool_phases'):
        controller.nominal_tool_phases={p.label:p for p in controller.phases}
    for i in range(controller.phase_index,len(controller.phases)):
        phase=controller.phases[i]
        if phase.label=='fat: park':continue
        nominal=controller.nominal_tool_phases[phase.label]
        target=np.eye(4);target[:3,:3]=(nominal.orientation*STL.gripper_reference.inv()).as_matrix()
        target[:3,3]=nominal.position-target[:3,:3]@np.array([0,0,.018])
        goal=target@np.linalg.inv(G_F)
        if np.linalg.norm(goal[:3,3]-phase.position)>.010:raise RuntimeError('FAT_VISUAL_CORRECTION_TOO_LARGE')
        controller.phases[i]=replace(phase,position=goal[:3,3],orientation=Rotation.from_matrix(goal[:3,:3]))
    return {'source':'rendered side-marker pixels and configured fixed fixture transform','T_world_fat':W_F.tolist()}


def run(output,view=False,michelson=False,hold_seconds=0.,model=None,data=None,fixture_pose=None,
        staging_xy=None,shared_viewer=None,shared_monitor=None,optical_config=None,assembly_optics=False,state_observer=None,speed=1.):
    speed=positive_speed(speed)
    from .vision_pick_place import reserve_output_directory
    output=reserve_output_directory(Path(output))
    owns_scene=model is None
    if owns_scene:
        if data is not None:raise ValueError('Shared data requires its model')
        m=make_scene(michelson);d=mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m,d,m.key('home').id);mujoco.mj_forward(m,d)
    else:
        if data is None:raise ValueError('Shared model requires data; refusing to reset the scene')
        m,d=model,data
    optical_config=BENCH if optical_config is None else optical_config
    started=float(d.time)
    import hashlib
    evidence={'bench':optical_config,'fixture_source':'post-assembly rendered global marker' if fixture_pose is not None else 'configured fixed fixture',
        'source_sha256':{Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [__file__,Path(__file__).with_name('michelson_optics.py'),Path(__file__).with_name('michelson_assembly_optics.py')]}}
    (output/'provenance.json').write_text(json.dumps(evidence,indent=2)+'\n')
    mujoco.mj_saveModel(m,str(output/'model.mjb'),None)
    geoms={g:'fat' for g in scene.component_collision_geoms(m,'fat')}
    extension=optical_config.get('driver_half_length_m',.016)-.016 if assembly_optics else 0.
    controller=CartesianController(m,d,phases=plan(d,fixture_pose=fixture_pose,staging_xy=staging_xy,shaft_extension=extension),object_geoms=geoms)
    fixture_geoms={}
    for name in (tuple(n for n,_ in scene.COMPONENTS) if assembly_optics else ('mirror','splitter','reference_mirror','optical_detector','laser_source')):
        if mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)>=0:
            fixture_geoms.update({g:name for g in scene.component_collision_geoms(m,name)})
    def optical_sample(index,n=256):
        if assembly_optics and n==256:n=1024
        folder=output/f'optical-{index}'
        folder.mkdir(parents=True,exist_ok=True)
        observation,intensity=optical_backend(m,d,n,capture_folder=folder,**({'config':optical_config} if assembly_optics else {}))
        if viewer is not None and assembly_optics:
            from .scene_recording import draw_paths
            with viewer.lock():draw_paths(viewer.user_scn,observation.get('paths',[]))
        np.savez_compressed(folder/'state.npz',qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,time=d.time)
        (folder/'measurement.json').write_text(json.dumps(observation,indent=2)+'\n')
        from .optical_sensor import publish
        publish(output/'detector',intensity,{'window_m':optical_config['detector_window_m'],'input_power_W':optical_config['input_power_W'],'full_scale_W_m2':optical_config['detector_full_scale_W_m2'],
            'frame_id':'michelson_detector_xz','display_axes':'X/Z',
            'task_label':f"Michelson integrated modulation visibility: {observation['visibility']*100:.2f}%"},float(d.time))
        return observation,intensity
    initial=d.body('fat').xpos.copy();grip_samples=0;held_samples=0;max_lift=0.;contacts=0
    both_since=None;drive_baseline=None;drive_response={"lower":0.,"upper":0.};return_refined=False
    record=[];last=None;viewer=shared_viewer;monitor=shared_monitor;optical_player=None;drive_start=None;visual_corrected=False;last_vision=-1.;vision_report=None;max_knobs={"lower":0.,"upper":0.};fringe_history=[];last_optics=-1.;fringe_status=None;fringe_policy=None
    if michelson:
        from .michelson_optics import FringeSearch,from_state
        optical_backend=from_state
        if assembly_optics:
            from .michelson_assembly_optics import from_state as optical_backend
        fringe_policy=FringeSearch()
        if assembly_optics:
            from .michelson_assembly_optics import PhaseGradientSearch
            fringe_policy=PhaseGradientSearch(RATIO)
        first,_=optical_sample('initial');fringe_history.append({'time_s':float(d.time),**first})
        from dataclasses import replace
        controller.phases=[replace(p,duration=190. if assembly_optics else 120.) if p.label=='fat: drive' else p for p in controller.phases]
    if view and shared_viewer is None:
        from mujoco import viewer as mjviewer
        viewer=mjviewer.launch_passive(m,d)
        viewer.cam.lookat[:]=[.05,.34,.86];viewer.cam.distance=1.;viewer.cam.azimuth=130;viewer.cam.elevation=-25
        from .camera_monitor import CameraMonitor
        monitor=CameraMonitor(m,fps=1/max(1.,speed),open_browser=False)
        monitor.update(m,d,'FAT CONTACT BENCH',force=True);monitor.start_player(output)
    if view and michelson:
        optical_player=subprocess.Popen([sys.executable,'-m','simulation.optical_sensor','--folder',str(output/'detector'),'--watch'])
    try:
        last_sync=-1.
        pacer=SimulationPacer(d.time,speed,enabled=viewer is not None)
        while d.time-started<(260 if assembly_optics else 200) and not controller.done and not controller.failed:
            controller(m,d)
            if state_observer is not None:state_observer(m,d,controller.phase.label)
            pads=set();engaged=set();force=np.zeros(6)
            for i,c in enumerate(d.contact[:d.ncon]):
                a,b=int(c.geom1),int(c.geom2)
                if len(controller.unsafe_contacts)<10:
                    for robot,fixture in ((a,b),(b,a)):
                        if robot in controller.arm_geoms and fixture in fixture_geoms:
                            controller.unsafe_contacts.append((float(d.time),controller.phase.label,'arm/'+fixture_geoms[fixture]))
                    for tool,fixture in ((a,b),(b,a)):
                        if tool in geoms and fixture in fixture_geoms:
                            intended=m.geom(tool).name in ('fat_lower_key','fat_upper_key') and any(t in m.geom(fixture).name for t in ('socket','lead'))
                            if not intended:controller.unsafe_contacts.append((float(d.time),controller.phase.label,'tool/'+fixture_geoms[fixture]))
                if a in geoms or b in geoms:
                    other=b if a in geoms else a
                    if other in controller.pad_geoms:
                        mujoco.mj_contactForce(m,d,i,force)
                        if force[0]>.001:pads.add(controller.pad_geoms[other])
                    if any(tag in (m.geom(other).name or '') for tag in ['socket','lead']):
                        contacts+=1
                        mujoco.mj_contactForce(m,d,i,force)
                        key_name=m.geom(a if a in geoms else b).name
                        for label in ['lower','upper']:
                            if key_name=='fat_'+label+'_key' and (m.geom(other).name or '').startswith(label) and force[0]>.005:engaged.add(label)
            if controller.phase.label in ('fat: lift','fat: transfer','fat: prealign','fat: insert','fat: seat','fat: drive'):
                grip_samples+=1;held_samples+=pads=={'left','right'}
            max_lift=max(max_lift,float(d.body('fat').xpos[2]-initial[2]))
            if controller.phase.label!=last:
                print('MECHANICS',d.time,controller.phase.label,flush=True);last=controller.phase.label
                record.append({'time_s':float(d.time),'phase':last,'fat_position_m':d.body('fat').xpos.tolist(),'pads':list(pads)})
            if controller.phase.label=='fat: insert':
                if engaged=={'lower','upper'}:
                    if both_since is None:both_since=float(d.time)
                    if d.time-both_since>.15:
                        from dataclasses import replace
                        j=controller.phase_index+1
                        for k in (j,j+1):
                            controller.phases[k]=replace(controller.phases[k],position=d.site('pinch_center').xpos.copy()+[0,.003,0],
                                orientation=Rotation.from_matrix(d.site('pinch_center').xmat.reshape(3,3)))
                        controller.advance(d)
                else:both_since=None
            return_observation=controller.phase.label in ('fat: return lift','fat: return high') and not return_refined
            if (controller.phase.label=='fat: insert' and d.time-last_vision>.5) or return_observation:
                try:
                    vision_report=refine_tool(m,d,controller,output,fixture_pose=fixture_pose)
                except RuntimeError as error:
                    controller.failed=str(error)
                    break
                visual_corrected=True;last_vision=float(d.time)
                if return_observation:return_refined=True
            for label in ['lower','upper']:
                max_knobs[label]=max(max_knobs[label],abs(float(d.joint(label+'_knob_joint').qpos[0])))
            if controller.phase.label=='fat: drive':
                if drive_start is None:
                    drive_start=float(d.time)
                    drive_baseline={label:float(d.joint(label+'_knob_joint').qpos[0]) for label in ['lower','upper']}
                    details={}
                    for label in ['lower','upper']:
                        details[label]={'driver_center_m':d.body(label+'_driver').xpos.tolist(),
                            'socket_center_m':d.body(label+'_knob').xpos.tolist(),
                            'motor_angle_rad':float(d.joint(label+'_motor_joint').qpos[0]),
                            'float_x_m':float(d.joint(label+'_float_x').qpos[0]),
                            'float_z_m':float(d.joint(label+'_float_z').qpos[0])}
                    (output/'engagement.json').write_text(json.dumps(details,indent=2))

                for label in ['lower','upper']:
                    drive_response[label]=max(drive_response[label],abs(float(d.joint(label+'_knob_joint').qpos[0])-drive_baseline[label]))
                if michelson:
                    if d.time-last_optics>3.:
                        measurement,intensity=optical_sample(f"step-{len(fringe_history):03d}")
                        fringe_status,commands=fringe_policy.decide(measurement)
                        fringe_history.append({'time_s':float(d.time),**measurement,'decision':fringe_status,'commands_rad':commands.tolist()})
                        print('FRINGE',measurement['visibility'],fringe_status,commands,flush=True)
                        for label,value in zip(['lower','upper'],commands):d.ctrl[m.actuator(label+'_motor').id]=value
                        last_optics=float(d.time)
                        if fringe_status!='move':controller.advance(d)
                else:
                    for label in ['lower','upper']:d.ctrl[m.actuator(label+'_motor').id]=.8
            if controller.unsafe_contacts:
                controller.failed='UNEXPECTED_PHYSICAL_CONTACT';break
            if monitor is not None:monitor.update(m,d,controller.phase.label)
            if viewer is not None:
                if not viewer.is_running():
                    controller.failed='VIEWER_CLOSED'
                    break
                if time.monotonic()-last_sync>=1/30:
                    viewer.set_texts((None,None,f'FAT contact / Michelson | requested {speed:g}x',controller.phase.label))
                    viewer.sync();last_sync=time.monotonic()
            mujoco.mj_step(m,d)
            pacer.wait(d.time)
        # Verify retention after release with the arm held, rather than sampling only the final controller tick.
        if controller.done:
            until=d.time+2.
            while d.time<until:
                controller(m,d);mujoco.mj_step(m,d)
                if state_observer is not None:state_observer(m,d,'post-release verification')
                pacer.wait(d.time)
        mujoco.mj_forward(m,d)
        knobs={n:float(d.joint(n+'_knob_joint').qpos[0]) for n in ['lower','upper']}
        tilt={n:float(d.joint('mirror_'+n).qpos[0]) for n in ['pitch','yaw']}
        result={'success':controller.done and not controller.unsafe_contacts and np.linalg.norm(d.body('fat').xpos-initial)<.01 and max_lift>.1 and held_samples/max(grip_samples,1)>.8 and contacts>0 and (assembly_optics or min(drive_response.values())>.1),
                'completed':controller.done,'failure':controller.failed or (None if controller.done else 'TIME_LIMIT'),'unsafe_contacts':controller.unsafe_contacts,'max_lift_m':max_lift,
                'requested_view_speed':speed,
                'bilateral_fraction':held_samples/max(grip_samples,1),'socket_contact_samples':contacts,
                'vision_refinement':vision_report,'drive_response_rad':drive_response,'max_knob_angles_rad':max_knobs,'knob_angles_rad':knobs,'mirror_tilt_rad':tilt,'phases':record,
                'michelson':{'enabled':michelson,'status':fringe_status,'history':fringe_history},'fat_return_error_m':float(np.linalg.norm(d.body('fat').xpos-initial)),
                'scope':('same-scene post-assembly contact adjustment; fresh visual fixture pose' if assembly_optics else 'known-fixture contact mechanics; provisional keyed socket and screw-lever constraints')}
        if michelson:
            final,field=optical_sample('released-512',512)
            fine,_=optical_sample('released-1024',1024)
            result['michelson']['released_grid_check']=fine
            result['michelson']['after_release']=final
            result['success']=bool(result['success'] and fringe_status=='accept' and final['valid'] and fine['valid'] and min(final['visibility'],fine['visibility'])>=optical_config['accept_visibility'] and abs(final['visibility']-fine['visibility'])<.005)
            if not result['success'] and not result['failure']:
                result['failure']='POST_RETURN_OPTICAL_ACCEPTANCE_FAILED' if fringe_status=='accept' else 'OPTICAL_FEEDBACK_FAILED'
            import cv2
            with mujoco.Renderer(m,width=1280,height=720) as renderer:
                camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
                camera.lookat[:]=[.05,.34,.86];camera.distance=1.;camera.azimuth=130;camera.elevation=-25
                renderer.update_scene(d,camera=camera)
                cv2.imwrite(str(output/'michelson-scene.png'),cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))
        np.savez_compressed(output/'state.npz',qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,time=d.time)
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        if michelson:
            from .michelson_optics import report_plot
            if final.get('phase_scan_power_W') and (output/'optical-initial/phase-00-detector16.png').exists():
                report_plot(output,result['michelson'],config=optical_config)
        print(json.dumps({k:v for k,v in result.items() if k not in ('phases','vision_refinement','michelson')}),flush=True)
        if viewer is not None and hold_seconds>0:
            if monitor is not None:monitor.update(m,d,'SUCCESS' if result['success'] else 'FAILED',force=True)
            end=time.monotonic()+hold_seconds
            while viewer.is_running() and time.monotonic()<end:
                viewer.sync();time.sleep(.03)
        return result
    finally:
        if monitor is not None and shared_monitor is None:monitor.close()
        if optical_player is not None and optical_player.poll() is None:
            optical_player.terminate()
            try:optical_player.wait(timeout=3)
            except subprocess.TimeoutExpired:optical_player.kill();optical_player.wait()
        if viewer is not None and shared_viewer is None:viewer.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--view',action='store_true');p.add_argument('--michelson',action='store_true');p.add_argument('--hold-seconds',type=float,default=30.)
    p.add_argument('--speed',type=positive_speed,default=1.,help='Live-view speed; physics timestep is unchanged')
    args=p.parse_args()
    if args.view and sys.platform=='darwin' and not os.environ.get('MJPYTHON_BIN'):
        launcher=Path(sys.executable).with_name('mjpython')
        os.execv(sys.executable,[sys.executable,str(launcher),'-m','simulation.fat_mechanics',*sys.argv[1:]])
    r=run(args.output,args.view,args.michelson,args.hold_seconds,speed=args.speed)
    raise SystemExit(0 if r['success'] else 1)
