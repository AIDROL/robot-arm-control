"""One persistent scene: stage modules -> visual assembly -> FAT -> interference."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import line_assembly as scene
from . import fat_mechanics as mechanics
from .housing_profile import STL
from .selected_housing import add_mount_targets
from .vision_pick_place import reserve_output_directory,run as place,require_pose
from .scene_recording import StateRecorder
from .timing import positive_speed

CONFIG=Path(__file__).resolve().parents[1]/'configs/optical_setups/michelson_full.json'


def load_layout(path=CONFIG):
    config=json.loads(Path(path).read_text())
    if config['schema_version']!='michelson-full-assembly-v1':raise ValueError('Unknown Michelson assembly schema')
    if tuple(c['id'] for c in config['components'])!=tuple(n for n,_ in scene.COMPONENTS):raise ValueError('Inventory order mismatch')
    targets={}
    for item in config['components']:
        R=Rotation.from_euler('z',item['yaw_rad']).as_matrix()
        T=np.eye(4);T[:3,:3]=R;T[:3,3]=np.array(item['optical_center_world_m'])-R@np.array([0,.004,-.014])
        if abs(T[2,3]-scene.TABLE_Z-STL.body_height)>1e-6:raise ValueError('Post height mismatch')
        if np.any(abs(T[:2,3]-[0,.13])>[.53,.47]):raise ValueError('Target outside supported table')
        targets[item['id']]=T
    return config,targets


def make_scene(config,targets):
    def configure(spec):
        mechanics.modify(spec)
        for label in ('lower','upper'):
            mesh=spec.mesh(label+'_hex');vertices=np.array(mesh.uservert).reshape(-1,3)
            vertices[:,1]*=config['driver_half_length_m']/.016;mesh.uservert=vertices.ravel()
            spec.joint(label+'_motor_joint').range=[-20,20]
            spec.actuator(label+'_motor').ctrlrange=[-20,20]
            spec.joint(label+'_knob_joint').frictionloss=config['knob_frictionloss_Nm']
        spec.body('fat').pos=[*config['fat_staging_xy_m'],.805]
        for index,item in enumerate(config['components']):
            name=item['id'];body=spec.body(name)
            body.pos=[(-.3+.15*index),.50,.874];body.quat=[1,0,0,0]
            if name!='mirror':add_mount_targets(spec,body,name)
            if name!='mirror':
                colors={'laser':[.3,.05,.05,1.],'splitter':[.3,.8,1.,.35],
                    'reference_mirror':[.8,.85,.9,1.],'detector':[.03,.03,.03,1.]}
                body.add_geom(name=name+'_optical_face',type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    pos=[0,.004,-.014],quat=[2**-.5,2**-.5,0,0],size=[.011,.0005],
                    rgba=colors[item['role']],contype=0,conaffinity=0,mass=0)
            # Kinematic holding represents a switchable magnetic base, engaged only
            # after contact-supported placement, at the attained pose (never a target snap).
            spec.add_equality(name=name+'_base_latch',type=mujoco.mjtEq.mjEQ_WELD,
                name1='world',name2=name,objtype=mujoco.mjtObj.mjOBJ_BODY,active=False,
                data=[0,0,0,0,0,0,1,0,0,0,1],solref=[.002,1],solimp=[.999,.9999,.0001,.5,2])
            marker=spec.geom('target_'+name);marker.pos[:2]=targets[name][:2,3]
            marker.quat=Rotation.from_matrix(targets[name][:3,:3]).as_quat()[[3,0,1,2]]
        bias=Rotation.from_euler('xz',[.00035,-.0003])
        spec.site('mirror_optical_axis').quat=(bias*Rotation.from_euler('x',-np.pi/2)).as_quat()[[3,0,1,2]]
        spec.geom('mirror_disc').quat=(bias*Rotation.from_euler('x',np.pi/2)).as_quat()[[3,0,1,2]]
    return scene.make_scene(randomize=False,include_fat=True,vision=True,mount_proxies=False,fat_key_proxies=False,
                            target_xs=tuple(targets[n][0,3] for n,_ in scene.COMPONENTS),spec_modifier=configure)


def latch_base(model,data,name):
    equality=model.equality(name+'_base_latch').id
    if data.eq_active[equality]:return
    pads={model.geom(f'{s}_finger_pad_{i}').id for s in ('left','right') for i in (1,2)}
    geoms=set(scene.component_collision_geoms(model,name));table=model.geom('workbench').id
    force=np.zeros(6);table_force=0.;pad_force=0.
    for i,c in enumerate(data.contact[:data.ncon]):
        a,b=int(c.geom1),int(c.geom2)
        if geoms.intersection((a,b)):
            mujoco.mj_contactForce(model,data,i,force)
            if table in (a,b):table_force+=max(0.,float(force[0]))
            if pads.intersection((a,b)):pad_force+=max(0.,float(force[0]))
    if table_force<.02 or pad_force>.05:raise RuntimeError('BASE_LATCH_REQUIRES_RELEASED_TABLE_SUPPORT:'+name)
    before=data.qpos.copy()
    model.eq_data[equality,3:6]=data.body(name).xpos
    model.eq_data[equality,6:10]=data.body(name).xquat
    data.eq_active[equality]=1
    mujoco.mj_forward(model,data)
    if not np.array_equal(before,data.qpos):raise RuntimeError('Latch unexpectedly changed generalized positions')


def run(output,view=False,assembly_only=False,resume=None,speed=1.):
    speed=positive_speed(speed)
    output=reserve_output_directory(Path(output));config,targets=load_layout()
    m=make_scene(config,targets);d=mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,m.key('home').id);mujoco.mj_forward(m,d)
    result={'success':False,'scope':'one persistent dynamic scene, empty experiment area with staged modules','placements':[],
            'started_from_empty_area':resume is None,
            'requested_view_speed':speed,
            'base_holding':'30 N contact-enabled attraction proxy during release, then ideal rigid latch at achieved pose; no magnetic field simulation',
            'config':config,'events':[]};viewer=None;monitor=None
    recorder=StateRecorder()
    magnetic=set()
    def enable_magnet(model,data,name):
        magnetic.add(name)
        event('MAGNET_FORCE_ENABLED',component=name,downward_force_N=30.)
    def fix_base(model,data,name):
        latch_base(model,data,name)
        magnetic.discard(name);data.xfrc_applied[model.body(name).id]=0
    def observe_step(model,data,phase):
        for name in magnetic:
            body=model.body(name).id;force=np.array([0.,0.,-30.])
            point=data.geom(name+'_magnetic_base').xpos
            data.xfrc_applied[body,:3]=force
            data.xfrc_applied[body,3:]=np.cross(point-data.xipos[body],force)
        recorder.capture(model,data,phase)
    def event(label,**extra):
        row={'time_s':float(d.time),'stage':label,**extra};result['events'].append(row)
        with (output/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('MICHELSON',row,flush=True)
    try:
        if resume is not None:
            previous=mujoco.MjModel.from_binary_path(str(Path(resume)/'assembly-model.mjb'))
            if (previous.nq,previous.nv,previous.nu)!=(m.nq,m.nv,m.nu):raise ValueError('Checkpoint topology mismatch')
            if [previous.joint(i).name for i in range(previous.njnt)]!=[m.joint(i).name for i in range(m.njnt)]:raise ValueError('Checkpoint joint order mismatch')
            for name,_ in scene.COMPONENTS:
                equality=m.equality(name+'_base_latch').id
                m.eq_data[equality]=previous.eq_data[previous.equality(name+'_base_latch').id]
            result['resume_scope']='saved generalized state and attained base poses; current explicit model configuration'
            saved=np.load(Path(resume)/'assembly-state.npz')
            mujoco.mj_setState(m,d,saved['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
            result['placements']=[x for x in json.loads((Path(resume)/'result.json').read_text())['placements'] if x['success']]
            event('RESUMED_ASSEMBLY',path=str(resume))
        if view:
            from mujoco import viewer as mjviewer
            from .camera_monitor import CameraMonitor
            viewer=mjviewer.launch_passive(m,d,show_right_ui=False)
            viewer.cam.lookat[:]=[0,.38,.88];viewer.cam.distance=1.25;viewer.cam.azimuth=135;viewer.cam.elevation=-30
            monitor=CameraMonitor(m,fps=1/max(1.,speed),open_browser=False);monitor.update(m,d,'EMPTY EXPERIMENT AREA',force=True);monitor.start_player(output)
        event('EMPTY_EXPERIMENT_AREA' if resume is None else 'ASSEMBLED_STATE')
        recorder.capture(m,d,'EMPTY EXPERIMENT AREA' if resume is None else 'RESUMED STATE')
        if len(result['placements'])<len(config['components']):
            for item in config['components']:
                if item['id'] in {x['id'] for x in result['placements'] if x['success']}:continue
                name=item['id'];event('PLACE_STARTED',component=name,role=item['role'])
                report=place(7,name,output/name,model=m,data=d,shared_viewer=viewer,shared_monitor=monitor,
                    target_pose=targets[name],placement_strategy='contact_supported',precision=True,require_precision_goal=False,
                    precision_position_tolerance=.0005,pickup_yaw_prior=0.,release_hook=fix_base,grip_command=120.,
                    state_observer=observe_step,support_hook=enable_magnet,speed=speed)
                result['placements'].append({'id':name,'role':item['role'],'success':report['success'],
                    'result_file':str(Path(report['output_directory'])/'result.json')})
                if not report['success']:raise RuntimeError('PLACEMENT_FAILED:'+name+':'+str(report['failure_code'])+':'+str(report.get('controller_failure')))
                fix_base(m,d,name);event('PLACED_AND_BASE_LATCHED',component=name)
                state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION))
                mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION)
                np.savez_compressed(output/'assembly-state.npz',state=state)
                mujoco.mj_saveModel(m,str(output/'assembly-model.mjb'),None)
                (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        event('ASSEMBLY_COMPLETE')
        if not assembly_only:
            from .global_camera import capture_observations,select_objects
            folder=output/'post-assembly-vision';folder.mkdir()
            cameras=capture_observations(m,d,folder);objects=select_objects(cameras)
            fixture=require_pose(objects['mirror'],float(d.time))
            (folder/'objects.json').write_text(json.dumps(objects,indent=2))
            event('FAT_STARTED')
            result['fat']=mechanics.run(output/'fat',view=view,michelson=True,model=m,data=d,
                fixture_pose=fixture,staging_xy=config['fat_staging_xy_m'],shared_viewer=viewer,shared_monitor=monitor,
                optical_config=config,assembly_optics=True,state_observer=observe_step,speed=speed)
            if not result['fat']['success']:raise RuntimeError('FAT_OR_OPTICS_FAILED')
        result['success']=True;event('COMPLETE')
    except Exception as error:
        result['failure']=str(error);event('FAILED',reason=str(error))
        mujoco.mj_saveModel(m,str(output/'failed-model.mjb'),None)
        state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION)
        np.savez_compressed(output/'failed-state.npz',state=state)
    finally:
        result['time_s']=float(d.time)
        recorder.capture(m,d,'COMPLETE' if result['success'] else 'FAILED',force=True)
        recorder.save(output,m,config)
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        if viewer is not None:
            viewer.set_texts((None,None,'COMPLETE' if result['success'] else 'FAILED',result.get('failure','Close scene window to exit')))
            end=time.monotonic()+30
            while viewer.is_running() and time.monotonic()<end:viewer.sync();time.sleep(.03)
            viewer.close()
        if monitor is not None:monitor.close()
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--view',action='store_true');parser.add_argument('--assembly-only',action='store_true');parser.add_argument('--resume',type=Path)
    parser.add_argument('--speed',type=positive_speed,default=1.,help='Live-view speed, e.g. 0.5, 1, 2 or 4; physics timestep is unchanged')
    args=parser.parse_args()
    if args.view and sys.platform=='darwin' and not os.environ.get('MJPYTHON_BIN'):
        os.execv(sys.executable,[sys.executable,str(Path(sys.executable).with_name('mjpython')),'-m','simulation.michelson_experiment',*sys.argv[1:]])
    result=run(args.output,args.view,args.assembly_only,args.resume,args.speed)
    raise SystemExit(0 if result['success'] else 1)
