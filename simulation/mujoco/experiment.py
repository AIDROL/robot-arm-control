"""Persistent-scene experiment: five visual placements -> FAT RGB alignment -> return."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

from . import line_assembly as scene
from . import global_camera as vision
from .fat_alignment import FatAlignmentController, TIP_TOLERANCE, ANGLE_TOLERANCE_DEG
from . import fat_vision
from .housing_profile import for_model
from .vision_pick_place import run as place, reserve_output_directory, write_json, require_pose


def run(seed,output,view=False,hide_fat_marker=False,resume_assembly=None,layout=None,optics_spec=None,adjust_optics=False):
    if adjust_optics and optics_spec is None:
        raise ValueError('Optical adjustment requires --optics-spec and --layout')
    if optics_spec is not None and layout is None:
        raise ValueError('Optical validation requires --layout')
    output=reserve_output_directory(output)
    result={'seed':seed,'success':False,'mechanical_success':False,'failure_code':None,'placements':[], 'fat_observations':[],
            'scope':'all placements -> FAT pickup -> side-marker visual alignment -> FAT return',
            'placement_acceptance':'existing 10 mm XY mechanical checks; 0.5 mm precision goal reported separately',
            'fat_hold_endpoints':{},
            'limitations':['FAT is a kinematic mocap pickup proxy; no force/torque or insertion physics',
                           'hole targets derived from calibrated marker offsets; not directly detected hole edges',
                           'no screw turning or optical feedback']}
    events=[]
    scene_args = {}
    if layout is not None:
        from .optical_layout import load
        adapted = load(layout)
        scene_args = {'component_names': adapted.names, 'layout_sha256': adapted.sha256}
        result['layout_adaptation'] = adapted.report
        write_json(output/'layout-adaptation.json', adapted.report)
    model=scene.make_scene(seed=seed,include_fat=True,vision=True,
                           target_xs=scene.FAT_TARGET_XS if layout is None else adapted.target_xs,
                           **scene_args)
    data=mujoco.MjData(model)
    geometry=for_model(model)
    result['housing_geometry']=geometry.id
    result['target_layout']=scene.target_layout_id(model)
    components=scene.components_for_model(model)
    result['components']=[name for name,_ in components]
    result['optical_axis_height_m']=geometry.body_height+geometry.screw_midpoint[2]
    result['limitations'].append('Post/base dimensions and masses are provisional; no magnetic force or fastening model')
    result['target_yaw_rad']=geometry.placement_yaw
    mujoco.mj_resetDataKeyframe(model,data,model.key('home').id)
    mujoco.mj_forward(model,data)
    if resume_assembly is not None:
        saved=np.load(resume_assembly,allow_pickle=False)
        if int(saved['seed'])!=seed:raise ValueError('Checkpoint seed mismatch')
        saved_profile=str(saved['housing_geometry']) if 'housing_geometry' in saved else 'proxy'
        if saved_profile!=geometry.id:raise ValueError('Checkpoint housing geometry mismatch; reassemble with the current model')
        saved_layout=str(saved['target_layout']) if 'target_layout' in saved else 'legacy_yaw0'
        if saved_layout!=result['target_layout']:
            raise ValueError('Checkpoint target layout mismatch; reassemble with the 90-degree placement layout')
        if saved['state'].size!=mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_INTEGRATION):
            raise ValueError('Checkpoint state size mismatch')
        mujoco.mj_setState(model,data,saved['state'],mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model,data)
        result['resumed_from']=str(resume_assembly)
        result['placements']=json.loads((resume_assembly.parent/'assembly-placements.json').read_text())
    viewer,monitor=None,None
    optical_player=None
    def event(stage,**fields):
        record={'stage':stage,'simulation_time':float(data.time),**fields}
        events.append(record)
        with (output/'events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(f"EXPERIMENT {stage} t={data.time:.2f}",flush=True)
    try:
        event('SCENE_READY')
        if view:
            from mujoco import viewer as mjviewer
            from .camera_monitor import CameraMonitor
            viewer=mjviewer.launch_passive(model,data,show_right_ui=False)
            cam=vision.overview_camera(model)
            viewer.cam.lookat[:]=cam.lookat
            viewer.cam.distance,viewer.cam.azimuth,viewer.cam.elevation=cam.distance,cam.azimuth,cam.elevation
            monitor=CameraMonitor(model,open_browser=False)
            monitor.update(model,data,'EXPERIMENT START',force=True)
            monitor.start_player(output)
        for name,_ in components:
            if resume_assembly is not None:continue
            event('PLACE_STARTED',object=name)
            report=place(seed,name,output/name,placement_strategy='settled',precision=True,
                         model=model,data=data,shared_viewer=viewer,shared_monitor=monitor,
                         require_precision_goal=False)
            result['placements'].append({'object':name,'success':report['success'],
                                         'precision_goal_met':report.get('precision_goal_met'),
                                         'result_file':str(Path(report['output_directory'])/'result.json')})
            if not report['success']:
                raise RuntimeError(f"PLACE_FAILED:{name}:{report['failure_code']}")
            event('PLACE_VERIFIED',object=name)
        event('ASSEMBLY_COMPLETE')
        if geometry.is_selected:
            from .beamline_geometry import inspect
            result['beamline_geometry']=inspect(model,data)
            write_json(output/'beamline-geometry.json',result['beamline_geometry'])
            if not result['beamline_geometry']['success']:
                raise RuntimeError('APERTURE_GEOMETRY_FAILED')
            event('APERTURE_GEOMETRY_VERIFIED')
        state=np.empty(mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(model,data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
        np.savez(output/'assembly-state.npz',state=state,seed=seed,housing_geometry=geometry.id,target_layout=result['target_layout'])
        write_json(output/'assembly-placements.json',result['placements'])
        optics_state=output/'assembly-state.npz'
        if adjust_optics:
            if view:
                import subprocess
                optical_player=subprocess.Popen([sys.executable,'-m','simulation.optical_sensor',
                    '--folder',str(output/'adjustment/detector'),'--watch'])
            from .optical_adjustment import run as adjust
            result['optical_adjustment']=adjust(model,data,layout,optics_spec,output/'adjustment',seed,viewer,monitor)
            if not result['optical_adjustment']['success']:
                raise RuntimeError('OPTICAL_ADJUSTMENT_FAILED:'+result['optical_adjustment']['reason'])
            mujoco.mj_getState(model,data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
            optics_state=output/'post-adjustment-state.npz'
            np.savez(optics_state,state=state,seed=seed,housing_geometry=geometry.id,target_layout=result['target_layout'])
        if optics_spec is not None:
            from .lightpipes_validation import export_actual, validate
            actual=export_actual(layout,optics_state,output/'actual-optical-layout.json')
            optical=validate(layout,optics_spec,output/'optics',actual)
            result['optical_validation']={'backend_verified':optical['backend_verified'],
                'actual_optical_pass':optical['actual_optical_pass'],
                'observation_valid':optical['actual_observation_valid'],'report_file':str(output/'optics/report.json')}
            event('OPTICS_EVALUATED',actual_pass=optical['actual_optical_pass'])
            result['optical_observation']=optical['cases']['actual']['detector_observation']
            if view and optical_player is None:
                import subprocess
                optical_player=subprocess.Popen([sys.executable,'-m','simulation.optical_sensor',
                    '--folder',str(output/'optics/actual-detector'),'--watch'])
        folder=output/'post_assembly';folder.mkdir()
        cameras=vision.capture_observations(model,data,folder)
        estimates=vision.select_objects(cameras)
        poses={name:require_pose(estimates[name],float(data.time)) for name,_ in components}
        write_json(folder/'observations.json',{'cameras':cameras,'objects':estimates})
        controller=FatAlignmentController(model,data,object_poses=poses,
                                           assembly_complete=True,visual_alignment=True)
        start=float(data.time);previous=None;last_sync=-1.;finished=None
        event('FAT_STARTED')
        while data.time-start<300:
            wall=time.monotonic()
            if viewer is not None and not viewer.is_running():raise RuntimeError('VIEWER_CLOSED')
            controller(model,data)
            if controller.failed:raise RuntimeError('FAT_MOTION_FAILED:'+controller.failed)
            if controller.pending_visual_target is not None:
                name=controller.pending_visual_target
                observation=fat_vision.capture(model,data,controller,name,output,hide_fat_marker)
                result['fat_observations'].append({'object':name,**observation})
                write_json(output/'fat-observations.json',result['fat_observations'])
                if not observation['valid']:raise RuntimeError(observation['reason']+':'+name)
                controller.apply_marker_observation(data,name,np.array(observation['T_world_marker']))
                event('FAT_VISUAL_REFINED',object=name)
            label=controller.phase.label
            if label!=previous:
                if previous is not None and previous.endswith(': hold'):
                    name=previous[4:].split(':')[0]
                    errors=[float(np.linalg.norm(data.site(f'fat_{key}_key_tip').xpos-
                                                data.site(f'{name}_{key}_screw_center').xpos))
                            for key,*_ in scene.FAT_KEY_OFFSETS]
                    result['fat_hold_endpoints'][name]={'tip_errors_m':errors,'max_tip_error_m':max(errors)}
                    event('FAT_HOLD_VERIFIED',object=name,max_tip_error_m=max(errors))
                event('FAT_PHASE',phase=label);previous=label
            if controller.unsafe_contacts:raise RuntimeError('FAT_UNSAFE_CONTACT')
            if monitor is not None:monitor.update(model,data,label)
            if viewer is not None and data.time-last_sync>=1/30:
                viewer.set_texts((None,None,'Experiment',label));viewer.sync();last_sync=float(data.time)
            if controller.done:
                if finished is None:finished=float(data.time)
                if data.time-finished>=1.:break
            mujoco.mj_step(model,data)
            if view:time.sleep(max(0.,model.opt.timestep-(time.monotonic()-wall)))
        if not controller.done:raise RuntimeError('FAT_TIMEOUT')
        result['fat_alignment']={name:{'both_tip_max_error_m':result['fat_hold_endpoints'].get(name,{}).get('max_tip_error_m',None),
                                       'angle_error_deg':controller.best_angle_error[name],
                                       'visually_refined':name in controller.visual_refined}
                                  for name,_ in components}
        for name,metrics in result['fat_alignment'].items():
            if (not metrics['visually_refined'] or metrics['both_tip_max_error_m'] is None or metrics['both_tip_max_error_m']>TIP_TOLERANCE
                    or metrics['angle_error_deg']>ANGLE_TOLERANCE_DEG):
                raise RuntimeError('FAT_ALIGNMENT_FAILED:'+name)
        result['fat_return_error_m']=float(np.linalg.norm(data.mocap_pos[controller.mocap_id]-controller.fat_start_position))
        if result['fat_return_error_m']>.015:raise RuntimeError('FAT_RETURN_FAILED')
        result['mechanical_success']=True
        result['success']=optics_spec is None or (result['optical_validation']['backend_verified'] and result['optical_validation']['actual_optical_pass'] and result['optical_validation']['observation_valid'])
        if not result['success']:result['failure_code']='OPTICAL_ACCEPTANCE_FAILED'
        event('VERIFIED' if result['success'] else 'OPTICAL_ACCEPTANCE_FAILED')
    except Exception as error:
        result['failure_code']=str(error)
        event('FAILED',reason=str(error))
    finally:
        result['simulation_seconds']=float(data.time)
        result['events']=events
        write_json(output/'result.json',result)
    print(f"EXPERIMENT success={result['success']} output={output}",flush=True)
    try:
        if viewer is not None:
            status='EXPERIMENT SUCCESS' if result['success'] else 'EXPERIMENT FAILED'
            if monitor is not None:
                monitor.update(model,data,status,force=True);monitor.finish(status)
            while viewer.is_running():
                viewer.set_texts((None,None,status,'Close scene window to exit'));viewer.sync();time.sleep(.05)
    finally:
        if viewer is not None:viewer.close()
        if monitor is not None:monitor.close()
        if optical_player is not None and optical_player.poll() is None:
            optical_player.terminate()
            try:optical_player.wait(timeout=3)
            except subprocess.TimeoutExpired:optical_player.kill();optical_player.wait()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--view',action='store_true')
    p.add_argument('--hide-fat-marker',action='store_true',help='Negative test: invalidate FAT RGB observation')
    p.add_argument('--resume-assembly',type=Path,help='Resume the saved completed assembly, without re-running placements')
    p.add_argument('--layout',type=Path,help='Structured three-component layout JSON (assembly adaptation)')
    p.add_argument('--optics-spec',type=Path,help='Optional LightPipes optical parameters and acceptance')
    p.add_argument('--adjust-optics',action='store_true',help='Bounded power-guided physical pinhole re-placement')
    args=p.parse_args()
    if args.view and sys.platform=='darwin' and not os.environ.get('MJPYTHON_BIN'):
        os.execv(sys.executable,[sys.executable,str(Path(sys.executable).with_name('mjpython')),
                                '-m','simulation.experiment',*sys.argv[1:]])
    r=run(args.seed,args.output,args.view,args.hide_fat_marker,args.resume_assembly,args.layout,args.optics_spec,args.adjust_optics)
    raise SystemExit(0 if r['success'] else 1)
