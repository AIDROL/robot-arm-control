"""Bounded power-only search driving physical vision-guided pinhole re-placement."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco

from .optical_sensor import usable, publish


class PowerSearch:
    def __init__(self,threshold=.8,max_actions=8,step_m=.0002,max_travel_m=.001):
        if (not 0<threshold<=1 or type(max_actions) is not int or not 1<=max_actions<=8
                or not 0<step_m<=.0002 or not 0<max_travel_m<=.001):
            raise ValueError('Invalid optical adjustment bounds')
        self.threshold=threshold;self.max_actions=max_actions;self.step=step_m
        self.max_travel=max_travel_m;self.travel=0.;self.actions=0;self.direction=1.;self.previous=None

    def decide(self,observation,now):
        ok,reason=usable(observation,now)
        if not ok:return {'action':'stop','reason':reason}
        fraction=observation.get('transmitted_fraction')
        if not isinstance(fraction,(float,int)) or not np.isfinite(fraction) or not 0<=fraction<=1.0001:return {'action':'stop','reason':'INVALID_POWER'}
        if fraction>=self.threshold:return {'action':'accept','reason':'POWER_THRESHOLD_MET'}
        if self.actions>=self.max_actions:return {'action':'stop','reason':'ACTION_LIMIT'}
        if self.previous is not None and fraction<self.previous:
            self.direction*=-1;self.step*=.5
        if self.step<.00000625:return {'action':'stop','reason':'MIN_STEP_REACHED'}
        if self.travel+self.step>self.max_travel+1e-12:return {'action':'stop','reason':'TRAVEL_LIMIT'}
        self.previous=fraction;self.actions+=1;self.travel+=self.step
        return {'action':'move','shift_world_m':[0.,self.direction*self.step,0.],
                'reason':'BOUNDED_POWER_PROBE','attempt':self.actions}


def run(model,data,layout,spec_path,output,seed=7,viewer=None,monitor=None):
    from .lightpipes_validation import export_actual,evaluate
    from .vision_pick_place import run as place
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    spec=json.loads(Path(spec_path).read_text());policy=PowerSearch(spec['acceptance']['min_transmitted_fraction'])
    records=[];initial_visual_xy=None
    result={'success':False,'reason':None,'history':records,
        'policy':'optical power only; fresh RGB-D relative targets; physical re-grasp',
        'limits':{'step_m':.0002,'max_actions':8,'total_command_travel_m':.001},
        'scope':'Y translation of pinhole only; no FAT turning or object-state assignment'}
    try:
        while True:
            folder=output/f'observation-{len(records):02d}';folder.mkdir()
            mujoco.mj_forward(model,data)
            from . import global_camera as vision
            from .vision_pick_place import require_pose
            cameras=vision.capture_observations(model,data,folder)
            visual=vision.select_objects(cameras)['pinhole']
            visual_pose=require_pose(visual,float(data.time))
            if initial_visual_xy is None:initial_visual_xy=visual_pose[:2,3].copy()
            if np.linalg.norm(visual_pose[:2,3]-initial_visual_xy)>.001:
                raise RuntimeError('VISUAL_DISPLACEMENT_LIMIT')
            (folder/'visual-observation.json').write_text(json.dumps(visual,indent=2)+'\n')
            state=np.empty(mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_INTEGRATION))
            mujoco.mj_getState(model,data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
            from .housing_profile import for_model
            from . import line_assembly as scene
            np.savez(folder/'state.npz',state=state,seed=seed,housing_geometry=for_model(model).id,target_layout=scene.target_layout_id(model))
            # Simulator is the optical plant, not an input available to policy.
            actual=export_actual(layout,folder/'state.npz',folder/'actual-layout.json')
            metrics,intensity=evaluate(actual,spec,spec['grids'][-1])
            pin=next(s['local_transmission'] for s in metrics['stages'] if s['id']=='pinhole')
            observation=publish(folder/'detector',intensity,spec,float(data.time),pin)
            # Publish the same pixel-derived packet to the independent viewer.
            import shutil
            live=output/'detector';live.mkdir(exist_ok=True)
            for name in ['detector16.png','preview.png','observation.json']:
                temporary=live/(name+'.tmp');shutil.copyfile(folder/'detector'/name,temporary)
                temporary.replace(live/name)
            (folder/'observation.json').write_text(json.dumps(observation,indent=2)+'\n')
            (folder/'optical-evaluation.json').write_text(json.dumps(metrics,indent=2)+'\n')
            decision=policy.decide(observation,float(data.time))
            row={'observation':observation,'visual_observation_file':str(folder/'visual-observation.json'),'decision':decision};records.append(row)
            print('OPTICAL ADJUST',len(records),observation['transmitted_fraction'],decision,flush=True)
            if decision['action']!='move':
                result['success']=decision['action']=='accept';result['reason']=decision['reason'];break
            report=place(seed,'pinhole',output/f'move-{policy.actions:02d}',model=model,data=data,
                shared_viewer=viewer,shared_monitor=monitor,placement_strategy='settled',precision=True,
                require_precision_goal=False,relative_shift_m=decision['shift_world_m'],camera_monitor=False)
            row['physical_action']={'success':report['success'],'failure_code':report['failure_code'],
                                    'result_file':str(Path(report['output_directory'])/'result.json')}
            if not report['success']:
                result['reason']='PHYSICAL_ACTION_FAILED';break
    except Exception as error:
        result['reason']=type(error).__name__+': '+str(error)
    finally:
        result['actions']=policy.actions;result['command_travel_m']=policy.travel
        (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layout',type=Path,default=Path('configs/optical_setups/three_mounts.json'))
    p.add_argument('--spec',type=Path,default=Path('configs/optical_setups/three_mounts_optics.json'))
    args=p.parse_args()
    from . import line_assembly as scene
    from .optical_layout import load
    from .housing_profile import for_model
    plan=load(args.layout);saved=np.load(args.checkpoint,allow_pickle=False)
    model=scene.make_scene(seed=int(saved['seed']),include_fat=True,vision=True,target_xs=plan.target_xs,component_names=plan.names,layout_sha256=plan.sha256)
    if str(saved['housing_geometry'])!=for_model(model).id or str(saved['target_layout'])!=scene.target_layout_id(model):
        raise ValueError('Checkpoint binding mismatch')
    data=mujoco.MjData(model);mujoco.mj_setState(model,data,saved['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(model,data)
    result=run(model,data,args.layout,args.spec,args.output,int(saved['seed']))
    print(json.dumps({'success':result['success'],'reason':result['reason'],'actions':result['actions']}))
    return 0 if result['success'] else 1


if __name__=='__main__':raise SystemExit(main())
