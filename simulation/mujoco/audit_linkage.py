"""Causal data-path audit, not a precision/repeatability experiment.

Interventions are test-only. No injected observations are executed as motions.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import global_camera as vision
from . import line_assembly as scene
from . import wrist_camera
from .fat_alignment import FatAlignmentController
from .vision_pick_place import reserve_output_directory, require_pose, WristController, write_json
from .task_space_control import phases_for_scene
from . import housing_profile as profiles


def audit(evidence, output):
    output=reserve_output_directory(output)
    checks={}
    complete=json.loads((evidence/'result.json').read_text())
    placement_file=evidence/'lens_1'/'result.json'
    if not placement_file.exists():
        placement_file=Path(next(p['result_file'] for p in complete['placements'] if p['object']=='lens_1'))
    placement_folder=placement_file.parent
    placement=json.loads(placement_file.read_text())
    geometry=profiles.from_id(placement.get('housing_geometry',profiles.PROXY.id))
    model=scene.make_scene(seed=7,include_fat=True,vision=True,target_xs=scene.FAT_TARGET_XS,
                           housing_model='stl' if geometry.id==profiles.STL.id else 'proxy')
    data=mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,data,model.key('home').id)
    mujoco.mj_forward(model,data)
    def observe(folder):
        folder.mkdir()
        cams=vision.capture_observations(model,data,folder)
        pose=require_pose(vision.select_objects(cams)['lens_1'],float(data.time))
        target=next(p.position.copy() for p in phases_for_scene(model,data,scene.FAT_TARGET_XS,
                    {'lens_1':pose}) if p.label=='lens_1: approach')
        return cams,pose,target
    before,T0,g0=observe(output/'before_object_move')
    injected=np.array([.012,-.008,0.])
    adr=int(model.jnt_qposadr[model.joint('lens_1_free').id])
    data.qpos[adr:adr+3]+=injected
    mujoco.mj_forward(model,data)
    after,T1,g1=observe(output/'after_object_move')
    checks['object_move_image_to_grasp']={
        'passed':bool(np.linalg.norm((T1[:3,3]-T0[:3,3])-injected)<.001
                      and np.allclose(g1-g0,T1[:3,3]-T0[:3,3])),
        'injected_world_delta_m':injected.tolist(),'estimated_world_delta_m':(T1[:3,3]-T0[:3,3]).tolist(),
        'grasp_target_delta_m':(g1-g0).tolist(),
        'image_hashes':{cam:[hashlib.sha256((output/'before_object_move'/f'{cam}_rgb.png').read_bytes()).hexdigest(),
                            hashlib.sha256((output/'after_object_move'/f'{cam}_rgb.png').read_bytes()).hexdigest()]
                        for cam in vision.CAMERAS}}
    # Deliberately replay the old image-derived pose while object truth is moved.
    # A planner secretly using object truth would now follow the injected move.
    replay=next(p.position for p in phases_for_scene(model,data,scene.FAT_TARGET_XS,{'lens_1':T0})
                if p.label=='lens_1: approach')
    checks['truth_changed_observation_frozen']={'passed':bool(np.allclose(replay,g0)),
        'target_delta_m':(replay-g0).tolist(),'note':'Counterfactual replay is audit-only, not normal sensor acquisition.'}
    empty={cam:{'estimates':[],'simulation_time':0.} for cam in vision.CAMERAS}
    try:
        require_pose(vision.select_objects(empty)['lens_1'],0.)
        rejected=False
    except ValueError as error:
        rejected=str(error)=='MARKER_NOT_DETECTED'
    checks['global_observation_missing']={'passed':rejected}

    fine=placement['fine_observation']
    rgb=cv2.cvtColor(cv2.imread(str(placement_folder/'wrist_rgb.png')),cv2.COLOR_BGR2RGB)
    depth=np.load(placement_folder/'wrist_depth_m.npy',allow_pickle=False)
    coarse=np.array(placement['coarse_control_pose']);K=np.array(fine['K']);WC=np.array(fine['T_world_camera'])
    f0,_=wrist_camera.refine(rgb,depth,K,WC,coarse,0,fine['timestamp'],geometry)
    f1,_=wrist_camera.refine(rgb,depth+.002,K,WC,coarse,0,fine['timestamp'],geometry)
    targets=[]
    for f in [f0,f1]:
        if not f['valid']:raise RuntimeError('Wrist evidence audit invalid: '+f['reason'])
        controller=WristController(model,data,scene.FAT_TARGET_XS,object_poses={'lens_1':coarse})
        controller.apply_refinement(data,np.array(f['T_world_object'])[:3,3]-coarse[:3,3])
        targets.append(next(p.position.copy() for p in controller.phases if p.label=='lens_1: approach'))
    delta=targets[1]-targets[0]
    checks['depth_to_final_grasp']={'passed':bool(.0015<np.linalg.norm(delta)<.003),
        'injected_optical_depth_delta_m':.002,'grasp_target_delta_m':delta.tolist()}
    invalid,_=wrist_camera.refine(rgb,np.full_like(depth,np.nan),K,WC,coarse,0,fine['timestamp'],geometry)
    checks['invalid_depth_no_pose_fallback']={'passed':not invalid['valid'] and 'T_world_object' not in invalid}

    assembled=json.loads((evidence/'post_assembly'/'observations.json').read_text())['objects']
    poses={name:np.array(assembled[name]['T_world_object']) for name,_ in scene.COMPONENTS}
    observation=json.loads((evidence/'fat-observations.json').read_text())[0]
    marker=np.array(observation['T_world_marker'])
    fat_targets=[]
    for dy in [0.,.002]:
        c=FatAlignmentController(model,data,object_poses=copy.deepcopy(poses),assembly_complete=True,visual_alignment=True)
        # Known tool calibration for target-generation audit; no FAT motion run.
        c.relative_tip_position=np.array([0.,.04,-.025])
        c.relative_rotation=Rotation.identity()
        c.phases=c._alignment_phases(data)
        c.phase_index=next(i for i,p in enumerate(c.phases) if p.label=='fat lens_1: prealign')
        others={p.label:p.position.copy() for p in c.phases if p.label.startswith('fat pinhole:')}
        changed=marker.copy();changed[1,3]+=dy
        c.apply_marker_observation(data,'lens_1',changed)
        assert all(np.array_equal(p.position,others[p.label]) for p in c.phases if p.label in others)
        fat_targets.append(next(p.position.copy() for p in c.phases if p.label=='fat lens_1: insert'))
    delta=fat_targets[1]-fat_targets[0]
    checks['fat_observation_to_hole_approach']={'passed':bool(np.allclose(delta,[0,.002,0],atol=1e-10)),
        'injected_marker_world_delta_m':[0,.002,0],'tool_target_delta_m':delta.tolist(),
        'unrelated_mount_targets_unchanged':True}
    try:
        FatAlignmentController(model,data,assembly_complete=True,visual_alignment=True)
        forbidden=False
    except ValueError:
        forbidden=True
    checks['fat_visual_mode_requires_observations']={'passed':forbidden}

    starts=[json.loads(Path(p['result_file']).read_text())['scene_start_time'] for p in complete['placements']]
    checks['persistent_scene']={'passed':len(starts)==5 and all(b>a for a,b in zip(starts,starts[1:])),
                                'placement_start_times':starts,'full_run_success':complete['success'],
                                'resumed_from':complete.get('resumed_from')}
    result={'success':all(c['passed'] for c in checks.values()),'checks':checks,
        'evidence':str(evidence.resolve()),
        'boundary':'This proves tested simulation data paths, not real hardware physics. FAT remains mocap-attached and interpolated home.'}
    write_json(output/'audit.json',result)
    lines=['# 链路因果审计','','| 检查 | 结果 |','| --- | --- |']
    lines += [f"| {name} | {'PASS' if c['passed'] else 'FAIL'} |" for name,c in checks.items()]
    lines += ['',result['boundary']]
    (output/'audit.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,default=Path('runs/full-chain-final'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    r=audit(args.evidence,args.output)
    raise SystemExit(0 if r['success'] else 1)
