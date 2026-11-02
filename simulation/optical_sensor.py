"""Independent ideal detector observation and native image viewer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np


def observe(intensity, window_m, input_power_W, simulation_time_s=None, full_scale_W_m2=1200.):
    """Linear 16-bit ideal sensor. Metrics derive from quantized pixels only."""
    a=np.asarray(intensity,dtype=float)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or a.shape[0]<8:
        raise ValueError('Detector requires a square intensity image')
    if not all(np.isfinite(v) and v>0 for v in (window_m,input_power_W,full_scale_W_m2)):
        raise ValueError('Invalid detector calibration')
    finite=bool(np.isfinite(a).all() and np.all(a>=0))
    clean=np.maximum(np.nan_to_num(a,nan=0.,posinf=0.,neginf=0.),0.)
    counts=np.rint(np.clip(clean/full_scale_W_m2,0,1)*65535).astype(np.uint16)
    signal=counts.astype(float)*full_scale_W_m2/65535
    n=len(a);dx=window_m/n
    total=float(signal.sum());power=total*dx*dx
    yy,zz=np.meshgrid((np.arange(n)-n//2)*dx,(np.arange(n)-n//2)*dx)
    centroid=None if total<=0 else [float((signal*v).sum()/total) for v in (yy,zz)]
    border=max(1,n//50);edges=np.ones(a.shape,dtype=bool);edges[border:-border,border:-border]=False
    edge_fraction=float(signal[edges].sum()/max(total,1e-300))
    reasons=[]
    if not finite:reasons.append('INVALID_INTENSITY')
    if power<1e-10 or np.count_nonzero(counts)<9:reasons.append('NO_SIGNAL')
    if np.any(clean>=full_scale_W_m2):reasons.append('SATURATED')
    if edge_fraction>.02:reasons.append('BEAM_AT_SENSOR_EDGE')
    obs={'schema_version':'optical-observation-v1','valid':not reasons,'reasons':reasons,
         'capture_id':str(time.time_ns()),'wall_time_unix_s':time.time(),
         'simulation_time_s':simulation_time_s,'frame_id':'detector_yz_offset_from_input_axis',
         'power_W':power,'transmitted_fraction':power/input_power_W,
         'centroid_m':centroid,'centroid_yz_m':centroid,'pixel_pitch_m':dx,'window_m':window_m,
         'shape':list(a.shape),'full_scale_W_m2':full_scale_W_m2,
         'reference_input_power_W':input_power_W,'edge_power_fraction':edge_fraction,
         'source':'quantized ideal 16-bit LightPipes detector pixels',
         'sensor_model':'linear; fixed gain; no dark current, read noise or lens distortion',
         'validity_scope':'measurement usability, independent of optical task acceptance'}
    return counts,obs


def display(counts, observation):
    # Fixed exposure across frames/cases. Do not auto-normalize weak beams.
    levels=np.log1p(1000*counts.astype(float)/65535)/np.log(1001)
    rgb=cv2.applyColorMap(np.rint(levels*255).astype(np.uint8),cv2.COLORMAP_INFERNO)
    rgb=cv2.flip(cv2.resize(rgb,(640,640)),0)  # physical +Z points up
    center=observation.get('centroid_m',observation.get('centroid_yz_m'))
    if center is not None:
        x=int(320+center[0]/observation['window_m']*640)
        y=int(320-center[1]/observation['window_m']*640)
        cv2.drawMarker(rgb,(x,y),(80,255,80),cv2.MARKER_CROSS,18,1)
    canvas=np.full((780,640,3),20,np.uint8);canvas[:640]=rgb
    pinhole=observation.get('model_pinhole_local_transmission')
    lines=[f'VALID={observation["valid"]} {", ".join(observation["reasons"]) or "OK"}',
           f'Power {observation["power_W"]*1e6:.3f} uW | total transmission {observation["transmitted_fraction"]*100:.3f}%',
           f'Centroid {observation.get("display_axes","Y/Z")} (mm): {None if center is None else [round(v*1e3,4) for v in center]}',
           f'Fixed LOG display 0-{observation["full_scale_W_m2"]:g} W/m2 | {observation.get("display_axes","Y/Z")} coordinates',
           f'Sim t={None if observation["simulation_time_s"] is None else round(observation["simulation_time_s"],2)} | ideal 16-bit optical detector',
           ('Pinhole local transmission (MODEL): '+f'{pinhole*100:.3f}%') if pinhole is not None else observation.get('task_label','Pinhole local transmission (MODEL): n/a')]
    for i,line in enumerate(lines):cv2.putText(canvas,line,(8,660+i*19),cv2.FONT_HERSHEY_SIMPLEX,.40,(230,230,230),1,cv2.LINE_AA)
    return canvas


def publish(folder, intensity, spec, simulation_time_s=None, model_pinhole_transmission=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    counts,obs=observe(intensity,spec['window_m'],spec['input_power_W'],simulation_time_s,
        full_scale_W_m2=spec.get('full_scale_W_m2',1200.))
    obs['model_pinhole_local_transmission']=model_pinhole_transmission
    obs['task_label']=spec.get('task_label','Pinhole local transmission (MODEL): n/a')
    obs['display_axes']=spec.get('display_axes','Y/Z')
    obs['frame_id']=spec.get('frame_id',obs['frame_id'])
    if obs['display_axes']=='X/Z':
        obs['centroid_xz_m']=obs.pop('centroid_yz_m')
    obs['pinhole_metric_source']='LightPipes model power ratio before/after aperture; not independently measured by output camera'
    preview=display(counts,obs)
    for name,data in [('detector16.png',counts),('preview.png',preview)]:
        temporary=folder/(name+'.tmp.png')
        if not cv2.imwrite(str(temporary),data):raise RuntimeError('Detector image encoding failed')
        obs[name+'_sha256']=hashlib.sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary,folder/name)
    temporary=folder/'observation.json.tmp'
    temporary.write_text(json.dumps(obs,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,folder/'observation.json')
    return obs


def usable(observation, now_simulation_s, max_age_s=.5):
    """Consumer-side gate: archived valid pixels can still be stale for control."""
    if not observation.get('valid'):
        return False, 'INVALID_MEASUREMENT'
    stamp=observation.get('simulation_time_s')
    if stamp is None or not np.isfinite(stamp) or not np.isfinite(now_simulation_s):
        return False, 'MISSING_SIMULATION_TIMESTAMP'
    if not 0<=now_simulation_s-stamp<=max_age_s:
        return False, 'STALE_OR_FUTURE_OBSERVATION'
    return True, 'OK'


def read(folder):
    folder=Path(folder);obs=json.loads((folder/'observation.json').read_text())
    raw=(folder/'preview.png').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=obs['preview.png_sha256']:
        raise ValueError('Detector frame and observation do not match')
    return obs,cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)


def play(folder, watch=False, max_seconds=None):
    title='Optical detector - LightPipes'
    cv2.namedWindow(title,cv2.WINDOW_NORMAL);cv2.resizeWindow(title,640,790)
    started=time.monotonic();previous=None;frames=0;last=None
    try:
        while max_seconds is None or time.monotonic()-started<max_seconds:
            try:
                obs,canvas=read(folder)
                if obs['capture_id']!=previous:frames+=1;previous=obs['capture_id']
                age=max(0.,time.time()-obs['wall_time_unix_s'])
                fresh=age<=2.
                label=('NEW/RECENT OBSERVATION' if fresh else 'HELD FRAME: awaiting new optical calculation') if watch else 'SAVED OBSERVATION'
                canvas=cv2.copyMakeBorder(canvas,26,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
                cv2.putText(canvas,label,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.39,(150,220,250),1,cv2.LINE_AA)
                cv2.imshow(title,canvas);last=obs
            except (OSError,ValueError,KeyError):
                canvas=np.zeros((200,640,3),np.uint8)
                cv2.putText(canvas,'Waiting for the next optical measurement',(10,70),cv2.FONT_HERSHEY_SIMPLEX,.5,(230,230,230),1)
                cv2.imshow(title,canvas)
            if cv2.waitKey(100)&255 in (27,ord('q')):break
            try:
                if cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
            except cv2.error:break
    finally:cv2.destroyWindow(title)
    return {'displayed_observations':frames,'last_capture_id':None if last is None else last['capture_id']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--watch',action='store_true');p.add_argument('--max-seconds',type=float)
    args=p.parse_args();print(json.dumps(play(args.folder,args.watch,args.max_seconds)))
