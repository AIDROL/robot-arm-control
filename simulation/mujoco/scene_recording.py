"""Record actual simulated states and replay them in native, separate windows."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import mujoco
import numpy as np
from .timing import positive_speed


class StateRecorder:
    def __init__(self,fps=30):
        self.period=1/fps;self.next_time=-1.;self.last_time=-1.;self.rows=[]

    def capture(self,model,data,phase,force=False):
        stamp=float(data.time)
        if stamp<self.last_time-1e-9:raise RuntimeError('Scene time reset during episode recording')
        self.last_time=stamp
        if stamp+1e-9<self.next_time and not force:return
        self.rows.append((stamp,data.qpos.copy(),str(phase)));self.next_time=stamp+self.period

    def save(self,folder,model,config):
        folder=Path(folder)
        mujoco.mj_saveModel(model,str(folder/'episode-model.mjb'),None)
        np.savez_compressed(folder/'episode.npz',time=np.array([r[0] for r in self.rows]),
            qpos=np.array([r[1] for r in self.rows]),phase=np.array([r[2] for r in self.rows]),
            config=json.dumps(config),source='actual dynamic rollout states; replay uses forward kinematics, not a new physics trial')


def draw_paths(scene,paths,clear=True):
    if clear:scene.ngeom=0
    for index,path in enumerate(paths):
        color=[1.,.2,.1,.8] if index==0 else [.1,.9,1.,.8]
        points=path['points_m']
        for a,b in zip(points[:-1],points[1:]):
            if scene.ngeom>=scene.maxgeom:return
            geom=scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),color)
            mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.0004,np.array(a),np.array(b))
            scene.ngeom+=1


def publish_recorded_detector(folder,config,path,observation,stamp):
    import cv2
    from simulation.optical_sensor import publish
    counts=cv2.imread(str(path/'phase-00-detector16.png'),cv2.IMREAD_UNCHANGED)
    if counts is None or counts.dtype!=np.uint16:raise ValueError('Missing or invalid recorded 16-bit detector frame')
    intensity=counts.astype(float)*config['detector_full_scale_W_m2']/65535
    return publish(folder,intensity,{'window_m':config['detector_window_m'],'input_power_W':config['input_power_W'],
        'full_scale_W_m2':config['detector_full_scale_W_m2'],'frame_id':'michelson_detector_xz','display_axes':'X/Z',
        'task_label':f"RECORDED visibility: {observation['visibility']*100:.2f}%"},stamp)


def play(folder,speed=1.,cameras=True,hold_seconds=30.,start_seconds=0.):
    from mujoco import viewer as mjviewer
    from simulation.optical_sensor import publish
    folder=Path(folder);saved=np.load(folder/'episode.npz',allow_pickle=False)
    speed=positive_speed(speed)
    stamps=saved['time'];positions=saved['qpos'];phases=saved['phase'];config=json.loads(str(saved['config']))
    if not 0<=start_seconds<=stamps[-1]-stamps[0]:raise ValueError('Replay start outside recorded duration')
    replay_begin=float(stamps[0])+start_seconds
    model=mujoco.MjModel.from_binary_path(str(folder/'episode-model.mjb'));data=mujoco.MjData(model)
    first=max(0,int(np.searchsorted(stamps,replay_begin,side='right')-1))
    data.qpos[:]=positions[first];data.time=float(stamps[first]);mujoco.mj_forward(model,data)
    optical=[]
    for p in (folder/'fat').glob('optical-*'):
        state=p/'state.npz';measurement=p/'measurement.json';image=p/'phase-00-detector16.png'
        if state.exists() and measurement.exists() and image.exists():
            optical.append((float(np.load(state)['time']),p,json.loads(measurement.read_text())))
    optical.sort(key=lambda x:x[0]);cursor=-1;monitor=None;player=None
    window=mjviewer.launch_passive(model,data,show_right_ui=False)
    window.cam.lookat[:]=[0,.38,.88];window.cam.distance=1.3;window.cam.azimuth=135;window.cam.elevation=-30
    try:
        if cameras:
            from .camera_monitor import CameraMonitor
            from .vision_pick_place import reserve_output_directory
            display_folder=reserve_output_directory(folder/'replay')
            monitor=CameraMonitor(model,fps=1,open_browser=False);monitor.start_player(display_folder)
            player=subprocess.Popen([sys.executable,'-m','simulation.optical_sensor','--folder',str(display_folder/'detector'),'--watch'])
        start=time.monotonic();end_sim=float(stamps[-1]);last_index=-1
        while window.is_running():
            elapsed=(time.monotonic()-start)*speed+replay_begin;index=min(len(stamps)-1,max(0,int(np.searchsorted(stamps,elapsed,side='right')-1)))
            if index!=last_index:
                data.qpos[:]=positions[index];data.time=float(stamps[index]);mujoco.mj_forward(model,data)
                if monitor is not None:monitor.update(model,data,'RECORDED | '+str(phases[index]))
                last_index=index
            while cursor+1<len(optical) and optical[cursor+1][0]<=data.time:
                cursor+=1;stamp,p,observation=optical[cursor]
                with window.lock():draw_paths(window.user_scn,observation.get('paths',[]))
                if cameras:
                    publish_recorded_detector(display_folder/'detector',config,p,observation,stamp)
            visibility='Laser off / assembly' if cursor<0 else f"Visibility {optical[cursor][2]['visibility']*100:.2f}%"
            window.set_texts((None,None,'RECORDED PHYSICAL SIMULATION',f'{phases[index]} | t={data.time:.2f}s | {speed:g}x | {visibility}'))
            window.sync()
            if elapsed>end_sim+hold_seconds*speed:break
            time.sleep(1/60)
    finally:
        if monitor is not None:monitor.close()
        if player is not None and player.poll() is None:
            player.terminate()
            try:player.wait(timeout=3)
            except subprocess.TimeoutExpired:player.kill();player.wait()
        window.close()


def export_video(folder,speed=3.,fps=20):
    """Render actual recorded scene states; optical sensors remain separate."""
    import cv2
    import shutil
    speed=positive_speed(speed)
    folder=Path(folder);saved=np.load(folder/'episode.npz',allow_pickle=False)
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:raise RuntimeError('ffmpeg is required for video export')
    model=mujoco.MjModel.from_binary_path(str(folder/'episode-model.mjb'));data=mujoco.MjData(model)
    stamps=saved['time'];config=json.loads(str(saved['config']));roles={x['id']:x['role'] for x in config['components']}
    optical=[]
    for p in (folder/'fat').glob('optical-*'):
        if (p/'state.npz').exists() and (p/'measurement.json').exists():
            optical.append((float(np.load(p/'state.npz')['time']),json.loads((p/'measurement.json').read_text())))
    optical.sort(key=lambda x:x[0]);cursor=-1
    output=folder/'full-experiment.mp4';width,height=960,540
    encoder=subprocess.Popen([ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
        '-r',str(fps),'-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.lookat[:]=[0,.35,.88];camera.distance=1.45;camera.azimuth=135;camera.elevation=-35
    model.vis.map.znear=.01
    try:
        with mujoco.Renderer(model,width=width,height=height) as renderer:
            for stamp in np.arange(stamps[0],stamps[-1]+speed/fps,speed/fps):
                index=min(len(stamps)-1,max(0,int(np.searchsorted(stamps,stamp,side='right')-1)))
                data.qpos[:]=saved['qpos'][index];data.time=float(stamps[index]);mujoco.mj_forward(model,data)
                while cursor+1<len(optical) and optical[cursor+1][0]<=data.time:cursor+=1
                renderer.update_scene(data,camera=camera)
                if cursor>=0:draw_paths(renderer.scene,optical[cursor][1].get('paths',[]),clear=False)
                rgb=renderer.render().copy();phase=str(saved['phase'][index]);name=phase.split(':')[0]
                title=f'{roles.get(name,name)} | {phase}'
                cv2.rectangle(rgb,(0,0),(width,58),(12,18,26),-1)
                cv2.putText(rgb,'RECORDED PHYSICAL SIMULATION | empty area -> assembly -> FAT -> optical feedback',(12,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1,cv2.LINE_AA)
                visibility='laser off / assembly' if cursor<0 else f"visibility {optical[cursor][1]['visibility']*100:.2f}%"
                cv2.putText(rgb,f'{title} | t={data.time:.1f}s | {speed:g}x | {visibility}',(12,44),cv2.FONT_HERSHEY_SIMPLEX,.43,(150,230,220),1,cv2.LINE_AA)
                encoder.stdin.write(rgb.tobytes())
    finally:
        encoder.stdin.close();code=encoder.wait()
    if code:raise RuntimeError('Video encoder failed')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--speed',type=positive_speed,default=1.);parser.add_argument('--no-cameras',action='store_true');parser.add_argument('--hold-seconds',type=float,default=30.)
    parser.add_argument('--export-video',action='store_true')
    parser.add_argument('--start-seconds',type=float,default=0.)
    args=parser.parse_args()
    if not args.export_video and sys.platform=='darwin' and not os.environ.get('MJPYTHON_BIN'):
        os.execv(sys.executable,[sys.executable,str(Path(sys.executable).with_name('mjpython')),'-m','simulation.scene_recording',*sys.argv[1:]])
    if args.export_video:print(export_video(args.folder,args.speed))
    else:play(args.folder,args.speed,not args.no_cameras,args.hold_seconds,args.start_seconds)
