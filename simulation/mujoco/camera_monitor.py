"""Bounded-memory live camera dashboard on loopback; rendering stays on caller thread."""
import json
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import webbrowser
import subprocess
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
from .wrist_camera import MIN_DEPTH_M, MAX_DEPTH_M

PAGE = '''<!doctype html><meta charset="utf-8"><title>机械臂相机监看</title>
<style>body{margin:20px;background:#111820;color:#e4edf4;font:16px system-ui}
header{display:flex;justify-content:space-between;align-items:center}h1{font-size:22px}
main{max-width:1000px}section{background:#202b36;padding:12px;border-radius:8px}
h2{font-size:16px;margin:0 0 10px}img{width:100%;display:block;aspect-ratio:16/9;object-fit:contain;background:#000}
p{color:#aebdcc}button{padding:8px 14px}#state{color:#8de4b1}</style>
<header><h1>机械臂 · 实时相机</h1><button id="toggle">暂停画面</button></header>
<p id="state">等待第一帧…</p><main></main>
<p>预览 640 × 360；定位仍使用原始分辨率。深度：近处蓝色，远处红色，黑色为无效或超出显示范围；显示范围 0.28–1.0 m。
检测使用原始分辨率，角点 0–3、ID 和 PnP 坐标轴仅供显示；SIM calibration 为仿真配置，非实机标定。
深度窗口复用同帧 RGB 的角点。末端相机为理想 RGB-D 代理。任务结束后画面静止属于正常状态，页面不会重播录像。</p>
<script>
const allCameras=[['global_a','全局相机 A · RGB'],['global_b','全局相机 B · RGB'],
 ['wrist_rgb','末端相机 · RGB'],['wrist_depth','末端相机 · 深度'],['fat_rgb','FAT 相机 · RGB']];
const selected=new URLSearchParams(location.search).get('camera')||'global_a';
const cameras=allCameras.filter(([id])=>id===selected);
document.title=(cameras[0]||['','相机监看'])[1];
const main=document.querySelector('main');
for(const [id,label] of cameras){const panel=document.createElement('section');
 const title=document.createElement('h2');title.textContent=label;panel.append(title);
 const img=document.createElement('img');img.id=id;img.alt=label;panel.append(img);main.append(panel)}
let paused=false,sequence=-1;
document.querySelector('#toggle').onclick=()=>{paused=!paused;document.querySelector('#toggle').textContent=paused?'继续播放':'暂停画面'};
async function poll(){try{const response=await fetch('/status',{cache:'no-store'});
 const s=await response.json();document.querySelector('#state').textContent=
 `${s.phase} · 仿真 ${s.simulation_time.toFixed(2)} s · 帧 ${s.sequence} · 实际预览 ${s.fps.toFixed(1)} fps`;
 if(!paused && s.sequence!==sequence){sequence=s.sequence;
 for(const [id] of cameras)document.getElementById(id).src='/frame/'+id+'?n='+sequence;}
 }catch(e){document.querySelector('#state').textContent='相机连接已关闭，请重新运行仿真。'}
 setTimeout(poll,200)}poll();</script>'''


class CameraMonitor:
    def __init__(self, model, fps=2, open_browser=True):
        self.frames = {}
        self.lock = threading.Lock()
        self.status = dict(sequence=0, simulation_time=0., phase='准备', fps=0.)
        self.period = 1. / fps
        self.last_capture = -float('inf')
        self.last_wall = None
        # Decode at native resolution: small global tags disappear at 640px.
        old = model.vis.global_.offwidth, model.vis.global_.offheight
        try:
            model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720
            self.renderer = mujoco.Renderer(model, width=1280, height=720)
        finally:
            model.vis.global_.offwidth, model.vis.global_.offheight = old
        self.global_renderer = None
        monitor = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split('?', 1)[0]
                with monitor.lock:
                    if path == '/':
                        data, kind = PAGE.encode(), 'text/html; charset=utf-8'
                    elif path == '/status':
                        data, kind = json.dumps(monitor.status).encode(), 'application/json'
                    elif path == '/snapshot':
                        data = json.dumps({'status': monitor.status,
                            'frames': {k: base64.b64encode(v).decode('ascii')
                                       for k, v in monitor.frames.items()}}).encode()
                        kind = 'application/json'
                    elif path.startswith('/frame/') and path[7:] in monitor.frames:
                        data, kind = monitor.frames[path[7:]], 'image/jpeg'
                    else:
                        self.send_error(404)
                        return
                self.send_response(200)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/'
        print(f'Live camera monitor: {self.url}', flush=True)
        if open_browser:
            for camera in ('global_a', 'global_b', 'wrist_rgb', 'wrist_depth', 'fat_rgb'):
                webbrowser.open_new(self.url + '?camera=' + camera)
        self.player = None

    def start_player(self, output):
        # Separate process gives OpenCV its own macOS GUI main thread and keeps
        # rendering/robot control out of the native player's event loop.
        self.player = subprocess.Popen([sys.executable, '-m', 'simulation.camera_player',
                                       '--url', self.url, '--output', str(output)],
                                      cwd=str(Path(__file__).resolve().parents[2]))

    def update(self, model, data, phase, force=False):
        if not force and data.time - self.last_capture < self.period:
            return None
        self.last_capture = float(data.time)
        images = {}
        from . import global_camera as vision
        from .vision_overlay import annotate
        if self.global_renderer is None:
            self.global_renderer = mujoco.Renderer(model,width=vision.WIDTH,height=vision.HEIGHT)
        detection_status = {}
        self.renderer.disable_depth_rendering()
        for name in ('global_a', 'global_b', 'wrist_rgbd', 'fat_rgb'):
            renderer = self.global_renderer if name in vision.CAMERAS else self.renderer
            near=model.vis.map.znear
            try:
                if name in vision.CAMERAS:model.vis.map.znear=max(near,.01)
                renderer.update_scene(data, camera=name)
                rgb = renderer.render().copy()
            finally:
                model.vis.map.znear=near
            key = 'wrist_rgb' if name == 'wrist_rgbd' else name
            K,_ = vision.calibration(model,data,name,rgb.shape[1],rgb.shape[0])
            images[key],records = annotate(rgb,K,key)
            detection_status[key]={'markers':records,'K':K.tolist(),
                                   'calibration_source':'configured_simulation',
                                   'simulation_time':float(data.time)}
            if name == 'wrist_rgbd':
                wrist_rgb,wrist_K,wrist_records = rgb,K,records
        self.renderer.enable_depth_rendering()
        try:
            self.renderer.update_scene(data, camera='wrist_rgbd')
            depth = self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()
        valid = np.isfinite(depth) & (depth >= MIN_DEPTH_M) & (depth < MAX_DEPTH_M)
        scaled = np.uint8(np.clip(np.nan_to_num((depth - MIN_DEPTH_M) / (MAX_DEPTH_M - MIN_DEPTH_M)), 0, 1) * 255)
        depth_bgr = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
        depth_bgr[~valid] = 0
        images['wrist_depth'],_ = annotate(wrist_rgb,wrist_K,'wrist_depth',
                                          detections=wrist_records,background=depth_bgr)
        detection_status['wrist_depth']={'source':'aligned wrist RGB detections; not depth marker detection',
                                         'simulation_time':float(data.time)}
        frames = {}
        for key, bgr in images.items():
            ok, encoded = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise RuntimeError('Camera preview encoding failed')
            frames[key] = encoded.tobytes()
        now = time.monotonic()
        with self.lock:
            self.frames = frames  # only the latest camera frames are retained
            self.status.update(sequence=self.status['sequence'] + 1,
                               simulation_time=float(data.time), phase=phase,
                               detections=detection_status,
                               fps=0. if self.last_wall is None else 1. / max(now-self.last_wall, 1e-6))
        self.last_wall = now
        return cv2.cvtColor(images['wrist_rgb'], cv2.COLOR_BGR2RGB)

    def finish(self, phase):
        with self.lock:
            self.status['phase'] = phase

    def close(self):
        if self.player is not None and self.player.poll() is None:
            self.player.terminate()
            try:
                self.player.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.player.kill()
                self.player.wait()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.renderer.close()
        if self.global_renderer is not None:self.global_renderer.close()
