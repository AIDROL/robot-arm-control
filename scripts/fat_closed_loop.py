"""Run or replay the matching-CAD FAT simulation acceptance from a terminal."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
BACKEND = ROOT


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def active_job(folder):
    """Identify our exact native run/queue; never change another process."""
    queued=False
    for process in Path('/proc').glob('[0-9]*'):
        try:
            argv=[x.decode() for x in (process/'cmdline').read_bytes().split(b'\0') if x]
            if '--output' in argv and any(Path(x).name=='isaac_michelson.py' for x in argv):
                if Path(argv[argv.index('--output')+1]).resolve()==folder.resolve():return 'running'
            if '--folder' in argv and 'run' in argv and any(Path(x).name=='fat_closed_loop.py' for x in argv):
                value=Path(argv[argv.index('--folder')+1])
                if not value.is_absolute():value=(process/'cwd').resolve()/value
                if value.resolve()==folder.resolve():queued=True
        except (OSError,ValueError,IndexError,UnicodeError):
            continue
    return 'queued' if queued else None


def run_status(folder):
    folder=Path(folder)
    last={}
    events=folder/'events.jsonl'
    if events.is_file():
        for line in reversed(events.read_text().splitlines()):
            try:last=json.loads(line);break
            except ValueError:continue
    result_path=folder/'result.json'
    row={'folder':str(folder),'phase':last.get('stage'),'time_s':last.get('time_s'),
         'phase_started_s':last.get('time_s'),
         'log':str(folder.with_suffix('.log'))}
    checkpoint=folder/'checkpoint-status.json'
    if checkpoint.is_file():
        try:
            stamp=json.loads(checkpoint.read_text())['time_s']
            if math.isfinite(stamp):
                row['checkpoint_time_s']=stamp
                row['time_s']=max(row['time_s'] or 0.,stamp)
        except (OSError,ValueError,KeyError,TypeError):pass
    latest=None
    for path in (folder/'optics').glob('*/measurement.json'):
        try:
            sample=json.loads(path.read_text())
            if not math.isfinite(sample['time_s']) or not math.isfinite(sample['visibility']):continue
            if latest is None or sample['time_s']>latest['time_s']:
                latest={'label':path.parent.name,'time_s':sample['time_s'],
                        'visibility':sample['visibility'],'valid':sample.get('valid',False)}
        except (OSError,ValueError,KeyError,TypeError):continue
    if latest:
        row['latest_optics']=latest
        row['time_s']=max(row['time_s'] or 0.,latest['time_s'])
    try:result=json.loads(result_path.read_text()) if result_path.is_file() else None
    except json.JSONDecodeError:
        return {**row,'state':'finishing' if active_job(folder) else 'invalid_result'}
    if result is not None and not result.get('success'):
        return {**row,'state':'failed','failure':result.get('failure','no successful FAT result')}
    if result is not None and (folder/'evidence-manifest.json').is_file():
        return {**row,'state':'sealed'}
    active=active_job(folder)
    if result is not None:return {**row,'state':'sealing' if active else 'unsealed'}
    return {**row,'state':active or ('incomplete' if folder.exists() else 'not_started')}


def readiness_error(status):
    state=status['state'];phase=status.get('phase') or '初始化'
    if state=='finishing':return '实验结果正在写入，尚未完成封存。日志：'+status['log']
    if state=='invalid_result':return '实验结果文件不完整，且未发现运行进程。日志：'+status['log']
    if state=='failed':return 'FAT closed-loop acceptance failed: '+str(status['failure'])
    if state in ('sealing','unsealed'):return '实验证据尚未封存，暂时不能回放。日志：'+status['log']
    if state=='queued':return '实验等待显存，尚未完成；可用 replay --wait 等待。日志：'+status['log']
    if state=='not_started':return '该目录尚无实验记录，也未发现排队进程：'+status['folder']
    if state=='incomplete':return f'实验尚未完成，最后阶段：{phase}；未发现对应运行进程，请检查日志：'+status['log']
    optics=status.get('latest_optics')
    progress=(f"；轨迹已保存至 {status['checkpoint_time_s']:.2f} s" if 'checkpoint_time_s' in status else '')
    progress+=(f"；最新光学测量 {optics['time_s']:.2f} s，可见度 {optics['visibility']*100:.2f}%"
              +('（测量无效）' if optics.get('valid') is False else '')) if optics else ''
    return f'实验尚未完成，最后阶段：{phase}（阶段起始 {status.get("phase_started_s",status.get("time_s"))} s）{progress}。日志：'+status['log']


def wait_for_evidence(folder,interval=3.):
    previous=None
    while True:
        status=run_status(folder)
        if status['state']=='sealed':return
        if status['state'] not in ('running','queued','sealing','finishing'):
            raise ValueError(readiness_error(status))
        token=(status['state'],status.get('phase'),status.get('latest_optics',{}).get('time_s'),status.get('checkpoint_time_s'))
        if token!=previous:print(readiness_error(status),flush=True);previous=token
        time.sleep(interval)


def seal(folder):
    manifest={str(p.relative_to(folder)):digest(p) for p in sorted(folder.rglob('*'))
              if p.is_file() and p.relative_to(folder).parts[0] not in ('live','presentation')
              and p.name!='evidence-manifest.json'}
    temporary=folder/'evidence-manifest.tmp'
    temporary.write_text(json.dumps(manifest,indent=2)+'\n')
    temporary.replace(folder/'evidence-manifest.json')


def verify(folder):
    status=run_status(folder)
    if status['state']!='sealed':raise ValueError(readiness_error(status))
    manifest=json.loads((folder/'evidence-manifest.json').read_text())
    for relative,expected in manifest.items():
        path=(folder/relative).resolve()
        if not path.is_relative_to(folder.resolve()) or digest(path)!=expected:
            raise ValueError('Evidence changed: '+relative)
    result=json.loads((folder/'result.json').read_text())
    fat=result.get('fat',{})
    if not result.get('success') or not fat.get('success'):
        raise ValueError('FAT closed-loop acceptance failed: '+str(result.get('failure','no successful FAT result')))
    required=('scene.usda','physx-trajectory.npz','physx-socket-seating.npz',
              'fat-result.json','fat-interface-audit.json','optics/initial/measurement.json',
              'optics/released-512/measurement.json','optics/released-1024/measurement.json','cameras.json')
    if not all(name in manifest for name in required):raise ValueError('Incomplete evidence bundle')
    audit=json.loads((folder/'fat-interface-audit.json').read_text())
    if not result.get('success') or not fat.get('success') or not audit.get('passed') or result.get('unsafe_contacts'):
        raise ValueError('FAT closed-loop acceptance did not pass')
    if not fat.get('history') or not any(row.get('decision')=='accept' for row in fat['history']):
        raise ValueError('No accepted optical feedback observation')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','replay','verify','status'))
    parser.add_argument('--folder',type=Path,required=True,help='New run directory, or existing accepted evidence directory')
    parser.add_argument('--assembly',action='store_true',help='Start from an empty experiment area and assemble all five modules first')
    parser.add_argument('--isaac-python',type=Path,default=Path(os.environ.get('ISAAC_SIM_PYTHON',str(Path.home()/'Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh'))))
    parser.add_argument('--backend-site-packages',type=Path,default=Path(os.environ.get('AIDROL_BACKEND_SITE_PACKAGES',str(ROOT/'.venv/lib/python3.11/site-packages'))))
    parser.add_argument('--camera',choices=('overview','global','wrist','fat'),default='overview')
    parser.add_argument('--speed',type=float,default=1.)
    parser.add_argument('--start-seconds',type=float)
    parser.add_argument('--paused',action='store_true')
    parser.add_argument('--wait',action='store_true',help='Wait for the active run and evidence seal before opening replay')
    parser.add_argument('--max-frames',type=int,default=0)
    parser.add_argument('--exit-after',type=float,default=0.)
    args=parser.parse_args();folder=args.folder.resolve()
    if args.action=='status':
        print(json.dumps(run_status(folder),ensure_ascii=False,indent=2));return 0
    if args.action=='verify':
        result=verify(folder);print(json.dumps({'passed':True,'scope':result['scope'],'fat':result['fat']},ensure_ascii=False,indent=2));return 0
    env=dict(os.environ)
    for name in ('PYTHONPATH','ROS_DISTRO','CONDA_PREFIX','CONDA_DEFAULT_ENV'):env.pop(name,None)
    env['AIDROL_BACKEND_SITE_PACKAGES']=str(args.backend_site_packages)
    if args.action=='run':
        if folder.exists():parser.error('Output directory already exists; use a new directory to preserve evidence')
        folder.parent.mkdir(parents=True,exist_ok=True)
        log=folder.with_suffix('.log')
        command=[str(args.isaac_python),str(BACKEND/'simulation/isaac/isaac_michelson.py'),
                 '--config',str(BACKEND/'configs/optical_setups/michelson_cad_fat.json'),
                 '--stage','full' if args.assembly else 'fat-bench','--output',str(folder)]
        print('Executing native PhysX; log:',log,flush=True)
    else:
        if args.wait:
            try:wait_for_evidence(folder)
            except KeyboardInterrupt:
                print('停止等待；原实验继续运行。',flush=True);return 130
        result=verify(folder)
        print('Historical FAT simulation replay; no new physical acceptance.',result['scope'],flush=True)
        directory=folder/'presentation'/uuid.uuid4().hex;directory.mkdir(parents=True)
        log=directory/'viewer.log'
        command=[str(args.isaac_python),str(BACKEND/'simulation/isaac/isaac_michelson_replay.py'),
                 '--folder',str(folder),'--state-dir',str(directory),'--camera',args.camera,
                 '--speed',str(args.speed),'--aruco','--wrist-depth',
                 '--max-frames',str(args.max_frames),'--exit-after',str(args.exit_after),
                 '--snapshot',str(directory/'viewport.png')]
        if args.start_seconds is not None:command+=['--start-seconds',str(args.start_seconds)]
        if args.paused:command.append('--freeze')
        print('Viewer output:',directory,flush=True)
    process=None
    try:
        if not args.isaac_python.is_file():raise ValueError('Isaac python.sh not found: '+str(args.isaac_python))
        last=None
        while True:
            gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            free=int(gpu.stdout.strip().splitlines()[0])
            minimum=int(env.get('SIM_MIN_FREE_GPU_MB','4500'))
            if free>=minimum:break
            message=f'等待显存释放：可用 {free} MiB，需要 {minimum} MiB；保留其他训练。'
            if message!=last:print(message,flush=True);last=message
            time.sleep(3)
        with log.open('w') as stream:
            process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            process.wait()
    except KeyboardInterrupt:
        if process and process.poll() is None:
            os.killpg(process.pid,signal.SIGINT);process.wait(timeout=30)
        return 130
    if args.action=='run':
        if not (folder/'result.json').is_file():raise ValueError('No native result; see '+str(log))
        seal(folder)
        result=verify(folder)
        print('FAT simulation acceptance passed; post-return visibility:',
              result['fat']['after_release']['visibility'],result['fat']['released_grid_check']['visibility'])
    else:
        report=json.loads((directory/'viewer-performance.json').read_text())
        if report.get('error'):raise ValueError(report['error'])
    return process.returncode


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,OSError,KeyError) as error:
        print(str(error),file=sys.stderr);raise SystemExit(1)
