"""Native OpenCV live camera player; no browser and no MuJoCo dependency."""
import argparse
import base64
import json
from pathlib import Path
import time
from urllib.request import build_opener, ProxyHandler
from urllib.parse import urlparse

import cv2
import numpy as np

VIEWS = [('global_a', 'Global A - RGB'), ('global_b', 'Global B - RGB'),
         ('wrist_rgb', 'Wrist - RGB'), ('wrist_depth', 'Wrist - depth (0.28-1.0 m)'),
         ('fat_rgb', 'FAT - side marker RGB')]
TITLES = {key: 'Robot Camera - ' + label for key, label in VIEWS}


def render_view(key, frames, status, paused=False):
    """One sensor per image/window; never combine sensor views."""
    canvas = np.full((412, 640, 3), (30, 25, 20), np.uint8)
    phase = str(status.get('phase', 'Waiting')).split(' · ')[0]
    phase = phase.encode('ascii', errors='ignore').decode().strip() or 'Preparing'
    message = (f"{'PAUSED' if paused else 'LIVE'} | {phase} | "
               f"t={status.get('simulation_time', 0):.2f}s | frame {status.get('sequence', 0)}")
    cv2.putText(canvas, message, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, .43,
                (160, 235, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'SPACE: pause pictures | Q/ESC: close all camera windows',
                (10, 43), cv2.FONT_HERSHEY_SIMPLEX, .4,
                (220, 220, 220), 1, cv2.LINE_AA)
    if key in frames:
        image = cv2.imdecode(np.frombuffer(base64.b64decode(frames[key]), np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            canvas[52:412] = cv2.resize(image, (640, 360))
    return canvas


def window_visible(title):
    try:
        return cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def play(url, output=None, max_seconds=None):
    parsed = urlparse(url)
    if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1':
        raise ValueError('Player accepts only the local simulation monitor URL')
    opener = build_opener(ProxyHandler({}))
    active = set(TITLES)
    paused, sequence, displayed = False, -1, 0
    frames, status = {}, {}
    started, next_fetch = time.monotonic(), 0.
    stats = {'window_visible': False, 'window_count': 0,
             'displayed_updates': 0, 'last_sequence': -1}
    try:
        for index, (key, _) in enumerate(VIEWS):
            title = TITLES[key]
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 640, 412)
            cv2.moveWindow(title, 20 + index % 3 * 660, 50 + index // 3 * 460)
            cv2.imshow(title, render_view(key, {}, {}))
        cv2.waitKey(1)
        stats.update(window_visible=all(window_visible(t) for t in TITLES.values()),
                     window_count=sum(window_visible(t) for t in TITLES.values()),
                     windows=TITLES)
        print('Five separate native camera windows opened', flush=True)
        while active and (max_seconds is None or time.monotonic() - started < max_seconds):
            active = {key for key in active if window_visible(TITLES[key])}
            if not active:
                break
            now = time.monotonic()
            if now >= next_fetch and not paused:
                try:
                    with opener.open(url.rstrip('/') + '/snapshot', timeout=.5) as response:
                        packet = json.load(response)
                    status, frames = packet['status'], packet['frames']
                    if status['sequence'] != sequence and all(k in frames for k in TITLES):
                        for key in active:
                            canvas = render_view(key, frames, status)
                            cv2.imshow(TITLES[key], canvas)
                            if output is not None:
                                output.mkdir(parents=True, exist_ok=True)
                                cv2.imwrite(str(output / f'camera-player-{key}.jpg'), canvas)
                        sequence = status['sequence']
                        displayed += 1
                        stats.update(displayed_updates=displayed, last_sequence=sequence,
                                     simulation_time=status['simulation_time'], phase=status['phase'],
                                     window_count=len(active), visible_cameras=sorted(active))
                        if output is not None:
                            (output / 'camera-player-status.json').write_text(json.dumps(stats, indent=2))
                except (OSError, ValueError, KeyError):
                    for key in active:
                        cv2.imshow(TITLES[key], render_view(key, {}, {
                            'phase': 'Connection unavailable - restart simulation'}))
                next_fetch = time.monotonic() + .1
            key = cv2.waitKey(15) & 0xff
            if key in (27, ord('q')):
                break
            if key == ord(' '):
                paused = not paused
                for camera in active:
                    cv2.imshow(TITLES[camera], render_view(camera, frames, status, paused))
    finally:
        cv2.destroyAllWindows()
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-seconds', type=float, help='Bounded GUI verification run')
    args = parser.parse_args()
    play(args.url, args.output, args.max_seconds)
