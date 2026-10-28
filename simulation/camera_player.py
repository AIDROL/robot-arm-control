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

TITLE = 'Robot Camera Player'
VIEWS = [('global_a', 'Global A - RGB'), ('global_b', 'Global B - RGB'),
         ('wrist_rgb', 'Wrist - RGB'), ('wrist_depth', 'Wrist - depth (0.03-1.0 m)')]


def compose(frames, status, paused=False):
    canvas = np.full((824, 1280, 3), (30, 25, 20), np.uint8)
    for index, (key, label) in enumerate(VIEWS):
        x, y = (index % 2) * 640, 40 + (index // 2) * 392
        cv2.putText(canvas, label, (x + 12, y + 23), cv2.FONT_HERSHEY_SIMPLEX, .6,
                    (230, 230, 230), 1, cv2.LINE_AA)
        if key in frames:
            image = cv2.imdecode(np.frombuffer(base64.b64decode(frames[key]), np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                canvas[y+32:y+392, x:x+640] = cv2.resize(image, (640, 360))
    phase = str(status.get('phase', 'Waiting')).split(' · ')[0]
    phase = phase.encode('ascii', errors='ignore').decode().strip() or 'Preparing'
    message = (f"{'PAUSED' if paused else 'LIVE'} | {phase} | "
               f"t={status.get('simulation_time', 0):.2f}s | frame {status.get('sequence', 0)} "
               '| SPACE: pause picture | Q/ESC: close player')
    cv2.putText(canvas, message, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .48,
                (160, 235, 180), 1, cv2.LINE_AA)
    return canvas


def play(url, output=None, max_seconds=None):
    parsed = urlparse(url)
    if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1':
        raise ValueError('Player accepts only the local simulation monitor URL')
    opener = build_opener(ProxyHandler({}))
    cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TITLE, 1120, 721)
    cv2.imshow(TITLE, compose({}, {}))
    cv2.waitKey(1)
    paused, sequence, displayed = False, -1, 0
    frames, status = {}, {}
    started, next_fetch = time.monotonic(), 0.
    stats = {'window_visible': bool(cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) >= 1),
             'displayed_updates': 0, 'last_sequence': -1}
    print('Native camera player window opened', flush=True)
    try:
        while max_seconds is None or time.monotonic() - started < max_seconds:
            now = time.monotonic()
            if now >= next_fetch and not paused:
                try:
                    with opener.open(url.rstrip('/') + '/snapshot', timeout=.5) as response:
                        packet = json.load(response)
                    status, frames = packet['status'], packet['frames']
                    if status['sequence'] != sequence and len(frames) == 4:
                        canvas = compose(frames, status)
                        cv2.imshow(TITLE, canvas)
                        sequence = status['sequence']
                        displayed += 1
                        stats.update(displayed_updates=displayed, last_sequence=sequence,
                                     simulation_time=status['simulation_time'], phase=status['phase'])
                        if output is not None:
                            output.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(output / 'camera-player.jpg'), canvas)
                            (output / 'camera-player-status.json').write_text(json.dumps(stats, indent=2))
                except (OSError, ValueError, KeyError) as error:
                    canvas = compose({}, {'phase': 'Connection unavailable - restart simulation'})
                    cv2.imshow(TITLE, canvas)
                next_fetch = time.monotonic() + .1
            key = cv2.waitKey(15) & 0xff
            if key in (27, ord('q')) or cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(' '):
                paused = not paused
                cv2.imshow(TITLE, compose(frames, status, paused))
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
