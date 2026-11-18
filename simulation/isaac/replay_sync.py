"""File-based clock bridge for a read-only native trajectory viewer."""
import bisect
import json
from pathlib import Path
import time


class ReplaySync:
    def __init__(self, directory, folder=None):
        self.directory = Path(directory) if directory else None
        self.revision = 0
        self.last_write = -1.
        timeline_path=self.directory/'timeline.json' if self.directory else None
        self.timeline = json.loads(timeline_path.read_text()) if timeline_path and timeline_path.is_file() else {}
        if not self.timeline and folder:
            folder=Path(folder)
            self.timeline['events']=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines() if line.strip()]
            rosi=folder.parent/'rosi.json'
            if rosi.is_file():
                self.timeline['steps']=json.loads(rosi.read_text())['steps']
        self.events = sorted(self.timeline.get('events', []), key=lambda e: e['time_s'])
        self.times = [e['time_s'] for e in self.events]

    def command(self, clock, select_camera):
        if not self.directory:
            return
        try:
            value = json.loads((self.directory / 'command.json').read_text())
        except (OSError, ValueError):
            return
        if value['revision'] <= self.revision:
            return
        action = value['action']
        if action == 'play':
            clock.set_paused(False)
        elif action == 'pause':
            clock.set_paused(True)
        elif action == 'seek':
            clock.seek(value['value'])
        elif action == 'speed':
            clock.set_speed(value['value'])
        elif action == 'restart':
            clock.restart()
        elif action == 'camera':
            select_camera(value['camera'])
        self.revision = value['revision']

    def write(self, clock, phase, camera, wall=None):
        if not self.directory:
            return
        wall = time.monotonic() if wall is None else wall
        if wall - self.last_write < .2:
            return
        cursor = clock.source_time
        events = self.events[:bisect.bisect_right(self.times, cursor)]
        started = next((e for e in reversed(events) if e['stage'] == 'ROSI_STEP_STARTED'), None)
        completed = any(e['stage'] == 'ROSI_STEP_COMPLETED' and e.get('i') == started['i']
                        for e in events) if started else False
        value = {'source_time_s': cursor, 'end_time_s': clock.end,
                 'paused': clock.paused or cursor >= clock.end, 'speed': clock.speed,
                 'phase': phase, 'camera': camera,
                 'step': {**started, 'completed': completed} if started else None,
                 'workflow_node': 'done' if cursor >= clock.end else 'sim',
                 'command_revision': self.revision, 'updated_wall_s': time.time()}
        temp = self.directory / 'state.tmp'
        temp.write_text(json.dumps(value, allow_nan=False))
        temp.replace(self.directory / 'state.json')
        self.last_write = wall
