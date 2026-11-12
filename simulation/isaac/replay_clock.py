"""Continuous, adjustable wall-clock playback of recorded simulation time."""
import math
import time


class ReplayClock:
    def __init__(self, start, end, speed=4., paused=False, wall=None):
        self.start, self.end = float(start), float(end)
        self.source_time = self.start
        self.paused = paused
        self.wall = time.monotonic() if wall is None else wall
        self.set_speed(speed, wall=self.wall)

    def advance(self, wall=None):
        wall = time.monotonic() if wall is None else wall
        if not self.paused:
            self.source_time = min(self.end, self.source_time + max(0., wall-self.wall)*self.speed)
        self.wall = wall
        return self.source_time

    def set_speed(self, speed, wall=None):
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError('Replay speed must be finite and positive')
        if hasattr(self, 'speed'):
            self.advance(wall)
        self.speed = float(speed)

    def set_paused(self, paused, wall=None):
        self.advance(wall)
        self.paused = paused

    def seek(self, source_time, wall=None):
        self.wall = time.monotonic() if wall is None else wall
        self.source_time = min(self.end, max(self.start, source_time))

    def restart(self, wall=None):
        self.seek(self.start, wall)
        self.paused = False
