"""Wall-clock pacing only; physics and controller time remain simulation time."""
import math
import time


def positive_speed(value):
    speed=float(value)
    if not math.isfinite(speed) or speed<=0:
        raise ValueError('Speed must be a finite positive number')
    return speed


class SimulationPacer:
    def __init__(self,simulation_time,speed=1.,enabled=True):
        self.speed=positive_speed(speed)
        self.enabled=enabled
        self.simulation_start=float(simulation_time)
        self.wall_start=time.monotonic()

    def wait(self,simulation_time):
        if not self.enabled:return
        deadline=self.wall_start+(float(simulation_time)-self.simulation_start)/self.speed
        delay=deadline-time.monotonic()
        if delay>0:time.sleep(delay)
