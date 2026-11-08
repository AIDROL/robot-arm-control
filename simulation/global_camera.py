"""Compatibility entry point; implementation lives in simulation.mujoco.global_camera."""
from pathlib import Path
import importlib
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    runpy.run_module("simulation.mujoco.global_camera", run_name="__main__", alter_sys=True)
else:
    sys.modules[__name__] = importlib.import_module("simulation.mujoco.global_camera")
