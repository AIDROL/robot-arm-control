"""Verify that the imported xArm7 articulates in Isaac Sim 5.1."""
from __future__ import annotations

from pathlib import Path

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdPhysics
import omni.usd

asset = Path(__file__).resolve().parents[2] / "assets/xarm7_isaac/xarm7.usd"
if not asset.is_file():
    raise FileNotFoundError(asset)
world = World(stage_units_in_meters=1.0, physics_dt=1 / 120)
add_reference_to_stage(usd_path=str(asset), prim_path="/World/XArm7")
app.update()
stage = omni.usd.get_context().get_stage()
print("articulation_roots", [str(p.GetPath()) for p in stage.Traverse()
      if p.HasAPI(UsdPhysics.ArticulationRootAPI)], flush=True)
world.reset()
arm = Articulation("/World/XArm7/link_base/link_base")
arm.initialize()
names = list(arm.dof_names)
required = [f"joint{i}" for i in range(1, 8)]
missing = set(required) - set(names)
if missing:
    raise RuntimeError(f"Missing arm joints: {sorted(missing)}; imported: {names}")
index = arm.get_dof_index("joint1")
start = float(arm.get_joint_positions()[0, index])
print("joint1_initial_gains", tuple(x[0, index] for x in arm.get_gains()), flush=True)
world.play()
arm.set_gains(kps=np.array([[1000.]]), kds=np.array([[50.]]),
              joint_indices=np.array([index]))
for _ in range(120):
    arm.set_joint_position_targets(np.array([[start + .1]]),
                                   joint_indices=np.array([index]))
    world.step(render=False)
end = float(arm.get_joint_positions()[0, index])
print("arm_dofs", names, flush=True)
print(f"joint1_start={start:.5f} target={start + .1:.5f} achieved={end:.5f}", flush=True)
if abs(end - start) < .02:
    raise RuntimeError("xArm7 joint1 did not move under a position command")
world.stop()
app.close()
