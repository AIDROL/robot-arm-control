"""Import the pinned Menagerie xArm7 MJCF with Isaac Sim 5.1's importer."""
from __future__ import annotations

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("assets/xarm7_isaac/xarm7.usd"))
args = parser.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import omni.kit.commands
from isaacsim.core.utils.extensions import enable_extension
from pxr import Usd

root = Path(__file__).resolve().parents[2]
source = root / "assets/xarm7_menagerie/xarm7.xml"
if not source.is_file():
    raise FileNotFoundError(source)
output = args.output.resolve()
output.parent.mkdir(parents=True, exist_ok=True)

enable_extension("isaacsim.asset.importer.mjcf")
app.update()
ok, config = omni.kit.commands.execute("MJCFCreateImportConfig")
if not ok:
    raise RuntimeError("Could not create MJCF import config")
config.set_fix_base(True)
config.set_import_inertia_tensor(True)
ok, imported_path = omni.kit.commands.execute(
    "MJCFCreateAsset", mjcf_path=str(source), import_config=config,
    prim_path="/XArm7", dest_path=str(output))
if not ok:
    raise RuntimeError("MJCF import failed")
app.update()

stage = Usd.Stage.Open(str(output))
if stage is None:
    raise RuntimeError(f"Importer did not write {output}")
joints = [(str(prim.GetPath()), prim.GetTypeName()) for prim in stage.Traverse()
          if prim.GetTypeName() in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint")]
revolute = [path for path, kind in joints if kind == "PhysicsRevoluteJoint"]
print("imported_path", imported_path, flush=True)
print("usd", output, flush=True)
print("revolute_count", len(revolute), flush=True)
print("joints", joints, flush=True)
if len(revolute) < 7:
    raise RuntimeError("Imported USD is missing xArm7's seven revolute arm joints")
app.close()
