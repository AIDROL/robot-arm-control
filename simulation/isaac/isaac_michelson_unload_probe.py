"""Explicit checkpoint diagnosis of motor preload and withdrawal, not acceptance."""
import argparse
import json
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--powered", action="store_true", help="Control comparison: retain position drives during withdrawal")
parser.add_argument("--unload-target", action="store_true", help="Hold measured shaft angles to remove servo preload")
parser.add_argument("--high-resolution-axis", action="store_true")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "anti_aliasing": 0})
signal.signal(signal.SIGINT, signal.default_int_handler)
sys.path.append(str(ROOT / ".venv/lib/python3.11/site-packages"))
import numpy as np
from scipy.spatial.transform import Rotation
from simulation.isaac.isaac_michelson_backend import IsaacMechanics
from simulation.isaac.isaac_michelson_control import Experiment

experiment = None
result = {"success": False, "started_from_empty_area": False,
          "scope": "development checkpoint motor unloading/withdrawal diagnosis only"}
try:
    sim = IsaacMechanics(app, args.output, dt=.002)
    refined_fixture = None
    if args.high_resolution_axis:
        sim.restore_assembly(args.source)
        refined_fixture = sim.capture("refined-global", high_resolution=True)[4]
    sim.restore_assembly(args.source, optical_checkpoint=args.checkpoint)
    state = np.load(args.source / "optics" / args.checkpoint / "native-state.npz")
    sim.command(state["command_names"].tolist(), state["commands"])
    experiment = Experiment(sim)
    experiment.active = "fat"
    experiment.grip_value = 255.
    experiment.event("RESTORED_OPTICAL_CHECKPOINT_FOR_DIAGNOSIS", source=str(args.source), checkpoint=args.checkpoint)
    experiment.hold(.5)
    before = experiment.optical_sample("before")
    if args.unload_target:
        sim.command([label + "_motor_joint" for label in ("lower", "upper")],
                    [float(sim.data.joint(label + "_motor_joint").qpos[0]) for label in ("lower", "upper")])
        experiment.event("MEASURED_SHAFT_ANGLE_HOLD")
    else:
        experiment.event("MOTORS_BACKDRIVABLE" if not args.powered else "POWERED_COMPARISON",
                         native_gains=sim.set_fat_motor_power(args.powered))
    experiment.hold(.5)
    unloaded = experiment.optical_sample("unloaded")
    vision = json.loads((args.source / "post-assembly-vision.json").read_text())
    fixture = np.array(vision["observations"]["4"])
    if args.high_resolution_axis:
        fixture = refined_fixture
    tcp = sim.site_pose("pinch_center")
    experiment.move("withdraw", tcp[:3, 3] - fixture[:3, 1] * .040,
                    Rotation.from_matrix(tcp[:3, :3]), 3.)
    experiment.hold(2.)
    after = experiment.optical_sample("after-withdrawal")
    result.update(before=before, unloaded=unloaded, after=after, success=after["valid"] and after["visibility"] >= .95)
except Exception as error:
    import traceback
    traceback.print_exc()
    result["failure"] = str(error)
finally:
    if experiment:
        experiment.save()
        result["unsafe_contacts"] = experiment.unsafe
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print("UNLOAD_PROBE", json.dumps({key: result[key] for key in ("success", "scope", "failure") if key in result}), flush=True)
    app.close()
