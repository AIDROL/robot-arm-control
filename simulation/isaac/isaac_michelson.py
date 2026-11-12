"""Isaac / PhysX Michelson migration and staged physical acceptance."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--view", action="store_true", help="Independent smooth live renderer and continuous preview cameras")
parser.add_argument("--view-fps", type=float, default=60., help="Live viewport frame rate target")
parser.add_argument("--camera-fps", type=float, default=15., help="Live camera preview publication rate")
parser.add_argument("--preview-width", type=int, default=480, help="Live camera preview width")
parser.add_argument("--stage", choices=("settle", "grip", "pick", "assembly", "full"), default="full")
parser.add_argument("--component", choices=("lens_1", "pinhole", "lens_2", "beam_splitter", "mirror"), default="lens_1")
parser.add_argument("--seconds", type=float, default=1.)
parser.add_argument("--dt", type=float, default=.002)
parser.add_argument("--resume", type=Path, help="Explicit development resume of a saved native five-component assembly")
parser.add_argument("--feedback-actions", type=int, default=60, help="Bounded development feedback probe; default retains the original 60-action limit")
args = parser.parse_args()
if not 0 < args.dt <= 1 / 120:
    parser.error("--dt must be positive and no larger than 1/120 s")
if not 1 <= args.feedback_actions <= 60:
    parser.error("--feedback-actions must be between 1 and 60")
if args.view_fps <= 0 or args.camera_fps <= 0 or args.preview_width < 160:
    parser.error("--view-fps and --camera-fps must be positive")
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=False)
sources = (("isaac_michelson.py", "isaac"), ("isaac_michelson_backend.py", "isaac"),
           ("isaac_michelson_control.py", "isaac"), ("michelson_assembly_optics.py", ""),
           ("michelson_experiment.py", "mujoco"), ("fat_mechanics.py", "mujoco"),
           ("live_stream.py", "isaac"), ("isaac_michelson_live.py", "isaac"),
           ("display_rendering.py", "isaac"), ("optical_device_visuals.py", ""))
(args.output / "source").mkdir()
for name, backend in sources:
    (args.output / "source" / name).write_bytes((ROOT / "simulation" / backend / name).read_bytes())
(args.output / "configuration.json").write_bytes((ROOT / "configs/optical_setups/michelson_full.json").read_bytes())
(args.output / "provenance.json").write_text(json.dumps({
    "source_sha256": {name: hashlib.sha256((ROOT / "simulation" / backend / name).read_bytes()).hexdigest() for name, backend in sources},
    "configuration_sha256": hashlib.sha256((ROOT / "configs/optical_setups/michelson_full.json").read_bytes()).hexdigest(),
    "source_roles": "MuJoCo asset compiler / kinematics only; native PhysX dynamics, contacts, state and Isaac images",
    "mechanics": "source geometry/inertia and finite-force drives; PhysX mimic screw-lever; passive elasto-plastic knob friction",
    "argv": sys.argv}, indent=2))

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "anti_aliasing": 0})
signal.signal(signal.SIGINT, signal.default_int_handler)
# Append project wheels after Isaac's own packages: its NumPy / OpenCV / SciPy
# stay authoritative. MuJoCo supplies the asset compiler and kinematics only.
sys.path.append(str(ROOT / ".venv/lib/python3.11/site-packages"))
import numpy as np
from simulation.isaac.isaac_michelson_backend import IsaacMechanics
from simulation.isaac.isaac_michelson_control import Experiment, DOWN

result = {"success": False, "engine": "Isaac Sim 5.1 / PhysX", "stage": args.stage,
          "started_from_empty_area": args.resume is None,
          "scope": "stage-specific acceptance; only stage=full includes assembly, FAT and fresh post-return optics",
          "base_holding": "30 N contact-supported release attraction proxy, then ideal fixed joint at attained pose; no magnetic field simulation"}
started = time.monotonic()
experiment = None
live_process = None
try:
    mechanics = IsaacMechanics(app, args.output, False, dt=args.dt)
    result["physics_dt_s"] = args.dt
    experiment = Experiment(mechanics, feedback_action_limit=args.feedback_actions)
    if args.view:
        mechanics.start_live_stream()
        launcher = next((p / "python.sh" for p in Path(sys.executable).resolve().parents
                         if (p / "python.sh").is_file() and (p / "setup_python_env.sh").is_file()
                         and (p / "apps").is_dir()), None)
        if launcher is None:
            raise RuntimeError("ISAAC_PYTHON_LAUNCHER_NOT_FOUND")
        live_log = (args.output / "live-view.log").open("w")
        live_process = subprocess.Popen([str(launcher), str(ROOT / "simulation/isaac/isaac_michelson_live.py"),
                                         "--folder", str(args.output), "--fps", str(args.view_fps),
                                         "--camera-fps", str(args.camera_fps), "--preview-width", str(args.preview_width)], stdout=live_log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        deadline = time.monotonic() + 90.
        while not (mechanics.live_folder / "ready.json").exists():
            if live_process.poll() is not None:
                raise RuntimeError("LIVE_VIEW_START_FAILED: see live-view.log")
            if time.monotonic() > deadline:
                raise RuntimeError("LIVE_VIEW_FIRST_FRAMES_TIMEOUT: see live-view.log")
            time.sleep(.05)
        result["live_display"] = "separate renderer; 100 ms display interpolation; native dynamics and control captures unchanged"
    experiment.event("EMPTY_EXPERIMENT_AREA")
    initial = {name: mechanics.site_pose(name + "_optical_axis")[:3, 3].tolist()
               for name in ("lens_1", "pinhole", "lens_2", "beam_splitter", "mirror")}
    contacts = 0
    for i in range(round(args.seconds / mechanics.dt)):
        experiment.tick()
        contacts += len(mechanics.contacts)
    experiment.move("clear staging view", np.array([0., .15, 1.12]), DOWN, 3.)
    observed = mechanics.capture("settled-global")
    positions = {name: mechanics.site_pose(name + "_optical_axis")[:3, 3].tolist()
                 for name in initial}
    result.update(initial_optical_centers_m=initial, settled_optical_centers_m=positions,
                  contact_samples=contacts, detected_ids=sorted(observed), time_s=mechanics.time,
                  tcp_pose=mechanics.site_pose("pinch_center").tolist())
    # This accepts only the scene/settling gate, never assembly or optics.
    scene_gate = bool(contacts and set(range(5)).issubset(observed)
                      and all(np.linalg.norm(np.array(initial[n]) - positions[n]) < .005 for n in initial))
    result["scene_gate_passed"] = scene_gate
    result["success"] = scene_gate if args.stage == "settle" else False
    if args.resume is not None:
        if args.stage != "full":
            raise ValueError("Resume is only for full-stage FAT development")
        resumed_placements = mechanics.restore_assembly(args.resume)
        experiment = Experiment(mechanics, feedback_action_limit=args.feedback_actions)
        experiment.placements = resumed_placements
        experiment.event("RESUMED_NATIVE_ASSEMBLY", source=str(args.resume.resolve()))
        result["resume_scope"] = "saved PhysX accepted placement prefix; only remaining placements are newly executed"
        placed = {row["id"] for row in resumed_placements}
        for marker, component in enumerate(mechanics.config["components"]):
            if component["id"] in placed:
                continue
            experiment.placement(component["id"], marker)
            experiment.active = None
            experiment.move("clear staging view", np.array([0., .15, 1.12]), DOWN, 3.)
        experiment.event("ASSEMBLY_GATE_COMPLETE", components=5)
        result["success"] = False
        result["fat"] = experiment.fat()
        result["success"] = result["fat"]["success"]
    elif args.stage in ("grip", "pick", "assembly", "full"):
        if not scene_gate:
            raise RuntimeError("SCENE_GATE_FAILED")
        inventory = mechanics.config["components"]
        items = [item for item in inventory if item["id"] == args.component] if args.stage in ("grip", "pick") else inventory
        for component in items:
            marker = inventory.index(component)
            experiment.placement(component["id"], marker, grip_only=args.stage == "grip")
            if args.stage != "grip":
                experiment.active = None
                experiment.move("clear staging view", np.array([0., .15, 1.12]), DOWN, 3.)
        experiment.event("GRIP_GATE_COMPLETE" if args.stage == "grip" else "ASSEMBLY_GATE_COMPLETE", components=len(items))
        result["success"] = all(p["success"] for p in experiment.placements)
        if args.stage == "full":
            result["success"] = False
            result["fat"] = experiment.fat()
            result["success"] = result["fat"]["success"]
except KeyboardInterrupt:
    result["success"] = False
    result["failure"] = "INTERRUPTED"
except Exception as error:
    import traceback
    traceback.print_exc()
    result["success"] = False
    result["failure"] = str(error)
    if experiment is not None:
        (args.output / "last-native-contacts.json").write_text(json.dumps(experiment.sim.contact_records(), indent=2))
        try:
            experiment.sim.capture("failure-global")
        except Exception:
            pass
finally:
    if experiment is not None:
        experiment.save()
        result.update(placements=experiment.placements, unsafe_contacts=experiment.unsafe,
                      max_native_fk_error_m=experiment.max_fk_error, active_metrics=experiment.metrics,
                      max_screw_lever_error_rad=experiment.max_transmission_error)
        if hasattr(experiment, "fat_result"):
            result["fat"] = experiment.fat_result
            (args.output / "fat-result.json").write_text(json.dumps(experiment.fat_result, indent=2))
        result["time_s"] = experiment.sim.time
    result["wall_seconds"] = time.monotonic() - started
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if experiment is not None and hasattr(experiment.sim, "live_folder"):
        from simulation.isaac.live_stream import write_json
        experiment.sim.publish_live(force=True, check_stop=False)
        write_json(experiment.sim.live_folder / "worker-done.json", {"success": result["success"], "time_s": result["time_s"]})
    summary = {key: result[key] for key in ("success", "stage", "started_from_empty_area", "time_s", "failure") if key in result}
    summary["output_directory"] = str(args.output)
    if "fat" in result:
        summary["fat_return_error_m"] = result["fat"].get("fat_return_error_m")
        summary["post_return_visibility"] = [result["fat"].get(key, {}).get("visibility")
                                               for key in ("after_release", "released_grid_check")]
    print("ISAAC_RESULT", json.dumps(summary), flush=True)
    app.close()
    if live_process is not None:
        try:
            live_process.wait(timeout=12.)
        except subprocess.TimeoutExpired:
            os.killpg(live_process.pid, signal.SIGTERM)
            live_process.wait(timeout=5.)
raise SystemExit(0 if result["success"] else 1)
