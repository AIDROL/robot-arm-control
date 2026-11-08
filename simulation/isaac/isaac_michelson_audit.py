"""Independent audit of native-state, contact and 16-bit detector evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def measure_scan(folder, config):
    frames = []
    for i in range(config["reference_phase_samples"]):
        frame = cv2.imread(str(Path(folder) / f"phase-{i:02d}-detector16.png"), cv2.IMREAD_UNCHANGED)
        if frame is None or frame.ndim != 2 or frame.dtype != np.uint16 or frame.shape[0] != frame.shape[1]:
            raise ValueError("MISSING_OR_INVALID_RAW_16BIT_PHASE_FRAME")
        frames.append(frame)
    if len({frame.shape for frame in frames}) != 1:
        raise ValueError("PHASE_SCAN_SHAPE_MISMATCH")
    n = frames[0].shape[0]
    scale = config["detector_full_scale_W_m2"] / 65535 * (config["detector_window_m"] / n)**2
    powers = np.array([frame.astype(np.float64).sum() * scale for frame in frames])
    phase = np.arange(len(frames)) * (2 * np.pi / len(frames))
    mean = float(powers.mean())
    cosine = float(2 * np.mean(powers * np.cos(phase)))
    sine = float(2 * np.mean(powers * np.sin(phase)))
    visibility = float(np.hypot(cosine, sine) / max(mean, 1e-30))
    valid = bool(mean > .2 * config["input_power_W"] and not any(np.any(frame == 65535) for frame in frames))
    return {"grid_size": n, "valid": valid, "visibility": float(np.clip(visibility, 0., 1.)),
            "raw_visibility_fit": visibility,
            "mean_port_power_W": mean, "phase_scan_power_W": powers.tolist()}


def audit(folder):
    folder = Path(folder)
    result = json.loads((folder / "result.json").read_text())
    config = json.loads((folder / "configuration.json").read_text())
    provenance = json.loads((folder / "provenance.json").read_text())
    checks = {"full_empty_area_execution": result["stage"] == "full" and result["started_from_empty_area"],
              "controller_completed": bool(result["success"]), "no_unsafe_contacts": not result["unsafe_contacts"]}
    checks["archived_sources_match"] = all(
        (folder / "source" / name).is_file() and hashlib.sha256((folder / "source" / name).read_bytes()).hexdigest() == value
        for name, value in provenance["source_sha256"].items())
    checks["configuration_matches"] = hashlib.sha256((folder / "configuration.json").read_bytes()).hexdigest() == provenance["configuration_sha256"]
    drives = json.loads((folder / "native-drives.json").read_text())
    checks["native_actuator_units"] = all(
        np.isclose(drives["fat"][name][parameter], expected, rtol=1e-4, atol=1e-10)
        for name in ("lower_motor_joint", "upper_motor_joint")
        for parameter, expected in (("kp", .03), ("kd", .00011), ("max_effort", .002), ("armature", 1e-6)))
    trace = np.load(folder / "physx-trajectory.npz")
    times = trace["time_s"]
    checks["continuous_native_timeline"] = bool(len(times) > 2 and times[0] == 0. and np.all(np.diff(times) > 0.)
                                                 and np.max(np.diff(times)) <= .1)
    names = trace["body_names"].tolist()
    destinations = []
    for item in config["components"]:
        R = Rotation.from_euler("z", item["yaw_rad"]).as_matrix()
        destinations.append((np.array(item["optical_center_world_m"]) - R @ np.array([0., .004, -.014]))[:2])
    initial_centers = np.array([trace["body_poses"][0, names.index(item["id"]), :2, 3] for item in config["components"]])
    # Stock modules and all destination footprints must start separated. The
    # 70 mm clearance covers the selected housing's horizontal half extents.
    checks["native_initial_area_is_empty"] = bool(np.min(np.linalg.norm(
        initial_centers[:, None, :] - np.array(destinations)[None, :, :], axis=2)) > .07)
    contacts = np.load(folder / "physx-contacts.npz")
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    placements = []
    for item in config["components"]:
        name = item["id"]
        T = trace["body_poses"][-1, names.index(name)]
        R = Rotation.from_euler("z", item["yaw_rad"]).as_matrix()
        target = np.array(item["optical_center_world_m"]) - R @ np.array([0., .004, -.014])
        xy = float(np.linalg.norm(T[:2, 3] - target[:2]))
        height = float(abs(T[2, 3] - .874))
        tilt = float(np.arccos(np.clip(T[2, 2], -1., 1.)))
        active = contacts["active"] == name
        moving = active & np.isin(contacts["phase"], [name + ": lift", name + ": transfer"])
        bilateral = (contacts["left_normal_N"] > .001) & (contacts["right_normal_N"] > .001)
        fraction = float(np.mean(bilateral[moving])) if moving.any() else 0.
        transfer = active & (contacts["phase"] == name + ": transfer")
        clearance = float(contacts["active_body_z_m"][transfer].min() - .874) if transfer.any() else 0.
        gap = float(contacts["active_body_to_tcp_m"][transfer].max()) if transfer.any() else float("inf")
        lift = float(contacts["active_body_z_m"][active].max() - .874) if active.any() else 0.
        latched = next((row for row in events if row["stage"] == "BASE_LATCHED_AT_ACHIEVED_POSE" and row["component"] == name), None)
        drift = float("inf")
        if latched:
            positions = trace["body_poses"][times >= latched["time_s"], names.index(name), :3, 3]
            drift = float(np.linalg.norm(positions - np.array(latched["pose"])[:3, 3], axis=1).max()) if len(positions) else float("inf")
        fine = json.loads((folder / (name + "-wrist.json")).read_text())
        row = {"id": name, "xy_error_m": xy, "height_error_m": height, "tilt_rad": tilt, "max_lift_m": lift,
               "min_carry_clearance_m": clearance, "max_grip_gap_m": gap,
               "bilateral_fraction": fraction, "post_latch_drift_m": drift,
               "valid_wrist_metric_depth": bool(fine["valid"] and fine["position_source"] == "rendered_metric_depth")}
        row["success"] = bool(xy < .010 and height < .005 and tilt < np.deg2rad(5.) and lift > .1 and fraction > .8
                              and clearance > .05 and gap < .05
                              and drift < .0001 and row["valid_wrist_metric_depth"])
        placements.append(row)
    checks["five_native_placements"] = all(row["success"] for row in placements)
    scans = {}
    for label in ("released-512", "released-1024"):
        path = folder / "optics" / label
        scans[label] = measure_scan(path, config)
        recorded = json.loads((path / "measurement.json").read_text())
        checks[label + "_reported_metric_matches_pixels"] = bool(
            abs(scans[label]["visibility"] - recorded["visibility"]) < 1e-10
            and abs(scans[label]["raw_visibility_fit"] - recorded["raw_visibility_fit"]) < 1e-10)
        returned = [row["time_s"] for row in events if row["stage"] == "POST_RETURN_SETTLING"]
        checks[label + "_is_fresh_after_return"] = bool(returned and recorded["time_s"] >= returned[-1] + 1.99)
        state = np.load(path / "native-state.npz")
        joint_indices = dict(zip(state["joint_names"].tolist(), state["joint_qpos_indices"].tolist()))
        checks[label + "_settled_transmission"] = all(
            abs(float(state["qpos"][joint_indices["mirror_" + axis]]) - .0016 * float(state["qpos"][joint_indices[knob + "_knob_joint"]])) <= 2e-6
            for axis, knob in (("pitch", "lower"), ("yaw", "upper")))
        planes = json.loads((path / "actual-optical-planes.json").read_text())
        bodies = dict(zip(state["body_names"].tolist(), state["body_poses"]))
        matches = []
        for component in config["components"]:
            role = component["role"]
            local = np.eye(4)
            local[:3, :3] = Rotation.from_euler("x", -np.pi / 2).as_matrix()
            body = component["id"]
            if role == "adjustable_mirror":
                body = "mirror_tilt"
                local[:3, :3] = (Rotation.from_euler("xz", [.00035, -.0003]) * Rotation.from_euler("x", -np.pi / 2)).as_matrix()
            else:
                local[:3, 3] = [0., .004, -.014]
            matches.append(np.allclose(bodies[body] @ local, planes[role], atol=1e-6, rtol=0.))
        checks[label + "_planes_match_native_bodies"] = bool(all(matches))
    values = [scan["visibility"] for scan in scans.values()]
    checks["post_return_optics"] = all(scan["valid"] for scan in scans.values()) and min(values) >= config["accept_visibility"]
    raw_grid_difference = abs(scans["released-512"]["raw_visibility_fit"] - scans["released-1024"]["raw_visibility_fit"])
    checks["optical_grid_convergence"] = raw_grid_difference < .005
    drive = (contacts["active"] == "fat") & (contacts["phase"] == "fat: drive")
    socket_fraction = float(np.mean((contacts["lower_socket_contact"] & contacts["upper_socket_contact"])[drive])) if drive.any() else 0.
    checks["native_bilateral_tool_contact"] = bool(drive.any() and all(
        np.any(contacts[label + "_socket_contact"][drive]) for label in ("lower", "upper"))
        and any(row["stage"] == "FAT_BILATERAL_ENGAGEMENT_CONFIRMED" for row in events))
    seating = np.load(folder / "physx-socket-seating.npz")
    checks["native_shafts_remain_inserted"] = bool(len(seating["time_s"]) > 0 and all(
        np.all((seating[label + "_depth_m"] >= .0035) & (seating[label + "_depth_m"] <= .0095)
               & (seating[label + "_radial_m"] < .0015)) for label in ("lower", "upper")))
    checks["continuous_seating_monitor"] = bool(drive.any() and len(seating["time_s"]) > 2
        and np.all(np.diff(seating["time_s"]) > 0.)
        and np.max(np.diff(seating["time_s"])) <= result["physics_dt_s"] * 1.01
        and seating["time_s"][0] <= contacts["time_s"][drive].min()
        and seating["time_s"][-1] >= contacts["time_s"][drive].max() - result["physics_dt_s"] * 1.01)
    fat_grip = (contacts["active"] == "fat") & np.isin(contacts["phase"],
        ["fat: lift", "fat: transfer", "fat: prealign", "fat: insert", "fat: seat", "fat: drive"])
    checks["native_bilateral_fat_grip"] = bool(fat_grip.any() and np.mean(
        (contacts["left_normal_N"][fat_grip] > .001) & (contacts["right_normal_N"][fat_grip] > .001)) > .8)
    joint_indices = dict(zip(trace["native_joint_names"].tolist(), trace["native_joint_qpos_indices"].tolist()))
    driving = trace["phase"] == "fat: drive"
    checks["both_knobs_physically_responded"] = bool(driving.any() and all(
        np.ptp(trace["measured_qpos"][driving, joint_indices[label + "_knob_joint"]]) > .01
        for label in ("lower", "upper")))
    fat = trace["body_poses"][-1, names.index("fat"), :3, 3]
    return_error = float(np.linalg.norm(fat - np.r_[config["fat_staging_xy_m"], .805]))
    checks["tool_return"] = return_error < .01
    return {"success": bool(all(checks.values())), "checks": checks, "placements": placements,
            "optical_scans_from_raw_pixels": scans, "bilateral_socket_fraction": socket_fraction,
            "fat_return_error_m": return_error, "time_s": float(times[-1]),
            "raw_grid_visibility_difference": raw_grid_difference,
            "scope": "independent saved-native-state/contact and quantized detector-image audit; ideal scalar optics assumptions remain"}


def plot_report(folder, report):
    """Export a figure from the saved detector counts and measurements."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder)
    result = json.loads((folder / "result.json").read_text())
    config = json.loads((folder / "configuration.json").read_text())
    samples = result["fat"]["history"]
    final = report["optical_scans_from_raw_pixels"]["released-512"]
    paths = (folder / "optics/initial", folder / "optics/released-512")
    observations = (samples[0], final)
    images = []
    phase_indices = []
    for path, measurement in zip(paths, observations):
        phase = int(np.argmax(measurement["phase_scan_power_W"]))
        counts = cv2.imread(str(path / f"phase-{phase:02d}-detector16.png"), cv2.IMREAD_UNCHANGED)
        images.append(counts.astype(float) * config["detector_full_scale_W_m2"] / 65535)
        phase_indices.append(phase)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), constrained_layout=True)
    width_mm = config["detector_window_m"] * 1000
    peak = max(float(image.max()) for image in images)
    for axis, image, title, phase in zip(axes[0], images, ("Before FAT adjustment", "Fresh measurement after tool return"), phase_indices):
        plotted = axis.imshow(image, extent=[-width_mm/2, width_mm/2, -width_mm/2, width_mm/2],
                              origin="lower", cmap="inferno", vmin=0., vmax=peak)
        axis.set(title=title + f" (scan phase {phase})", xlabel="Detector X (mm)", ylabel="Detector Z (mm)")
    fig.colorbar(plotted, ax=axes[0].tolist(), label="Recorded intensity (W/m²)", shrink=.85)
    t0 = samples[0]["time_s"]
    axes[1, 0].plot([row["time_s"] - t0 for row in samples], [row["visibility"] * 100 for row in samples], "o-", ms=3,
                    label="Pixel feedback measurements")
    for label, scan in report["optical_scans_from_raw_pixels"].items():
        measurement = json.loads((folder / "optics" / label / "measurement.json").read_text())
        axes[1, 0].scatter(measurement["time_s"] - t0, scan["visibility"] * 100, marker="x", s=45, label=label)
    axes[1, 0].axhline(config["accept_visibility"] * 100, color="gray", ls="--", label="Post-return acceptance")
    axes[1, 0].set(xlabel="Time since post-assembly observation (s)", ylabel="Integrated modulation visibility (%)", ylim=(-2, 103))
    axes[1, 0].legend(fontsize=8, loc="lower right")
    phase = np.arange(config["reference_phase_samples"]) * 2 * np.pi / config["reference_phase_samples"]
    axes[1, 1].plot(phase, np.array(samples[0]["phase_scan_power_W"]) * 1000, "o-", label="Before adjustment")
    axes[1, 1].plot(phase, np.array(final["phase_scan_power_W"]) * 1000, "o-", label="After tool return (512)")
    axes[1, 1].set(xlabel="Ideal reference-phase scan (rad)", ylabel="Power integrated from raw 16-bit pixels (mW)")
    axes[1, 1].legend(fontsize=8)
    for axis in axes[1]:
        axis.grid(alpha=.25)
    fig.suptitle("Isaac Sim / PhysX Michelson verification\nActual optical planes + ideal scalar coherent optics", fontsize=13)
    fig.savefig(folder / "michelson-verification.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    try:
        report = audit(args.folder)
    except (KeyError, ValueError, OSError) as error:
        report = {"success": False, "failure": str(error)}
    (args.folder / "independent-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["success"]:
        plot_report(args.folder, report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["success"] else 1)
