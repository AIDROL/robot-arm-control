"""Compare the supplied CAD shaft geometry with the active contact mechanism.

This is a geometric admission check, not a closed-loop acceptance result.
It runs on the source model without stepping either physics engine.
"""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from pxr import Gf, Usd, UsdGeom


ASSETS = Path(__file__).resolve().parents[2] / "assets/fat"


def audit(model):
    profile = json.loads((ASSETS / "fat_mobile_v1.json").read_text())
    path = ASSETS / profile["converted"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != profile["converted_sha256"]:
        raise ValueError("FAT_CAD_ASSET_HASH_MISMATCH")
    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    rotation, anchor = np.array(profile["rotation_cad_to_body"]), np.array(profile["anchor_cad_m"])
    shafts = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or "tn__m2_" not in str(prim.GetPath()):
            continue
        transform = cache.GetLocalToWorldTransform(prim)
        points = np.array([transform.Transform(Gf.Vec3d(*p)) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()])
        low, high = points.min(axis=0), points.max(axis=0)
        centre = (low + high) / 2
        shafts.append({"cad_part": str(prim.GetPath()), "centre_body_m": (rotation @ (centre-anchor)).tolist(),
                       "axis_body": (rotation @ np.array([0., 0., -1.])).tolist(),
                       "shaft_length_m": float(high[2]-low[2])})
    if len(shafts) != 2:
        raise ValueError(f"FAT_CAD_SHAFT_COUNT_INVALID:{len(shafts)}")
    shafts.sort(key=lambda s:s["centre_body_m"][0])
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    def distance(names):
        return float(np.linalg.norm(data.body(names[0]).xpos-data.body(names[1]).xpos))*1000
    cad_spacing = float(np.linalg.norm(np.array(shafts[1]["centre_body_m"])-shafts[0]["centre_body_m"]))*1000
    driver_spacing = distance(["lower_driver", "upper_driver"])
    socket_spacing = distance(["lower_knob", "upper_knob"])
    # Even allowing arbitrary translation/rotation, a pair's spacing mismatch
    # leaves at least half the difference as radial error at each shaft.
    best_radial_error = abs(cad_spacing-socket_spacing)/2
    tolerance = 1.5  # Existing socket_seating() admission threshold, millimeters.
    bounds = profile["cad_world_bounds_m"]
    cad_size = np.abs(rotation) @ (np.array(bounds["max"])-bounds["min"])
    calibrated = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'fat_cad_grasp_left') >= 0
    if calibrated:
        points=[]
        for i in range(model.ngeom):
            root=int(model.geom_bodyid[i])
            while root and model.body(root).name != 'fat':root=int(model.body_parentid[root])
            if not root or not (model.geom_contype[i] or model.geom_conaffinity[i]):continue
            kind=model.geom_type[i];size=model.geom_size[i]
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                mesh=model.geom_dataid[i];a=model.mesh_vertadr[mesh];n=model.mesh_vertnum[mesh]
                v=model.mesh_vert[a:a+n]
            else:
                ext=size if kind == mujoco.mjtGeom.mjGEOM_BOX else np.array([size[0],size[0],size[1]])
                v=np.array([[x,y,z] for x in (-ext[0],ext[0]) for y in (-ext[1],ext[1]) for z in (-ext[2],ext[2])])
            world=v@data.geom_xmat[i].reshape(3,3).T+data.geom_xpos[i]
            points.extend((world-data.body('fat').xpos)@data.body('fat').xmat.reshape(3,3))
        points=np.array(points);proxy_size=points.max(0)-points.min(0)
        axis_errors=[]
        for label,shaft in zip(('lower','upper'),shafts):
            local=data.body('fat').xmat.reshape(3,3).T@(data.body(label+'_driver').xpos-data.body('fat').xpos)
            axis_errors.append(float(np.linalg.norm(local[[0,2]]-np.array(shaft['centre_body_m'])[[0,2]])))
    else:
        proxy_size=np.asarray(model.geom('fat_housing').size)*2
    checks = [
        {"name":"cad_socket_pair_radial_fit", "passed":best_radial_error < tolerance,
         "minimum_error_mm":best_radial_error, "limit_mm":tolerance},
        {"name":"cad_driver_pair_spacing", "passed":abs(cad_spacing-driver_spacing)<.1,
         "difference_mm":abs(cad_spacing-driver_spacing), "limit_mm":.1},
        {"name":"cad_housing_collision_dimensions", "passed":bool(np.allclose(cad_size,proxy_size,atol=.001,rtol=0)),
         "cad_size_mm":(cad_size*1000).tolist(), "proxy_size_mm":(proxy_size*1000).tolist()},
    ]
    if calibrated:
        checks.append({'name':'cad_driver_axis_positions','passed':max(axis_errors)<.0001,
                       'max_error_mm':max(axis_errors)*1000,'limit_mm':.1})
    return {"schema_version":"fat-interface-audit/1.0", "passed":all(c["passed"] for c in checks),
            "scope":"geometric admission only; no FAT motion or optics executed",
            "source_step_sha256":profile["source_sha256"], "cad_sha256":profile["converted_sha256"],
            "cad_shafts":shafts, "cad_axis_spacing_mm":cad_spacing,
            "physics_driver_spacing_mm":driver_spacing, "fixture_socket_spacing_mm":socket_spacing,
            "checks":checks,
            "mechanical_profile":"cad_v1_matching_fixture" if calibrated else "legacy_proxy",
            "remaining_calibrations":["hardware mass/inertia, friction and motor torque",
                                      "physical camera extrinsics", "hardware knob to mirror response"],
            "closed_loop_acceptance":"not_run"}


def require_compatible(model, output):
    result = audit(model)
    (Path(output)/"fat-interface-audit.json").write_text(json.dumps(result,indent=2))
    if not result["passed"]:
        raise RuntimeError("FAT_CAD_INTERFACE_NOT_CALIBRATED: see fat-interface-audit.json")
    return result
