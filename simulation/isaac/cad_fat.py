"""Inline the supplied FAT CAD mesh in the native Isaac scene as visual geometry.

The archive's STEP dimensions are retained in meters. No collision or mass API
is attached to these display meshes; the existing contact model is separate.
"""
import hashlib
import json
from pathlib import Path

from pxr import Gf, Usd, UsdGeom


ASSETS = Path(__file__).resolve().parents[2] / "assets/fat"


def add_fat_visual(stage, body_path, output=None, driver_paths=None):
    profile = json.loads((ASSETS / "fat_mobile_v1.json").read_text())
    if hashlib.sha256((ASSETS / profile["source"]).read_bytes()).hexdigest() != profile["source_sha256"]:
        raise RuntimeError("FAT_CAD_STEP_HASH_MISMATCH")
    source = ASSETS / profile["converted"]
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_hash != profile["converted_sha256"]:
        raise RuntimeError("FAT_CAD_ASSET_HASH_MISMATCH")
    cad = Usd.Stage.Open(str(source))
    if not cad or abs(UsdGeom.GetStageMetersPerUnit(cad) - 1.) > 1e-12:
        raise RuntimeError("FAT_CAD_UNITS_INVALID")
    if UsdGeom.GetStageUpAxis(cad) != UsdGeom.Tokens.z:
        raise RuntimeError("FAT_CAD_UP_AXIS_INVALID")
    meshes = [prim for prim in cad.Traverse() if prim.IsA(UsdGeom.Mesh)]
    if len(meshes) != 15:
        raise RuntimeError(f"FAT_CAD_MESH_COUNT_INVALID:{len(meshes)}")

    anchor = Gf.Vec3d(*profile["anchor_cad_m"])
    rotation = profile["rotation_cad_to_body"]
    def body_point(point):
        delta = point - anchor
        return Gf.Vec3f(*(sum(row[i] * delta[i] for i in range(3)) for row in rotation))

    root = UsdGeom.Xform.Define(stage, f"{body_path}/FAT_CAD")
    root.GetPrim().SetCustomDataByKey("source_step_sha256", profile["source_sha256"])
    root.GetPrim().SetCustomDataByKey("visual_only", True)
    root.GetPrim().SetCustomDataByKey("source_cad_usd_sha256", actual_hash)
    cache = UsdGeom.XformCache()
    top_surface = None
    for index, prim in enumerate(meshes):
        original = UsdGeom.Mesh(prim)
        world = cache.GetLocalToWorldTransform(prim)
        cad_points = [world.Transform(Gf.Vec3d(*point)) for point in original.GetPointsAttr().Get()]
        points = [body_point(point) for point in cad_points]
        path = f"{root.GetPath()}/part_{index:02d}"
        if driver_paths:
            from simulation.mujoco.fat_cad_mechanics import ROTATING_PARTS, SHAFT_XZ, SHAFT_Y
            for label, indices in ROTATING_PARTS.items():
                if index in indices:
                    path = f"{driver_paths[label]}/CAD_part_{index:02d}"
                    x,z = SHAFT_XZ[label]
                    points = [p-Gf.Vec3f(x,SHAFT_Y,z) for p in points]
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(original.GetFaceVertexCountsAttr().Get())
        mesh.CreateFaceVertexIndicesAttr(original.GetFaceVertexIndicesAttr().Get())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        name = str(prim.GetPath())
        if "/ArduinoNanoBox/" in name:
            color = (.16, .20, .26)
        elif "/motor" in name:
            color = (.29, .33, .39)
        elif "tn__fat_" in name:
            color = (.22, .27, .33)
            top_y = max(p[1] for p in cad_points)
            top_surface = [p for p in cad_points if abs(p[1] - top_y) < 1e-6]
        else:
            color = (.64, .68, .72)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        mesh.GetPrim().SetCustomDataByKey("cad_part_path", name)
    if not top_surface:
        raise RuntimeError("FAT_CAD_TOP_SURFACE_MISSING")
    center_cad = Gf.Vec3d(*[(min(p[i] for p in top_surface) + max(p[i] for p in top_surface))/2
                          for i in range(3)])
    center = body_point(center_cad)
    marker = profile["top_marker"]
    outer = marker["edge_m"] + 2 * marker["quiet_zone_m"]
    if any(max(p[i] for p in top_surface)-min(p[i] for p in top_surface) < outer for i in (0, 2)):
        raise RuntimeError("FAT_CAD_MARKER_DOES_NOT_FIT_TOP_SURFACE")
    _add_top_marker(stage, str(root.GetPath()), center, marker)
    if output is not None:
        (Path(output) / "fat-cad.json").write_text(json.dumps({
            **profile, "rendered_mesh_count": len(meshes),
            "top_marker_center_body_m": list(center),
            "usd_path": str(root.GetPath()), "physics": "separate CAD-aligned contact prisms" if driver_paths else "visual_only",
            "shaft_visual_articulation": bool(driver_paths),
        }, indent=2))
    return root.GetPath()


def _add_top_marker(stage, root, center, marker):
    import cv2

    pattern = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), marker["id"], 6)
    tag = UsdGeom.Xform.Define(stage, root + "/TopArUco")
    tag.AddTranslateOp().Set(Gf.Vec3d(*center))
    tag.GetPrim().SetCustomDataByKey("aruco_id", marker["id"])

    def tile(name, x, y, size, value, z):
        cube = UsdGeom.Cube.Define(stage, str(tag.GetPath()) + "/" + name)
        cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        cube.AddScaleOp().Set(Gf.Vec3f(size, size, .00005))
        cube.CreateDisplayColorAttr([Gf.Vec3f(value, value, value)])

    edge = marker["edge_m"]
    tile("quiet_zone", 0., 0., edge + 2 * marker["quiet_zone_m"], 1., .00010)
    for row in range(6):
        for col in range(6):
            if pattern[row, col] == 0:
                tile(f"cell_{row}_{col}", (col-2.5)*edge/6, (2.5-row)*edge/6, edge/6, 0., .00017)
