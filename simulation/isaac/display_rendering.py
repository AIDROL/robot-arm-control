"""Shared display-only RTX settings, atomic pose updates and viewport probes.

Do not apply this profile to control/acceptance cameras in the physics worker.
"""
import json
from pathlib import Path


def configure(viewport, resolution=(1440, 810), stage=None):
    import carb.settings
    import omni.replicator.core as rep

    viewport.resolution = resolution
    viewport.resolution_scale = 1.
    # Quality reconstruction leaves enough GPU headroom for uniform 60 Hz.
    # Performance mode at low sensor resolution caused the earlier smearing.
    rep.settings.set_render_rtx_realtime(antialiasing="DLSS")
    settings = carb.settings.get_settings()
    settings.set_int("/rtx/post/dlss/execMode", 2)
    # Zero-angle directional shadows are stable; the original sampled dome
    # shadows shimmered. Keep the lighting edits in the display process only.
    settings.set_bool("/rtx/shadows/enabled", True)
    for key in ("/rtx/reflections/enabled",
                "/rtx/indirectDiffuse/enabled", "/rtx/post/motionblur/enabled",
                "/rtx-transient/dlssg/enabled"):
        settings.set_bool(key, False)
    settings.set_float("/rtx/sceneDb/ambientLightIntensity", .5)
    settings.set_int("/rtx/background/source/type", 2)
    settings.set("/rtx/background/source/color", (.045, .055, .075))
    settings.set_bool("/app/viewport/grid/enabled", True)
    if stage is not None:
        restore_ground(stage)
        studio_lighting(stage)
        surface_materials(stage)



def restore_ground(stage):
    """Restore the existing physical floor's display; never add a collider."""
    from pxr import Gf, UsdGeom

    for prim in stage.Traverse():
        if prim.GetCustomDataByKey("source_geom_name") == "floor":
            # The source floor used a checker texture; the importer only
            # copied its white material multiplier. Keep a readable fallback.
            prim.SetDisplayName("Ground Plane")
            UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(.20, .22, .25)])
            UsdGeom.Imageable(prim).MakeVisible()


def studio_lighting(stage):
    from pxr import Gf, UsdGeom, UsdLux

    dome = stage.GetPrimAtPath("/World/Lighting")
    if dome:
        UsdGeom.Imageable(dome).MakeInvisible()
    for name, intensity, color, rotation in (
            ("Key", 2000., (1., 1., 1.), (25., -35., -25.)),
            ("Fill", 650., (.78, .86, 1.), (35., 35., 20.))):
        light = UsdLux.DistantLight.Define(stage, "/World/Display" + name + "Light")
        light.CreateIntensityAttr(intensity)
        light.CreateColorAttr(Gf.Vec3f(*color))
        light.CreateAngleAttr(0.)
        xform = UsdGeom.Xformable(light)
        xform.ClearXformOpOrder()
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))


def surface_materials(stage):
    """Add display surfaces without replacing the physics-purpose bindings."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    cache = {}
    for prim in list(stage.Traverse()):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        geom = UsdGeom.Gprim(prim)
        colors = geom.GetDisplayColorAttr().Get()
        if not colors or geom.GetVisibilityAttr().Get() == "invisible":
            continue
        color = tuple(map(float, colors[0]))
        # Keep the archived object colors, labels and detector markings intact.
        # Coated parts have a satin finish; dark support hardware is rougher.
        roughness = .34 if max(color)-min(color) < .15 else .42
        readable_screen = (prim.GetCustomDataByKey("source_geom_name") or "").endswith("_detector_screen")
        key = color, roughness, readable_screen
        if key not in cache:
            material = UsdShade.Material.Define(stage, f"/World/DisplayMaterials/m{len(cache)}")
            shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(.12)
            if readable_screen:
                # A neutral display fill makes the blank screen recognizable
                # inside the carrier shadow. This is not measured irradiance.
                shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.25, .25, .25))
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            cache[key] = material
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(cache[key])
        # The MJCF-to-USD export preserved STL triangles but omitted their
        # smooth shading. Restore normals on curved arm shells only; the
        # source mesh, collision geometry and object markings stay unchanged.
        if prim.IsA(UsdGeom.Mesh) and prim.GetParent().GetName().startswith("link"):
            import numpy as np
            mesh = UsdGeom.Mesh(prim)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            if np.all(counts == 3) and not mesh.GetNormalsAttr().HasAuthoredValueOpinion():
                normals = smooth_triangle_normals(np.asarray(mesh.GetPointsAttr().Get()),
                    np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3))
                mesh.CreateNormalsAttr(normals)
                mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)


def smooth_triangle_normals(points, faces, crease_degrees=45.):
    """Area-weighted corner normals; never smooth across a sharp mesh edge."""
    import numpy as np

    corners = points[faces]
    cross = np.cross(corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0])
    unit = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-15)
    flat = faces.ravel()
    counts = np.bincount(flat, minlength=len(points))
    offsets = np.cumsum(counts)-counts
    order = np.argsort(flat)
    degree = counts[flat]
    owner = np.repeat(np.arange(len(flat)), degree)
    neighbor_index = np.repeat(offsets[flat], degree) + np.arange(len(owner)) - np.repeat(np.cumsum(degree)-degree, degree)
    neighbors = order[neighbor_index] // 3
    keep = np.einsum('ij,ij->i', unit[owner//3], unit[neighbors]) >= np.cos(np.deg2rad(crease_degrees))
    result = np.zeros((len(flat), 3))
    np.add.at(result, owner[keep], cross[neighbors[keep]])
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-15)
    return result.astype(np.float32)


def settings_snapshot(viewport):
    import carb.settings

    settings = carb.settings.get_settings()
    keys = ("/rtx/post/aa/op", "/rtx/post/dlss/execMode", "/rtx/post/motionblur/enabled", "/rtx/shadows/enabled",
            "/rtx-transient/dlssg/enabled", "/rtx/reflections/enabled", "/rtx/indirectDiffuse/enabled",
            "/app/viewport/grid/enabled")
    result = {"settings": {key: settings.get(key) for key in keys},
              "viewport_resolution": list(viewport.resolution), "profile": "studio display only"}
    if (result["settings"]["/rtx/post/aa/op"] != 3 or result["settings"]["/rtx/post/dlss/execMode"] != 2
            or result["settings"]["/rtx/shadows/enabled"] is not True
            or result["settings"]["/app/viewport/grid/enabled"] is not True):
        raise RuntimeError("DISPLAY_RENDER_SETTINGS_OVERRIDDEN")
    return result


def add_cameras(stage, descriptors):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from pxr import Gf, UsdGeom

    paths = {}
    for name, descriptor in descriptors.items():
        path = descriptor.get("prim_path") or descriptor["path"]
        camera = UsdGeom.Camera.Define(stage, path)
        if "local_transform" in descriptor:
            transform = np.array(descriptor["local_transform"])
            xform = UsdGeom.Xformable(camera.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, transform[:3, 3])))
            q = Rotation.from_matrix(transform[:3, :3]).as_quat()
            xform.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
        camera.CreateClippingRangeAttr(Gf.Vec2f(.001, 10.))
        camera.CreateHorizontalApertureAttr(descriptor["horizontal_aperture"])
        camera.CreateVerticalApertureAttr(descriptor["horizontal_aperture"]*9/16)
        camera.CreateFocalLengthAttr(descriptor["focal_length"])
        paths[name] = path
    return paths


class PoseWriter:
    def __init__(self, prims):
        self.attributes = [(prim.GetAttribute("xformOp:translate"), prim.GetAttribute("xformOp:orient"))
                           for prim in prims]
        self.previous = None

    def update(self, positions, quaternions):
        import numpy as np
        from pxr import Gf, Sdf

        if self.previous is not None and np.array_equal(positions, self.previous[0]) and np.array_equal(quaternions, self.previous[1]):
            return
        # Publish an entire pose at once, rather than many individual USD notices.
        with Sdf.ChangeBlock():
            for (translate, orient), p, q in zip(self.attributes, positions, quaternions):
                translate.Set(Gf.Vec3d(*map(float, p)))
                orient.Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
        self.previous = positions.copy(), quaternions.copy()


class ViewportProbe:
    def __init__(self, folder, viewport):
        self.folder = Path(folder) if folder else None
        self.viewport = viewport
        self.captures = []
        if self.folder:
            self.folder.mkdir(parents=True, exist_ok=True)

    def update(self, frame):
        if self.folder and 60 <= frame < 90:
            from omni.kit.viewport.utility import capture_viewport_to_file
            self.captures.append(capture_viewport_to_file(self.viewport, str(self.folder/f"viewport-{frame:03d}.png")))

    def save(self, name, report):
        if self.folder:
            (self.folder/name).write_text(json.dumps(report, indent=2))
