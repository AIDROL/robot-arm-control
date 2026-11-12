"""Recognizable schematic devices, aligned to the existing optical planes.

These are visual geometry, not vendor CAD, collision shapes or an RTX wave
optics simulation. Detector irradiance remains in the measured optics output.
"""

ROLE_LABELS = {'laser': 'Laser source', 'splitter': 'Beam splitter',
               'reference_mirror': 'Reference mirror', 'detector': 'Detector screen',
               'adjustable_mirror': 'Adjustable mirror'}


def device_geometry(role, detector_window=.02):
    """Body-local centres and half sizes; optical axis is local +Y at z=-.014."""
    if role == 'laser':
        return [
            ('laser_barrel', 'cylinder', (0., -.012, -.014), (.010, .015), (.10, .12, .15, 1.)),
            ('laser_support', 'box', (0., -.012, -.037), (.004, .006, .013), (.20, .22, .24, 1.)),
            ('laser_back_cap', 'cylinder', (0., -.028, -.014), (.0105, .001), (.24, .27, .30, 1.)),
            ('laser_front_ring', 'cylinder', (0., .0035, -.014), (.011, .0003), (.50, .53, .57, 1.)),
            ('laser_aperture', 'cylinder', (0., .00385, -.014), (.0045, .00015), (.9, .025, .015, 1.)),
        ]
    if role == 'detector':
        # The active square matches the actual configured detector window;
        # a blank white surface does not pretend to be a measured fringe image.
        half = detector_window / 2
        return [
            ('detector_support', 'box', (0., 0., -.042), (.004, .003, .008), (.20, .22, .24, 1.)),
            ('detector_frame', 'box', (0., 0., -.014),
             (max(.024, half+.006), .003, max(.020, half+.006)), (.055, .065, .08, 1.)),
            ('detector_screen', 'box', (0., .0037, -.014),
             (half, .0003, half), (.96, .97, .98, 1.)),
        ]
    return []


def add_mjcf_device(body, name, role, detector_window=.02):
    import mujoco
    from math import sqrt

    for suffix, kind, position, size, color in device_geometry(role, detector_window):
        arguments = dict(name=name+'_'+suffix, pos=position, size=size, rgba=color,
                         contype=0, conaffinity=0, mass=0.)
        if kind == 'cylinder':
            arguments.update(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                             quat=[sqrt(.5), sqrt(.5), 0., 0.])
        else:
            arguments['type'] = mujoco.mjtGeom.mjGEOM_BOX
        body.add_geom(**arguments)


def upgrade_isaac_display(stage, config):
    """Retrofit old replay copies too, without rewriting their saved USD/evidence."""
    from pxr import Gf, Usd, UsdGeom

    cameras = {}
    for component in config['components']:
        name, role = component['id'], component['role']
        body = stage.GetPrimAtPath('/World/Bodies/'+name)
        if not body:
            continue
        body.SetDisplayName(ROLE_LABELS[role]+' ('+name+')')
        geometry = device_geometry(role, config['detector_window_m'])
        if not geometry:
            continue
        existing = {p.GetCustomDataByKey('source_geom_name'): p for p in Usd.PrimRange(body)}
        old_face = existing.get(name+'_optical_face')
        if old_face:
            UsdGeom.Imageable(old_face).MakeInvisible()
        # The legacy generic mirror mount is not a laser/detector housing.
        # Its physics shapes remain, but its opaque display proxies occluded
        # the replacement screen at the calibrated optical plane.
        for source_name, prim in existing.items():
            if source_name and source_name.startswith((name+'_frame_', name+'_lower_', name+'_upper_')):
                UsdGeom.Imageable(prim).MakeInvisible()
        root = str(body.GetPath())+'/OpticalDevice'
        UsdGeom.Xform.Define(stage, root)
        for suffix, kind, position, size, color in geometry:
            source_name = name+'_'+suffix
            if source_name in existing:
                continue
            path = root+'/'+suffix
            if kind == 'cylinder':
                shape = UsdGeom.Cylinder.Define(stage, path)
                shape.CreateRadiusAttr(size[0])
                shape.CreateHeightAttr(size[1]*2)
                shape.CreateAxisAttr('Z')
            else:
                shape = UsdGeom.Cube.Define(stage, path)
                shape.CreateSizeAttr(1.)
            xform = UsdGeom.Xformable(shape.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(*position))
            if kind == 'cylinder':
                xform.AddRotateXOp().Set(90.)
            else:
                xform.AddScaleOp().Set(Gf.Vec3f(*(v*2 for v in size)))
            shape.CreateDisplayColorAttr([Gf.Vec3f(*color[:3])])
            shape.GetPrim().SetCustomDataByKey('source_geom_name', source_name)
            shape.GetPrim().SetCustomDataByKey('schematic_display_only', True)
        camera = UsdGeom.Camera.Define(stage, str(body.GetPath())+'/DeviceCamera')
        camera.CreateClippingRangeAttr(Gf.Vec2f(.002, 10.))
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateFocalLengthAttr(10.)
        transform = Gf.Matrix4d(1.).SetLookAt(Gf.Vec3d(.05, .10, .07),
                                            Gf.Vec3d(0., 0., -.02), Gf.Vec3d(0., 0., 1.)).GetInverse()
        xform = UsdGeom.Xformable(camera)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(transform)
        cameras['laser' if role == 'laser' else 'detector'] = str(camera.GetPath())
    return cameras
