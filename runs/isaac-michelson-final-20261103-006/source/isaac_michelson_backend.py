"""PhysX scene built from the repository's calibrated mechanical description.

MuJoCo is used as an asset/kinematics reader only: this module never calls
mj_step, mj_forward, collision or the MuJoCo contact solver. Dynamics, contacts,
joint drives and rendered images belong to Isaac Sim / PhysX.
Import after starting SimulationApp and making the project dependencies available.
"""
from pathlib import Path
import json
import time

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import omni.usd
import omni.physx
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, RigidPrim
from isaacsim.sensors.camera import Camera
from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

from simulation.michelson_experiment import load_layout, make_scene


def quat(rotation):
    xyzw = Rotation.from_matrix(np.asarray(rotation)).as_quat()
    return Gf.Quatf(float(xyzw[3]), Gf.Vec3f(*map(float, xyzw[:3])))


def matrix(position, quaternion):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
    result[:3, 3] = position
    return result


def transform(prim, T):
    api = UsdGeom.Xformable(prim)
    api.ClearXformOpOrder()
    api.AddTranslateOp().Set(Gf.Vec3d(*map(float, T[:3, 3])))
    api.AddOrientOp().Set(quat(T[:3, :3]))


class IsaacMechanics:
    def __init__(self, app, output, view=False, dt=.001, width=1280, height=720):
        self.app, self.output, self.dt = app, Path(output), dt
        self.config, self.targets = load_layout()
        self.model = make_scene(self.config, self.targets)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        self.world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=dt)
        self.stage = omni.usd.get_context().get_stage()
        context = self.world.get_physics_context()
        context.set_solver_type("TGS")
        context.enable_gpu_dynamics(False)
        context.set_broadphase_type("MBP")
        self.body_paths, self.joint_paths, self.geom_paths = {}, {}, {}
        self.link_owners, self.owner_transforms = {}, {}
        self.articulations, self.joint_handles = {}, {}
        self.sensors = {}
        self.stiction_reference = {"lower_knob_joint": 0., "upper_knob_joint": 0.}
        self.last_commands = {}
        self.magnets = set()
        self.contacts = []
        self.time = 0.
        self.steps = 0
        self.profile = {"physics_s": 0., "state_readback_s": 0.}
        self.view = view
        self._materials = {}
        self.drive_specs = {}
        self._build_bodies()
        self._build_geometries()
        self._build_joints()
        self._build_constraints()
        light = UsdLux.DomeLight.Define(self.stage, "/World/Lighting")
        light.CreateIntensityAttr(1000.)
        self.world.reset()
        for name in ("link_base", "mirror", "fat"):
            path = self.body_paths[self.model.body(name).id]
            articulation = Articulation(path, name=name)
            articulation.initialize()
            self.articulations[name] = articulation
            print("ISAAC_ARTICULATION", name, list(articulation.dof_names), flush=True)
            for i, joint_name in enumerate(articulation.dof_names):
                self.joint_handles[joint_name] = (articulation, i)
        self.rigid = RigidPrim(list(self.body_paths.values()), name="mechanical_bodies")
        self.rigid.initialize()
        self.rigid_ids = list(self.body_paths)
        drives = {}
        for name, articulation in self.articulations.items():
            kp, kd = articulation.get_gains()
            effort = articulation.get_max_efforts()
            armature = articulation.get_armatures()
            drives[name] = {joint: {"kp": float(kp[0, i]), "kd": float(kd[0, i]),
                                    "max_effort": float(effort[0, i]), "armature": float(armature[0, i])}
                            for i, joint in enumerate(articulation.dof_names)}
            for joint, measured in drives[name].items():
                if joint in self.drive_specs:
                    for parameter in ("kp", "kd", "max_effort"):
                        expected = self.drive_specs[joint][parameter]
                        if not np.isclose(measured[parameter], expected, rtol=1e-4, atol=1e-9):
                            raise RuntimeError(f"NATIVE_DRIVE_UNITS_MISMATCH:{joint}:{parameter}:{measured[parameter]}:{expected}")
                j = self.model.joint(joint).id
                expected_armature = float(self.model.dof_armature[self.model.jnt_dofadr[j]])
                if not np.isclose(measured["armature"], expected_armature, rtol=1e-4, atol=1e-12):
                    raise RuntimeError("NATIVE_ARMATURE_MISMATCH:" + joint)
        (self.output / "native-drives.json").write_text(json.dumps(drives, indent=2))
        home = self.model.key("home").qpos
        for name, (articulation, index) in self.joint_handles.items():
            joint = self.model.joint(name).id
            value = float(home[self.model.jnt_qposadr[joint]])
            articulation.set_joint_positions(np.array([[value]]), joint_indices=np.array([index]))
            if name.startswith("joint") or name.endswith("driver_joint"):
                articulation.set_joint_position_targets(np.array([[value]]), joint_indices=np.array([index]))
        self._subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact)
        self.world.play()
        self.camera = Camera("/World/GlobalCamera", resolution=(width, height))
        self.camera.initialize()
        self.camera.set_clipping_range(.01, 10.)
        # A calibrated overhead image covers the staging slots and experiment area.
        self.camera.set_world_pose(position=np.array([0., .33, 1.9]),
                                   orientation=np.array([1., 0., 0., 0.]), camera_axes="usd")
        self.camera.set_focal_length(2.0)
        self.camera.set_horizontal_aperture(2.0955)
        self.camera.set_vertical_aperture(2.0955 * height / width)
        self.K = np.asarray(self.camera.get_intrinsics_matrix())
        self.W_C = matrix(*self.camera.get_world_pose(camera_axes="usd")) @ np.diag([1., -1., -1., 1.])
        if view:
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(eye=np.array([1.4, -1., 1.6]), target=np.array([0., .3, .85]))
        self.stage.GetRootLayer().Export(str(self.output / "scene.usda"))
        self.sync()

    def _new_body(self, path, T, mass, com=None, inertia=None, inertia_rotation=None):
        prim = UsdGeom.Xform.Define(self.stage, path).GetPrim()
        transform(prim, T)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        phys = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        phys.CreateSolverPositionIterationCountAttr(64)
        phys.CreateSolverVelocityIterationCountAttr(8)
        phys.CreateSleepThresholdAttr(0.)
        phys.CreateMaxDepenetrationVelocityAttr(.5)
        api = UsdPhysics.MassAPI.Apply(prim)
        api.CreateMassAttr(float(max(mass, 1e-6)))
        api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, np.zeros(3) if com is None else com)))
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, np.full(3, 1e-9) if inertia is None else inertia)))
        if inertia_rotation is not None:
            api.CreatePrincipalAxesAttr(quat(inertia_rotation))
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        return prim

    def _build_bodies(self):
        m, d = self.model, self.data
        # Merge fixed children into their dynamic ancestor, preserving its mass
        # and inertia with the parallel-axis theorem.
        for i in range(1, m.nbody):
            owner = i
            while m.body_jntnum[owner] == 0 and m.body_parentid[owner] != 0:
                owner = int(m.body_parentid[owner])
            self.link_owners[i] = owner
        for owner in sorted(set(self.link_owners.values())):
            name = m.body(owner).name or f"body_{owner}"
            T = matrix(d.xpos[owner], d.xquat[owner])
            self.owner_transforms[owner] = T
            members = [i for i, value in self.link_owners.items() if value == owner]
            total = sum(m.body_mass[i] for i in members)
            if total <= 0:
                continue
            inv = np.linalg.inv(T)
            centers = {i: (inv @ np.r_[d.xipos[i], 1.])[:3] for i in members}
            com = sum(m.body_mass[i] * centers[i] for i in members) / total
            inertia = np.zeros((3, 3))
            for i in members:
                R = T[:3, :3].T @ d.ximat[i].reshape(3, 3)
                offset = centers[i] - com
                inertia += R @ np.diag(m.body_inertia[i]) @ R.T
                inertia += m.body_mass[i] * ((offset @ offset) * np.eye(3) - np.outer(offset, offset))
            eig, axes = np.linalg.eigh(inertia)
            if np.linalg.det(axes) < 0:
                axes[:, 0] *= -1
            path = f"/World/Bodies/{name}"
            self._new_body(path, T, total, com, np.maximum(eig, 1e-10), axes)
            self.body_paths[owner] = path
        for name in ("link_base", "mirror", "fat"):
            prim = self.stage.GetPrimAtPath(self.body_paths[m.body(name).id])
            UsdPhysics.ArticulationRootAPI.Apply(prim)
            articulation = PhysxSchema.PhysxArticulationAPI.Apply(prim)
            articulation.CreateEnabledSelfCollisionsAttr(True)
            articulation.CreateSolverPositionIterationCountAttr(64)
            articulation.CreateSolverVelocityIterationCountAttr(8)

    def _material(self, friction):
        key = round(float(friction), 3)
        if key not in self._materials:
            mat = UsdShade.Material.Define(self.stage, f"/World/Materials/friction_{str(key).replace('.', '_')}")
            api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            api.CreateStaticFrictionAttr(key)
            api.CreateDynamicFrictionAttr(key)
            api.CreateRestitutionAttr(0.)
            PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr("max")
            self._materials[key] = mat
        return self._materials[key]

    def _build_geometries(self):
        m, d = self.model, self.data
        for i in range(m.ngeom):
            body = int(m.geom_bodyid[i])
            owner = self.link_owners.get(body, 0)
            collision = bool(m.geom_contype[i] or m.geom_conaffinity[i])
            if not collision and m.geom_rgba[i, 3] == 0:
                continue
            parent = self.body_paths.get(owner, "/World/Static")
            path = f"{parent}/g{i:04d}"
            kind, size = int(m.geom_type[i]), m.geom_size[i]
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                geom = UsdGeom.Cube.Define(self.stage, path)
                geom.CreateSizeAttr(1.)
                scale = size * 2
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                geom = UsdGeom.Sphere.Define(self.stage, path)
                geom.CreateRadiusAttr(float(size[0]))
                scale = None
            elif kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
                cls = UsdGeom.Cylinder if kind == mujoco.mjtGeom.mjGEOM_CYLINDER else UsdGeom.Capsule
                geom = cls.Define(self.stage, path)
                geom.CreateRadiusAttr(float(size[0]))
                geom.CreateHeightAttr(float(size[1] * 2))
                geom.CreateAxisAttr("Z")
                scale = None
            elif kind == mujoco.mjtGeom.mjGEOM_PLANE:
                geom = UsdGeom.Cube.Define(self.stage, path)
                geom.CreateSizeAttr(1.)
                # MuJoCo planes can have a finite visual extent (the table
                # surface). Keep that extent instead of covering the room.
                xy = np.where(size[:2] > 0, size[:2] * 2, 20.)
                scale = np.r_[xy, .01 if collision else .0002]
            elif kind == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = int(m.geom_dataid[i])
                va, vn = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
                fa, fn = m.mesh_faceadr[mesh], m.mesh_facenum[mesh]
                vertices = m.mesh_vert[va:va + vn]
                faces = m.mesh_face[fa:fa + fn]
                geom = UsdGeom.Mesh.Define(self.stage, path)
                geom.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in vertices])
                geom.CreateFaceVertexCountsAttr([3] * fn)
                geom.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
                geom.CreateSubdivisionSchemeAttr("none")
                geom.CreateDoubleSidedAttr(True)
                scale = None
            else:
                raise ValueError(f"Unsupported geometry {i}: {kind}")
            T = np.eye(4)
            T[:3, :3] = d.geom_xmat[i].reshape(3, 3)
            T[:3, 3] = d.geom_xpos[i]
            if owner in self.body_paths:
                T = np.linalg.inv(self.owner_transforms[owner]) @ T
            if kind == mujoco.mjtGeom.mjGEOM_PLANE:
                T[2, 3] -= scale[2] / 2
            transform(geom.GetPrim(), T)
            geom.GetPrim().SetCustomDataByKey("source_geom_name", m.geom(i).name or "")
            if scale is not None:
                UsdGeom.Xformable(geom).AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))
            rgba = m.geom_rgba[i].copy()
            if m.geom_matid[i] >= 0:
                rgba = m.mat_rgba[m.geom_matid[i]].copy()
            geom.CreateDisplayColorAttr([Gf.Vec3f(*map(float, rgba[:3]))])
            # Contact decomposition is invisible; source visuals retain the STL.
            if m.geom_group[i] == 3 or rgba[3] == 0:
                geom.CreateVisibilityAttr("invisible")
            self.geom_paths[path] = i
            if collision:
                UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
                api = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
                api.CreateContactOffsetAttr(.0001)
                api.CreateRestOffsetAttr(0.)
                # MuJoCo condim=4/6 includes spin friction (coefficient has
                # length units); PhysX defaults to zero torsional friction.
                # A minimum patch radius gives the matching mu*N*r torque
                # bound for a single-anchor contact patch. Multi-anchor
                # patches already have their distributed friction moment.
                if m.geom_condim[i] >= 4 and m.geom_friction[i, 0] > 0:
                    radius = float(m.geom_friction[i, 1] / m.geom_friction[i, 0])
                    api.CreateMinTorsionalPatchRadiusAttr(radius)
                if kind == mujoco.mjtGeom.mjGEOM_MESH:
                    UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr("convexHull")
                UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(self._material(m.geom_friction[i, 0]),
                    bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics")

    def _joint(self, path, parent_path, child_path, world_frame, kind, parent_T, child_T):
        joint = kind.Define(self.stage, path)
        if parent_path:
            joint.CreateBody0Rel().SetTargets([parent_path])
        joint.CreateBody1Rel().SetTargets([child_path])
        for index, T in enumerate((parent_T, child_T)):
            local = np.linalg.inv(T) @ world_frame
            getattr(joint, f"CreateLocalPos{index}Attr")().Set(Gf.Vec3f(*map(float, local[:3, 3])))
            getattr(joint, f"CreateLocalRot{index}Attr")().Set(quat(local[:3, :3]))
        return joint

    def _build_joints(self):
        m, d = self.model, self.data
        for body, child_path in self.body_paths.items():
            parent = int(m.body_parentid[body])
            parent_owner = self.link_owners.get(parent, 0)
            parent_path = self.body_paths.get(parent_owner)
            parent_T = self.owner_transforms.get(parent_owner, np.eye(4))
            child_T = self.owner_transforms[body]
            joints = list(range(m.body_jntadr[body], m.body_jntadr[body] + m.body_jntnum[body]))
            if not joints:
                self._joint(f"/World/Joints/fix_{m.body(body).name}", parent_path, child_path,
                            child_T, UsdPhysics.FixedJoint, parent_T, child_T)
                continue
            for offset, j in enumerate(joints):
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    continue
                name = m.joint(j).name
                target_path = child_path
                if offset < len(joints) - 1:
                    target_path = f"/World/Bodies/virtual_{name}"
                    self._new_body(target_path, child_T, 1e-6)
                world = child_T.copy()
                world[:3, 3] = (child_T @ np.r_[m.jnt_pos[j], 1.])[:3]
                axis = m.jnt_axis[j]
                axis_R = Rotation.align_vectors(axis[None, :], np.array([[0., 0., 1.]]))[0].as_matrix()
                world[:3, :3] = child_T[:3, :3] @ axis_R
                hinge = m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
                kind = UsdPhysics.RevoluteJoint if hinge else UsdPhysics.PrismaticJoint
                path = f"/World/Joints/{name}"
                joint = self._joint(path, parent_path, target_path, world, kind, parent_T, child_T)
                joint.CreateAxisAttr("Z")
                PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim()).CreateArmatureAttr(
                    float(m.dof_armature[m.jnt_dofadr[j]]))
                factor = 180. / np.pi if hinge else 1.
                if m.jnt_limited[j]:
                    joint.CreateLowerLimitAttr(float(m.jnt_range[j, 0] * factor))
                    joint.CreateUpperLimitAttr(float(m.jnt_range[j, 1] * factor))
                elif name in ("lower_knob_joint", "upper_knob_joint"):
                    # PhysX mimic requires finite reference limits. These are
                    # exactly the mirror's existing +/-0.02 rad travel / ratio.
                    joint.CreateLowerLimitAttr(float(-.02 / .0016 * factor))
                    joint.CreateUpperLimitAttr(float(.02 / .0016 * factor))
                damping = float(m.dof_damping[m.jnt_dofadr[j]])
                stiffness = float(m.jnt_stiffness[j])
                max_force = 1e6
                target = float(m.qpos_spring[m.jnt_qposadr[j]])
                for a in range(m.nu):
                    if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_trnid[a, 0] == j:
                        stiffness = float(m.actuator_gainprm[a, 0])
                        damping += float(-m.actuator_biasprm[a, 2])
                        max_force = float(max(abs(m.actuator_forcerange[a])))
                if name in ("left_driver_joint", "right_driver_joint"):
                    stiffness, damping, max_force, target = 50., 5., 25., 0.
                if name in self.stiction_reference:
                    # Passive elasto-plastic dry-friction approximation. The
                    # yield torque retains the source's 0.3 mN m threshold.
                    stiffness, damping = 1., .00002
                    max_force = float(m.dof_frictionloss[m.jnt_dofadr[j]])
                    target = 0.
                if stiffness or damping:
                    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular" if hinge else "linear")
                    drive.CreateTypeAttr("force")
                    # USD angular gains use degrees; PhysX tensor gains and
                    # the source mechanical description use radians. Limits
                    # and target angles need the opposite conversion above.
                    drive.CreateStiffnessAttr(stiffness / factor)
                    drive.CreateDampingAttr(damping / factor)
                    drive.CreateMaxForceAttr(max_force)
                    drive.CreateTargetPositionAttr(target * factor)
                    self.drive_specs[name] = {"kp": stiffness, "kd": damping, "max_effort": max_force}
                self.joint_paths[name] = path
                parent_path, parent_T = target_path, child_T

    def _build_constraints(self):
        m = self.model
        # Preserve source parent/child and explicit collision exclusions even
        # where several source DOFs introduce virtual PhysX links.
        exclusions = {}
        for body in self.body_paths:
            parent = self.link_owners.get(int(m.body_parentid[body]), 0)
            if parent in self.body_paths and parent != body:
                exclusions.setdefault(body, set()).add(parent)
        for signature in m.exclude_signature:
            a, b = int(signature) >> 16, int(signature) & 65535
            a, b = self.link_owners[a], self.link_owners[b]
            if a != b:
                exclusions.setdefault(a, set()).add(b)
        for body, others in exclusions.items():
            api = UsdPhysics.FilteredPairsAPI.Apply(self.stage.GetPrimAtPath(self.body_paths[body]))
            api.CreateFilteredPairsRel().SetTargets([self.body_paths[i] for i in sorted(others)])
        for e in range(m.neq):
            if not m.eq_active0[e]:
                continue
            kind = m.eq_type[e]
            if kind == mujoco.mjtEq.mjEQ_CONNECT:
                a, b = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
                oa, ob = self.link_owners[a], self.link_owners[b]
                T = matrix(self.data.xpos[a], self.data.xquat[a])
                frame = T.copy()
                frame[:3, 3] = (T @ np.r_[m.eq_data[e, :3], 1.])[:3]
                joint = self._joint(f"/World/Joints/loop_{e}", self.body_paths[oa], self.body_paths[ob],
                                    frame, UsdPhysics.SphericalJoint,
                                    self.owner_transforms[oa], self.owner_transforms[ob])
                joint.CreateExcludeFromArticulationAttr(True)
            elif kind == mujoco.mjtEq.mjEQ_JOINT:
                a, b = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
                name_a, name_b = m.joint(a).name, m.joint(b).name
                if name_a == "left_driver_joint":
                    # Both drivers receive the same finite-force motor command.
                    continue
                mimic = PhysxSchema.PhysxMimicJointAPI.Apply(self.stage.GetPrimAtPath(self.joint_paths[name_a]), "rotZ")
                mimic.CreateReferenceJointRel().SetTargets([self.joint_paths[name_b]])
                mimic.CreateReferenceJointAxisAttr("rotZ")
                mimic.CreateGearingAttr(float(-m.eq_data[e, 1]))
                mimic.CreateOffsetAttr(float(-m.eq_data[e, 0] * 180. / np.pi))
            else:
                raise ValueError(f"Unsupported active mechanical equality {e}: {kind}")
        self.latches = {}
        for name in ("lens_1", "pinhole", "lens_2", "beam_splitter", "mirror"):
            body = m.body(name).id
            T = self.owner_transforms[body]
            latch = self._joint(f"/World/Joints/{name}_base_latch", None, self.body_paths[body],
                                T, UsdPhysics.FixedJoint, np.eye(4), T)
            latch.CreateExcludeFromArticulationAttr(True)
            latch.CreateJointEnabledAttr(False)
            self.latches[name] = latch

    def latch(self, name):
        """Engage the ideal base at the attained pose, without changing state."""
        T = self.body_poses[self.model.body(name).id].copy()
        joint = self.latches[name]
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(*map(float, T[:3, 3])))
        joint.GetLocalRot0Attr().Set(quat(T[:3, :3]))
        joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.))
        joint.GetLocalRot1Attr().Set(Gf.Quatf(1.))
        joint.GetJointEnabledAttr().Set(True)
        self.world.render()
        return T

    def _on_contact(self, headers, data):
        for header in headers:
            a = self.geom_paths.get(str(PhysicsSchemaTools.intToSdfPath(header.collider0)))
            b = self.geom_paths.get(str(PhysicsSchemaTools.intToSdfPath(header.collider1)))
            if a is None or b is None:
                continue
            for c in data[header.contact_data_offset:header.contact_data_offset + header.num_contact_data]:
                impulse = np.asarray(c.impulse, dtype=float)
                normal = np.asarray(c.normal, dtype=float)
                self.contacts.append((a, b, float(abs(impulse @ normal) / self.dt), np.asarray(c.position)))

    def step(self, render=False):
        start = time.perf_counter()
        for name in self.magnets:
            body = self.model.body(name).id
            T = self.body_poses[body]
            point = (T @ np.array([0., 0., -.108, 1.]))[:3]
            self.rigid.apply_forces_and_torques_at_pos(forces=np.array([[0., 0., -30.]]),
                positions=np.array([point]), indices=np.array([self.rigid_ids.index(body)]), is_global=True)
        for name, reference in self.stiction_reference.items():
            if name not in self.joint_handles:
                continue
            joint = self.model.joint(name).id
            measured = float(self.data.qpos[self.model.jnt_qposadr[joint]])
            yield_angle = float(self.model.dof_frictionloss[self.model.jnt_dofadr[joint]])
            displacement = measured - reference
            if abs(displacement) > yield_angle:
                reference = measured - np.sign(displacement) * yield_angle
                self.stiction_reference[name] = reference
            self.command([name], [reference])
        self.contacts = []
        self.world.step(render=render)
        self.profile["physics_s"] += time.perf_counter() - start
        self.steps += 1
        self.time = self.steps * self.dt
        start = time.perf_counter()
        self.sync()
        self.profile["state_readback_s"] += time.perf_counter() - start

    def sync(self):
        m, d = self.model, self.data
        for articulation in self.articulations.values():
            positions = articulation.get_joint_positions()[0]
            velocities = articulation.get_joint_velocities()[0]
            for index, name in enumerate(articulation.dof_names):
                j = m.joint(name).id
                d.qpos[m.jnt_qposadr[j]] = positions[index]
                d.qvel[m.jnt_dofadr[j]] = velocities[index]
        positions, orientations = self.rigid.get_world_poses()
        self.body_poses = {body: matrix(p, q) for body, p, q in zip(self.rigid_ids, positions, orientations)}
        for body, T in self.body_poses.items():
            j = m.body_jntadr[body]
            if m.body_jntnum[body] and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                q = Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]
                adr = m.jnt_qposadr[j]
                d.qpos[adr:adr + 7] = np.r_[T[:3, 3], q]
        # The native world anchor has finite solver compliance. Feed the
        # measured robot base TF into the independent FK/Jacobian model instead
        # of assuming the nominal mounting transform stayed mathematically exact.
        base = m.body("link_base").id
        actual_base = self.body_poses[base]
        m.body_pos[base] = actual_base[:3, 3]
        m.body_quat[base] = Rotation.from_matrix(actual_base[:3, :3]).as_quat()[[3, 0, 1, 2]]
        d.time = self.time
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)

    def command(self, names, values):
        grouped = {}
        for name, value in zip(names, values):
            self.last_commands[name] = float(value)
            articulation, index = self.joint_handles[name]
            grouped.setdefault(articulation, ([], []))
            grouped[articulation][0].append(index)
            grouped[articulation][1].append(value)
        for articulation, (indices, targets) in grouped.items():
            articulation.set_joint_position_targets(np.array([targets]), joint_indices=np.array(indices))

    def set_fat_motor_power(self, enabled):
        """Backdrivable power-off mode retains only passive rotor damping."""
        readback = {}
        for name in ("lower_motor_joint", "upper_motor_joint"):
            arm, index = self.joint_handles[name]
            arm.set_gains(kps=np.array([[.03 if enabled else 0.]]),
                          kds=np.array([[.00011 if enabled else .00001]]),
                          joint_indices=np.array([index]))
            kp, kd = arm.get_gains()
            readback[name] = {"kp": float(kp[0, index]), "kd": float(kd[0, index])}
        return readback

    def restore_assembly(self, folder, optical_checkpoint=None):
        """Explicit development resume from a recorded native assembled state.

        This never counts as a fresh empty-area assembly acceptance run.
        """
        folder = Path(folder)
        import hashlib
        provenance = json.loads((folder / "provenance.json").read_text())
        config_path = Path(__file__).resolve().parents[1] / "configs/optical_setups/michelson_full.json"
        if hashlib.sha256(config_path.read_bytes()).hexdigest() != provenance["configuration_sha256"]:
            raise ValueError("CHECKPOINT_LAYOUT_MISMATCH")
        events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
        latched = {row["component"] for row in events if row["stage"] == "BASE_LATCHED_AT_ACHIEVED_POSE"}
        placements = [row for row in events if row["stage"] == "PLACEMENT_EVALUATED" and row["success"] and row["id"] in latched]
        inventory = [item["id"] for item in self.config["components"]]
        # A development continuation's accepted prefix may itself have been
        # restored. Its native assembled checkpoint still contains every body;
        # preserve the explicit provenance rather than inventing new events.
        declared = json.loads((folder / "result.json").read_text()).get("placements", [])
        if len(declared) > len(placements) and all(row.get("success") for row in declared):
            placements = declared
        if not placements or [row["id"] for row in placements] != inventory[:len(placements)]:
            raise ValueError("RESUME_REQUIRES_ACCEPTED_LATCHED_INVENTORY_PREFIX")
        completed = next((row for row in events if row["stage"] == "ASSEMBLY_GATE_COMPLETE" and row["components"] == 5), None)
        if completed is None:
            next_name = inventory[len(placements)]
            completed = next(row for row in events if row["stage"] == next_name + ": pregrasp")
        saved = np.load(folder / "physx-trajectory.npz")
        index = int(np.searchsorted(saved["time_s"], completed["time_s"], side="right") - 1)
        qpos = saved["measured_qpos"][index]
        if qpos.shape != self.data.qpos.shape:
            raise ValueError("CHECKPOINT_TOPOLOGY_MISMATCH")
        poses = dict(zip(saved["body_names"].tolist(), saved["body_poses"][index]))
        saved_time = float(saved["time_s"][index])
        if optical_checkpoint is not None:
            # Explicit development diagnosis only, never fresh acceptance.
            state = np.load(folder / "optics" / optical_checkpoint / "native-state.npz")
            qpos = state["qpos"]
            poses = dict(zip(state["body_names"].tolist(), state["body_poses"]))
            saved_time = float(state["time_s"])
        for name in ("lens_1", "pinhole", "lens_2", "beam_splitter", "mirror", "fat"):
            T = poses[name]
            q = Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]
            if name in self.articulations:
                self.articulations[name].set_world_poses(positions=np.array([T[:3, 3]]), orientations=np.array([q]))
            else:
                body = self.model.body(name).id
                self.rigid.set_world_poses(positions=np.array([T[:3, 3]]), orientations=np.array([q]),
                                           indices=np.array([self.rigid_ids.index(body)]))
        for name, (articulation, handle) in self.joint_handles.items():
            joint = self.model.joint(name).id
            value = float(qpos[self.model.jnt_qposadr[joint]])
            articulation.set_joint_positions(np.array([[value]]), joint_indices=np.array([handle]))
            if name.startswith("joint") or name.endswith(("driver_joint", "motor_joint")):
                self.command([name], [value])
            if name in self.stiction_reference:
                self.stiction_reference[name] = value
                self.command([name], [value])
        self.steps = round(saved_time / self.dt)
        self.time = self.steps * self.dt
        self.step()
        for item in placements:
            self.latch(item["id"])
        self.sync()
        return placements

    def grip(self, value):
        self.command(["left_driver_joint", "right_driver_joint"], [value * .00333] * 2)

    def site_pose(self, name):
        site = self.model.site(name).id
        body = int(self.model.site_bodyid[site])
        owner = self.link_owners[body]
        # Fixed site-to-body calibration; actual body pose comes from PhysX.
        local = np.linalg.inv(matrix(self.data.xpos[owner], self.data.xquat[owner]))
        calculated = np.eye(4)
        calculated[:3, :3] = self.data.site_xmat[site].reshape(3, 3)
        calculated[:3, 3] = self.data.site_xpos[site]
        return self.body_poses[owner] @ local @ calculated

    def capture(self, label, high_resolution=False):
        previous_resolution = self.camera.get_resolution()
        if high_resolution:
            self.camera.set_resolution((3840, 2160))
            self.K = np.asarray(self.camera.get_intrinsics_matrix())
        for _ in range(8):
            self.world.render()
        rgba = self.camera.get_rgba()
        if rgba is None or rgba.ndim != 3 or rgba.shape[0] == 0:
            raise RuntimeError("ISAAC_CAMERA_FRAME_MISSING")
        rgb = np.ascontiguousarray(rgba[:, :, :3], dtype=np.uint8)
        cv2.imwrite(str(self.output / f"{label}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        observed = {}
        h = .024 / 2
        square = np.array([[-h, h, 0.], [h, h, 0.], [h, -h, 0.], [-h, -h, 0.]])
        if ids is not None:
            for marker, pixels in zip(ids.flatten(), corners):
                if marker > 4:
                    continue
                ok, rv, tv = cv2.solvePnP(square, pixels.reshape(4, 2).astype(float), self.K,
                                         np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
                if not ok or tv[2, 0] <= 0:
                    continue
                C_M = np.eye(4)
                C_M[:3, :3], C_M[:3, 3] = cv2.Rodrigues(rv)[0], tv.ravel()
                # A resting module's support plane is a calibration input,
                # not simulator object truth. It avoids the severe depth /
                # tilt ambiguity of a distant 24 mm monocular marker.
                plane_z = .76 + .114 + .05235
                world_corners = []
                for u, v in pixels.reshape(4, 2):
                    ray = self.W_C[:3, :3] @ np.linalg.solve(self.K, [u, v, 1.])
                    distance = (plane_z - self.W_C[2, 3]) / ray[2]
                    if distance <= 0:
                        raise RuntimeError("TOP_MARKER_SUPPORT_PLANE_BEHIND_CAMERA")
                    world_corners.append(self.W_C[:3, 3] + distance * ray)
                points = np.array(world_corners)
                from simulation.isaac_michelson_control import upright_marker_rotation
                pose = np.eye(4)
                pose[:3, :3] = upright_marker_rotation(points)
                pose[:3, 3] = points.mean(axis=0) - [0., 0., .05235]
                observed[int(marker)] = pose
        (self.output / f"{label}-vision.json").write_text(json.dumps({
            "time_s": self.time, "source": "Isaac RGB corners + calibrated table/top-marker plane; resting modules only",
            "K": self.K.tolist(), "T_world_camera_optical": self.W_C.tolist(),
            "observations": {k: v.tolist() for k, v in observed.items()}}, indent=2))
        if high_resolution:
            self.camera.set_resolution(previous_resolution)
            self.K = np.asarray(self.camera.get_intrinsics_matrix())
        return observed

    def capture_sensor(self, name, label, depth=False):
        """Render a robot/tool-mounted camera using its fixed local calibration."""
        if name not in self.sensors:
            source = self.model.camera(name).id
            body = int(self.model.cam_bodyid[source])
            owner = self.link_owners[body]
            owner_T = matrix(self.data.xpos[owner], self.data.xquat[owner])
            body_T = matrix(self.data.xpos[body], self.data.xquat[body])
            W_C = body_T @ matrix(self.model.cam_pos[source], self.model.cam_quat[source])
            local = np.linalg.inv(owner_T) @ W_C
            camera = Camera(self.body_paths[owner] + "/Sensors/" + name, resolution=(1280, 720))
            camera.initialize()
            q = Rotation.from_matrix(local[:3, :3]).as_quat()[[3, 0, 1, 2]]
            camera.set_local_pose(translation=local[:3, 3], orientation=q, camera_axes="usd")
            camera.set_clipping_range(.001, 10.)
            aperture = 2.0955 * 720 / 1280
            camera.set_horizontal_aperture(2.0955)
            camera.set_vertical_aperture(aperture)
            camera.set_focal_length(aperture / (2 * np.tan(np.deg2rad(self.model.cam_fovy[source]) / 2)))
            if depth:
                camera.add_distance_to_image_plane_to_frame()
            self.sensors[name] = (camera, owner, local)
        camera, owner, local = self.sensors[name]
        for _ in range(8):
            self.world.render()
        rgb = np.ascontiguousarray(camera.get_rgba()[:, :, :3], dtype=np.uint8)
        # Read the native annotator directly, like get_rgba(). A camera created
        # after physics has advanced may not yet have an event-cached frame.
        depth_image = camera.get_depth() if depth else None
        if depth and depth_image is None:
            raise RuntimeError("ISAAC_METRIC_DEPTH_FRAME_MISSING")
        if depth:
            depth_image = np.asarray(depth_image).squeeze()
        K = np.asarray(camera.get_intrinsics_matrix())
        W_C = self.body_poses[owner] @ local @ np.diag([1., -1., -1., 1.])
        cv2.imwrite(str(self.output / (label + ".png")), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if depth:
            np.save(self.output / (label + "-depth-m.npy"), depth_image)
        return rgb, depth_image, K, W_C

    def contact_records(self):
        return [{"geom_a": a, "name_a": self.model.geom(a).name or "",
                 "body_a": self.model.body(int(self.model.geom_bodyid[a])).name,
                 "geom_b": b, "name_b": self.model.geom(b).name or "",
                 "body_b": self.model.body(int(self.model.geom_bodyid[b])).name,
                 "normal_force_N": force, "position_m": point.tolist()}
                for a, b, force, point in self.contacts]
