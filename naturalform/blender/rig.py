"""Armature creation and posing of built creatures."""

import math

import bpy
import numpy as np
from mathutils import Matrix, Vector

from ..core.creature import NaturalFormError, _suggest
from ..core.sdf import axis_angle_matrix, euler_matrix, rotation_between, vec


def _override():
    """A context override with a window / 3D view, for operators called from
    timers or other context-less places (e.g. an MCP server's executor)."""
    wm = bpy.context.window_manager
    for win in (wm.windows if wm else []):
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                region = next((r for r in area.regions if r.type == "WINDOW"), None)
                return {"window": win, "screen": win.screen, "area": area, "region": region}
        return {"window": win, "screen": win.screen}
    return {}


def mode_set(mode, obj=None):
    """bpy.ops.object.mode_set that also works without a UI context."""
    try:
        return bpy.ops.object.mode_set(mode=mode)
    except RuntimeError:
        ov = _override()
        if obj is not None:
            ov.update(active_object=obj, object=obj, selected_objects=[obj], selected_editable_objects=[obj])
        if not ov:
            raise
        with bpy.context.temp_override(**ov):
            return bpy.ops.object.mode_set(mode=mode)


def ensure_object_mode():
    obj = bpy.context.view_layer.objects.active
    if obj is not None and obj.mode != "OBJECT":
        try:
            mode_set("OBJECT", obj)
        except RuntimeError:
            pass


def default_roll(head, tail):
    d = np.asarray(tail) - np.asarray(head)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.array([0.0, -1.0, 0.0])
    d = d / n
    # bone Z axis: forward (-Y) for vertical bones, up (+Z) otherwise
    return np.array([0.0, -1.0, 0.0]) if abs(d[2]) > 0.7 else np.array([0.0, 0.0, 1.0])


def create_armature(name, bones, order, collection):
    """bones: dict name -> core Bone; order: parent-first list of names."""
    ensure_object_mode()
    data = bpy.data.armatures.new(name)
    arm = bpy.data.objects.new(name, data)
    collection.objects.link(arm)
    try:
        data.display_type = "STICK"
    except Exception:
        pass
    arm.show_in_front = True
    vl = bpy.context.view_layer
    prev_active = vl.objects.active
    for o in list(bpy.context.selected_objects):
        o.select_set(False)
    arm.select_set(True)
    vl.objects.active = arm
    mode_set("EDIT", arm)
    try:
        ebs = data.edit_bones
        for n in order:
            b = bones[n]
            eb = ebs.new(n)
            eb.head = Vector(b.head.tolist())
            eb.tail = Vector(b.tail.tolist())
            roll = b.roll if b.roll is not None else default_roll(b.head, b.tail)
            try:
                eb.align_roll(Vector(roll.tolist()))
            except Exception:
                pass
            eb.use_deform = b.deform
        for n in order:
            b = bones[n]
            if b.parent and b.parent in ebs:
                eb = ebs[n]
                eb.parent = ebs[b.parent]
                if b.connect and (ebs[b.parent].tail - eb.head).length < 1e-5:
                    eb.use_connect = True
    finally:
        mode_set("OBJECT", arm)
    if prev_active is not None and prev_active.name in vl.objects:
        vl.objects.active = prev_active
    return arm


def _resolve_armature(target):
    obj = bpy.data.objects.get(target) if isinstance(target, str) else target
    if obj is None:
        raise NaturalFormError("no object named %r.%s" % (target, _suggest(target, bpy.data.objects.keys())))
    if obj.type == "ARMATURE":
        return obj
    if obj.parent is not None and obj.parent.type == "ARMATURE":
        return obj.parent
    for m in getattr(obj, "modifiers", []):
        if m.type == "ARMATURE" and m.object is not None:
            return m.object
    raise NaturalFormError("%r is not a NaturalForm creature with a rig (build it with rig=True)" % obj.name)


def _depth(pb):
    d = 0
    while pb.parent is not None:
        pb = pb.parent
        d += 1
    return d


def pose_rig(target, pose, reset=False):
    """Pose a built creature's armature.

    Same syntax as ``Creature.pose``: {bone: [rx, ry, rz]} (world-axis
    degrees), {bone: {"aim": [x, y, z]}}, {"axis": [...], "angle": deg}.
    Directions/axes are in the creature's own space (as if it stood at the
    origin facing -Y), so poses work no matter where it was placed.
    ``reset=True`` clears the current pose first; otherwise rotations add
    on top of it.
    """
    arm = _resolve_armature(target)
    if isinstance(pose, str):
        presets = arm.get("nf_poses")
        import json
        presets = json.loads(presets) if presets else {}
        if pose not in presets:
            raise NaturalFormError("unknown pose %r for %s. Presets: %s" % (pose, arm.name, ", ".join(presets) or "none"))
        pose = presets[pose]
        reset = True
    pbs = arm.pose.bones
    unknown = [b for b in pose if b not in pbs]
    if unknown:
        raise NaturalFormError("unknown bone(s) %s.%s Bones: %s" % (unknown, _suggest(unknown[0], pbs.keys()), ", ".join(pbs.keys())))
    if reset:
        for pb in pbs:
            pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    for name in sorted(pose, key=lambda n: _depth(pbs[n])):
        pb = pbs[name]
        spec = pose[name]
        M = np.array(pb.matrix)  # armature space
        head = M[:3, 3].copy()
        cur = M[:3, 1].copy() * pb.bone.length
        if isinstance(spec, dict):
            if "aim" in spec:
                R = rotation_between(cur, vec(spec["aim"]))
                if spec.get("twist"):
                    R = axis_angle_matrix(R @ cur, math.radians(float(spec["twist"]))) @ R
            elif "axis" in spec:
                R = axis_angle_matrix(vec(spec["axis"]), math.radians(float(spec.get("angle", 0))))
            elif "rotate" in spec:
                R = euler_matrix(spec["rotate"])
            else:
                raise NaturalFormError("pose spec needs 'aim', 'axis'+'angle' or 'rotate': %r" % spec)
        else:
            R = euler_matrix(spec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = head - R @ head
        pb.matrix = Matrix((T @ M).tolist())
        bpy.context.view_layer.update()
    return {"armature": arm.name, "posed_bones": sorted(pose)}


def reset_pose(target):
    arm = _resolve_armature(target)
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    return {"armature": arm.name, "reset": True}


def bone_positions(target):
    """World-space head/tail of every bone (posed)."""
    arm = _resolve_armature(target)
    mw = arm.matrix_world
    out = {}
    for pb in arm.pose.bones:
        out[pb.name] = {"head": [round(v, 4) for v in mw @ pb.head],
                        "tail": [round(v, 4) for v in mw @ pb.tail],
                        "parent": pb.parent.name if pb.parent else None}
    return out
