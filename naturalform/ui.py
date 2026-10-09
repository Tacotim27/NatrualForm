"""Blender UI: sidebar panel, Add-menu entries and operators."""

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, StringProperty

from . import api
from .core.creature import NaturalFormError

_KINDS = [("human", "Human", "People of any age"), ("quadruped", "Quadruped", "Four-legged animals"),
          ("bird", "Bird", "Birds"), ("fish", "Fish", "Fish, sharks, dolphins"),
          ("serpent", "Serpent", "Snakes, worms, eels")]
_ENUM_CACHE = {}


def _species_items(self, context):
    kind = self.kind
    if kind == "human":
        items = [("male", "Man", ""), ("female", "Woman", ""), ("neutral", "Neutral", "")]
    else:
        items = [(s, s.replace("_", " ").title(), "") for s in api.TEMPLATES[kind]["presets"]]
    _ENUM_CACHE[("species", kind)] = items
    return items


def _pose_items(self, context):
    kind = self.kind
    items = [("NONE", "Default", "")] + [(p, p.replace("_", " ").title(), "") for p in api.TEMPLATES[kind]["poses"]]
    _ENUM_CACHE[("pose", kind)] = items
    return items


class NF_Settings(bpy.types.PropertyGroup):
    kind: EnumProperty(name="Kind", items=_KINDS, default="human")
    species: EnumProperty(name="Species", items=_species_items)
    age: FloatProperty(name="Age", default=30.0, min=0.5, max=100.0)
    height: FloatProperty(name="Height", description="0 = typical size", default=0.0, min=0.0, max=20.0,
                          unit="LENGTH")
    clothing: EnumProperty(name="Clothing", items=[(c, c.title(), "") for c in
                                                     ("casual", "shorts", "dress", "suit", "underwear", "none")])
    hair_style: EnumProperty(name="Hair", items=[("DEFAULT", "Default", "")] + [
        (h, h.title(), "") for h in ("short", "long", "bob", "ponytail", "bun", "afro", "buzz", "mohawk", "bald")])
    pose: EnumProperty(name="Pose", items=_pose_items)
    detail: EnumProperty(name="Detail", default="medium", items=[
        ("preview", "Preview", "fastest"), ("low", "Low", ""), ("medium", "Medium", ""),
        ("high", "High", "slow"), ("ultra", "Ultra", "very slow")])
    rig: BoolProperty(name="Rig", default=True)
    use_color: BoolProperty(name="Custom colour", default=False)
    color: FloatVectorProperty(name="Colour", subtype="COLOR_GAMMA", size=3, min=0.0, max=1.0,
                               default=(0.6, 0.45, 0.3))
    name: StringProperty(name="Name", default="")


def _settings_kwargs(s):
    kw = {"detail": s.detail, "rig": s.rig, "location": tuple(bpy.context.scene.cursor.location)}
    if s.name:
        kw["name"] = s.name
    if s.pose and s.pose != "NONE":
        kw["pose"] = s.pose
    if s.height > 0:
        kw["height" if s.kind in ("human", "quadruped", "bird") else "length"] = s.height
    if s.kind == "human":
        kw.update(sex=s.species, age=s.age, clothing=s.clothing)
        if s.hair_style != "DEFAULT":
            kw["hair_style"] = s.hair_style
        if s.use_color:
            kw["top_color"] = tuple(s.color)
        return api.create_human, kw
    if s.use_color:
        kw["color"] = tuple(s.color)
    fn = {"quadruped": api.create_quadruped, "bird": api.create_bird, "fish": api.create_fish,
          "serpent": api.create_serpent}[s.kind]
    return (lambda **k: fn(s.species, **k)), kw


class NATURALFORM_OT_create(bpy.types.Operator):
    bl_idname = "naturalform.create"
    bl_label = "Create Creature"
    bl_description = "Create the creature configured in the NaturalForm panel at the 3D cursor"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene.naturalform
        fn, kw = _settings_kwargs(s)
        try:
            r = fn(**kw)
        except NaturalFormError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        self.report({"INFO"}, "Created %s (%d vertices, %.1fs)" % (r["name"], r["vertices"], r["time_s"]))
        return {"FINISHED"}


class NATURALFORM_OT_quick_add(bpy.types.Operator):
    bl_idname = "naturalform.quick_add"
    bl_label = "Add Creature"
    bl_description = "Add a creature at the 3D cursor"
    bl_options = {"REGISTER", "UNDO"}

    what: StringProperty(default="human")

    def execute(self, context):
        try:
            r = api.create(self.what, location=tuple(context.scene.cursor.location))
        except NaturalFormError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        self.report({"INFO"}, "Created %s" % r["name"])
        return {"FINISHED"}


def _active_creature(context):
    obj = context.active_object
    while obj is not None and "nf_spec" not in obj:
        obj = obj.parent
    return obj


class NATURALFORM_OT_pose(bpy.types.Operator):
    bl_idname = "naturalform.pose"
    bl_label = "Apply Pose"
    bl_description = "Pose the active creature's rig with a preset"
    bl_options = {"REGISTER", "UNDO"}

    preset: StringProperty(default="stand")

    @classmethod
    def poll(cls, context):
        return _active_creature(context) is not None

    def execute(self, context):
        obj = _active_creature(context)
        try:
            api.pose(obj.name, self.preset)
        except NaturalFormError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        return {"FINISHED"}


class NATURALFORM_OT_rebuild(bpy.types.Operator):
    bl_idname = "naturalform.rebuild"
    bl_label = "Rebuild"
    bl_description = "Re-mesh the active creature with the panel's detail level"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _active_creature(context) is not None

    def execute(self, context):
        obj = _active_creature(context)
        try:
            api.rebuild(obj.name, detail=context.scene.naturalform.detail)
        except NaturalFormError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        return {"FINISHED"}


class NATURALFORM_OT_copy_guide(bpy.types.Operator):
    bl_idname = "naturalform.copy_guide"
    bl_label = "Copy LLM Guide"
    bl_description = "Copy the NaturalForm guide to the clipboard (paste it into your AI assistant)"

    def execute(self, context):
        from .docs import guide
        context.window_manager.clipboard = guide() + "\n" + guide("custom") + "\n" + guide("pose")
        self.report({"INFO"}, "NaturalForm guide copied to the clipboard")
        return {"FINISHED"}


class VIEW3D_PT_naturalform(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NaturalForm"
    bl_label = "NaturalForm"

    def draw(self, context):
        s = context.scene.naturalform
        col = self.layout.column(align=True)
        col.prop(s, "kind")
        col.prop(s, "species", text="Sex" if s.kind == "human" else "Species")
        if s.kind == "human":
            col.prop(s, "age")
            col.prop(s, "clothing")
            col.prop(s, "hair_style")
        col.prop(s, "height", text="Height / Length")
        col.prop(s, "pose")
        col.prop(s, "detail")
        col.prop(s, "rig")
        row = col.row(align=True)
        row.prop(s, "use_color", text="")
        sub = row.row(align=True)
        sub.enabled = s.use_color
        sub.prop(s, "color", text="Clothes colour" if s.kind == "human" else "Colour")
        col.prop(s, "name")
        self.layout.operator(NATURALFORM_OT_create.bl_idname, icon="ADD")
        obj = _active_creature(context)
        if obj is not None:
            box = self.layout.box()
            box.label(text="Active: %s" % obj.name)
            flow = box.grid_flow(columns=3, align=True)
            arm = obj if obj.type == "ARMATURE" else None
            presets = []
            if arm is not None and arm.get("nf_poses"):
                import json
                presets = list(json.loads(arm["nf_poses"]))
            for ptxt in presets[:15]:
                op = flow.operator(NATURALFORM_OT_pose.bl_idname, text=ptxt.replace("_", " ").title())
                op.preset = ptxt
            box.operator(NATURALFORM_OT_rebuild.bl_idname, icon="FILE_REFRESH")
        self.layout.separator()
        self.layout.operator(NATURALFORM_OT_copy_guide.bl_idname, icon="COPYDOWN")
        self.layout.label(text="LLMs: import naturalform as nf; nf.help()")


class VIEW3D_MT_naturalform_add(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_naturalform_add"
    bl_label = "NaturalForm"

    def draw(self, context):
        lay = self.layout
        for label, what in (("Man", "man"), ("Woman", "woman"), ("Child", "child"), ("Dog", "dog"),
                            ("Cat", "cat"), ("Horse", "horse"), ("Cow", "cow"), ("Lion", "lion"),
                            ("Elephant", "elephant"), ("Sparrow", "sparrow"), ("Eagle", "eagle"),
                            ("Fish", "trout"), ("Shark", "shark"), ("Snake", "snake")):
            op = lay.operator(NATURALFORM_OT_quick_add.bl_idname, text=label)
            op.what = what


def _add_menu(self, context):
    self.layout.menu(VIEW3D_MT_naturalform_add.bl_idname, icon="OUTLINER_OB_ARMATURE")


CLASSES = (NF_Settings, NATURALFORM_OT_create, NATURALFORM_OT_quick_add, NATURALFORM_OT_pose,
           NATURALFORM_OT_rebuild, NATURALFORM_OT_copy_guide, VIEW3D_PT_naturalform, VIEW3D_MT_naturalform_add)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.naturalform = bpy.props.PointerProperty(type=NF_Settings)
    bpy.types.VIEW3D_MT_add.append(_add_menu)


def unregister():
    bpy.types.VIEW3D_MT_add.remove(_add_menu)
    del bpy.types.Scene.naturalform
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
