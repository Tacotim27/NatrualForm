"""Public, LLM-friendly API of NaturalForm.

Everything here takes plain Python values (numbers, strings, lists, dicts)
and returns plain dicts, so it is easy to drive from an MCP
``execute_blender_code`` call:

    import naturalform as nf
    nf.help()                                   # prints the guide
    r = nf.create("horse", location=(2, 0, 0))
    print(r)
"""

import difflib
import json

from .core.colors import NAMED as _NAMED_COLORS
from .core.creature import Creature, NaturalFormError
from .templates import bird as _bird
from .templates import fish as _fish
from .templates import human as _human
from .templates import quadruped as _quad
from .templates import serpent as _serp

COLORS = sorted(_NAMED_COLORS)

# build options understood by every create_* function
BUILD_OPTIONS = ("name", "location", "rotation", "scale", "detail", "rig", "pose", "collection", "replace",
                 "smooth", "select")

TEMPLATES = {
    "human": {"spec": _human.human, "params": _human.PARAMS, "presets": ("male", "female", "child", "elder"),
              "poses": _human.POSES},
    "quadruped": {"spec": _quad.quadruped, "params": _quad.PARAMS, "presets": tuple(_quad.PRESETS),
                  "poses": _quad.POSES},
    "bird": {"spec": _bird.bird, "params": _bird.PARAMS, "presets": tuple(_bird.PRESETS), "poses": _bird.POSES},
    "fish": {"spec": _fish.fish, "params": _fish.PARAMS, "presets": tuple(_fish.PRESETS), "poses": _fish.POSES},
    "serpent": {"spec": _serp.serpent, "params": _serp.PARAMS, "presets": tuple(_serp.PRESETS),
                "poses": ("rest",)},
}

_HUMAN_WORDS = {
    "human": {}, "person": {}, "man": {"sex": "male"}, "woman": {"sex": "female"},
    "male": {"sex": "male"}, "female": {"sex": "female"}, "boy": {"sex": "male", "age": 9},
    "girl": {"sex": "female", "age": 9}, "child": {"age": 8}, "kid": {"age": 8}, "baby": {"age": 1, "clothing": "underwear"},
    "toddler": {"age": 2.5}, "teen": {"age": 15}, "teenager": {"age": 15}, "elder": {"age": 75},
    "old man": {"sex": "male", "age": 75}, "old woman": {"sex": "female", "age": 75},
}

_ALIASES = {"puppy": ("quadruped", "dog", {"size": 0.5}), "kitten": ("quadruped", "cat", {"size": 0.6}),
            "lamb": ("quadruped", "sheep", {"size": 0.55}), "foal": ("quadruped", "horse", {"size": 0.6}),
            "calf": ("quadruped", "cow", {"size": 0.55}), "hen": ("bird", "chicken", {}),
            "rooster": ("bird", "chicken", {}), "bunny": ("quadruped", "rabbit", {}),
            "rat": ("quadruped", "mouse", {"size": 2.0, "color": "gray"}), "stag": ("quadruped", "deer", {}),
            "doe": ("quadruped", "deer", {"horns": "none"}), "ox": ("quadruped", "bull", {}),
            "piglet": ("quadruped", "pig", {"size": 0.4}), "pony": ("quadruped", "pony", {}),
            "kitty": ("quadruped", "cat", {}), "doggo": ("quadruped", "dog", {})}


def _split(kwargs):
    build = {k: kwargs.pop(k) for k in list(kwargs) if k in BUILD_OPTIONS}
    return kwargs, build


def _unique_name(base):
    import bpy
    name, i = base, 0
    while bpy.data.objects.get(name) is not None or bpy.data.objects.get(name + "_mesh") is not None:
        i += 1
        name = "%s.%03d" % (base, i)
    return name


def _finish(creature, build, default_name):
    """Auto-named creatures get a unique name (two create("dog") calls give
    two dogs); an explicit ``name=`` replaces an existing creature."""
    if not build.get("name"):
        build["name"] = _unique_name(default_name)
        build.setdefault("replace", False)
    return creature.build(**build)


# ---------------------------------------------------------------------------
# spec constructors (editable Creature objects)
# ---------------------------------------------------------------------------

def human(**params):
    """Editable human spec (see ``help('human')``). Call ``.build()`` on it."""
    return _human.human(**params)


def quadruped(species="dog", **params):
    """Editable four-legged animal spec (see ``help('quadruped')``)."""
    return _quad.quadruped(species, **params)


def bird(species="sparrow", **params):
    """Editable bird spec (see ``help('bird')``)."""
    return _bird.bird(species, **params)


def fish(species="trout", **params):
    """Editable fish / shark / dolphin spec (see ``help('fish')``)."""
    return _fish.fish(species, **params)


def serpent(species="snake", **params):
    """Editable snake / worm / eel spec (see ``help('serpent')``)."""
    return _serp.serpent(species, **params)


def spec(kind, **params):
    """Editable spec for any species or kind name (same routing as ``create``)."""
    tmpl, species, extra = _route(kind)
    extra.update(params)
    if tmpl == "human":
        return _human.human(**extra)
    return TEMPLATES[tmpl]["spec"](species, **extra)


# ---------------------------------------------------------------------------
# one-call creation
# ---------------------------------------------------------------------------

def create_human(**kwargs):
    """Create a human in the scene. Accepts every human parameter plus the
    build options (name, location, rotation, scale, detail, rig, pose,
    collection). Returns a dict describing the created objects."""
    params, build = _split(kwargs)
    return _finish(_human.human(**params), build, "Human")


def create_quadruped(species="dog", **kwargs):
    params, build = _split(kwargs)
    return _finish(_quad.quadruped(species, **params), build, species.replace("_", " ").title().replace(" ", ""))


def create_bird(species="sparrow", **kwargs):
    params, build = _split(kwargs)
    if build.get("pose") == "fly" and "wings" not in params and species not in ("penguin",):
        params["wings"] = "spread"
    return _finish(_bird.bird(species, **params), build, species.title())


def create_fish(species="trout", **kwargs):
    params, build = _split(kwargs)
    return _finish(_fish.fish(species, **params), build, species.title())


def create_serpent(species="snake", **kwargs):
    params, build = _split(kwargs)
    return _finish(_serp.serpent(species, **params), build, species.replace("_", " ").title().replace(" ", ""))


def _route(kind):
    k = str(kind).strip().lower().replace("-", "_")
    if k in TEMPLATES and k != "human":
        default = {"quadruped": "dog", "bird": "sparrow", "fish": "trout", "serpent": "snake"}[k]
        return k, default, {}
    if k.replace("_", " ") in _HUMAN_WORDS:
        return "human", None, dict(_HUMAN_WORDS[k.replace("_", " ")])
    if k in _ALIASES:
        t, sp, extra = _ALIASES[k]
        return t, sp, dict(extra)
    for t in ("quadruped", "bird", "fish", "serpent"):
        if k in TEMPLATES[t]["presets"]:
            return t, k, {}
    if k.endswith("s") and k[:-1] != "" and any(k[:-1] in TEMPLATES[t]["presets"] for t in TEMPLATES):
        return _route(k[:-1])
    names = list(_HUMAN_WORDS) + list(_ALIASES) + [s for t in TEMPLATES for s in TEMPLATES[t]["presets"]]
    close = difflib.get_close_matches(k, names, n=4)
    raise NaturalFormError(
        "no template for %r.%s Use one of: %s. For anything else build a custom creature with "
        "nf.Creature (see nf.help('custom'))." % (kind, (" Did you mean %s?" % ", ".join(close)) if close else "",
                                                 ", ".join(sorted(set(names)))))


def create(kind, **kwargs):
    """Create any supported creature by name: 'human', 'woman', 'child',
    'dog', 'horse', 'eagle', 'shark', 'cobra', ... Extra keyword arguments
    are template parameters and/or build options."""
    tmpl, species, extra = _route(kind)
    extra.update(kwargs)
    params_, build_ = _split(extra)
    default = str(kind).replace("_", " ").title().replace(" ", "") if kind not in TEMPLATES else None
    if tmpl == "human":
        return _finish(_human.human(**params_), build_, default or "Human")
    if tmpl == "bird" and build_.get("pose") == "fly" and "wings" not in params_ and species != "penguin":
        params_["wings"] = "spread"
    spec_fn = {"quadruped": _quad.quadruped, "bird": _bird.bird, "fish": _fish.fish,
               "serpent": _serp.serpent}[tmpl]
    return _finish(spec_fn(species, **params_), build_,
                   default or species.replace("_", " ").title().replace(" ", ""))


def build(creature, **options):
    """Build a :class:`Creature` (custom or edited template). Same options as
    the create_* functions. Also accepts a dict / JSON spec."""
    if isinstance(creature, (dict, str)):
        creature = Creature.from_dict(creature)
    if not isinstance(creature, Creature):
        raise NaturalFormError("build() needs a Creature, a dict or a JSON string")
    return creature.build(**options)


# ---------------------------------------------------------------------------
# working with built creatures
# ---------------------------------------------------------------------------

def _root(name):
    import bpy
    obj = bpy.data.objects.get(name)
    if obj is None:
        close = difflib.get_close_matches(str(name), bpy.data.objects.keys(), n=3)
        raise NaturalFormError("no object named %r.%s" % (name, (" Did you mean %s?" % ", ".join(close)) if close else ""))
    if "nf_spec" not in obj and obj.parent is not None and "nf_spec" in obj.parent:
        obj = obj.parent
    if "nf_spec" not in obj:
        raise NaturalFormError("%r was not made by NaturalForm" % obj.name)
    return obj


def load(name):
    """Return the editable :class:`Creature` a built object was made from.
    Edit it and call ``.build(name=..., replace=True)`` to update the object
    in place (keeps its location / rotation if you pass them back)."""
    return Creature.from_dict(json.loads(_root(name)["nf_spec"]))


def rebuild(name, pose=None, detail=None, **params):
    """Rebuild a creature made by a template with changed parameters, e.g.
    ``nf.rebuild("Human", age=70, hair_color="white")``. Keeps its name and
    placement. Without template parameters it just re-meshes (e.g. with a
    new ``pose`` or ``detail``)."""
    obj = _root(name)
    loc = tuple(obj.location)
    rot = tuple(round(a * 57.29577951308232, 4) for a in obj.rotation_euler)
    scl = tuple(obj.scale)
    old_pose = obj.get("nf_pose") or None
    if old_pose:
        try:
            old_pose = json.loads(old_pose)
        except ValueError:
            pass
    build_opts = dict(name=obj.name, location=loc, rotation=rot, scale=scl, replace=True,
                      pose=pose if pose is not None else old_pose)
    if detail is not None:
        build_opts["detail"] = detail
    c = Creature.from_dict(json.loads(obj["nf_spec"]))
    if params:
        t = c.meta.get("template")
        if t not in TEMPLATES:
            raise NaturalFormError("%r is a custom creature; edit it with nf.load(%r) instead" % (name, name))
        species = c.meta.get("params", {}).get("species")
        merged = dict(c.meta.get("user_params", {}))
        merged.update(params)
        c = TEMPLATES[t]["spec"](**merged) if t == "human" else TEMPLATES[t]["spec"](species, **merged)
    return c.build(**build_opts)


def pose(name, pose, reset=None):
    """Pose a built creature through its armature (no re-meshing).

    ``pose``: a preset name (see ``list_poses``) or {bone: spec} where spec is
    [rx, ry, rz] world-axis degrees, {"aim": [x, y, z]} (point the bone along
    a direction in the creature's own space; it faces -Y, left is +X, up is
    +Z) or {"axis": [x, y, z], "angle": deg}. Presets replace the current
    pose; dicts add to it unless ``reset=True``.
    For a cleaner, fully re-meshed result use ``rebuild(name, pose=...)``."""
    from .blender.rig import pose_rig
    return pose_rig(_root(name), pose, reset=bool(reset))


def reset_pose(name):
    from .blender.rig import reset_pose as _rp
    return _rp(_root(name))


def bones(name):
    """World-space head / tail of every bone of a built creature."""
    from .blender.rig import bone_positions
    return bone_positions(_root(name))


def describe(name):
    """Summary of a built creature: template, parameters, parts and bones."""
    obj = _root(name)
    c = Creature.from_dict(json.loads(obj["nf_spec"]))
    return {"name": obj.name, "kind": obj.get("nf_kind"), "template": c.meta.get("template"),
            "params": c.meta.get("params"), "pose": obj.get("nf_pose") or None,
            "location": [round(v, 4) for v in obj.location], "summary": c.summary(bones=False)}


def focus(target=None, view="three_quarter", margin=1.3):
    """Frame ``target`` (object name, list of names, or None = selection) in
    every 3D viewport, seen from ``view``: front, back, left, right, side,
    top, three_quarter. Useful right before taking a viewport screenshot."""
    import bpy
    import mathutils
    import numpy as np
    dirs = {"front": (0, -1, 0.12), "back": (0, 1, 0.12), "left": (1, 0, 0.1), "side": (1, 0, 0.1),
            "right": (-1, 0, 0.1), "top": (0.0, -0.05, 1), "three_quarter": (0.8, -1, 0.45)}
    if view not in dirs:
        raise NaturalFormError("view must be one of %s" % ", ".join(dirs))
    if target is None:
        objs = list(bpy.context.selected_objects)
    else:
        names = [target] if isinstance(target, str) else list(target)
        objs = []
        for n in names:
            o = bpy.data.objects.get(n)
            if o is None:
                raise NaturalFormError("no object named %r" % n)
            objs.append(o)
            objs.extend(o.children_recursive)
    pts = []
    for o in objs:
        if o.type == "MESH":
            mw = o.matrix_world
            for v in o.bound_box:
                pts.append(mw @ mathutils.Vector(v))
        else:
            pts.append(o.matrix_world.translation.copy())
    if not pts:
        raise NaturalFormError("nothing to frame")
    P = np.array([tuple(p) for p in pts])
    lo, hi = P.min(axis=0), P.max(axis=0)
    center = mathutils.Vector(((lo + hi) / 2).tolist())
    radius = float(np.linalg.norm(hi - lo)) / 2 or 1.0
    d = mathutils.Vector(dirs[view]).normalized()
    n = 0
    wm = bpy.context.window_manager
    for win in (wm.windows if wm else []):
        for area in win.screen.areas:
            if area.type != "VIEW_3D":
                continue
            r3d = area.spaces.active.region_3d
            r3d.view_perspective = "PERSP"
            r3d.view_location = center
            r3d.view_rotation = (-d).to_track_quat("-Z", "Y")
            r3d.view_distance = radius * margin * 2.2
            area.tag_redraw()
            n += 1
    return {"views_updated": n, "center": [round(v, 3) for v in center], "radius": round(radius, 3),
            "note": "" if n else "no 3D viewport available (background mode)"}


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def list_species():
    """Every creature name ``create()`` understands, grouped by template."""
    out = {t: list(v["presets"]) for t, v in TEMPLATES.items()}
    out["human"] = sorted(_HUMAN_WORDS)
    out["aliases"] = sorted(_ALIASES)
    return out


def list_poses(kind="human"):
    """Pose preset names for a template or species."""
    tmpl = kind if kind in TEMPLATES else _route(kind)[0]
    return list(TEMPLATES[tmpl]["poses"])


def params(kind="human"):
    """Parameter table {name: [default, description]} for a template/species."""
    tmpl = kind if kind in TEMPLATES else _route(kind)[0]
    out = {k: [v[0], v[1]] for k, v in TEMPLATES[tmpl]["params"].items()}
    if kind not in TEMPLATES and tmpl != "human":
        species = _route(kind)[1]
        mod = {"quadruped": _quad, "bird": _bird, "fish": _fish, "serpent": _serp}[tmpl]
        for k, v in mod.PRESETS[species].items():
            if k in out:
                out[k][0] = v
    return out


def help(topic=None):
    """Print the guide (``topic``: None, 'human', 'quadruped', 'bird', 'fish',
    'serpent', 'custom', 'pose', 'materials', 'colors', or a species)."""
    from .docs import guide
    print(guide(topic))
