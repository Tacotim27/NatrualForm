"""The guide printed by ``naturalform.help()`` - written for LLM agents."""

OVERVIEW = """\
NATURALFORM - organic creatures for Blender (humans, animals, custom beasts)
=============================================================================
Use it from Python (e.g. an MCP `execute_blender_code` call):

    import naturalform as nf
    r = nf.create("woman", age=30, hair_style="ponytail", clothing="casual",
                  location=(0, 0, 0), rotation=(0, 0, 45), pose="walk")
    print(r)          # dict: object names, size, bones, pose presets ...

ONE-CALL CREATION  nf.create(name, **params, **build_options)
  humans    : human person man woman boy girl child baby toddler teen elder ...
  quadruped : dog labrador dalmatian dachshund corgi greyhound wolf fox cat lion
              lioness tiger leopard cheetah horse pony zebra donkey deer cow bull
              pig sheep goat ram bear polar_bear elephant giraffe camel rhino
              rabbit mouse  (+ aliases puppy kitten foal calf lamb bunny rat ...)
  bird      : sparrow robin crow pigeon eagle owl parrot duck chicken penguin
              flamingo seagull hummingbird swan
  fish      : goldfish clownfish trout salmon tuna shark angelfish koi piranha
              dolphin orca
  serpent   : snake python cobra rattlesnake viper green_mamba worm eel
  Or call nf.create_human / create_quadruped(species) / create_bird(species) /
  create_fish(species) / create_serpent(species) directly.

BUILD OPTIONS (every create_* / .build())
  name="Rex"            root object name. Given explicitly, it REPLACES an existing
                        creature of that name; omitted, a unique name is chosen
                        (Dog, Dog.001, ...) so repeated calls add more creatures
  location=(x,y,z)      metres; creatures stand ON this point (fish: centred)
  rotation=(rx,ry,rz)   degrees. Creatures face -Y; rotation=(0,0,180) faces +Y
  scale=1.0
  detail="medium"       preview | low | medium | high | ultra (or an int).
                        preview/low ~1 s, medium 2-6 s, high 5-15 s.
  pose="walk"           preset or pose dict, applied BEFORE meshing (clean joints)
  rig=True              armature + weights, so it can be posed later
  collection="Animals"  put the objects in this collection
Result: {"name": root object (move this one), "mesh": mesh object, "armature",
         "dimensions_m", "pose_presets", "vertices", "time_s", ...}

CONVENTIONS  metres, +Z up, ground at z=0, creatures face -Y (the Blender
front view looks at their face), their LEFT side is +X, names ending .L/.R are
left/right. Real-world sizes by default (human 1.75 m, horse 1.6 m at the
withers, cat 0.25 m); use height= / length= / size= to change.

AFTER CREATION
  nf.pose("Horse", "run")                       re-pose via the rig (fast)
  nf.pose("Human", {"head": [0, 0, 30]})        bone: world XYZ degrees
  nf.pose("Human", {"upper_arm.R": {"aim": [0, -1, 0.3]}})   point a bone
  nf.rebuild("Human", age=70, beard="full")     change params, keep placement
  nf.rebuild("Dog", pose="sit")                 re-mesh in a new pose
  nf.bones("Human") / nf.describe("Human")      inspect joints / parameters
  nf.focus("Horse", view="side")                frame it in the viewport (then
                                                take a viewport screenshot)
DISCOVERY  nf.help("human"|"quadruped"|"bird"|"fish"|"serpent"|"custom"|
           "pose"|"materials"|"colors"|<species>), nf.list_species(),
           nf.list_poses("dog"), nf.params("horse")  (defaults for a species)
TIPS
  * Prefer templates + parameters; customise with nf.spec(...) + edits (see
    help('custom')) before building a creature from scratch.
  * Re-running a create call with the same explicit name= updates that
    creature instead of adding a duplicate.
  * Colours: names (help('colors')), "#rrggbb", (r,g,b) 0-1 or 0-255.
  * If a call times out in your MCP client, Blender still finishes the job:
    check the scene, or use detail="low" for drafts.
"""

CUSTOM = """\
CUSTOM CREATURES  (nf.Creature) - for dragons, centaurs, aliens, mushrooms ...
-----------------------------------------------------------------------------
Shapes are smooth-blended SDF primitives; chains create bones, so the result
is rigged automatically. Units: metres, face -Y, left = +X, ground z=0.

    c = nf.Creature("Dragon")
    # spine / limbs / tails / necks: tapered capsules through points (+ bones)
    c.chain("body", [(0,-0.6,1.0), (0,-0.15,1.05), (0,0.3,1.02), (0,0.75,0.95)],
            radii=[0.3,0.38,0.34,0.24], bones=["chest","spine","hips"],   # 4 points = 3 bones
            squash=0.85, squash_dir=(1,0,0))
    c.chain("neck", [(0,-0.7,1.1), (0,-1.1,1.6), (0,-1.3,1.9)], [0.2,0.14,0.12],
            parent="chest", blend=0.12)
    c.ellipsoid("head", (0,-1.45,1.95), (0.16,0.28,0.15), bone="neck_1")
    c.chain("leg.L", [(0.3,-0.4,0.9), (0.38,-0.5,0.45), (0.36,-0.4,0.05)],
            [0.14,0.09,0.07], parent="chest", mirror=True)   # makes leg.R too
    c.chain("hind_leg.L", [(0.3,0.5,0.95), (0.42,0.7,0.5), (0.36,0.55,0.05)],
            [0.17,0.1,0.07], parent="hips", mirror=True)
    c.chain("tail", [(0,0.8,0.95), (0,1.5,0.7), (0.3,2.2,0.45)], [0.2,0.1,0.03], parent="hips")
    c.chain("wing.L", [(0.3,-0.3,1.3),(1.2,-0.1,1.7),(2.4,0.2,1.6)], [0.12,0.3,0.05],
            parent="chest", squash=0.12, squash_dir=(0,0,1), material="wing", mirror=True)
    c.eye("eye.L", (0.1,-1.62,2.02), 0.035, look=(0.5,-1,0.1), color="eye_yellow",
          pupil="slit", sclera=None, mirror=True)       # add eyes AFTER the head
    c.material("body", kind="scales", color="olive_green", belly_color="khaki")
    c.material("wing", kind="membrane", color=(120,40,40))
    print(c.build(location=(0,0,0), pose={"neck_0": [-20,0,0]}))

Shape methods (all accept bone=, blend=, material=, op=, mirror=, clip=, layer=)
  chain(name, points, radii, parent=, bones=, blend=, joint_blend=, squash=,
        squash_dir=, visible=True, make_bones=True)
  ellipsoid(name, center, radii, rotate=[rx,ry,rz] | align=direction)
  sphere(name, center, radius)     capsule(name, start, end, radius, end_radius=)
  box(name, center, half_size, rounding=)     torus(name, center, R, r, align=)
  eye(name, center, radius, look=, color=, pupil=round|slit|horizontal|none,
      iris_size=, sclera="white"|None, socket=True)
  bone(name, head, tail, parent=)            (bones without geometry)
  blend   smooth-union radius with everything added before (default 30% of the
          part size). Keep it small for small details - large blends inflate.
  op      "union" (add) | "subtract" (carve, e.g. nostrils, sockets) |
          "intersect"
  clip    [(point, normal), ...] cut the shape with planes; the side the normal
          points to is kept (hairlines, hems, flat bottoms)
  layer   separate surface meshed on top of the body (hair, clothes, shells,
          horns). Parts only blend/carve within their own layer.
  mirror  also create the .R copy (x -> -x) of a .L part
Editing: remove(name), get(name), scale_part(name, f), inflate_part(name, m),
  transform(translate=, rotate=, scale=, names=[...]), clothe(name, sources,
  thickness, material, clip=), merge(other, prefix=, parent=), copy(),
  surface_point(origin, direction) (ray-cast onto the skin, e.g. to place a
  horn), set_material(names, key), summary(), to_json(), Creature.from_dict().
Start from a template:  c = nf.spec("horse"); c.remove("tail"); ... c.build()
Pegasus:  c = nf.spec("horse", color="white", mane_color="platinum")
          c.chain("wing.L", [(0.2,-0.45,1.55), (0.9,-0.35,2.0), (1.8,-0.1,2.3)], [0.08,0.22,0.04],
                  parent="chest", squash=0.15, squash_dir=(0,0,1), mirror=True)
          c.build(name="Pegasus")
"""

POSE = """\
POSING
------
Two ways, same syntax:
  * pose=... when building  -> the body is re-meshed in that pose (best look)
  * nf.pose(name, ...) after -> rotates the armature (fast, keeps the mesh)
Pose = preset name or {bone_name: spec}:
  [rx, ry, rz]                 rotate around WORLD axes at the bone's head (deg)
  {"aim": [x, y, z]}           point the bone in a direction (creature space:
                               -Y forward, +X its left, +Z up)
  {"aim": [...], "twist": d}   then twist around its own axis
  {"axis": [x,y,z], "angle": d}
Children follow parents; bones are processed parent-first.
Sign help (creature faces -Y, its left is +X):
  +X degrees swing a downward limb BACKWARDS, -X forwards; on a neck/head +X
  tips the face down, -X lifts it.
  +Z degrees turn the head to the creature's LEFT, -Z to its right.
  +Y degrees tilt the top of a bone toward the creature's LEFT.
Prefer "aim" for limbs - it is unambiguous.
Human bones: hips spine chest neck head shoulder.L upper_arm.L forearm.L hand.L
  thumb_0..2.L index_0..2.L middle_ ring_ pinky_ thigh.L shin.L foot.L toe.L
  (+ .R, eye.L/eye.R, ponytail_*). Presets: stand A T walk run sit sit_ground
  wave hands_up point hands_on_hips kneel think arms_crossed
Quadruped bones: hips spine chest neck_0 neck_1 head upper_arm.L forearm.L
  front_foot.L thigh.L shin.L hind_foot.L tail_0.. (trunk_0.. for elephants).
  Presets: stand walk run sit lie graze alert look_left look_right
Bird bones: hips chest neck head wing.L (folded) | wing_0..2.L (spread) tail
  thigh.L leg.L foot.L. Presets: stand perch peck look_left look_right fly
  (create_bird(..., pose="fly") builds spread wings automatically).
Fish bones: head spine_0..2 tail. Presets: straight swim turn_left turn_right
  dive leap.  Serpents: spine_0..15 head (shape= straight|wave|coil|raised).
Use nf.bones(name) to read joint positions of a built creature.
"""

MATERIALS = """\
MATERIALS  c.material(key, kind=, color=, **options) - procedural, no UVs
  kinds: skin fur wool scales feathers hair keratin lips mouth teeth membrane plain
  options: belly_color (countershading), pattern (stripes spots rosettes patches
  giraffe mottled), pattern_color, pattern_scale (feature size in metres),
  roughness, subsurface, sheen, bump
Template parameters cover the usual cases: color=, belly_color=, pattern=,
pattern_color=, skin_color=, hair_color=, eye_color=, top_color= ...
Patterns use the rest pose position (attribute nf_rest) so they stick to the
skin when the rig moves.
"""


def _template_doc(name):
    from . import api
    t = api.TEMPLATES[name]
    from .templates.common import doc_table
    head = {
        "human": "nf.create_human(**params)  or nf.create('woman'|'boy'|'elder', ...)",
        "quadruped": "nf.create_quadruped(species, **params) or nf.create('horse', ...)",
        "bird": "nf.create_bird(species, **params) or nf.create('owl', ...)",
        "fish": "nf.create_fish(species, **params) or nf.create('shark', ...)",
        "serpent": "nf.create_serpent(species, **params) or nf.create('cobra', ...)",
    }[name]
    lines = [name.upper(), "-" * len(name), head]
    if name != "human":
        lines.append("species: " + ", ".join(t["presets"]))
    lines.append("pose presets: " + ", ".join(t["poses"]))
    lines.append("parameters (name, default, meaning) - presets override defaults,"
                 " see nf.params('<species>'):")
    lines.append(doc_table(t["params"]))
    if name == "human":
        lines.append("examples:\n  nf.create_human(sex='female', age=34, height=1.68, skin_color='tan',"
                     " hair_style='bob', hair_color='black', clothing='dress', pose='wave')\n"
                     "  nf.create('old man', beard='full', clothing='suit', fat=0.6)\n"
                     "  nf.create('girl', age=6, clothing='shorts', top_color='pink', location=(1,0,0))")
    elif name == "quadruped":
        lines.append("examples:\n  nf.create('dog', color='black', ear_type='pointed', size=1.2, pose='sit')\n"
                     "  nf.create('horse', color='white', mane_color='platinum', pose='run')\n"
                     "  nf.create_quadruped('cat', color='ginger_fur', pattern='stripes',"
                     " pattern_color='rust', pose='lie')")
    elif name == "bird":
        lines.append("examples:\n  nf.create('eagle', pose='fly', location=(0,0,4))\n"
                     "  nf.create('parrot', color='green', wing_color='blue', head_color='yellow')")
    elif name == "fish":
        lines.append("examples:\n  nf.create('shark', length=3, pose='swim', location=(0,0,-2))\n"
                     "  nf.create('goldfish', location=(0,0,0.2))  # fish are centred on location")
    elif name == "serpent":
        lines.append("examples:\n  nf.create('python', shape='coil')\n"
                     "  nf.create('snake', shape='wave', color='green', length=2)")
    return "\n".join(lines)


def guide(topic=None):
    if topic is None:
        return OVERVIEW
    t = str(topic).lower().strip()
    if t in ("custom", "creature", "builder"):
        return CUSTOM
    if t in ("pose", "poses", "posing", "rig"):
        return POSE
    if t in ("material", "materials"):
        return MATERIALS
    if t in ("color", "colors", "colour", "colours"):
        from .core.colors import NAMED
        return "Named colours (or use '#rrggbb' / (r,g,b)):\n" + ", ".join(sorted(NAMED))
    from . import api
    if t in api.TEMPLATES:
        return _template_doc(t)
    try:
        tmpl, species, extra = api._route(t)
    except Exception as e:  # noqa: BLE001 - we want the message
        return str(e)
    out = _template_doc(tmpl)
    if tmpl != "human":
        p = api.params(t)
        out += "\n\n%s defaults: " % t + ", ".join("%s=%r" % (k, v[0]) for k, v in p.items()
                                                   if v[0] not in (None, False, "none"))
    return out
