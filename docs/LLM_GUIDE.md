# NaturalForm - guide for LLM agents

Paste this file into your assistant's system prompt / project instructions
(or let the assistant run `import naturalform as nf; nf.help()` inside
Blender - it prints the same text). It assumes the assistant can run Python
in Blender, e.g. through the `execute_blender_code` tool of
[blender-mcp](https://github.com/ahujasid/blender-mcp).


## Overview

```text
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
```

## Custom creatures

```text
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
```

## Posing

```text
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
```

## Materials

```text
MATERIALS  c.material(key, kind=, color=, **options) - procedural, no UVs
  kinds: skin fur wool scales feathers hair keratin lips mouth teeth membrane plain
  options: belly_color (countershading), pattern (stripes spots rosettes patches
  giraffe mottled), pattern_color, pattern_scale (feature size in metres),
  roughness, subsurface, sheen, bump
Template parameters cover the usual cases: color=, belly_color=, pattern=,
pattern_color=, skin_color=, hair_color=, eye_color=, top_color= ...
Patterns use the rest pose position (attribute nf_rest) so they stick to the
skin when the rig moves.
```

## Human parameters

```text
HUMAN
-----
nf.create_human(**params)  or nf.create('woman'|'boy'|'elder', ...)
pose presets: stand, A, T, walk, run, sit, sit_ground, wave, hands_up, point, hands_on_hips, kneel, think, arms_crossed
parameters (name, default, meaning) - presets override defaults, see nf.params('<species>'):
  height           None         metres (default from sex and age)
  sex              'male'       male | female | neutral
  masculinity      None         0 (female) .. 1 (male); overrides sex
  age              30           years; children get big heads and short limbs, elders stoop
  muscle           0.5          0..1
  fat              0.3          0..1
  style            'realistic'  realistic | stylized | chibi
  head_scale       1.0          multiplier
  shoulder_width   1.0          multiplier
  hip_width        1.0          multiplier
  waist            1.0          waist width multiplier
  torso_length     1.0          multiplier
  leg_length       1.0          multiplier
  arm_length       1.0          multiplier
  neck_length      1.0          multiplier
  neck_thickness   1.0          multiplier
  breast_size      None         0..1 (default 0.5 for adult women, 0 otherwise)
  eye_size         1.0          multiplier
  nose_size        1.0          multiplier
  nose_length      1.0          multiplier (how far it projects)
  ear_size         1.0          multiplier
  jaw_width        1.0          multiplier
  chin_size        1.0          multiplier
  lip_fullness     1.0          multiplier
  brow_ridge       1.0          multiplier
  cheekbones       1.0          multiplier
  skin_color       'light'      porcelain pale fair light medium olive tan brown dark deep ebony, hex or rgb
  eye_color        'brown'      brown blue green hazel gray amber, hex or rgb
  hair_style       None         bald buzz short bob long ponytail bun afro mohawk (default by sex)
  hair_color       None         black dark_brown brown light_brown blonde platinum red auburn ginger gray white, hex/rgb
  eyebrows         True         add eyebrows
  beard            'none'       none mustache goatee full
  clothing         'casual'     none underwear casual shorts dress suit
  top_color        None         colour of the shirt / dress / jacket
  bottom_color     None         colour of trousers / shorts / skirt
  shoe_color       None         colour of shoes
  fingers          True         separate fingers (False = mitten hands, faster)
examples:
  nf.create_human(sex='female', age=34, height=1.68, skin_color='tan', hair_style='bob', hair_color='black', clothing='dress', pose='wave')
  nf.create('old man', beard='full', clothing='suit', fat=0.6)
  nf.create('girl', age=6, clothing='shorts', top_color='pink', location=(1,0,0))
```

## Quadruped parameters

```text
QUADRUPED
---------
nf.create_quadruped(species, **params) or nf.create('horse', ...)
species: dog, labrador, dalmatian, dachshund, corgi, greyhound, wolf, fox, cat, lion, lioness, tiger, leopard, cheetah, horse, pony, zebra, donkey, deer, cow, bull, pig, sheep, goat, ram, bear, polar_bear, elephant, giraffe, camel, rhino, rabbit, mouse
pose presets: stand, walk, run, sit, lie, graze, alert, look_left, look_right
parameters (name, default, meaning) - presets override defaults, see nf.params('<species>'):
  height           None         withers (shoulder) height in metres; default = species size * size
  size             1.0          multiplier on the species' typical size
  family           'digitigrade' leg layout: digitigrade (dog/cat), unguligrade (horse/deer), plantigrade (bear), columnar (elephant), leaper (rabbit)
  foot             'paw'        paw | hoof | cloven | pad | plantigrade
  body_length      0.72         shoulder-to-hip distance / withers height
  chest_depth      0.46         chest depth / withers height
  chest_width      0.32         body width / withers height
  belly_tuck       0.35         0 (barrel belly) .. 1 (greyhound waist)
  hip_height       1.0          croup height / withers height
  fat              0.3          0..1 body mass
  leg_length       1.0          multiplier (short legs = dachshund / corgi)
  leg_thickness    1.0          multiplier
  foot_size        1.0          multiplier
  neck_length      0.36         / withers height
  neck_angle       40.0         degrees above horizontal
  neck_thickness   1.0          multiplier
  head_length      0.42         / withers height (skull + snout)
  head_width       0.26         / withers height
  head_angle       12.0         degrees the face points below horizontal
  snout_length     0.5          fraction of the head length
  snout_width      0.55         fraction of the head width
  eye_size         1.0          multiplier
  eye_angle        28.0         0 = eyes face forward (predators) .. 70 = on the sides (prey)
  ear_type         'floppy'     pointed | floppy | round | long | small | elephant | none
  ear_size         1.0          multiplier
  tail_type        'thin'       thin | bushy | short | tuft | horse | fluffy | curly | none
  tail_length      0.7          / withers height
  tail_angle       15.0         degrees above horizontal at the base (negative = hanging)
  tail_curl        10.0         degrees of bend per segment (positive curls up)
  horns            'none'       none | antlers | goat | ram | bull | straight | ossicones | rhino
  horn_size        1.0          multiplier
  mane             'none'       none | horse | lion | mohawk
  trunk            False        elephant trunk
  tusks            False        tusks
  hump             0.0          0..1 back hump (camel / bison)
  material         'fur'        fur | wool | skin | scales
  color            'fawn'       main colour
  belly_color      None         underside colour (countershading)
  pattern          'none'       none | stripes | spots | rosettes | patches | giraffe | mottled
  pattern_color    'black'      pattern colour
  pattern_scale    None         feature size in metres (default by size)
  nose_type        'round'      round | disc (pig) | wide (horse, cow)
  nose_color       'black'      nose / muzzle tip colour
  hoof_color       'hoof'       hooves / claws colour
  horn_color       'horn'       horns / tusks colour
  mane_color       None         mane / tail-hair colour (default: darker main colour)
  eye_color        'eye_brown'  iris colour
  pupil            'round'      round | slit | horizontal
  sclera           False        show eye whites
examples:
  nf.create('dog', color='black', ear_type='pointed', size=1.2, pose='sit')
  nf.create('horse', color='white', mane_color='platinum', pose='run')
  nf.create_quadruped('cat', color='ginger_fur', pattern='stripes', pattern_color='rust', pose='lie')
```

## Bird parameters

```text
BIRD
----
nf.create_bird(species, **params) or nf.create('owl', ...)
species: sparrow, robin, crow, pigeon, eagle, owl, parrot, duck, chicken, penguin, flamingo, seagull, hummingbird, swan
pose presets: stand, perch, peck, look_left, look_right, fly
parameters (name, default, meaning) - presets override defaults, see nf.params('<species>'):
  height           None         standing height in metres (default: species size * size)
  size             1.0          multiplier on the species' typical size
  body_angle       30.0         degrees the body axis rises toward the head (10 duck .. 80 penguin)
  body_width       0.55         / body length
  body_depth       0.6          / body length
  fat              0.3          0..1 plumpness
  neck_length      0.12         / body length
  neck_angle       60.0         degrees above horizontal
  neck_shape       'straight'   straight | s (flamingo / swan)
  head_size        1.0          multiplier
  beak             'cone'       cone | stout | hook | flat | long | bent
  beak_length      0.25         / body length
  beak_width       1.0          multiplier
  eye_size         1.0          multiplier
  eye_angle        70.0         0 = facing forward (owl) .. 80 = on the sides
  facial_disc      False        owl face
  face_color       None         facial disc colour
  ear_tufts        False        owl ear tufts
  crest            False        crest feathers on the head
  comb             False        chicken comb and wattle
  wings            'folded'     folded | spread | raised
  wing_length      1.0          multiplier
  wing_span        2.2          spread wing span / body length
  flippers         False        penguin flippers instead of wings
  tail             'fan'        fan | long | short | upright
  tail_length      0.45         / body length
  leg_length       0.35         visible leg length / body length
  leg_thickness    1.0          multiplier
  feet             'perching'   perching | webbed | none
  color            'tawny'      main plumage colour
  belly_color      None         underside colour
  breast_color     None         breast colour
  head_color       None         head colour
  wing_color       None         wing / tail colour
  pattern          'none'       none | mottled | spots | stripes
  pattern_color    'charcoal'   pattern colour
  beak_color       'charcoal'   beak colour
  leg_color        (190, 150, 120) legs and feet colour
  eye_color        'black'      iris colour
examples:
  nf.create('eagle', pose='fly', location=(0,0,4))
  nf.create('parrot', color='green', wing_color='blue', head_color='yellow')
```

## Fish parameters

```text
FISH
----
nf.create_fish(species, **params) or nf.create('shark', ...)
species: goldfish, clownfish, trout, salmon, tuna, shark, angelfish, koi, piranha, dolphin, orca
pose presets: straight, swim, turn_left, turn_right, dive, leap
parameters (name, default, meaning) - presets override defaults, see nf.params('<species>'):
  length           None         body length in metres (default: species size * size)
  size             1.0          multiplier on the species' typical size
  body_height      0.28         / length
  body_width       0.45         width / height
  head_size        1.0          multiplier
  snout            0.0          0..1 pointed snout / beak (dolphin)
  eye_size         1.0          multiplier
  tail             'fork'       fork | round | lunate | shark | fluke | double
  tail_size        1.0          multiplier
  dorsal           'normal'     normal | long | sail | shark | dolphin | none
  fin_size         1.0          multiplier for all fins
  pectoral_size    1.0          multiplier
  gill_slits       0            number of shark gill slits (0 = bony fish gill cover)
  material         'scales'     scales | skin
  color            (150, 160, 170) main colour
  belly_color      'white'      underside colour
  fin_color        None         fin colour (default: main colour)
  pattern          'none'       none | stripes | spots | patches | mottled
  pattern_color    'black'      pattern colour
  pattern_scale    None         feature size in metres
  eye_color        'gold'       iris colour
examples:
  nf.create('shark', length=3, pose='swim', location=(0,0,-2))
  nf.create('goldfish', location=(0,0,0.2))  # fish are centred on location
```

## Serpent parameters

```text
SERPENT
-------
nf.create_serpent(species, **params) or nf.create('cobra', ...)
species: snake, python, cobra, rattlesnake, viper, green_mamba, worm, eel
pose presets: rest
parameters (name, default, meaning) - presets override defaults, see nf.params('<species>'):
  length           None         total length in metres (default: species size * size)
  size             1.0          multiplier on the species' typical size
  thickness        0.035        max body radius / length
  shape            'wave'       straight | wave | coil | raised
  waves            1.5          number of S-bends for shape=wave
  head             'round'      round | viper (triangular) | none (worm)
  head_size        1.0          multiplier
  hood             False        cobra hood
  rattle           False        rattlesnake rattle
  tongue           False        forked tongue sticking out
  segments         False        segmented look (worms)
  eye_size         1.0          multiplier
  pupil            'round'      round | slit
  material         'scales'     scales | skin
  color            (80, 110, 50) main colour
  belly_color      'cream'      underside colour
  pattern          'none'       none | stripes | spots | patches | rosettes | mottled
  pattern_color    'black'      pattern colour
  pattern_scale    None         feature size in metres
  eye_color        'eye_amber'  iris colour
examples:
  nf.create('python', shape='coil')
  nf.create('snake', shape='wave', color='green', length=2)
```
