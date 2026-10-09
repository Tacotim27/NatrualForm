"""Parametric four-legged animals with species presets.

Built in units of the withers (shoulder) height, then scaled. Faces -Y,
left = +X, feet on the ground. Every preset is just a dict of parameters,
so any of them can be overridden, e.g. ``quadruped("dog", ear_type="pointed",
tail_type="bushy", color="black")``.
"""

import math

import numpy as np

from ..core.creature import Creature, NaturalFormError
from .common import choice, clamp, lerp, resolve, sym, unit

FAMILIES = ("digitigrade", "unguligrade", "plantigrade", "columnar", "leaper")
EARS = ("pointed", "floppy", "round", "long", "small", "elephant", "none")
TAILS = ("thin", "bushy", "short", "tuft", "horse", "fluffy", "curly", "none")
HORNS = ("none", "antlers", "goat", "ram", "bull", "straight", "ossicones", "rhino")
MANES = ("none", "horse", "lion", "mohawk")
FEET = ("paw", "hoof", "cloven", "pad", "plantigrade")

PARAMS = {
    "height": (None, "withers (shoulder) height in metres; default = species size * size"),
    "size": (1.0, "multiplier on the species' typical size"),
    "family": ("digitigrade", "leg layout: digitigrade (dog/cat), unguligrade (horse/deer), plantigrade (bear), columnar (elephant), leaper (rabbit)"),
    "foot": ("paw", "paw | hoof | cloven | pad | plantigrade"),
    "body_length": (0.72, "shoulder-to-hip distance / withers height"),
    "chest_depth": (0.46, "chest depth / withers height"),
    "chest_width": (0.32, "body width / withers height"),
    "belly_tuck": (0.35, "0 (barrel belly) .. 1 (greyhound waist)"),
    "hip_height": (1.0, "croup height / withers height"),
    "fat": (0.3, "0..1 body mass"),
    "leg_length": (1.0, "multiplier (short legs = dachshund / corgi)"),
    "leg_thickness": (1.0, "multiplier"),
    "foot_size": (1.0, "multiplier"),
    "neck_length": (0.36, "/ withers height"),
    "neck_angle": (40.0, "degrees above horizontal"),
    "neck_thickness": (1.0, "multiplier"),
    "head_length": (0.42, "/ withers height (skull + snout)"),
    "head_width": (0.26, "/ withers height"),
    "head_angle": (12.0, "degrees the face points below horizontal"),
    "snout_length": (0.5, "fraction of the head length"),
    "snout_width": (0.55, "fraction of the head width"),
    "eye_size": (1.0, "multiplier"),
    "eye_angle": (28.0, "0 = eyes face forward (predators) .. 70 = on the sides (prey)"),
    "ear_type": ("floppy", "pointed | floppy | round | long | small | elephant | none"),
    "ear_size": (1.0, "multiplier"),
    "tail_type": ("thin", "thin | bushy | short | tuft | horse | fluffy | curly | none"),
    "tail_length": (0.7, "/ withers height"),
    "tail_angle": (15.0, "degrees above horizontal at the base (negative = hanging)"),
    "tail_curl": (10.0, "degrees of bend per segment (positive curls up)"),
    "horns": ("none", "none | antlers | goat | ram | bull | straight | ossicones | rhino"),
    "horn_size": (1.0, "multiplier"),
    "mane": ("none", "none | horse | lion | mohawk"),
    "trunk": (False, "elephant trunk"),
    "tusks": (False, "tusks"),
    "hump": (0.0, "0..1 back hump (camel / bison)"),
    "material": ("fur", "fur | wool | skin | scales"),
    "color": ("fawn", "main colour"),
    "belly_color": (None, "underside colour (countershading)"),
    "pattern": ("none", "none | stripes | spots | rosettes | patches | giraffe | mottled"),
    "pattern_color": ("black", "pattern colour"),
    "pattern_scale": (None, "feature size in metres (default by size)"),
    "nose_type": ("round", "round | disc (pig) | wide (horse, cow)"),
    "nose_color": ("black", "nose / muzzle tip colour"),
    "hoof_color": ("hoof", "hooves / claws colour"),
    "horn_color": ("horn", "horns / tusks colour"),
    "mane_color": (None, "mane / tail-hair colour (default: darker main colour)"),
    "eye_color": ("eye_brown", "iris colour"),
    "pupil": ("round", "round | slit | horizontal"),
    "sclera": (False, "show eye whites"),
}

_DOG = dict(family="digitigrade", foot="paw", ear_type="floppy", ear_size=1.2, tail_type="thin", color="tawny",
            belly_color="cream", eye_angle=28.0, head_length=0.38, head_width=0.3, snout_length=0.44,
            leg_thickness=0.9, _size=0.57)
PRESETS = {
    "dog": dict(_DOG),
    "labrador": dict(_DOG, color="beige", _size=0.57),
    "dalmatian": dict(_DOG, color="white", pattern="spots", pattern_color="black", pattern_scale=0.045,
                      belly_color=None, _size=0.58, belly_tuck=0.45),
    "dachshund": dict(_DOG, leg_length=0.45, body_length=1.25, _size=0.42, color="chestnut", ear_size=1.2),
    "corgi": dict(_DOG, leg_length=0.55, body_length=1.0, ear_type="pointed", ear_size=1.25, tail_type="short",
                  color="ginger_fur", belly_color="white", _size=0.38),
    "greyhound": dict(_DOG, belly_tuck=0.9, chest_depth=0.5, fat=0.1, head_length=0.45, snout_length=0.58,
                      snout_width=0.42, ear_type="small", tail_length=0.85, tail_angle=-30, color="gray", _size=0.72),
    "wolf": dict(_DOG, ear_type="pointed", tail_type="bushy", tail_angle=-25, tail_curl=-6, color="wolf_gray",
                 belly_color="cream", pattern="mottled", pattern_color="charcoal", pattern_scale=0.25,
                 eye_color="eye_amber", chest_width=0.28, _size=0.8),
    "fox": dict(_DOG, ear_type="pointed", ear_size=1.3, tail_type="bushy", tail_length=0.95, tail_angle=-5,
                tail_curl=4, color="fox_red", belly_color="white", snout_width=0.42, snout_length=0.55,
                eye_color="eye_amber", pupil="slit", chest_width=0.28, leg_thickness=0.85, _size=0.38),
    "cat": dict(family="digitigrade", foot="paw", body_length=0.85, chest_depth=0.5, chest_width=0.42,
                belly_tuck=0.2, fat=0.4, neck_length=0.26, neck_angle=35, head_length=0.4, head_width=0.42,
                snout_length=0.24, snout_width=0.62, head_angle=5, ear_type="pointed", ear_size=0.85,
                tail_type="thin", tail_length=1.0, tail_angle=40, tail_curl=10, leg_thickness=0.9,
                leg_length=0.88, eye_size=1.45, eye_angle=18, pupil="slit", eye_color="eye_green", color="gray",
                belly_color="white", pattern="stripes", pattern_color="charcoal", pattern_scale=0.05, _size=0.25),
    "lion": dict(family="digitigrade", foot="paw", body_length=0.95, chest_depth=0.5, chest_width=0.32,
                 belly_tuck=0.2, neck_length=0.3, neck_angle=25, head_length=0.4, head_width=0.32,
                 snout_length=0.42, snout_width=0.65, head_angle=10, ear_type="round", ear_size=0.7,
                 tail_type="tuft", tail_length=0.85, tail_angle=-40, tail_curl=6, eye_angle=18,
                 eye_color="eye_amber", color="lion_gold", belly_color="cream", mane="lion",
                 mane_color="chestnut", nose_color=(150, 90, 80), _size=1.05),
    "lioness": dict(family="digitigrade", foot="paw", body_length=0.95, chest_depth=0.48, chest_width=0.3,
                    neck_length=0.3, neck_angle=25, head_length=0.38, head_width=0.3, snout_length=0.42,
                    snout_width=0.62, head_angle=10, ear_type="round", ear_size=0.75, tail_type="tuft",
                    tail_length=0.85, tail_angle=-40, eye_angle=18, eye_color="eye_amber", color="lion_gold",
                    belly_color="cream", nose_color=(150, 90, 80), _size=0.95),
    "tiger": dict(family="digitigrade", foot="paw", body_length=1.0, chest_depth=0.5, chest_width=0.32,
                  belly_tuck=0.2, neck_length=0.3, neck_angle=25, head_length=0.4, head_width=0.34,
                  snout_length=0.4, snout_width=0.65, head_angle=10, ear_type="round", ear_size=0.65,
                  tail_type="thin", tail_length=0.9, tail_angle=-35, tail_curl=8, eye_angle=18,
                  eye_color="eye_amber", color="tiger_orange", belly_color="white", pattern="stripes",
                  pattern_color="black", pattern_scale=0.16, nose_color=(170, 100, 90), _size=1.0),
    "leopard": dict(family="digitigrade", foot="paw", body_length=1.0, chest_depth=0.46, chest_width=0.28,
                    neck_length=0.3, neck_angle=25, head_length=0.38, head_width=0.32, snout_length=0.38,
                    snout_width=0.62, ear_type="round", ear_size=0.75, tail_type="thin", tail_length=1.1,
                    tail_angle=-20, tail_curl=12, eye_angle=18, eye_color="eye_green", color="lion_gold",
                    belly_color="cream", pattern="rosettes", pattern_color="charcoal", pattern_scale=0.07,
                    _size=0.65),
    "cheetah": dict(family="digitigrade", foot="paw", body_length=0.85, chest_depth=0.48, chest_width=0.24,
                    belly_tuck=0.85, fat=0.1, neck_length=0.32, neck_angle=30, head_length=0.3, head_width=0.26,
                    snout_length=0.35, snout_width=0.6, ear_type="round", ear_size=0.7, tail_type="thin",
                    tail_length=0.95, tail_angle=-30, tail_curl=10, leg_thickness=0.75, eye_angle=18,
                    eye_color="eye_amber", color="sandy", belly_color="white", pattern="spots",
                    pattern_color="black", pattern_scale=0.04, _size=0.8),
    "horse": dict(family="unguligrade", foot="hoof", body_length=0.64, chest_depth=0.44, chest_width=0.3,
                  belly_tuck=0.1, hip_height=1.0, neck_length=0.5, neck_angle=52, neck_thickness=1.05,
                  head_length=0.42, head_width=0.2, head_angle=48, snout_length=0.55, snout_width=0.62,
                  ear_type="small", ear_size=1.0, tail_type="horse", tail_length=0.62, tail_angle=-55,
                  eye_angle=55, mane="horse", color="bay", belly_color=None, mane_color="black_hair", nose_type="wide",
                  leg_thickness=0.9, nose_color=(60, 50, 48), _size=1.6),
    "pony": dict(family="unguligrade", foot="hoof", body_length=0.68, chest_depth=0.48, chest_width=0.34,
                 belly_tuck=0.0, fat=0.5, neck_length=0.46, neck_angle=48, neck_thickness=1.1, head_length=0.46,
                 head_width=0.25, head_angle=48, snout_length=0.55, snout_width=0.65, ear_type="small",
                 tail_type="horse", tail_length=0.65, tail_angle=-55, eye_angle=55, mane="horse", nose_type="wide",
                 color="cream", mane_color="platinum", leg_length=0.85, nose_color=(60, 50, 48), _size=1.15),
    "zebra": dict(family="unguligrade", foot="hoof", body_length=0.64, chest_depth=0.46, chest_width=0.32,
                  belly_tuck=0.05, neck_length=0.48, neck_angle=45, neck_thickness=1.05, head_length=0.46,
                  head_width=0.22, head_angle=48, snout_length=0.56, snout_width=0.62, ear_type="small",
                  ear_size=1.3, tail_type="tuft", tail_length=0.55, tail_angle=-60, eye_angle=55, mane="mohawk",
                  color="white", pattern="stripes", pattern_color="black", pattern_scale=0.11,
                  mane_color="black_hair", nose_color="black", nose_type="wide", _size=1.35),
    "donkey": dict(family="unguligrade", foot="hoof", body_length=0.66, chest_depth=0.48, chest_width=0.32,
                   belly_tuck=0.0, neck_length=0.42, neck_angle=40, head_length=0.48, head_width=0.26,
                   head_angle=45, snout_length=0.55, snout_width=0.65, ear_type="long", ear_size=1.1,
                   tail_type="tuft", tail_length=0.6, tail_angle=-60, eye_angle=55, mane="mohawk",
                   color="ash", belly_color="cream", mane_color="charcoal", nose_color="cream", nose_type="wide",
                   _size=1.15),
    "deer": dict(family="unguligrade", foot="cloven", body_length=0.62, chest_depth=0.42, chest_width=0.26,
                 belly_tuck=0.3, neck_length=0.5, neck_angle=55, neck_thickness=0.8, head_length=0.42,
                 head_width=0.2, head_angle=30, snout_length=0.5, snout_width=0.55, ear_type="pointed",
                 ear_size=1.2, tail_type="short", tail_length=0.15, tail_angle=30, eye_size=1.3, eye_angle=60,
                 horns="antlers", leg_thickness=0.7, color="tawny", belly_color="white", nose_color="black",
                 _size=1.0),
    "cow": dict(family="unguligrade", foot="cloven", body_length=0.85, chest_depth=0.55, chest_width=0.42,
                belly_tuck=0.0, fat=0.6, hip_height=1.02, neck_length=0.32, neck_angle=20, neck_thickness=1.25,
                head_length=0.42, head_width=0.3, head_angle=35, snout_length=0.48, snout_width=0.78,
                ear_type="small", ear_size=1.1, tail_type="tuft", tail_length=0.75, tail_angle=-75,
                eye_angle=60, horns="bull", horn_size=0.5, color="white", pattern="patches",
                pattern_color="black", pattern_scale=0.4, nose_color="pink_skin", nose_type="wide", leg_length=0.9,
                _size=1.4),
    "bull": dict(family="unguligrade", foot="cloven", body_length=0.85, chest_depth=0.6, chest_width=0.48,
                 belly_tuck=0.05, fat=0.5, neck_length=0.3, neck_angle=15, neck_thickness=1.5, head_length=0.42,
                 head_width=0.34, head_angle=35, snout_length=0.46, snout_width=0.78, ear_type="small",
                 tail_type="tuft", tail_length=0.75, tail_angle=-75, eye_angle=60, horns="bull", hump=0.4,
                 color="chocolate", nose_color="charcoal", nose_type="wide", leg_length=0.9, _size=1.5),
    "pig": dict(family="unguligrade", foot="cloven", body_length=0.95, chest_depth=0.6, chest_width=0.5,
                belly_tuck=0.0, fat=0.7, neck_length=0.18, neck_angle=10, neck_thickness=1.4, head_length=0.46,
                head_width=0.38, head_angle=15, snout_length=0.4, snout_width=0.55, ear_type="floppy", nose_type="disc",
                ear_size=1.1, tail_type="curly", tail_length=0.3, tail_angle=40, eye_size=0.8, eye_angle=50,
                material="skin", color="pink_skin", nose_color=(220, 140, 140), leg_length=0.7,
                leg_thickness=1.1, _size=0.75),
    "sheep": dict(family="unguligrade", foot="cloven", body_length=0.8, chest_depth=0.6, chest_width=0.5,
                  belly_tuck=0.0, fat=1.0, neck_length=0.3, neck_angle=35, neck_thickness=1.3, head_length=0.38,
                  head_width=0.24, head_angle=40, snout_length=0.5, snout_width=0.6, ear_type="small",
                  ear_size=1.2, tail_type="short", tail_length=0.15, eye_angle=60, pupil="horizontal",
                  eye_color="eye_amber", material="wool", color="cream", nose_color="charcoal",
                  leg_thickness=0.75, _size=0.75),
    "goat": dict(family="unguligrade", foot="cloven", body_length=0.75, chest_depth=0.5, chest_width=0.32,
                 belly_tuck=0.1, neck_length=0.38, neck_angle=50, head_length=0.4, head_width=0.22,
                 head_angle=35, snout_length=0.5, snout_width=0.6, ear_type="small", ear_size=1.3,
                 tail_type="short", tail_length=0.15, tail_angle=50, eye_angle=60, pupil="horizontal",
                 eye_color="eye_yellow", horns="goat", color="white", nose_color="pink_skin",
                 leg_thickness=0.75, _size=0.7),
    "ram": dict(family="unguligrade", foot="cloven", body_length=0.8, chest_depth=0.6, chest_width=0.48,
                belly_tuck=0.0, fat=0.9, neck_length=0.3, neck_angle=35, neck_thickness=1.3, head_length=0.4,
                head_width=0.26, head_angle=40, snout_length=0.5, snout_width=0.6, ear_type="small",
                tail_type="short", tail_length=0.15, eye_angle=60, pupil="horizontal", eye_color="eye_amber",
                horns="ram", color="cream", nose_color="charcoal", leg_thickness=0.8, _size=0.8),
    "bear": dict(family="plantigrade", foot="plantigrade", body_length=0.85, chest_depth=0.55,
                 chest_width=0.45, belly_tuck=0.0, fat=0.8, hip_height=1.0, neck_length=0.3, neck_angle=20,
                 neck_thickness=1.4, head_length=0.42, head_width=0.36, head_angle=12, snout_length=0.45,
                 snout_width=0.55, ear_type="round", ear_size=0.9, tail_type="short", tail_length=0.1,
                 eye_size=0.8, eye_angle=25, leg_thickness=1.4, color="brown", nose_color="black",
                 hoof_color="charcoal", _size=1.0),
    "polar_bear": dict(family="plantigrade", foot="plantigrade", body_length=0.95, chest_depth=0.55,
                       chest_width=0.42, fat=0.8, neck_length=0.42, neck_angle=20, neck_thickness=1.2,
                       head_length=0.45, head_width=0.3, head_angle=12, snout_length=0.48, snout_width=0.55,
                       ear_type="round", ear_size=0.7, tail_type="short", tail_length=0.08, eye_size=0.8,
                       leg_thickness=1.35, color="ivory", nose_color="black", hoof_color="charcoal", _size=1.2),
    "elephant": dict(family="columnar", foot="pad", body_length=0.62, chest_depth=0.5, chest_width=0.36,
                     belly_tuck=0.0, fat=0.4, hip_height=0.95, neck_length=0.16, neck_angle=10,
                     neck_thickness=1.6, head_length=0.4, head_width=0.36, head_angle=25, snout_length=0.2,
                     snout_width=0.7, ear_type="elephant", ear_size=1.0, tail_type="tuft", tail_length=0.45,
                     tail_angle=-70, eye_size=0.55, eye_angle=60, trunk=True, tusks=True, leg_thickness=0.8,
                     material="skin", color="elephant_gray", nose_color="elephant_gray",
                     hoof_color="ivory", horn_color="ivory", _size=3.0),
    "giraffe": dict(family="unguligrade", foot="cloven", body_length=0.42, chest_depth=0.36, chest_width=0.22,
                    belly_tuck=0.25, hip_height=0.82, neck_length=0.95, neck_angle=62, neck_thickness=0.75,
                    head_length=0.27, head_width=0.13, head_angle=45, snout_length=0.55, snout_width=0.7,
                    ear_type="small", ear_size=0.9, tail_type="tuft", tail_length=0.4, tail_angle=-70,
                    eye_size=1.2, eye_angle=60, horns="ossicones", mane="mohawk", leg_length=1.0,
                    leg_thickness=0.62, color="cream", pattern="giraffe", pattern_color="chestnut",
                    pattern_scale=0.22, mane_color="chestnut", nose_color="chocolate", _size=3.0),
    "camel": dict(family="unguligrade", foot="pad", body_length=0.62, chest_depth=0.42, chest_width=0.3,
                  belly_tuck=0.2, neck_length=0.6, neck_angle=38, neck_thickness=0.85, head_length=0.35,
                  head_width=0.18, head_angle=10, snout_length=0.55, snout_width=0.7, ear_type="small",
                  ear_size=0.7, tail_type="tuft", tail_length=0.35, tail_angle=-65, eye_angle=55, hump=1.0,
                  leg_thickness=0.75, color="sandy", nose_color="tan", _size=1.9),
    "rhino": dict(family="columnar", foot="pad", body_length=0.95, chest_depth=0.6, chest_width=0.5,
                  belly_tuck=0.0, fat=0.8, neck_length=0.22, neck_angle=5, neck_thickness=1.6, head_length=0.55,
                  head_width=0.32, head_angle=30, snout_length=0.5, snout_width=0.75, ear_type="small",
                  ear_size=1.3, tail_type="tuft", tail_length=0.3, tail_angle=-70, eye_size=0.6, eye_angle=65,
                  horns="rhino", leg_length=0.85, leg_thickness=0.85, material="skin", color="elephant_gray",
                  nose_color="elephant_gray", hoof_color="charcoal", _size=1.6),
    "rabbit": dict(family="leaper", foot="paw", body_length=0.8, chest_depth=0.6, chest_width=0.5,
                   belly_tuck=0.0, fat=0.6, hip_height=1.15, neck_length=0.2, neck_angle=40, neck_thickness=1.2,
                   head_length=0.48, head_width=0.45, head_angle=20, snout_length=0.3, snout_width=0.7,
                   ear_type="long", ear_size=1.25, tail_type="fluffy", tail_length=0.2, eye_size=1.3,
                   eye_angle=65, color="ash", belly_color="white", nose_color="pink_skin", _size=0.2),
    "mouse": dict(family="leaper", foot="paw", body_length=1.15, chest_depth=0.6, chest_width=0.55,
                  belly_tuck=0.0, fat=0.6, hip_height=1.15, neck_length=0.15, neck_angle=20, head_length=0.6,
                  head_width=0.42, head_angle=15, snout_length=0.5, snout_width=0.5, ear_type="round",
                  ear_size=1.8, tail_type="thin", tail_length=1.8, tail_angle=5, tail_curl=-6, eye_size=1.5,
                  eye_angle=55, leg_thickness=0.7, color="gray", belly_color="cream", nose_color="pink_skin",
                  _size=0.04),
}


def _leg_profiles(family):
    """(front, hind) joint lists: (dy, z) in withers-height units, dy relative
    to the leg anchor (+dy = backwards). 4 joints = 3 bones."""
    if family == "unguligrade":
        return ([(0.0, 0.64), (0.07, 0.52), (0.04, 0.28), (0.045, 0.085)],
                [(0.0, 0.72), (-0.11, 0.55), (0.12, 0.33), (0.08, 0.085)])
    if family == "plantigrade":
        return ([(0.0, 0.6), (0.06, 0.36), (0.02, 0.08), (-0.06, 0.035)],
                [(0.0, 0.68), (-0.09, 0.4), (0.03, 0.09), (-0.08, 0.035)])
    if family == "columnar":
        return ([(0.0, 0.62), (0.03, 0.42), (0.01, 0.13), (0.0, 0.06)],
                [(0.0, 0.66), (-0.05, 0.42), (0.02, 0.13), (0.0, 0.06)])
    if family == "leaper":
        return ([(0.0, 0.5), (0.05, 0.3), (0.0, 0.08), (-0.06, 0.03)],
                [(0.0, 0.62), (-0.2, 0.36), (0.12, 0.12), (-0.24, 0.03)])
    return ([(0.0, 0.62), (0.07, 0.43), (0.0, 0.12), (-0.04, 0.035)],
            [(0.0, 0.7), (-0.12, 0.47), (0.1, 0.22), (0.03, 0.035)])


def _pose_presets(family):
    P = {"stand": {}}
    P["walk"] = {
        "upper_arm.L": [-18, 0, 0], "forearm.L": [-6, 0, 0],
        "upper_arm.R": [14, 0, 0], "forearm.R": [30, 0, 0], "front_foot.R": [20, 0, 0],
        "thigh.R": [-16, 0, 0], "shin.R": [-10, 0, 0],
        "thigh.L": [16, 0, 0], "shin.L": [20, 0, 0],
        "neck_0": [-4, 0, 0],
    }
    P["run"] = {
        "upper_arm.L": [-45, 0, 0], "forearm.L": [-10, 0, 0], "upper_arm.R": [-30, 0, 0], "forearm.R": [-5, 0, 0],
        "thigh.L": [40, 0, 0], "shin.L": [10, 0, 0], "thigh.R": [30, 0, 0], "shin.R": [5, 0, 0],
        "neck_0": [-12, 0, 0], "tail_0": [-15, 0, 0],
    }
    P["sit"] = sym({"hips": [-38, 0, 0],
                    "upper_arm.L": {"aim": [0, -0.08, -1]}, "forearm.L": {"aim": [0, -0.02, -1]},
                    "front_foot.L": {"aim": [0, -0.6, -1]},
                    "thigh.L": {"aim": [0.15, -1, -0.15]}, "shin.L": {"aim": [0, 1, -0.35]},
                    "hind_foot.L": {"aim": [0, -1, -0.05]}, "neck_0": [28, 0, 0], "tail_0": [30, 0, 0]})
    P["lie"] = sym({"upper_arm.L": {"aim": [0.05, -1, -0.35]}, "forearm.L": {"aim": [0, -1, -0.05]},
                    "front_foot.L": {"aim": [0, -1, -0.05]},
                    "thigh.L": {"aim": [0.25, -1, -0.2]}, "shin.L": {"aim": [0.1, 1, -0.2]},
                    "hind_foot.L": {"aim": [0.05, -1, -0.05]}, "tail_0": [-20, 0, 0]})
    P["graze"] = {"neck_0": [55, 0, 0], "neck_1": [10, 0, 0], "head": [10, 0, 0]}
    P["alert"] = {"neck_0": [-12, 0, 0], "neck_1": [-6, 0, 0], "head": [-8, 0, 0], "tail_0": [-15, 0, 0]}
    P["look_left"] = {"neck_1": [0, 0, 25], "head": [0, 0, 20]}
    P["look_right"] = {"neck_1": [0, 0, -25], "head": [0, 0, -20]}
    if family == "unguligrade":
        P["walk"].update({"front_foot.R": [35, 0, 0], "shin.R": [-5, 0, 0], "hind_foot.L": [25, 0, 0]})
    return P


POSES = tuple(_pose_presets("digitigrade"))


def quadruped(species="dog", **kwargs):
    """Return a :class:`Creature` for a four-legged animal (see PRESETS / PARAMS)."""
    if species not in PRESETS:
        import difflib
        close = difflib.get_close_matches(str(species), PRESETS, n=3)
        raise NaturalFormError("unknown quadruped species %r.%s Presets: %s" % (
            species, (" Did you mean %s?" % ", ".join(close)) if close else "", ", ".join(PRESETS)))
    preset = dict(PRESETS[species])
    base_size = preset.pop("_size", 1.0)
    p = resolve(PARAMS, kwargs, "quadruped(%s)" % species, preset)
    fam = choice(p["family"], FAMILIES, "family")
    choice(p["ear_type"], EARS, "ear_type")
    choice(p["tail_type"], TAILS, "tail_type")
    choice(p["horns"], HORNS, "horns")
    choice(p["mane"], MANES, "mane")
    choice(p["foot"], FEET, "foot")
    fat = clamp(p["fat"])

    c = Creature(species.capitalize(), "quadruped")
    c.default_material = "fur"
    c.meta["ground"] = True

    L = float(p["body_length"])
    cd = float(p["chest_depth"])
    cw = float(p["chest_width"]) * (1 + 0.25 * fat)
    hh = float(p["hip_height"])
    lt = float(p["leg_thickness"])
    front_prof, hind_prof = _leg_profiles(fam)
    ll = float(p["leg_length"])
    fa_z = front_prof[0][1]
    drop = (1.0 - ll) * fa_z  # how much the body sinks with shorter legs

    def B(x, y, z):  # body point (dropped with short legs)
        return np.array([x, y, z - drop], dtype=np.float64)

    top = 1.0  # withers
    chest_bot = top - cd
    belly_bot = chest_bot + clamp(p["belly_tuck"]) * cd * 0.45 - 0.05 * fat
    yS, yH = -L / 2.0, L / 2.0
    hip_top = hh * top

    # ------------------------------------------------------------- spine (bones only)
    c.chain("spine", [B(0, yH + 0.08, hip_top - 0.1), B(0, yH * 0.3, (hip_top + top) / 2 - 0.07),
                      B(0, yS * 0.4, top - 0.08), B(0, yS - 0.04, top - 0.12)], 0.05,
            bones=["hips", "spine", "chest"], visible=False)

    # ------------------------------------------------------------- torso
    chest_c = B(0, yS + 0.1, (top + chest_bot) / 2 - 0.01)
    c.ellipsoid("ribcage", chest_c, (cw / 2, 0.24 + 0.2 * L, cd / 2), rotate=[-4, 0, 0], bone="chest", blend=0.0)
    c.sphere("breast", B(0, yS - 0.13, chest_bot + cd * 0.38), cd * 0.3 * (0.9 + 0.2 * fat), bone="chest", blend=0.06)
    ab_top = (top + hip_top) / 2 - 0.02
    c.ellipsoid("abdomen", B(0, 0.08 * L, (ab_top + belly_bot) / 2), (cw / 2 * (0.88 + 0.12 * fat), 0.3 * L + 0.08,
                (ab_top - belly_bot) / 2), bone="spine", blend=0.08)
    rump_c = B(0, yH - 0.03, hip_top - 0.16)
    c.ellipsoid("rump", rump_c, (cw / 2 * 0.95, 0.2, 0.18 + 0.03 * fat), bone="hips", blend=0.07)
    c.capsule("back", B(0, yS + 0.02, top - 0.07), B(0, yH, hip_top - 0.07), 0.07 + 0.03 * fat, bone="spine",
              blend=0.06, squash=1.6 + cw, squash_dir=(1, 0, 0))
    if fam == "leaper":  # rabbits / rodents: round body over folded hind legs
        c.ellipsoid("haunches", B(0, yH - 0.05, hip_top * 0.55), (cw * 0.55, 0.3, hip_top * 0.42), bone="hips",
                    blend=0.08)
    c.ellipsoid("shoulder_blade", B(cw * 0.3, yS + 0.03, top - 0.14), (0.065 * lt ** 0.5, 0.11, 0.15),
                rotate=[-22, 0, 0], bone="chest", blend=0.04, mirror=True)
    if p["hump"] > 0:
        hmp = float(p["hump"])
        c.ellipsoid("hump", B(0, lerp(yS * 0.8, 0.0, hmp), top + 0.08 * hmp),
                    (cw * 0.32, 0.2 * hmp + 0.1, 0.16 * hmp + 0.04), bone="spine", blend=0.08)

    # ------------------------------------------------------------- legs
    fs = float(p["foot_size"])
    front_anchor = np.array([cw * 0.32, yS, 0.0])
    hind_anchor = np.array([cw * 0.3, yH, 0.0])
    radii_f = np.array([0.09, 0.056, 0.037, 0.035]) * lt
    radii_h = np.array([0.11, 0.065, 0.04, 0.035]) * lt
    if fam == "unguligrade":
        radii_f = np.array([0.1, 0.06, 0.036, 0.032]) * lt
        radii_h = np.array([0.12, 0.07, 0.04, 0.032]) * lt
    elif fam == "columnar":
        radii_f = np.array([0.12, 0.1, 0.09, 0.1]) * lt
        radii_h = np.array([0.13, 0.1, 0.09, 0.1]) * lt
    radii_f = radii_f * (1 + 0.15 * fat)
    radii_h = radii_h * (1 + 0.2 * fat)

    def leg_points(anchor, prof):
        z0 = prof[0][1]
        k = (z0 - drop) / z0  # vertical squash so the anchor follows the body
        pts = []
        for dy, z in prof:
            pts.append(np.array([anchor[0], anchor[1] + dy * k, z * k]))
        return pts

    fpts = leg_points(front_anchor, front_prof)
    hpts = leg_points(hind_anchor, hind_prof)
    c.chain("front_leg.L", fpts, radii_f, parent="chest", bones=["upper_arm.L", "forearm.L", "front_foot.L"],
            blend=0.05, mirror=True)
    c.chain("hind_leg.L", hpts, radii_h, parent="hips", bones=["thigh.L", "shin.L", "hind_foot.L"],
            blend=0.06, mirror=True)
    # muscles
    c.ellipsoid("forearm_muscle", fpts[1] + (fpts[2] - fpts[1]) * 0.3 + [0, -0.01, 0],
                np.array([0.055, 0.12, 0.06]) * lt * (1 + 0.2 * fat), align=fpts[2] - fpts[1],
                bone="forearm.L", blend=0.03, mirror=True)
    c.ellipsoid("haunch", hpts[0] + (hpts[1] - hpts[0]) * 0.45 + [0.01, 0.02, 0],
                np.array([0.1, 0.19, 0.13]) * lt * (1 + 0.25 * fat), align=hpts[1] - hpts[0],
                bone="thigh.L", blend=0.05, mirror=True)
    if fam in ("digitigrade", "unguligrade", "leaper"):
        c.ellipsoid("gaskin", hpts[1] + (hpts[2] - hpts[1]) * 0.35 + [0, 0.012, 0],
                    np.array([0.05, 0.1, 0.055]) * lt, align=hpts[2] - hpts[1], bone="shin.L", blend=0.03,
                    mirror=True)
    _feet(c, p, fpts, hpts, lt, fs)

    # ------------------------------------------------------------- neck & head
    nl = float(p["neck_length"])
    na = math.radians(float(p["neck_angle"]))
    nt = float(p["neck_thickness"]) * (1 + 0.2 * fat)
    n0 = B(0, yS - 0.02, top - 0.14)
    ndir = np.array([0.0, -math.cos(na), math.sin(na)])
    n1 = n0 + ndir * nl * 0.5 + np.array([0, -0.02, 0.01]) * nl
    n2 = n0 + ndir * nl
    hw_ = float(p["head_width"])
    neck_r = np.array([0.16, 0.12, 0.09]) * nt * (0.55 + 0.45 * hw_ / 0.26)
    neck_r[2] = min(neck_r[2], hw_ * 0.3)
    neck_r[1] = min(neck_r[1], (neck_r[0] + neck_r[2]) * 0.55)
    c.chain("neck", [n0, n1, n2], neck_r, parent="chest", bones=["neck_0", "neck_1"], blend=0.07,
            squash=0.82, squash_dir=(1, 0, 0))
    _head(c, p, n2, fat)

    # ------------------------------------------------------------- tail, mane
    _tail(c, p, B(0, yH + 0.16, hip_top - 0.07))
    _mane(c, p, n0, n2)
    _materials(c, p, base_size)

    c.poses.update(_pose_presets(fam))
    # finer meshing around the head (cut across the upper neck)
    hl = float(p["head_length"])
    c.meta["detail_regions"] = [{"bone": "head", "at": 0.45, "radius": hl * 0.75 + 0.06, "factor": 2.5,
                                 "cut_bone": "neck_1", "cut_at": 0.6, "layers": ["body"]}]

    # ------------------------------------------------------------- scale
    withers = top - drop
    if p["height"]:
        s = float(p["height"]) / withers
    else:
        s = base_size * float(p["size"])
    c.transform(scale=s)
    c.snap_to_ground()
    c.meta.pop("_head", None)
    c.meta["params"] = dict(p, species=species)
    c.meta["user_params"] = dict(kwargs)
    c.meta["template"] = "quadruped"
    return c


def _feet(c, p, fpts, hpts, lt, fs):
    foot = p["foot"]
    for name, pts, bone in (("front", fpts, "front_foot.L"), ("hind", hpts, "hind_foot.L")):
        end = pts[-1]
        prev = pts[-2]
        if foot in ("hoof", "cloven"):
            r = 0.04 * lt ** 0.6 * fs
            top = end + np.array([0, -0.012, 0.0])
            bot = np.array([end[0], end[1] - 0.03 * fs, 0.0])
            c.capsule("%s_pastern" % name, prev + (end - prev) * 0.6, top, r * 0.85, end_radius=r * 0.9,
                      bone=bone, blend=0.015, mirror=True)
            c.capsule("%s_hoof" % name, top + [0, 0, 0.01], bot + [0, 0, 0.012], r * 0.95, end_radius=r * 1.2,
                      bone=bone, blend=0.006, material="hoof", mirror=True,
                      clip=[(bot + [0, 0, 0.004], (0, 0, 1))], clip_blend=0.002)
            if foot == "cloven":
                c.box("%s_cleft" % name, bot + [0, -r * 0.9, r * 0.8], (r * 0.08, r * 0.9, r * 1.0), rounding=0.0,
                      bone=bone, blend=0.002, op="subtract", mirror=True)
        elif foot == "pad":
            r = 0.1 * lt ** 0.8 * fs
            c.capsule("%s_foot" % name, prev + (end - prev) * 0.4, np.array([end[0], end[1], 0.02]), r * 0.9,
                      end_radius=r, bone=bone, blend=0.03, mirror=True, clip=[((0, 0, 0.003), (0, 0, 1))],
                      clip_blend=0.004)
            for k, dx in enumerate((-0.5, 0.0, 0.5)):
                c.sphere("%s_nail%d" % (name, k), np.array([end[0] + dx * r * 0.9, end[1] - r * 0.85, 0.025]),
                         r * 0.2, bone=bone, blend=0.006, material="hoof", mirror=True)
        elif foot == "plantigrade":
            r = 0.055 * lt ** 0.7 * fs
            c.ellipsoid("%s_paw" % name, np.array([end[0], end[1] + 0.02, r * 0.55]), (r * 1.05, r * 1.9, r * 0.6),
                        bone=bone, blend=0.03, mirror=True)
            for k, dx in enumerate((-0.6, -0.2, 0.2, 0.6)):
                c.capsule("%s_claw%d" % (name, k), np.array([end[0] + dx * r, end[1] - r * 1.5, r * 0.35]),
                          np.array([end[0] + dx * r * 1.1, end[1] - r * 2.1, r * 0.08]), r * 0.13,
                          end_radius=r * 0.05, bone=bone, blend=0.004, material="hoof", mirror=True)
        else:  # paw
            r = 0.04 * lt ** 0.7 * fs
            c.ellipsoid("%s_paw" % name, np.array([end[0], end[1] - r * 0.4, r * 0.7]), (r * 1.15, r * 1.45, r * 0.75),
                        bone=bone, blend=0.03, mirror=True)
            for k, dx in enumerate((-0.6, -0.2, 0.2, 0.6)):
                c.sphere("%s_toe%d" % (name, k), np.array([end[0] + dx * r * 0.95, end[1] - r * 1.35 + abs(dx) * r * 0.4,
                                                            r * 0.42]), r * 0.42, bone=bone, blend=r * 0.25, mirror=True)


def _head(c, p, base, fat):
    hl = float(p["head_length"])
    hw = float(p["head_width"])
    ha = math.radians(float(p["head_angle"]))
    fwd = np.array([0.0, -math.cos(ha), -math.sin(ha)])
    up = np.array([0.0, -math.sin(ha), math.cos(ha)])
    side = np.array([1.0, 0.0, 0.0])
    HF = np.column_stack([side, fwd, up])  # head frame: x side, y forward, z up
    sl = float(p["snout_length"])
    sw = float(p["snout_width"])
    cran_len = hl * (1 - sl)
    cran_c = base + fwd * cran_len * 0.5 + up * hw * 0.16
    c.bone("head", base, base + fwd * hl, parent="neck_1", connect=True)
    c.ellipsoid("cranium", cran_c, (hw * 0.5, max(cran_len * 0.62, hw * 0.42), hw * 0.44), rotate=HF,
                bone="head", blend=0.04)
    # jowls / cheeks widen the back of the face (big on horses and cattle)
    jw = 0.32 if p["foot"] in ("hoof", "cloven") else 0.24
    c.ellipsoid("jowl", cran_c + side * hw * 0.2 - up * hw * 0.2 + fwd * cran_len * 0.1,
                (hw * 0.24, cran_len * 0.42 + hw * 0.08, hw * jw), rotate=HF, bone="head", blend=0.04, mirror=True)
    # snout: starts a little below the forehead ("stop") and tapers to the nose
    s0 = cran_c + fwd * cran_len * 0.42 - up * hw * 0.14
    s1 = base + fwd * hl * 0.96 - up * hw * 0.1
    rs0 = hw * 0.5 * sw * 0.9
    rs1 = hw * 0.5 * sw * (0.62 if p["foot"] not in ("hoof", "cloven") else 0.75)
    c.capsule("snout", s0, s1, rs0, end_radius=rs1, bone="head", blend=0.04, squash=0.86, squash_dir=up)
    c.capsule("jaw", cran_c - up * hw * 0.26 + fwd * cran_len * 0.1, s1 - up * rs1 * 0.75 - fwd * rs1 * 0.5,
              rs0 * 0.6, end_radius=rs1 * 0.55, bone="head", blend=0.03)
    nose = p["nose_type"]
    if nose == "disc":
        c.ellipsoid("nose", s1 + fwd * rs1 * 0.5, (rs1 * 0.95, rs1 * 0.4, rs1 * 0.8), rotate=HF, bone="head",
                    blend=rs1 * 0.15, material="nose")
        tipc = s1 + fwd * rs1 * 0.9
    elif nose == "wide":
        c.ellipsoid("nose", s1 + fwd * rs1 * 0.35, (rs1 * 1.0, rs1 * 0.6, rs1 * 0.85), rotate=HF, bone="head",
                    blend=rs1 * 0.3, material="nose")
        tipc = s1 + fwd * rs1 * 0.95
    else:
        c.sphere("nose", s1 + fwd * rs1 * 0.6 + up * rs1 * 0.2, rs1 * 0.5, bone="head", blend=rs1 * 0.25,
                 material="nose")
        tipc = s1 + fwd * rs1 * 1.05 + up * rs1 * 0.2
    if p["foot"] == "paw":
        # whisker pads
        c.sphere("whisker_pad", s1 + side * rs1 * 0.45 - up * rs1 * 0.25 - fwd * rs1 * 0.1, rs1 * 0.62,
                 bone="head", blend=rs1 * 0.35, mirror=True)
    c.ellipsoid("mouth_line", s1 - up * rs1 * 0.52 + fwd * rs1 * 0.1, (rs1 * 1.1, rs1 * 1.3, rs1 * 0.05),
                rotate=HF, bone="head", blend=rs1 * 0.08, op="subtract")
    c.sphere("nostril", tipc - fwd * rs1 * 0.06 + side * rs1 * 0.3 - up * rs1 * 0.05, rs1 * 0.13, bone="head",
             blend=rs1 * 0.06, op="subtract", mirror=True)
    # eyes
    ea = math.radians(float(p["eye_angle"]))
    er = 0.075 * hw * float(p["eye_size"])
    look = unit(fwd * math.cos(ea) + side * math.sin(ea) + up * 0.08)
    probe = cran_c + fwd * cran_len * 0.32 + up * hw * 0.05
    hit = c.surface_point(probe, look)
    ec = (np.array(hit) if hit is not None else probe + look * hw * 0.4) - look * er * 0.45
    c.eye("eye.L", ec, er, look=look, bone="head", color=p["eye_color"], pupil=p["pupil"],
          iris_size=0.55 if p["sclera"] else 0.82, pupil_size=0.45 if p["pupil"] == "round" else 0.6,
          sclera="white" if p["sclera"] else None, socket_blend=er * 0.5, mirror=True)
    c.sphere("brow_ridge", ec + up * er * 1.0 - look * er * 0.2, er * 1.05, bone="head", blend=er * 0.6, mirror=True,
             clip=[(ec + up * er * 0.55, up)], clip_blend=er * 0.2)
    _ears(c, p, base, cran_c, fwd, up, side, hw, cran_len)
    _horns(c, p, base, cran_c, fwd, up, side, hw, cran_len)
    if p["trunk"]:
        tr = rs1 * 0.95
        pts = [s1, s1 + fwd * hl * 0.25 - up * hl * 0.25, s1 + fwd * hl * 0.3 - up * hl * 0.7,
               s1 + fwd * hl * 0.25 - up * hl * 1.15, s1 + fwd * hl * 0.38 - up * hl * 1.45]
        c.chain("trunk", pts, [tr * 1.1, tr * 0.85, tr * 0.7, tr * 0.55, tr * 0.42], parent="head",
                bones=["trunk_%d" % i for i in range(4)], blend=0.06, joint_blend=0.03)
    if p["tusks"]:
        t0 = s0 - up * rs0 * 0.6 + side * rs0 * 0.55
        hs_ = float(p["horn_size"])
        pts = [t0, t0 + fwd * hl * 0.3 * hs_ - up * hl * 0.35 * hs_, t0 + fwd * hl * 0.65 * hs_ - up * hl * 0.45 * hs_,
               t0 + fwd * hl * 0.85 * hs_ - up * hl * 0.3 * hs_]
        c.chain("tusk.L", pts, np.array([0.11, 0.09, 0.065, 0.03]) * hw, parent="head", make_bones=False,
                blend=0.01, material="horn", mirror=True)
    c.meta["_head"] = {"cran_c": cran_c, "fwd": fwd, "up": up, "hw": hw, "cran_len": cran_len}


def _ears(c, p, base, cran_c, fwd, up, side, hw, cran_len):
    et = p["ear_type"]
    es = float(p["ear_size"])
    if et == "none":
        return
    root = cran_c + up * hw * 0.34 + side * hw * 0.3 - fwd * cran_len * 0.15
    if et == "pointed":
        tip = root + (up * 0.95 + side * 0.25 - fwd * 0.12) * hw * 0.62 * es
        c.capsule("ear.L", root, tip, hw * 0.17 * es ** 0.5, end_radius=hw * 0.02, squash=0.3, squash_dir=fwd,
                  bone="head", blend=0.02, mirror=True)
        c.ellipsoid("ear_hollow.L", root + (tip - root) * 0.45 + fwd * hw * 0.05,
                    (hw * 0.08 * es ** 0.5, hw * 0.035, np.linalg.norm(tip - root) * 0.4),
                    rotate=_frame(side, fwd, unit(tip - root)), bone="head", blend=0.01, op="subtract", mirror=True)
    elif et == "long":
        tip = root + (up * 0.97 + side * 0.15 + -fwd * 0.2) * hw * 1.35 * es
        c.capsule("ear.L", root, tip, hw * 0.17 * es ** 0.5, end_radius=hw * 0.14, squash=0.32, squash_dir=fwd,
                  bone="head", blend=0.02, mirror=True)
        c.ellipsoid("ear_hollow.L", root + (tip - root) * 0.5 + fwd * hw * 0.04,
                    (hw * 0.08, hw * 0.03, np.linalg.norm(tip - root) * 0.42), rotate=_frame(side, fwd, unit(tip - root)),
                    bone="head", blend=0.01, op="subtract", mirror=True)
    elif et == "floppy":
        root = cran_c + up * hw * 0.3 + side * hw * 0.36 - fwd * cran_len * 0.2
        tip = root + (side * 0.35 - up * 1.0 + fwd * 0.12) * hw * 0.85 * es
        c.capsule("ear.L", root, tip, hw * 0.16 * es ** 0.5, end_radius=hw * 0.13 * es ** 0.5, squash=0.28,
                  squash_dir=side, bone="head", blend=0.02, mirror=True)
    elif et == "round":
        c.ellipsoid("ear.L", root + up * hw * 0.1, (hw * 0.16 * es, hw * 0.07 * es, hw * 0.16 * es),
                    rotate=_frame(side, fwd, up), bone="head", blend=0.02, mirror=True)
        c.ellipsoid("ear_hollow.L", root + up * hw * 0.12 + fwd * hw * 0.05, (hw * 0.1 * es, hw * 0.04 * es, hw * 0.1 * es),
                    rotate=_frame(side, fwd, up), bone="head", blend=0.01, op="subtract", mirror=True)
    elif et == "small":
        tip = root + (up * 0.8 + side * 0.55 - fwd * 0.25) * hw * 0.42 * es
        c.capsule("ear.L", root, tip, hw * 0.12 * es ** 0.5, end_radius=hw * 0.025, squash=0.35, squash_dir=fwd,
                  bone="head", blend=0.015, mirror=True)
    elif et == "elephant":
        cen = cran_c + side * hw * 0.55 - fwd * hw * 0.1 - up * hw * 0.25
        c.ellipsoid("ear.L", cen, (hw * 0.05, hw * 0.55 * es, hw * 0.7 * es),
                    rotate=_frame(unit(side + fwd * 0.35), unit(fwd - side * 0.35), up), bone="head", blend=0.04,
                    mirror=True)


def _frame(x, y, z):
    x = unit(x)
    z = unit(z - np.dot(z, x) * x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _horns(c, p, base, cran_c, fwd, up, side, hw, cran_len):
    h = p["horns"]
    if h == "none":
        return
    k = float(p["horn_size"])
    root = cran_c + up * hw * 0.38 + side * hw * 0.2 + fwd * cran_len * 0.05
    L = "horn"
    if h == "antlers":
        beam = [root, root + (up * 0.8 + side * 0.5 - fwd * 0.3) * hw * 1.2 * k,
                root + (up * 1.7 + side * 0.9 - fwd * 0.4) * hw * 1.2 * k,
                root + (up * 2.5 + side * 1.0 - fwd * 0.1) * hw * 1.2 * k]
        c.chain("antler.L", beam, np.array([0.07, 0.055, 0.045, 0.025]) * hw * k ** 0.5 * 1.5, parent="head",
                make_bones=False, blend=0.01, material="horn", mirror=True, layer=L)
        for i, (a, d) in enumerate(((0.35, up * 0.6 + fwd * 0.8), (0.65, up * 0.8 + fwd * 0.5),
                                    (0.85, up * 0.6 - fwd * 0.6 + side * 0.3))):
            seg = int(a * 3)
            st = beam[seg] + (beam[min(seg + 1, 3)] - beam[seg]) * (a * 3 - seg)
            c.capsule("tine%d.L" % i, st, st + unit(d) * hw * 0.65 * k, hw * 0.045 * k ** 0.5,
                      end_radius=hw * 0.018, bone="head", blend=0.01, material="horn", mirror=True, layer=L)
        return
    if h == "goat":
        pts = [root, root + (up * 0.9 - fwd * 0.45 + side * 0.1) * hw * 0.7 * k,
               root + (up * 1.1 - fwd * 1.1 + side * 0.18) * hw * 0.7 * k,
               root + (up * 0.8 - fwd * 1.6 + side * 0.25) * hw * 0.7 * k]
        radii = np.array([0.12, 0.09, 0.06, 0.02]) * hw * k ** 0.5
    elif h == "ram":
        pts = []
        for t in np.linspace(0, 1.6 * math.pi, 9):
            r = hw * 0.42 * k * (1 - 0.25 * t / (1.6 * math.pi))
            pts.append(root + side * (hw * 0.12 + t * 0.03 * hw * k) + up * r * math.sin(t) * 0.9
                       - fwd * r * (1 - math.cos(t)) * 0.9 + up * 0.05 * hw)
        radii = np.linspace(0.16, 0.05, len(pts)) * hw * k ** 0.5
    elif h == "bull":
        pts = [root - up * hw * 0.05, root + side * hw * 0.5 * k + up * hw * 0.05,
               root + side * hw * 0.85 * k + up * hw * 0.25 * k + fwd * hw * 0.15 * k,
               root + side * hw * 0.95 * k + up * hw * 0.55 * k + fwd * hw * 0.3 * k]
        radii = np.array([0.13, 0.11, 0.07, 0.025]) * hw * k ** 0.5
    elif h == "straight":
        pts = [root, root + (up * 0.7 - fwd * 0.7 + side * 0.1) * hw * 1.6 * k]
        radii = np.array([0.09, 0.02]) * hw * k ** 0.5
    elif h == "ossicones":
        pts = [root - up * hw * 0.05, root + up * hw * 0.55 * k + side * hw * 0.08 - fwd * hw * 0.1]
        radii = np.array([0.12, 0.1]) * hw * k ** 0.5
        c.sphere("ossicone_tip.L", pts[-1], 0.14 * hw * k ** 0.5, bone="head", blend=0.01, material="mane",
                 mirror=True)
        c.chain("horn.L", pts, radii, parent="head", make_bones=False, blend=0.02, mirror=True)
        return
    elif h == "rhino":
        nose = cran_c + fwd * cran_len * 0.9 + fwd * float(p["head_length"]) * float(p["snout_length"]) * 0.7 + up * hw * 0.25
        c.capsule("horn_front", nose, nose + (up * 0.9 + fwd * 0.3) * hw * 1.1 * k, hw * 0.22 * k ** 0.5,
                  end_radius=hw * 0.02, bone="head", blend=0.03, material="horn", layer=L)
        n2 = nose - fwd * hw * 0.55 + up * hw * 0.12
        c.capsule("horn_back", n2, n2 + (up * 0.95 + fwd * 0.2) * hw * 0.5 * k, hw * 0.15 * k ** 0.5,
                  end_radius=hw * 0.02, bone="head", blend=0.03, material="horn", layer=L)
        return
    c.chain("horn.L", pts, radii, parent="head", make_bones=False, blend=0.008, material="horn", mirror=True,
            layer=L)


def _tail(c, p, root):
    t = p["tail_type"]
    if t == "none":
        return
    tl = float(p["tail_length"])
    a = math.radians(float(p["tail_angle"]))
    curl = math.radians(float(p["tail_curl"]))
    if t == "fluffy":
        c.sphere("tail", root + [0, 0.03, 0.02], 0.09 * max(tl / 0.2, 0.6), bone="hips", blend=0.03)
        return
    n = 5 if t != "short" else 2
    if t == "short":
        tl = min(tl, 0.2)
    if t == "curly":
        pts = []
        for i, ang in enumerate(np.linspace(0, 2.2 * math.pi, 9)):
            r = 0.05 * tl / 0.3
            pts.append(root + np.array([r * math.sin(ang) * 0.6, 0.03 + r * (1 - math.cos(ang)) * 0.5 + i * 0.006,
                                        r * math.sin(ang) + i * 0.004]))
        c.chain("tail", pts, np.linspace(0.025, 0.012, len(pts)), parent="hips",
                bones=["tail_%d" % i for i in range(len(pts) - 1)], blend=0.02, joint_blend=0.005)
        return
    seg = tl / n
    pts = [root]
    ang = a
    for i in range(n):
        d = np.array([0.0, math.cos(ang), math.sin(ang)])
        pts.append(pts[-1] + d * seg)
        ang += curl
    base_r = {"thin": 0.045, "bushy": 0.06, "short": 0.06, "tuft": 0.04, "horse": 0.07}[t]
    radii = list(np.linspace(base_r, base_r * 0.35, n + 1))
    mat = None
    layer = None
    if t == "bushy":
        radii = [base_r * k for k in (0.9, 1.45, 1.75, 1.6, 1.15, 0.45)][:n + 1]
    if t == "horse":
        # hair fall: thick, flattened, mane-coloured
        radii = [0.06, 0.08, 0.1, 0.1, 0.085, 0.04][:n + 1]
        mat = "mane"
        layer = "mane"
    c.chain("tail", pts, radii, parent="hips", bones=["tail_%d" % i for i in range(n)], blend=0.04,
            joint_blend=0.02, material=mat, layer=layer,
            squash=0.55 if t == "horse" else 1.0, squash_dir=(1, 0, 0))
    if t == "horse":
        c.chain("tail_dock", [root, pts[1]], [0.055, 0.045], parent="hips", make_bones=False, blend=0.04)
    if t == "tuft":
        c.ellipsoid("tail_tuft", pts[-1], (0.05, 0.05, 0.085), align=pts[-1] - pts[-2], bone="tail_%d" % (n - 1),
                    blend=0.015, material="mane")


def _mane(c, p, n0, n2):
    m = p["mane"]
    if m == "none":
        return
    d = n2 - n0
    nrm = unit(np.cross(np.array([1.0, 0, 0]), d))  # top of the neck
    if nrm[2] < 0:
        nrm = -nrm
    if m in ("horse", "mohawk"):
        pts = [n0 + d * t + nrm * (0.13 - 0.06 * t) * float(p["neck_thickness"]) for t in (0.05, 0.4, 0.75, 1.02)]
        r = 0.07 if m == "horse" else 0.05
        c.chain("mane", pts, [r * 0.8, r, r * 0.95, r * 0.7], parent="neck_0", make_bones=False, blend=0.02,
                material="mane", squash=0.3 if m == "horse" else 0.25, squash_dir=(1, 0, 0), layer="mane")
        if m == "horse":
            c.capsule("mane_fall", pts[1] + [0.05, 0, 0], pts[2] + [0.07, 0, -0.06], 0.06, end_radius=0.05,
                      bone="neck_1", blend=0.02, material="mane", squash=0.35, squash_dir=(1, 0, 0), layer="mane")
    elif m == "lion":
        h = c.meta.get("_head")
        fwd, hw, cc = h["fwd"], h["hw"], h["cran_c"]
        cut = (cc + fwd * h["cran_len"] * 0.15, -fwd)
        c.ellipsoid("mane", cc - fwd * hw * 0.35 - np.array([0, 0, 0.04]), (hw * 0.95, hw * 0.85, hw * 1.0),
                    bone="head", blend=0.05, material="mane", layer="mane", clip=[cut], clip_blend=0.03)
        c.ellipsoid("mane_neck", n0 + d * 0.55, (hw * 0.8, np.linalg.norm(d) * 0.5, hw * 0.85), align=d,
                    bone="neck_0", blend=0.06, material="mane", layer="mane")
        c.ellipsoid("mane_chest", n0 + d * 0.2 - nrm * 0.1, (hw * 0.6, hw * 0.55, hw * 0.75), align=d,
                    bone="neck_0", blend=0.06, material="mane", layer="mane")


def _materials(c, p, base_size):
    from ..core.colors import color as parse
    kind = p["material"]
    size = base_size * float(p["size"]) if not p["height"] else float(p["height"])
    ps = p["pattern_scale"] or 0.12 * size
    c.material("fur", kind=kind, color=p["color"], belly_color=p["belly_color"], pattern=p["pattern"],
               pattern_color=p["pattern_color"], pattern_scale=ps,
               scale_size={"scales": 0.01 * size, "wool": 0.02 * size}.get(kind),
               subsurface_scale=0.002 * size)
    c.material("nose", kind="skin", color=p["nose_color"], roughness=0.35, subsurface=0.1, bump=0.3)
    c.material("hoof", kind="keratin", color=p["hoof_color"])
    c.material("horn", kind="keratin", color=p["horn_color"])
    mc = p["mane_color"]
    if mc is None:
        try:
            base = parse(p["color"])
            mc = tuple(int(255 * max(0.0, x * 0.45) ** (1 / 2.2)) for x in base)
        except Exception:
            mc = "charcoal"
    c.material("mane", kind="hair", color=mc)
