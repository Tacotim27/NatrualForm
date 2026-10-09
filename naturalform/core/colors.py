"""Colour parsing: names, hex strings, 0-1 floats or 0-255 ints -> linear RGB."""

import difflib

# sRGB values (0-255). Natural tones first, then common names.
NAMED = {
    # skin tones (light -> deep)
    "porcelain": (247, 225, 208), "pale": (241, 211, 190), "fair": (234, 196, 168),
    "light": (224, 180, 150), "medium": (203, 152, 115), "olive": (186, 142, 100),
    "tan": (176, 125, 88), "brown": (141, 94, 62), "dark": (104, 66, 44),
    "deep": (72, 45, 31), "ebony": (52, 33, 24),
    # hair
    "black_hair": (22, 18, 16), "dark_brown_hair": (50, 34, 24), "brown_hair": (84, 56, 36),
    "light_brown_hair": (130, 94, 62), "blonde": (214, 178, 112), "platinum": (230, 220, 196),
    "red_hair": (150, 62, 30), "auburn": (110, 44, 26), "ginger": (176, 86, 40),
    "gray_hair": (140, 136, 132), "grey_hair": (140, 136, 132), "white_hair": (226, 222, 214),
    # fur / animal
    "fawn": (196, 160, 112), "tawny": (190, 136, 72), "sandy": (204, 178, 134),
    "chestnut": (122, 60, 32), "bay": (110, 58, 30), "chocolate": (80, 48, 30),
    "cream": (240, 228, 200), "ivory": (238, 232, 214), "charcoal": (54, 54, 56),
    "ash": (150, 146, 140), "rust": (170, 80, 40), "ginger_fur": (205, 110, 45),
    "tiger_orange": (214, 112, 34), "lion_gold": (196, 152, 88), "wolf_gray": (128, 124, 118),
    "fox_red": (200, 96, 40), "pink_skin": (230, 160, 150), "elephant_gray": (120, 116, 112),
    "horn": (210, 196, 166), "hoof": (52, 44, 40), "bone": (226, 216, 190),
    "beak_yellow": (236, 176, 52), "salmon": (250, 128, 114), "olive_green": (110, 120, 50),
    # eyes
    "eye_brown": (92, 56, 28), "eye_blue": (66, 120, 176), "eye_green": (86, 132, 74),
    "eye_hazel": (128, 98, 48), "eye_gray": (130, 140, 146), "eye_amber": (196, 128, 32),
    "eye_yellow": (210, 180, 50),
    # generic
    "white": (245, 245, 245), "black": (16, 16, 16), "gray": (128, 128, 128),
    "grey": (128, 128, 128), "red": (200, 40, 36), "orange": (230, 120, 30),
    "yellow": (240, 210, 60), "green": (70, 140, 60), "blue": (50, 90, 190),
    "purple": (120, 60, 150), "pink": (240, 150, 170), "gold": (212, 175, 55),
    "silver": (192, 192, 196), "teal": (40, 128, 128), "beige": (220, 200, 170),
    "khaki": (190, 176, 120), "maroon": (110, 30, 36), "navy": (30, 40, 90),
}

_ALIASES = {
    "eye": {"brown": "eye_brown", "blue": "eye_blue", "green": "eye_green", "hazel": "eye_hazel",
            "gray": "eye_gray", "grey": "eye_gray", "amber": "eye_amber", "yellow": "eye_yellow"},
    "hair": {"black": "black_hair", "dark_brown": "dark_brown_hair", "brown": "brown_hair",
             "light_brown": "light_brown_hair", "red": "red_hair", "gray": "gray_hair",
             "grey": "gray_hair", "white": "white_hair"},
}


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def color(value, context=None):
    """Return a linear RGB tuple from a name, hex string or numbers.

    ``context`` ('eye', 'hair') lets plain names like "blue" pick the right
    tone (an eye blue vs. a paint blue).
    """
    if value is None:
        return None
    if isinstance(value, str):
        key = value.strip().lower().replace(" ", "_").replace("-", "_")
        if key.startswith("#"):
            h = key[1:]
            if len(h) == 3:
                h = "".join(ch * 2 for ch in h)
            if len(h) != 6:
                raise ValueError("bad hex colour %r" % value)
            rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
            return tuple(srgb_to_linear(c / 255.0) for c in rgb)
        if context and key in _ALIASES.get(context, {}):
            key = _ALIASES[context][key]
        if key not in NAMED:
            close = difflib.get_close_matches(key, NAMED.keys(), n=4)
            raise ValueError("unknown colour %r. Use a hex string, an (r,g,b) tuple or one of the "
                             "names in naturalform.COLORS%s" % (value, (" - did you mean %s?" % close) if close else ""))
        return tuple(srgb_to_linear(c / 255.0) for c in NAMED[key])
    vals = [float(v) for v in value]
    if len(vals) == 4:
        vals = vals[:3]
    if len(vals) != 3:
        raise ValueError("colour needs 3 numbers, got %r" % (value,))
    if max(vals) > 1.0:
        return tuple(srgb_to_linear(min(v, 255.0) / 255.0) for v in vals)
    # floats 0-1 are interpreted as sRGB (what people pick in colour pickers)
    return tuple(srgb_to_linear(max(v, 0.0)) for v in vals)


def mix(a, b, t):
    return tuple(x * (1 - t) + y * t for x, y in zip(a, b))


def scale(c, f):
    return tuple(min(max(x * f, 0.0), 1.0) for x in c)
