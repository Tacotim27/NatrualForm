"""Shared helpers for templates: parameter resolution, maths, pose mirroring."""

import difflib

import numpy as np

from ..core.creature import NaturalFormError, flip_name, split_side


def resolve(schema, given, what, preset=None):
    """Merge defaults <- preset <- given; unknown keys raise a helpful error.

    ``schema`` maps name -> (default, description).
    """
    out = {k: v[0] for k, v in schema.items()}
    if preset:
        out.update(preset)
    for k, v in given.items():
        if k not in schema:
            close = difflib.get_close_matches(k, schema.keys(), n=3)
            raise NaturalFormError(
                "%s: unknown parameter %r.%s Valid parameters: %s" % (
                    what, k, (" Did you mean %s?" % ", ".join(close)) if close else "",
                    ", ".join(schema)))
        if v is not None:
            out[k] = v
    return out


def choice(value, options, what):
    if value not in options:
        close = difflib.get_close_matches(str(value), options, n=3)
        raise NaturalFormError("%s must be one of %s, got %r.%s" % (
            what, ", ".join(map(str, options)), value, (" Did you mean %s?" % ", ".join(close)) if close else ""))
    return value


def interp(x, xs, ys):
    return float(np.interp(x, xs, ys))


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))


def smoothstep(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0))
    return t * t * (3 - 2 * t)


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def mirror_spec(spec):
    """Mirror a pose spec from the left (+X) side to the right side."""
    if isinstance(spec, dict):
        out = dict(spec)
        if "aim" in out:
            a = list(out["aim"])
            out["aim"] = [-a[0], a[1], a[2]]
        if "axis" in out:
            a = list(out["axis"])
            out["axis"] = [a[0], -a[1], -a[2]]
        if "twist" in out:
            out["twist"] = -out["twist"]
        if "rotate" in out:
            r = list(out["rotate"])
            out["rotate"] = [r[0], -r[1], -r[2]]
        return out
    r = list(spec)
    return [r[0], -r[1], -r[2]]


def sym(pose):
    """Return pose with every ``.L`` entry mirrored onto ``.R`` (unless the
    ``.R`` side is given explicitly)."""
    out = dict(pose)
    for k, v in pose.items():
        base, side = split_side(k)
        if side in (".L", "_L", ".l"):
            other = flip_name(k)
            if other not in pose:
                out[other] = mirror_spec(v)
    return out


def doc_table(schema):
    return "\n".join("  %-16s %-12r %s" % (k, v[0], v[1]) for k, v in schema.items())
