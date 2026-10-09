"""NaturalForm - procedural organic creatures (humans, animals, custom beasts)
for Blender, designed to be driven by LLM agents through an MCP connection.

    import naturalform as nf
    nf.help()
    nf.create("woman", clothing="dress", pose="wave")
    nf.create("horse", location=(2, 0, 0), pose="run")
"""

bl_info = {
    "name": "NaturalForm",
    "author": "NaturalForm contributors",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > NaturalForm, Add > NaturalForm, Python: import naturalform",
    "description": "Procedural humans, animals and organic creatures - an LLM-friendly toolkit",
    "category": "Add Mesh",
}

__version__ = "1.0.0"

import sys

from .api import (BUILD_OPTIONS, COLORS, TEMPLATES, bird, bones, build, create, create_bird, create_fish,
                  create_human, create_quadruped, create_serpent, describe, fish, focus, help, human,
                  list_poses, list_species, load, params, pose, quadruped, rebuild, reset_pose, serpent, spec)
from .core.creature import Creature, NaturalFormError

__all__ = [
    "Creature", "NaturalFormError", "COLORS", "TEMPLATES", "BUILD_OPTIONS",
    "help", "create", "create_human", "create_quadruped", "create_bird", "create_fish", "create_serpent",
    "human", "quadruped", "bird", "fish", "serpent", "spec", "build", "load", "rebuild", "pose",
    "reset_pose", "bones", "describe", "focus", "list_species", "list_poses", "params",
]

_ALIAS = "naturalform"


def _install_alias():
    """Make ``import naturalform`` work however the add-on was installed
    (Blender 4.2+ extensions live in ``bl_ext.<repo>.naturalform``)."""
    if __name__ == _ALIAS:
        return
    current = sys.modules.get(_ALIAS)
    if current is not None and current is not sys.modules[__name__]:
        return  # another package already owns the name; leave it alone
    for modname, mod in list(sys.modules.items()):
        if modname == __name__ or modname.startswith(__name__ + "."):
            sys.modules[_ALIAS + modname[len(__name__):]] = mod


def _remove_alias():
    if __name__ == _ALIAS:
        return
    for modname in [m for m in sys.modules if m == _ALIAS or m.startswith(_ALIAS + ".")]:
        mod = sys.modules[modname]
        if getattr(mod, "__name__", "").startswith(__name__):
            del sys.modules[modname]


def register():
    from . import ui
    ui.register()
    _install_alias()


def unregister():
    from . import ui
    ui.unregister()
    _remove_alias()
