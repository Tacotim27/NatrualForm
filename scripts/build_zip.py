"""Package the add-on for installation.

    python scripts/build_zip.py

Creates in dist/:
  naturalform-<version>.zip         Blender 4.2+: Edit > Preferences > Get Extensions >
                                    (v menu) Install from Disk...
  naturalform-<version>-legacy.zip  Edit > Preferences > Add-ons > Install (legacy add-on;
                                    also works in Blender 4.2+ for "legacy" installs)
"""

import os
import re
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "naturalform")


def files():
    for base, dirs, names in os.walk(PKG):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for n in names:
            if n.endswith((".pyc", ".pyo")):
                continue
            full = os.path.join(base, n)
            yield full, os.path.relpath(full, PKG)


def main():
    manifest = open(os.path.join(PKG, "blender_manifest.toml"), encoding="utf-8").read()
    version = re.search(r'^version\s*=\s*"([^"]+)"', manifest, re.M).group(1)
    out = os.path.join(ROOT, "dist")
    os.makedirs(out, exist_ok=True)
    ext = os.path.join(out, "naturalform-%s.zip" % version)
    legacy = os.path.join(out, "naturalform-%s-legacy.zip" % version)
    with zipfile.ZipFile(ext, "w", zipfile.ZIP_DEFLATED) as z:
        for full, rel in files():
            z.write(full, rel)  # manifest at the zip root (extension layout)
    with zipfile.ZipFile(legacy, "w", zipfile.ZIP_DEFLATED) as z:
        for full, rel in files():
            z.write(full, os.path.join("naturalform", rel))  # package folder (legacy layout)
    print("wrote", ext)
    print("wrote", legacy)


if __name__ == "__main__":
    main()
