"""Run the test suite.

    python tests/run_tests.py                         # with the `bpy` wheel, or
    blender -b --factory-startup --python tests/run_tests.py

Blender-only tests are skipped automatically when ``bpy`` is unavailable.
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))


def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromName("test_core"))
    suite.addTests(loader.loadTestsFromName("test_templates"))
    try:
        import bpy  # noqa: F401
        suite.addTests(loader.loadTestsFromName("test_blender"))
    except ImportError:
        print("bpy not available - skipping Blender integration tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    main()
