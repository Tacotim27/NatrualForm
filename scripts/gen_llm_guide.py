"""Regenerate docs/LLM_GUIDE.md from the in-Blender help text.

    python scripts/gen_llm_guide.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from naturalform.docs import guide  # noqa: E402

HEADER = """# NaturalForm - guide for LLM agents

Paste this file into your assistant's system prompt / project instructions
(or let the assistant run `import naturalform as nf; nf.help()` inside
Blender - it prints the same text). It assumes the assistant can run Python
in Blender, e.g. through the `execute_blender_code` tool of
[blender-mcp](https://github.com/ahujasid/blender-mcp).

"""


def main():
    parts = [HEADER]
    for topic, title in ((None, "Overview"), ("custom", "Custom creatures"), ("pose", "Posing"),
                         ("materials", "Materials"), ("human", "Human parameters"),
                         ("quadruped", "Quadruped parameters"), ("bird", "Bird parameters"),
                         ("fish", "Fish parameters"), ("serpent", "Serpent parameters")):
        parts.append("## %s\n\n```text\n%s\n```\n" % (title, guide(topic).rstrip()))
    path = os.path.join(ROOT, "docs", "LLM_GUIDE.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print("wrote", path)


if __name__ == "__main__":
    main()
