"""Build showcase scenes with NaturalForm - the same calls an LLM would send
through an MCP ``execute_blender_code`` tool.

Run inside Blender (Text Editor > Run Script, or
``blender --python examples/gallery.py -- humans``).
Scenes: humans, animals, wildlife (birds, fish and snakes).
"""

import sys

import naturalform as nf


def humans():
    nf.create("man", name="Man", clothing="casual", top_color="teal", location=(-1.5, 0, 0), pose="walk")
    nf.create("woman", name="Woman", clothing="dress", hair_style="long", hair_color="blonde",
              skin_color="fair", location=(-0.5, 0, 0), pose="wave")
    nf.create("girl", name="Girl", age=7, clothing="shorts", hair_style="ponytail", hair_color="black",
              skin_color="brown", top_color="pink", location=(0.4, 0, 0))
    nf.create("old man", name="Grandpa", clothing="suit", beard="full", fat=0.55, location=(1.3, 0, 0),
              pose="arms_crossed")


def animals():
    nf.create("giraffe", name="Giraffe", location=(2.6, 2.2, 0), rotation=(0, 0, 35))
    nf.create("horse", name="Horse", location=(-2.4, 1.6, 0), rotation=(0, 0, -70), pose="walk")
    nf.create("cow", name="Cow", location=(0.2, 1.9, 0), rotation=(0, 0, -15))
    nf.create("dog", name="Dog", location=(-1.3, -0.4, 0), rotation=(0, 0, -30), pose="sit")
    nf.create("cat", name="Cat", location=(-0.5, -0.9, 0), rotation=(0, 0, 25))
    nf.create("sheep", name="Sheep", location=(0.6, -0.2, 0), rotation=(0, 0, 30), pose="graze")
    nf.create("pig", name="Pig", location=(1.8, -0.6, 0), rotation=(0, 0, 60))


def wildlife():
    nf.create("elephant", name="Elephant", location=(-1.6, 3.0, 0), rotation=(0, 0, -25))
    nf.create("deer", name="Deer", location=(1.4, 2.2, 0), rotation=(0, 0, 40))
    nf.create("lion", name="Lion", location=(-1.8, -0.2, 0), rotation=(0, 0, -40))
    nf.create("fox", name="Fox", location=(0.0, -0.6, 0), rotation=(0, 0, 15))
    nf.create("eagle", name="Eagle", location=(1.5, 0.6, 2.3), rotation=(0, 0, 20), pose="fly")
    nf.create("flamingo", name="Flamingo", location=(2.6, 0.2, 0), rotation=(0, 0, 30))
    nf.create("penguin", name="Penguin", location=(1.1, -0.9, 0))
    nf.create("cobra", name="Cobra", location=(-0.8, -1.3, 0), rotation=(0, 0, -10))


SCENES = {"humans": humans, "animals": animals, "wildlife": wildlife}

if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ["humans"]
    for name in argv:
        SCENES[name]()
