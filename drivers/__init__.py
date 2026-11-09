"""
Game drivers (MAME: src/mame/<manufacturer>/). Each driver is one arcade machine.

DRIVERS maps the short name used on the command line to "module:Class".
"""

DRIVERS = {
    "mpatrol": "drivers.mpatrol:MoonPatrol",
}


def get_driver(name: str):
    import importlib
    module_name, class_name = DRIVERS[name].split(":")
    return getattr(importlib.import_module(module_name), class_name)
