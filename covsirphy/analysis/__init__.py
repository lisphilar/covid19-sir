#!/usr/bin/env python
# -*- coding: utf-8 -*-

from importlib import import_module
from pathlib import Path
import sys


# Enable to import the objects like
# from (top directory).(sub directory) import (object)
directory = Path(__file__).parent
modules = directory.glob("./*[!_].py")

for m in modules:
    m_imported = import_module(f"{__name__}.{m.stem}")
    for (k, v) in m_imported.__dict__.items():
        if not k.startswith("__"):
            sys.modules[__name__].__dict__[k] = v
