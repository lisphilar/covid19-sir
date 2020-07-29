#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import wraps


def show_docstring(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
