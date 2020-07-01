#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings


def deprecate(old, new=None):
    """
    Decorator to raise deprecation warning.

    Args:
        old (str): description of the old method/function
        new (str): description of the new method/function
    """
    def _deprecate(func):
        def wrapper(*args, **kwargs):
            if new is None:
                comment = f"{old} is deprecated"
            else:
                comment = f"Please use {new} rather than {old}"
            warnings.warn(
                comment,
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return _deprecate
