#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings


def deprecate(old, new=None, version=None):
    """
    Decorator to raise deprecation warning.

    Args:
        old (str): description of the old method/function
        new (str or None): description of the new method/function
        version (str or None): version number, like 2.7.3-alpha
    """
    def _deprecate(func):
        def wrapper(*args, **kwargs):
            if new is None:
                comment = f"{old} is deprecated, version > {version}"
            else:
                comment = f"Please use {new} rather than {old}, version > {version}"
            warnings.warn(
                comment,
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return _deprecate
