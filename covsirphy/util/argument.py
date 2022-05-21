#!/usr/bin/env python
# -*- coding: utf-8 -*-

from inspect import signature
from covsirphy.util.error import deprecate


@deprecate("find_args", version="2.25.0-mu")
def find_args(func_list, **kwargs):
    """
    Find values of enabled arguments of the function from the keyword arguments.

    Args:
        func_list (list[function] or function): target function
        kwargs: keyword arguments

    Returns:
        dict: dictionary of enabled arguments
    """
    if not isinstance(func_list, list):
        func_list = [func_list]
    enabled_nest = [
        list(signature(func).parameters.keys()) for func in func_list
    ]
    enabled_set = set(sum(enabled_nest, list()))
    enabled_set = enabled_set - {"self", "cls"}
    return {k: v for (k, v) in kwargs.items() if k in enabled_set}
