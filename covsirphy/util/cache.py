#!/usr/bin/env python
# -*- coding: utf-8 -*-

def show_info(func):
    def wrapper(*args, **kwargs):
        return func.cache_info(*args, **kwargs)
    return wrapper
