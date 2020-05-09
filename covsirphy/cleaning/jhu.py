#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.cleaning.cbase import CleaningBase

__all__ = ["JHUData"]


class JHUData(CleaningBase):
    """
    Class for data cleaning of JHU/ dataset.
    """

    def __init__(self, filename):
        super().__init__(filename)
