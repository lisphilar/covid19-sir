#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.cleaning import CleaningBase


class JHUData(CleaningBase):
    """
    Class for data cleaning of JHU/ dataset.
    """

    def __init__(self, filename):
        super().__init__(filename)
