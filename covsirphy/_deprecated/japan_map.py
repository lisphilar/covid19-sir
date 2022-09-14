#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from covsirphy.util.error import deprecate


@deprecate(old="covsirphy.jpn_map", new="JHUData.map('Japan')", version="2.16")
def jpn_map(prefectures, values, title, cmap_name="Reds", filename=None):
    """
    Show colored Japan prefecture map.

    Args:
        prefectures (list[str] or pd.Series[str]): prefecture name.
        values (int or float): value of each prefectures
        title (str): title of the figure
        cmap_name (str): Please refere to
            https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        filename (str): filename of the figure, or None (show figure)
    """
    raise NotImplementedError()
