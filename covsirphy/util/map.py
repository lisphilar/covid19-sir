#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import japanmap
import pandas as pd


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
    # Prefecture code created in
    # https://www.kaggle.com/lisphilar/eda-of-japan-dataset
    # Primary data: http://nlftp.mlit.go.jp/ksj/gml/codelist/PrefCd.html
    # cf. https://www.japanvisitor.com/japan-travel/prefectures-map
    pref_code_dict = dict(
        [
            ('Hokkaido', 1), ('Aomori', 2), ('Iwate', 3),
            ('Miyagi', 4), ('Akita', 5), ('Yamagata', 6),
            ('Fukushima', 7), ('Ibaraki', 8), ('Tochigi', 9),
            ('Gunma', 10), ('Saitama', 11), ('Chiba', 12),
            ('Tokyo', 13), ('Kanagawa', 14), ('Niigata', 15),
            ('Toyama', 16), ('Ishikawa', 17), ('Fukui', 18),
            ('Yamanashi', 19), ('Nagano', 20), ('Gifu', 21),
            ('Shizuoka', 22), ('Aichi', 23), ('Mie', 24),
            ('Shiga', 25), ('Kyoto', 26), ('Osaka', 27), ('Hyogo', 28),
            ('Nara', 29), ('Wakayama', 30), ('Tottori', 31),
            ('Shimane', 32), ('Okayama', 33), ('Hiroshima', 34),
            ('Yamaguchi', 35), ('Tokushima', 36), ('Kagawa', 37),
            ('Ehime', 38), ('Kochi', 39), ('Fukuoka', 40),
            ('Saga', 41), ('Nagasaki', 42), ('Kumamoto', 43),
            ('Oita', 44), ('Miyazaki', 45), ('Kagoshima', 46),
            ('Okinawa', 47)
        ]
    )
    # Data to dataframe
    df = pd.DataFrame({"Name": prefectures, "Value": values})
    df["Code"] = df["Name"].map(pref_code_dict)
    df["Japanese_name"] = df["Code"].apply(lambda x: japanmap.pref_names[x])
    df = df.set_index("Japanese_name")
    # Color code
    cmap = cm.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(
        vmin=df["Value"].min(), vmax=df["Value"].max())

    def fcol(x):
        return "#" + bytes(cmap(norm(x), bytes=True)[:3]).hex()
    # Show figure
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.tick_params(length=0)
    plt.imshow(japanmap.picture(df["Value"].apply(fcol)))
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable._A = []
    plt.colorbar(mappable)
    plt.title(title)
    if filename is None:
        plt.show()
        return None
    plt.savefig(
        filename, bbox_inches="tight", transparent=False, dpi=300
    )
    plt.clf()
    return None
