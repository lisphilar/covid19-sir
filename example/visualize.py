#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.cleaning.jhu import JHUData


def main():
    # JHU dataset
    jhu_file = "input/covid_19_data.csv"
    # Show raw dataframe
    jhu_data = JHUData(jhu_file)
    print(jhu_data.raw.tail())


if __name__ == "__main__":
    main()
