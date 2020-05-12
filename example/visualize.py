#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy import JHUData


def main():
    # JHU dataset
    jhu_file = "input/covid_19_data.csv"
    # Show raw dataframe
    jhu_data = JHUData(jhu_file)
    jhu_data.cleaned()
    jhu_data.total()
    # TODO: output the dataframe as a CSV file
    # TODO: Visualization of total data


if __name__ == "__main__":
    main()
