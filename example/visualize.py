#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.cleaning.jhu import JHUData


def main():
    # JHU dataset
    # Show raw dataframe
    raw = JHUData("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
    print(raw)


if __name__ == "__main__":
    main()
