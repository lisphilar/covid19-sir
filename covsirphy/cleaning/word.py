#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Word(object):
    """
    Word definition.
    """
    # Variables of SIR-like model
    N = "Population"
    S = "Susceptible"
    C = "Confirmed"
    CI = "Infected"
    F = "Fatal"
    R = "Recovered"
    FR = "Fatal or Recovered"
    V = "Vacctinated"
    E = "Exposed"
    W = "Waiting"
    # Column names
    DATE = "Date"
    T = "Elapsed"
    COUNTRY = "Country"
    PROVINCE = "Province"
    COLUMNS = [DATE, COUNTRY, PROVINCE, C, CI, F, R]
    NLOC_COLUMNS = [DATE, C, CI, F, R]
    VALUE_COLUMNS = [C, CI, F, R]
    # Date format: 22Jan2020 etc.
    DATE_FORMAT = "%d%b%Y"
    # Separator of country and province
    SEP = "/"
    # EDA
    RATE_COLUMNS = [
        "Fatal per Confirmed",
        "Recovered per Confirmed",
        "Fatal per (Fatal or Recovered)"
    ]
