#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.util.error import deprecate
from covsirphy.util.term import Term


class PhaseSeries(Term):
    """
    A series of phases.

    Args:
        first_date (str): the first date of the series, like 22Jan2020
        last_date (str): the last date of the records, like 25May2020
        population (int): initial value of total population in the place
    """

    @deprecate("PhaseSeries", new="ODEHandler", version="2.19.1-zeta-fu1")
    def __init__(self, first_date, last_date, population):
        raise NotImplementedError
