#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.util.filer import Filer
from covsirphy.util.term import Term


class DataDownloader(Term):
    """Class to download datasets from the recommended data servers.

    Args:
        directory (str or pathlib.Path): directory to save downloaded datasets
        update_interval (int): update interval of downloading dataset
        verbose (int): level of verbosity when downloading

    Note:
        If @verbose is 0, no descriptions will be shown.
        If @verbose is 1 or larger, URL and database name will be shown.

    Note:
        Location layers are fixed to ["ISO3", "Province", "City"]
    """
    LAYERS = [Term.ISO3, Term.PROVINCE, Term.CITY]

    def __init__(self, directory="input", update_interval=12, verbose=1):
        self._filer = Filer(directory=directory)
        self._interval = self._ensure_natural_int(update_interval, include_zero=True, name="update_interval")
        self._verbose = self._ensure_natural_int(verbose, include_zero=True, name="verbose")
