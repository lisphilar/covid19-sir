#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings


def deprecate(old, new=None, version=None):
    """
    Decorator to raise deprecation warning.

    Args:
        old (str): description of the old method/function
        new (str or None): description of the new method/function
        version (str or None): version number, like 2.7.3-alpha
    """
    def _deprecate(func):
        def wrapper(*args, **kwargs):
            if new is None:
                comment = f"{old} is deprecated, version > {version}"
            else:
                comment = f"Please use {new} rather than {old}, version > {version}"
            warnings.warn(
                comment,
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return _deprecate


class SubsetNotFoundError(KeyError, ValueError):
    """
    Error when subset was failed with specified arguments.

    Args:
        country (str): country name
        country_alias (str or None): country name used in the dataset
        province (str or None): province name
        start_date (str or None): start date, like 22Jan2020
        end_date (str or None): end date, like 01Feb2020
        date (str or None): specified date, like 22Jan2020
        message (str or None): the other messages
    """

    def __init__(self, country, country_alias=None, province=None,
                 start_date=None, end_date=None, date=None, message=None):
        # Area
        if country_alias is None or country == country_alias:
            c_alias_str = ""
        else:
            c_alias_str = f" ({country_alias})"
        province_str = "" if province is None else f"/{province}"
        self.area = f"{country}{c_alias_str}{province_str}"
        # Date
        if date is None:
            start_str = "" if start_date is None else f" from {start_date}"
            end_str = "" if end_date is None else f" to {end_date}"
            self.date = f"{start_str}{end_str}"
        else:
            self.date = f" on {date}"
        # The other messages
        self.message = "" if message is None else f" {message}"

    def __str__(self):
        return f"Records{self.message} in {self.area}{self.date}"
