#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
from copy import deepcopy
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term


class _SubsetManager(Term):
    """
    Class to get subset with location information.

    Args:
        layers (list[str]): names of administration layers with the order (upper layers precede, e.g. ["Country", "Province"])
    """

    def __init__(self, layers):
        self._layers = Validator(layers, "layers").sequence(candidates=None)

    def layer(self, data, geo=None):
        """Return the data at the selected layer.

        Args:
            data (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation date
                    - columns defined by `SubsetManager(layers)` argument: note that "-" means total values of the upper layer
                    - the other columns of values
            geo (tuple(list[str] or tuple(str) or str) or str or None): location names to filter or None (top-level layer)

        Raises:
            TypeError: @geo has un-expected types
            ValueError: the length of @geo is equal to or larger than the length of layers

        Returns:
            pandas.DataFrame: as-is @data

        Note:
           Note that records with NAs as country names will be always removed.

        Note:
            When `geo=None` or `geo=(None,)`, returns country-level data, assuming we have country/province/city as layers here.

        Note:
            When `geo=("Japan",)` or `geo="Japan"`, returns province-level data in Japan.

        Note:
            When `geo=(["Japan", "UK"],)`, returns province-level data in Japan and UK.

        Note:
            When `geo=("Japan", "Kanagawa")`, returns city-level data in Kanagawa/Japan.

        Note:
            When `geo=("Japan", ["Tokyo", "Kanagawa"])`, returns city-level data in Tokyo/Japan and Kanagawa/Japan.
        """
        if geo is not None and not isinstance(geo, (list, tuple, str)):
            raise TypeError(
                f"@geo must be a tuple(list[str] or tuple(str) or str) or str or None, but {geo} was applied.")
        geo_converted = (geo,) if (geo is None or isinstance(geo, str)) else deepcopy(geo)
        df = Validator(data, "data").dataframe(columns=self._layers, empty_ok=False)
        df[self._layers] = df[self._layers].fillna(self.NA)
        df = df.loc[df[self._layers[0]] != self.NA]
        for (i, sel) in enumerate(geo_converted):
            if sel is None:
                try:
                    return df.loc[df[self._layers[i + 1]] == self.NA].reset_index(drop=True)
                except IndexError:
                    return df.reset_index(drop=True)
            if not isinstance(sel, (str, list, tuple)):
                raise TypeError(f"@geo must be a tuple(list[str] or tuple(str) or str) or None, but {geo} was applied.")
            if i >= len(self._layers):
                raise ValueError(f"The length of @geo must be smaller than that of layers, but {geo} was applied.")
            df = df.loc[df[self._layers[i]].isin([sel] if isinstance(sel, str) else sel)]
            if i == len(geo_converted) - 1 and i < len(self._layers) - 1:
                df = df.loc[df[self._layers[i + 1]] != self.NA]
                with contextlib.suppress(IndexError):
                    df = df.loc[df[self._layers[i + 2]] == self.NA]
        return df.reset_index(drop=True)

    def filter(self, data, geo=None):
        """Filter the data with geography information.

        Args:
            data (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation date
                    - columns defined by `SubsetManager(layers)` argument: note that "-" means total values of the upper layer
                    - the other columns of values
            geo (tuple(list[str] or tuple(str) or str) or str or None): location names for the layers to filter or None (all data at the top level)

        Raises:
            TypeError: @geo has un-expected types
            ValueError: the length of @geo is larger than the length of layers

        Returns:
            pandas.DataFrame: as-is @data

        Note:
           Note that records with NAs as country names will be always removed.

        Note:
            When `geo=None` or `geo=(None,)`, returns all country-level data, assuming we have country/province/city as layers here.

        Note:
            When `geo=("Japan",)` or `geo="Japan"`, returns country-level data in Japan.

        Note:
            When `geo=(["Japan", "UK"],)`, returns country-level data of Japan and UK.

        Note:
            When `geo=("Japan", "Tokyo")`, returns province-level data of Tokyo/Japan.

        Note:
            When `geo=("Japan", ["Tokyo", "Kanagawa"])`, returns province-level data of Tokyo/Japan and Kanagawa/Japan.

        Note:
            When `geo=("Japan", "Kanagawa", "Yokohama")`, returns city-level data of Yokohama/Kanagawa/Japan.

        Note:
            When `geo=(("Japan", "Kanagawa", ["Yokohama", "Kawasaki"])`, returns city-level data of Yokohama/Kanagawa/Japan and Kawasaki/Kanagawa/Japan.
        """
        if geo is None or geo == (None,) or geo == [None]:
            return self.layer(data=data, geo=None)
        if not isinstance(geo, (list, tuple, str)):
            raise TypeError(
                f"@geo must be a tuple(list[str] or tuple(str) or str) or str or None, but {geo} was applied.")
        geo_converted = (geo,) if isinstance(geo, str) else deepcopy(geo)
        if len(geo_converted) > len(self._layers):
            raise ValueError(f"The length of @geo cannot be larger than that of layers, but {geo} was applied.")
        *geo_formers, geo_last = geo_converted
        df = self.layer(data=data, geo=geo_formers or None)
        if not isinstance(geo_last, (str, list, tuple)):
            raise TypeError(
                f"The last value of @geo must be a list[str] or tuple(str) or str, but {geo_last} was applied.")
        selectors = [geo_last] if isinstance(geo_last, str) else geo_last
        return df.loc[df[self._layers[len(geo_converted) - 1]].isin(selectors)].reset_index(drop=True)
