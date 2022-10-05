#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
import pandas as pd
from covsirphy.util.config import config
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term


class _LayerAdjuster(Term):
    """Class to adjust location layers of time-series data.

    Args:
        layers (list[str] or None): list of layers of geographic information or None (["ISO3", "Province", "City"])
        country (str or None): layer name of countries or None (countries are not included in the layers)
        date (str): column name of observation dates

    Raises:
        ValueError: @layers has duplicates

    Note:
        Country level data specified with @country will be stored with ISO3 codes.
    """
    # Internal term
    _ID = "Location_ID"

    def __init__(self, layers=None, country="ISO3", date="Date"):
        # Countries will be specified with ISO3 codes and this requires conversion
        self._country = None if country is None else str(country)
        # Layers of location information
        self._layers = Validator(layers or [self._country, self.PROVINCE, self.CITY], "layers").sequence()
        if len(set(self._layers)) != len(self._layers):
            raise ValueError(f"@layer has duplicates, {self._layers}")
        # Date column
        self._date = str(date)
        # Location data
        self._loc_df = pd.DataFrame(columns=[self._ID, *self._layers])
        # All available data
        self._rec_df = pd.DataFrame(columns=[self._ID, self._date])
        # Citations
        self._citation_dict = {}

    def all(self, variables=None):
        """Return all available data.

        Args:
            variables (list[str] or None): list of variables to collect or None (all available variables)

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - columns defined by covsirphy.LayerAdjuster(layers)
                    - Date (pandas.Timestamp): observation dates
                    - columns defined by @variables
        """
        identifiers = [*self._layers, self._date]
        df = self._loc_df.merge(self._rec_df, how="right", on=self._ID).drop(self._ID, axis=1)
        df = df.sort_values(identifiers, ignore_index=True)
        df = df.loc[:, [*identifiers, *sorted(set(df.columns) - set(identifiers), key=df.columns.tolist().index)]]
        if variables is None:
            return df
        all_variables = df.columns.tolist()
        sel_variables = Validator(variables, "variables").sequence(candidates=set(all_variables) - set(identifiers))
        return df.loc[:, [*self._layers, self._date, *sel_variables]]

    def citations(self, variables=None):
        """
        Return citation list of the secondary data sources.

        Args:
            variables (list[str] or None): list of variables to collect or None (all available variables)

        Returns:
            list[str]: citation list
        """
        all_columns = [col for col in self._rec_df.columns if col not in (self._ID, self._date)]
        columns = Validator(variables, "variables").sequence(default=all_columns, candidates=all_columns)
        return Validator([v for k, v in self._citation_dict.items() if k in columns]).sequence(flatten=True, unique=True)

    def register(self, data, layers=None, date="Date", variables=None, citations=None, convert_iso3=True, **kwargs):
        """Register new data.

        Args:
            data (pandas.DataFrame): new data
                Index
                    reset index
                Columns
                    - columns defined by @data_layers
                    - columns defined by @date
                    - columns defined by @variables
            layers (list[str] or None): layers of the data or None (as the same _LayerAdjuster(layer))
            date (str): column name of observation dates of the data
            variables (list[str] or None): list of variables to add or None (all available columns)
            citations (list[str] or str or None): citations of the dataset or None (["my own dataset"])
            convert_iso3 (bool): whether convert country names to ISO3 codes or not
            **kwargs: keyword arguments of pandas.to_datetime() including "dayfirst (bool): whether date format is DD/MM or not"

        Raises:
            ValueError: @data_layers has duplicates

        Returns:
            covsirphy.LayerAdjuster: self
        """
        data_layers = Validator(layers, "layers").sequence(default=self._layers)
        if len(set(data_layers)) != len(data_layers):
            raise ValueError(f"@layer has duplicates, {data_layers}")
        df = Validator(data, "data").dataframe(columns=[*data_layers, date, *(variables or [])])
        # Convert date type
        df.rename(columns={date: self._date}, inplace=True)
        df[self._date] = pd.to_datetime(df[self._date], **kwargs).dt.round("D")
        with contextlib.suppress(TypeError):
            df[self.DATE] = df[self.DATE].dt.tz_convert(None)
        # Convert country names to ISO3 codes
        if convert_iso3 and self._country is not None and self._country in df:
            df.loc[:, self._country] = self._to_iso3(df[self._country])
        # Prepare necessary layers and fill in None with "NA"
        if data_layers is not None and data_layers != self._layers:
            df = self._prepare_layers(df, data_layers=data_layers)
        df[self._layers] = df[self._layers].astype("string").fillna(self.NA)
        # Locations
        loc_df = df.loc[:, self._layers]
        loc_df = pd.concat([self._loc_df, loc_df], axis=0, ignore_index=True)
        loc_df.drop_duplicates(subset=self._layers, keep="first", ignore_index=True, inplace=True)
        loc_df.loc[loc_df[self._ID].isna(), self._ID] = f"id{len(loc_df)}-" + \
            loc_df[loc_df[self._ID].isna()].index.astype("str")
        self._loc_df = loc_df.reset_index()[[self._ID, *self._layers]]
        # Records
        df = df.merge(self._loc_df, how="left", on=self._layers).drop(self._layers, axis=1)
        if variables is not None:
            columns = [self._ID, self._date, *Validator(variables, "variables").sequence()]
            df = df.loc[:, columns]
        rec_df = self._rec_df.reindex(columns=list(set(self._rec_df.columns) | set(df.columns)))
        rec_df = rec_df.set_index([self._ID, self._date]).combine_first(df.set_index([self._ID, self._date]))
        self._rec_df = rec_df.reset_index()
        # Citations
        new_citations = Validator(
            [citations] if isinstance(citations, str) else (citations or ["my own dataset"]), "citations").sequence()
        citation_dict = {col: self._citation_dict.get(col, []) + new_citations for col in variables or df.columns}
        self._citation_dict.update(citation_dict)
        return self

    def _prepare_layers(self, data, data_layers):
        """Prepare necessary layers, adding NAs and renaming layers.

        Args:
            data (pandas.DataFrame): local dataset
                Index
                    reset index
                Columns
                    - columns defined by @data_layers
                    - the other columns
            data_layers (list[str]): layers of the data

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - columns defined by LayerAdjuster(layers)
                    - the other columns
        """
        df = data.copy()
        expected, actual = self._align_layers(data_layers)
        # Adjust layer names and records
        for (layer, data_layer) in zip(expected, actual):
            if data_layer is None:
                df.loc[:, layer] = self.NA
                config.info(f"\t[INFO] New layer '{layer}' was added to the data with NAs.")
            elif layer is None:
                if data_layer == actual[-1]:
                    df.loc[df[data_layer] != self.NA, self._layers] = self.NA
                    config.info(f"\t[INFO] Records which has actual values at '{data_layer}' layer were disabled.")
                df = df.drop(data_layer, axis=1)
                config.info(f"\t[INFO] '{data_layer}' layer was removed.")
            elif layer != data_layer:
                df.rename(columns={data_layer: layer}, inplace=True)
                config.info(f"\t[INFO] '{data_layer}' layer was renamed to {layer}.")
        return df.reset_index(drop=True)

    def _align_layers(self, data_layers):
        """Perform sequence alignment of the layers of new data with the layers defined by LayerAdjuster(layers).

        Args:
            data_layers (list[str]): layers of the data

        Returns:
            tuple(list[str], list[str]): list of aligned layers, that of defined and that of new data

        Note:
            Example of sequence alignment: [A, B, C], [A, C] -> [A, B, C], [A, None, C]
        """
        expected, actual = [], []
        for layer in self._layers:
            current = [data_layer for data_layer in data_layers if data_layer not in actual]
            if layer in current:
                new = current[:current.index(layer) + 1]
                expected.extend([None for _ in range(len(new) - 1)] + [layer])
                actual.extend(new)
            else:
                expected.append(layer)
                actual.append(None)
        not_included = [data_layer for data_layer in data_layers if data_layer not in actual]
        expected += [None for _ in range(len(not_included))]
        actual += not_included
        for (i, (e, a)) in enumerate(zip(expected[:], actual[:]), start=1):
            if i == len(expected):
                break
            if e is not None and expected[i] is None and a is None and actual[i] is not None:
                expected[i - 1:i + 1] = [e]
                actual[i - 1:i + 1] = [actual[i]]
        return expected, actual
