#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
import pandas as pd
from covsirphy.util.term import Term
from covsirphy.collecting.geography import Geography


class DataCollector(Term):
    """Class for collecting data for the specified location.

    Args:
        layers (list[str] or None): list of layers of geographic information or None (["ISO3", "Province", "City"])
        country (str or None): layer name of countries or None (countries are not included in the layers)
        update_interval (int): update interval of downloading dataset
        verbose (int): level of verbosity when downloading

    Raises:
        ValueError: @layers has duplicates

    Note:
        Country level data specified with @country will be stored with ISO3 codes.

    Note:
        If @update_interval hours have passed since the last update of downloaded datasets,
        the dawnloaded datasets will be updated automatically.

    Note:
        If @verbose is 0, no descriptions will be shown.
        If @verbose is 1 or larger, URL and database name will be shown.
    """
    # Internal term
    _ID = "Location_ID"
    _GOOGLE_ID = "Google_ID"
    # OxCGRT Indicators
    _OXCGRT_COLS_RAW = [
        "school_closing",
        "workplace_closing",
        "cancel_events",
        "gatherings_restrictions",
        "transport_closing",
        "stay_home_restrictions",
        "internal_movement_restrictions",
        "international_movement_restrictions",
        "information_campaigns",
        "testing_policy",
        "contact_tracing",
        "stringency_index",
    ]
    OXCGRT_VARS = [v.capitalize() for v in _OXCGRT_COLS_RAW]
    # Mobility indicators
    _MOBILITY_COLS_RAW = [
        "mobility_grocery_and_pharmacy",
        "mobility_parks",
        "mobility_transit_stations",
        "mobility_retail_and_recreation",
        "mobility_residential",
        "mobility_workplaces",
    ]
    MOBILITY_VARS = [v.capitalize() for v in _MOBILITY_COLS_RAW]

    def __init__(self, layers=None, country="ISO3", update_interval=12, verbose=1):
        self._update_interval = self._ensure_natural_int(update_interval, name="update_interval", include_zero=True)
        self._verbose = self._ensure_natural_int(verbose, name="verbose", include_zero=True)
        # Countries will be specified with ISO3 codes and this requires conversion
        self._country = None if country is None else str(country)
        # Location data
        self._layers = self._ensure_list(target=layers or [self._country, self.PROVINCE, self.CITY], name="layers")
        if len(set(self._layers)) != len(self._layers):
            raise ValueError(f"@layer has duplicates, {self._layers}")
        self._loc_df = pd.DataFrame(columns=[self._ID, *self._layers])
        # All available data
        self._rec_df = pd.DataFrame(columns=[self._ID, self.DATE])
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
                    - columns defined by covsirphy.DataCollector(layers)
                    - Date (pandas.Timestamp): observation dates
                    - columns defined by @variables
        """
        identifiers = [*self._layers, self.DATE]
        df = self._loc_df.merge(self._rec_df, how="right", on=self._ID).drop(self._ID, axis=1)
        df = df.sort_values(identifiers, ignore_index=True)
        df = df.loc[:, [*identifiers, *sorted(set(df.columns) - set(identifiers), key=df.columns.tolist().index)]]
        if variables is None:
            return df
        all_variables = df.columns.tolist()
        sel_variables = self._ensure_list(target=variables, candidates=all_variables, name="variables")
        return df.drop(list(set(all_variables) - set(sel_variables)), axis=1)

    def subset(self, geo=None, variables=None):
        """Create a subset with the geographic information.

        Args:
            geo (tuple(list[str] or tuple(str) or str)): location names defined in covsirphy.Geography class
            variables (list[str] or None): list of variables to collect or None (all available variables)

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation dates
                    - columns defined by @variables

        Note:
            Please refer to covsirphy.Geography.filter() regarding @geo argument.

        Note:
            Layers will be dropped from the dataframe.
        """
        geo_converted = self._geo_with_iso3(geo=geo)
        all_df = self.all(variables=variables)
        if all_df.empty:
            return all_df
        geography = Geography(layers=self._layers)
        df = geography.filter(data=all_df, geo=geo_converted)
        return df.drop(self._layers, axis=1).groupby(self.DATE).first().reset_index()

    def _geo_with_iso3(self, geo=None):
        """Update the geographic information, converting country names to ISO3 codes.

        Args:
            geo (tuple(list[str] or tuple(str) or str)): location names defined in covsirphy.Geography class

        Returns:
            list[str] or tuple(str) or str): location names defined in covsirphy.Geography class
        """
        geo_converted = [geo] if isinstance(geo, str) else list(geo or [None])
        geo_converted += [None] * (len(self._layers) - len(geo_converted))
        if self._country is not None:
            geo_converted = [
                self._to_iso3(info) if layer == self._country else info for (layer, info) in zip(self._layers, geo_converted)]
        return [info for info in geo_converted if info is not None] or None

    def citations(self, variables=None):
        """
        Return citation list of the secondary data sources.

        Args:
            variables (list[str] or None): list of variables to collect or None (all available variables)

        Returns:
            list[str]: citation list
        """
        all_columns = [col for col in self._rec_df.columns if col not in (self._ID, self.DATE)]
        columns = self._ensure_list(target=variables or all_columns, candidates=all_columns, name="variables")
        return self.flatten([v for (k, v) in self._citation_dict.items() if k in columns], unique=True)

    def manual(self, data, date="Date", data_layers=None, variables=None, citations=None, convert_iso3=True, **kwargs):
        """Add data manually.

        Args:
            data (pandas.DataFrame): local dataset or None (un-available)
                Index
                    reset index
                Columns
                    - columns defined by @data_layers
                    - columns defined by @date
                    - columns defined by @variables
            date (str): column name of date
            data_layers (list[str]): layers of the data
            variables (list[str] or None): list of variables to add or None (all available columns)
            citations (list[str] or str or None): citations of the dataset or None (["my own dataset"])
            convert_iso3 (bool): whether convert country names to ISO3 codes or not
            **kwargs: keyword arguments of pandas.to_datetime() including "dayfirst (bool): whether date format is DD/MM or not"

        Raises:
            ValueError: @data_layers has duplicates

        Returns:
            covsirphy.DataCollector: self
        """
        if len(set(data_layers)) != len(data_layers):
            raise ValueError(f"@layer has duplicates, {data_layers}")
        self._ensure_dataframe(
            target=data, name="data", columns=[*(data_layers or self._layers), date, *(variables or [])])
        df = data.copy()
        # Convert date type
        df.rename(columns={date: self.DATE}, inplace=True)
        df[self.DATE] = pd.to_datetime(df[self.DATE], **kwargs)
        # Convert country names to ISO3 codes
        if convert_iso3 and self._country is not None and self._country in df:
            df.loc[:, self._country] = self._to_iso3(df[self._country])
        # Prepare necessary layers and fill in None with "NA"
        if data_layers is not None and data_layers != self._layers:
            df = self._prepare_layers(df, data_layers=data_layers)
        df.loc[:, self._layers] = df[self._layers].fillna(self.NA)
        # Locations
        loc_df = df.loc[:, self._layers]
        loc_df = pd.concat([self._loc_df, loc_df], axis=0, ignore_index=True)
        loc_df.drop_duplicates(subset=self._layers, keep="first", ignore_index=True, inplace=True)
        loc_df.loc[loc_df[self._ID].isna(), self._ID] = f"id{len(loc_df)}-" + \
            loc_df[loc_df[self._ID].isna()].index.astype("str")
        self._loc_df = loc_df.reset_index()[[self._ID, *self._layers]]
        # Records
        df = df.merge(self._loc_df, how="left", on=self._layers).drop(self._layers, axis=1).dropna()
        if variables is not None:
            columns = [self._ID, self.DATE, *self._ensure_list(target=variables, name="variables")]
            df = df.loc[:, columns]
        rec_df = self._rec_df.reindex(columns=list(set(self._rec_df.columns) | set(df.columns)))
        rec_df = rec_df.set_index([self._ID, self.DATE]).combine_first(df.set_index([self._ID, self.DATE]))
        self._rec_df = rec_df.reset_index()
        # Citations
        new_citations = self._ensure_list(
            [citations] if isinstance(citations, str) else (citations or ["my own dataset"]), name="citations")
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
                    - columns defined by DataCollector(layers)
                    - the other columns
        """
        df = data.copy()
        expected, actual = self._align_layers(data_layers)
        # Adjust layer names and records
        for (layer, data_layer) in zip(expected, actual):
            if data_layer is None:
                df.loc[:, layer] = self.NA
                self._print_v0(f"\t[INFO] New layer '{layer}' was added to the data with NAs.")
            elif layer is None:
                if data_layer == actual[-1]:
                    df.loc[df[data_layer] != self.NA, self._layers] = self.NA
                    self._print_v0(f"\t[INFO] Records which has actual values at '{data_layer}' layer were disabled.")
                df = df.drop(data_layer, axis=1)
                self._print_v0(f"\t[INFO] '{data_layer}' layer was removed.")
            elif layer != data_layer:
                df.rename(columns={data_layer: layer}, inplace=True)
                self._print_v0(f"\t[INFO] '{data_layer}' layer was renamed to {layer}.")
        return df.reset_index(drop=True)

    def _align_layers(self, data_layers):
        """Perform sequence alignment of the layers of new data with the layers defined by DataCollector(layers).

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

    def _print_v0(self, sentence):
        """Stdout the sentence when the verbosity level was set to 1 or higher.

        Args:
            sentence (str): the sentence to stdout
        """
        if self._verbose:
            print(sentence)

    def auto(self, geo=None):
        """Download datasets of the country specified with geographic information from remote servers automatically.

        Args:
            geo (tuple(list[str] or tuple(str) or str)): location names defined in covsirphy.Geography class

        Returns:
            covsirphy.DataCollector: self
        """
        if geo is None or self._country is None or self._country not in self._layers:
            iso3, geo_converted = None, deepcopy(geo)
        else:
            geo_converted = self._geo_with_iso3(geo=geo)
            name = geo_converted if isinstance(geo_converted, str) else geo_converted[self._layers.index(self._country)]
            iso3 = self._to_iso3(name)
        # COVID-19 Data Hub
        dh_dict = self._auto_covid19dh(iso3=iso3)
        place_df = dh_dict.pop("place_data").fillna(self.NA)
        geography = Geography(layers=[self._country or self.ISO3, self.PROVINCE, self.CITY])
        sel_df = geography.filter(data=place_df, geo=geo_converted)
        self.manual(**dh_dict)
        # Google Cloud Plat Form
        self.manual(**self._auto_google(place_df=sel_df))
        # Our World In Data
        self.manual(**self._auto_owid())
        return self
