#!/usr/bin/env python
# -*- coding: utf-8 -*-

import country_converter as coco
import pandas as pd
from covsirphy.util.term import Term
from covsirphy.util.filer import Filer
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
        If @verbose is 0, no description will be shown.
        If @verbose is 1 or larger, URL and database name will be shown.
        If @verbose is 2, detailed citation list will be show, if available.
    """
    # Internal term
    _ID = "Location_ID"
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
        df = self._loc_df.merge(self._rec_df, how="right", on=self._ID).drop(self._ID, axis=1)
        df = df.sort_values([*self._layers, self.DATE], ignore_index=True)
        if variables is None:
            return df
        all_variables = df.columns.tolist()
        sel_variables = self._ensure_list(target=variables, candidates=all_variables, name="variables")
        return df.drop(list(set(all_variables) - set(sel_variables)), axis=1)

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

    def manual(self, data, date="Date", data_layers=None, variables=None, citations=None, **kwargs):
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
            **kwargs: keyword arguments of pandas.to_datetime() including "dayfirst (bool): whether date format is DD/MM or not"

        Raises:
            ValueError: @data_layers has duplicates

        Returns:
            covsirphy.DataCollector: self
        """
        if len(set(data_layers)) != len(data_layers):
            raise ValueError(f"@layer has duplicates, {data_layers}")
        df = self._ensure_dataframe(
            target=data, name="data", columns=[*(data_layers or self._layers), date, *(variables or [])])
        # Convert date type
        df.rename(columns={date: self.DATE}, inplace=True)
        df[self.DATE] = pd.to_datetime(df[self.DATE], **kwargs)
        # Convert country names to ISO3 codes
        if self._country is not None and self._country in df:
            df.loc[:, self._country] = df[self._country].apply(self._to_iso3)
        # Prepare necessary layers
        if data_layers is not None and data_layers != self._layers:
            df = self._prepare_layers(df, data_layers=data_layers)
        # Locations
        loc_df = self._loc_df.merge(df, how="right", on=self._layers)
        loc_df.drop_duplicates(subset=self._layers, keep="first", ignore_index=True, inplace=True)
        loc_df.loc[loc_df[self._ID].isna(), self._ID] = f"id{len(loc_df)}-" + \
            loc_df[loc_df[self._ID].isna()].index.astype("str")
        self._loc_df = loc_df.reset_index()[[self._ID, *self._layers]].fillna(self.NA)
        # Records
        columns = [self._ID, self.DATE, *self._ensure_list(target=variables or [], name="variables")]
        df = df.merge(self._loc_df, how="left", on=self._layers)
        df = df.loc[:, columns].reset_index(drop=True)
        self._rec_df = pd.concat([self._rec_df, df], axis=0, ignore_index=True)
        # Citations
        new_citations = self._ensure_list(
            [citations] if isinstance(citations, str) else (citations or ["my own dataset"]), name="citations")
        citation_dict = {col: self._citation_dict.get(col, []) + new_citations for col in variables}
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
                    df.loc[df[data_layer] == self.NA, self._layers] = self.NA
                    self._print_v0(f"\t[INFO] Records which have NAs at '{data_layer}' layer was disabled.")
                df = df.drop(data_layer, axis=1)
                self._print_v0(f"\t[INFO] '{data_layer}' layer was removed.")
            elif layer != data_layer:
                df.rename(columns={data_layer: layer}, inplace=True)
                self._print_v0(f"\t[INFO] '{data_layer}' layer was renamed to {layer}.")
        return df.reset_index()

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

    @staticmethod
    def _to_iso3(name):
        """Convert country name to ISO3 codes.

        Args:
            name (str or list[str]): country name(s)

        Returns:
            str or list[str]: ISO3 code(s) or as-is when not found

        Note:
            "UK" will be converted to "GBR".
        """
        names = ["GBR" if elem == "UK" else elem for elem in ([name] if isinstance(name, str) else name)]
        return coco.convert(names, to="ISO3", not_found=None)

    def collect(self, geo=None, variables=None):
        """Collect necessary data from remote server and local data.

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

        """
        geo_converted = [geo] if isinstance(geo, str) else (geo or [None]).copy()
        geo_converted += [None] * (len(self._layers) - len(geo_converted))
        if self._country is not None:
            geo_converted = [
                self._to_iso3(info) if layer == self._country else info for (layer, info) in zip(self._layers, geo_converted)]
        # Collect data of the area
        all_df = self.all(variables=variables)
        geography = Geography(layers=self._layers)
        df = geography.filter(data=all_df, geo=geo_converted)
        return df.drop(self._layers, axis=1).groupby(self.DATE).sum().reset_index()

    def auto(self, iso3=None, directory="input", basename_dict=None):
        """Download datasets from remote servers automatically.

        Args:
            iso3 (str or None): ISO3 code of country that must be included or None (all available countries)
            directory (str or pathlib.Path): directory to save downloaded datasets
            basename_dict (dict[str, str]): basename of downloaded CSV files,
                "covid19dh": COVID-19 Data Hub (default: covid19dh.csv),
                "owid": Our World In Data (default: ourworldindata.csv),
                "google: COVID-19 Open Data by Google Cloud Platform (default: google_cloud_platform.csv),
                "japan": COVID-19 Dataset in Japan (default: covid_japan.csv).

        Returns:
            covsirphy.DataCollector: self
        """
        # Filenames to save remote datasets
        TITLE_DICT = {
            "covid19dh": "covid19dh",
            "owid": "ourworldindata",
            "google": "google_cloud_platform",
            "japan": "covid_japan",
        }
        filer = Filer(directory=directory, prefix=None, suffix=None, numbering=None)
        file_dict = {
            k: filer.csv(title=(basename_dict or {}).get(k, v))["path_or_buf"] for (k, v) in TITLE_DICT.items()}
        # COVID-19 Data Hub
        self.manual(**self._auto_covid19dh(iso3=iso3, path=file_dict["covid19dh"]))
        # Our World In Data
        # Google Cloud Plat Form
        # Japan dataset via CovsirPhy project
        return self

    def _auto_covid19dh(self, iso3, path):
        """Prepare records of the number of confirmed/fatal/recovered cases, the number of PCR tests and OXCGRT indicators.

        Args:
            iso3 (str or None): ISO3 code of country that must be included or None (all available countries)
            path (str): path of the CSV file to save dataset in local environment

        Returns:
            dict[str, pandas.DataFrame or str or list[str]]]
                - data (pandas.DataFrame):
                    Index
                        reset index
                    Columns
                        - Date (pandas.Timestamp): observation dates
                        - ISO3 (str): ISO3 codes of countries
                        - Province (str): province/prefecture/state name
                        - City (str): city name
                        - Confirmed (numpy.int64): the number of confirmed cases
                        - Fatal (numpy.int64): the number of fatal cases
                        - Recovered (numpy.int64): the number of recovered cases
                        - Tests (numpy.int64): the number of PCR tests
                        - School_closing (numpy.int64): one of the OxCGRT indicator
                        - Workplace_closing (numpy.int64): one of the OxCGRT indicator
                        - Cancel_events (numpy.int64): one of the OxCGRT indicator
                        - Gatherings_restrictions (numpy.int64): one of the OxCGRT indicator
                        - Transport_closing (numpy.int64): one of the OxCGRT indicator
                        - Stay_home_restrictions (numpy.int64): one of the OxCGRT indicator
                        - Internal_movement_restrictions (numpy.int64): one of the OxCGRT indicator
                        - International_movement_restrictions (numpy.int64): one of the OxCGRT indicator
                        - Information_campaigns (numpy.int64): one of the OxCGRT indicator
                        - Testing_policy (numpy.int64): one of the OxCGRT indicator
                        - Contact_tracing (numpy.int64): one of the OxCGRT indicator
                        - Stringency_index (numpy.int64): one of the OxCGRT indicator
                - data_layers (list[str]): ["ISO3", "Province", "City"]
                - citations (list[str]): citation of COVID-19 Data Hub
        """
        col_dict = {
            "date": self.DATE, "iso_alpha_3": self.ISO3,
            "administrative_area_level_2": self.PROVINCE, "administrative_area_level_3": self.CITY,
            "confirmed": self.C, "deaths": self.F, "recovered": self.R, "tests": self.TESTS, "population": self.N,
            **dict(zip(self._OXCGRT_COLS_RAW, self.OXCGRT_VARS)),
        }
        # Get raw data from server
        if iso3 is None:
            url = "https://storage.covid19datahub.io/level/1.csv.zip"
        else:
            url = f"https://storage.covid19datahub.io/country/{iso3}.csv.zip"
        df = pd.read_csv(
            url, header=0, use_cols=list(col_dict.values()), parse_dates="date", date_parser=lambda x: pd.datetime.strptime(x, "%Y-%m-%d"))
        df.rename(columns=col_dict, inplace=True)
        citation = '(Secondary source)' \
            ' Guidotti, E., Ardia, D., (2020), "COVID-19 Data Hub",' \
            ' Journal of Open Source Software 5(51):2376, doi: 10.21105/joss.02376.'
        return {"data": df, "data_layers": [self.ISO3, self.PROVINCE, self.CITY], "citations": citation}
