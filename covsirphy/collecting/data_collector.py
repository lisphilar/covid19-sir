#!/usr/bin/env python
# -*- coding: utf-8 -*-

import country_converter as coco
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
        If @verbose is 0, no description will be shown.
        If @verbose is 1 or larger, URL and database name will be shown.
        If @verbose is 2, detailed citation list will be show, if available.
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
        self._rec_df = self._rec_df.combine_first(df)
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
            name (str or list[str] or None): country name(s)

        Returns:
            str or list[str] or None: ISO3 code(s) or as-is when not found

        Note:
            "UK" will be converted to "GBR".
        """
        if name is None:
            return None
        names = ["GBR" if elem == "UK" else elem for elem in ([name] if isinstance(name, str) else name)]
        return coco.convert(names, to="ISO3", not_found=None)

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
        geo_converted = [geo] if isinstance(geo, str) else (geo or [None]).copy()
        geo_converted += [None] * (len(self._layers) - len(geo_converted))
        if self._country is not None:
            geo_converted = [
                self._to_iso3(info) if layer == self._country else info for (layer, info) in zip(self._layers, geo_converted)]
        all_df = self.all(variables=variables)
        if all_df.empty:
            return all_df
        geography = Geography(layers=self._layers)
        df = geography.filter(data=all_df, geo=geo_converted)
        return df.drop(self._layers, axis=1).groupby(self.DATE).sum().reset_index()

    def auto(self, geo=None):
        """Download datasets of the country specified with geographic information from remote servers automatically.

        Args:
            geo (tuple(list[str] or tuple(str) or str)): location names defined in covsirphy.Geography class

        Returns:
            covsirphy.DataCollector: self
        """
        if geo is None or self._country is None:
            iso3 = None
        else:
            iso3 = geo if isinstance(geo, str) else geo[self._layers.index(self._country)]
        # COVID-19 Data Hub
        dh_dict = self._auto_covid19dh(iso3=iso3)
        place_df = dh_dict.pop("places")
        self.manual(**dh_dict)
        # Google Cloud Plat Form
        self.manual(**self._auto_google(place_df=place_df))
        # Our World In Data
        self.manual(**self._auto_owid())
        # Japan dataset in CovsirPhy project
        if iso3 == "JPN":
            self.manual(**self._auto_cs_japan())
        return self

    def _read_csv(self, filepath_or_buffer, col_dict, date="date", date_format="%Y-%m-%d"):
        """Read CSV data.

        Args:
            filepath_or_buffer (str, path object or file-like object): file path or URL
            col_dict (dict[str, str]): dictionary to convert column names
            date (str): column name of date
            date_format (str): format of date column, like %Y-%m-%d
        """
        df = pd.read_csv(
            filepath_or_buffer, header=0, use_cols=list(col_dict.keys()),
            parse_dates=date, date_parser=lambda x: pd.datetime.strptime(x, date_format))
        return df.rename(columns=col_dict)

    def _auto_covid19dh(self, iso3):
        """Download records from "COVID-19 Data Hub" server.
        https://covid19datahub.io/

        Args:
            iso3 (str or None): ISO3 code of country that must be included or None (all available countries)

        Returns:
            dict[str, pandas.DataFrame or str or list[str]]]
                - data (pandas.DataFrame):
                    Index
                        reset index
                    Columns
                        - Date (pandas.Timestamp): observation dates
                        - self._country (str) or ISO3 (str): ISO3 codes of countries
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
                - data_layers (list[str]): [self._country (str) or "ISO3", "Province", "City"]
                - citations (list[str]): citation of "COVID-19 Data Hub"
                - place_data (pandas.DataFrame):
                    Index
                        reset index
                    Columns
                        - self._country (str) or ISO3 (str): ISO3 codes of countries
                        - Province (str): province/prefecture/state name
                        - City (str): city name
                        - Google_ID (str): the place_id used in Google Mobility Reports.

        Note:
            Regarding Google_ID, refer to https://github.com/GoogleCloudPlatform/covid-19-open-data/blob/main/docs/table-index.md
             and https://www.google.com/covid19/mobility/
        """
        country = self._country or self.ISO3
        col_dict = {
            "date": self.DATE, "iso_alpha_3": country,
            "administrative_area_level_2": self.PROVINCE, "administrative_area_level_3": self.CITY,
            "confirmed": self.C, "deaths": self.F, "recovered": self.R, "tests": self.TESTS, "population": self.N,
            "key_google_mobility": self._GOOGLE_ID,
            **dict(zip(self._OXCGRT_COLS_RAW, self.OXCGRT_VARS)),
        }
        # Get raw data from server
        if iso3 is None:
            url = "https://storage.covid19datahub.io/level/1.csv.zip"
        else:
            url = f"https://storage.covid19datahub.io/country/{iso3}.csv.zip"
        df = self._read_csv(url, col_dict=col_dict, date="date", date_format="%Y-%m-%d")
        # Citation
        citation = '(Secondary source)' \
            ' Guidotti, E., Ardia, D., (2020), "COVID-19 Data Hub",' \
            ' Journal of Open Source Software 5(51):2376, doi: 10.21105/joss.02376.'
        # Google IDs
        place_df = df[[country, self.PROVINCE, self.CITY, self._GOOGLE_ID]].drop_duplicates(ignore_index=True)
        df.drop(self._GOOGLE_ID, axis=1, inplace=True)
        return {"data": df, "data_layers": [country, self.PROVINCE, self.CITY], "citations": citation, "place_data": place_df}

    def _auto_google(self, place_df):
        """Download records from "Google Cloud Platform - COVID-19 Open-Data" server.
        https://github.com/GoogleCloudPlatform/covid-19-open-data

        Args:
            place_data (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - self._country (str) or ISO3 (str): ISO3 codes of countries
                    - Province (str): province/prefecture/state name
                    - City (str): city name
                    - Google_ID (str): the place_id used in Google Mobility Reports.

        Returns:
            dict[str, pandas.DataFrame or str or list[str]]]
                - data (pandas.DataFrame):
                    Index
                        reset index
                    Columns
                        - Date (pandas.Timestamp): observation dates
                        - self._country (str) or ISO3 (str): ISO3 codes of countries
                        - Province (str): province/prefecture/state name
                        - City (str): city name
                        - Mobility_grocery_and_pharmacy: % to baseline in visits (grocery markets, pharmacies etc.)
                        - Mobility_parks: % to baseline in visits (parks etc.)
                        - Mobility_transit_stations: % to baseline in visits (public transport hubs etc.)
                        - Mobility_retail_and_recreation: % to baseline in visits (restaurant, museums etc.)
                        - Mobility_residential: % to baseline in visits (places of residence)
                        - Mobility_workplaces: % to baseline in visits (places of work)
                - data_layers (list[str]): [self._country (str) or "ISO3", "Province", "City"]
                - citations (list[str]): citation of "Google Cloud Platform - COVID-19 Open-Data"
        """
        country = self._country or self.ISO3
        # Convert place_id to location_key
        index_url = "https://storage.googleapis.com/covid19-open-data/v3/index.csv"
        key_df = self._read_csv(
            index_url, col_dict=dict.fromkeys(["location_key", "place_id"]), date=None, date_format=None)
        key_dict = key_df.set_index("place_id").to_dict()
        keys = [key_dict.get(place) for place in place_df[self._GOOGLE_ID].unique()]
        # Get records
        col_dict = {
            "date": self.DATE, "place_id": self._GOOGLE_ID, **dict(zip(self._MOBILITY_COLS_RAW, self.MOBILITY_VARS))}
        if place_df[country].nunique() == 1:
            dataframes = []
            for key in keys:
                if key is None:
                    continue
                url = f"https://storage.googleapis.com/covid19-open-data/v3/location/{key}.csv"
                new_df = self._read_csv(url, col_dict=col_dict, date="date", date_format="%Y-%m-%d")
                dataframes.append(new_df)
            df = pd.concat(dataframes, axis=0, ignore_index=True)
        else:
            url = "https://storage.googleapis.com/covid19-open-data/v3/mobility.csv"
            df = self._read_csv(url, col_dict=col_dict, date="date", date_format="%Y-%m-%d")
        # Arrange data
        df = (df.set_index([self._ID, self._GOOGLE_ID]) + 100).reset_index()
        df = df.merge(place_df, how="left", on=self._GOOGLE_ID).drop(self._GOOGLE_ID, axis=1)
        # Citation
        citation = "O. Wahltinez and others (2020)," \
            " COVID-19 Open-Data: curating a fine-grained, global-scale data repository for SARS-CoV-2, " \
            " Work in progress, https://goo.gle/covid-19-open-data"
        return {"data": df, "data_layers": [country, self.PROVINCE, self.CITY], "citations": citation}

    def _auto_owid(self):
        """Download records from "Our World In Data" server.
        https://github.com/owid/covid-19-data/tree/master/public/data
        https://ourworldindata.org/coronavirus

        Returns:
            dict[str, pandas.DataFrame or str or list[str]]]
                - data (pandas.DataFrame):
                    Index
                        reset index
                    Columns
                        - Date (pandas.Timestamp): observation dates
                        - self._country (str) or ISO3 (str): ISO3 codes of countries
                        - Vaccinations (int): cumulative number of vaccinations
                        - Vaccinations_boosters (int): cumulative number of booster vaccinations
                        - Vaccinated_once (int): cumulative number of people who received at least one vaccine dose
                        - Vaccinated_full (int): cumulative number of people who received all doses prescrived by the protocol
                        - Tests: the number of tests
                - data_layers (list[str]): [self._country (str) or "ISO3"]
                - citations (list[str]): citation of "Our World In Data"
        """
        country = self._country or self.ISO3
        # Vaccinations
        v_col_dict = {
            "date": self.DATE, "iso_code": country,
            "total_vaccinations": self.VAC, "total_boosters": self.VAC_BOOSTERS,
            "people_vaccinated": self.V_ONCE, "people_fully_vaccinated": self.V_FULL,
        }
        URL_V = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv"
        v_df = self._read_csv(URL_V, col_dict=v_col_dict, date="date", date_format="%Y-%m-%d")
        # PCR tests
        p_col_dict = {
            "Date": self.DATE, "ISO code": country, "Cumulative total": self.TESTS, }
        URL_P = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/testing/covid-testing-all-observations.csv"
        p_df = self._read_csv(URL_P, col_dict=p_col_dict, date="Date", date_format="%Y-%m-%d")
        # Merge datasets
        df = v_df.merge(p_df, how="outer", on=[country, self.DATE])
        df = df.loc[~df[country].str.contains("OWID_")]
        # Citation
        citation = "Hasell, J., Mathieu, E., Beltekian, D. et al." \
            " A cross-country database of COVID-19 testing. Sci Data 7, 345 (2020)." \
            " https://doi.org/10.1038/s41597-020-00688-8"
        return {"data": df, "data_layers": [country], "citations": citation}

    def _auto_cs_japan(self):
        """Download records from "CovsirPhy project - COVID-19 Dataset in Japan" server.
        https://github.com/lisphilar/covid19-sir/tree/master/data

        Returns:
            dict[str, pandas.DataFrame or str or list[str]]]
                - data (pandas.DataFrame):
                    Index
                        reset index
                    Columns
                        - Date (pandas.Timestamp): observation dates
                        - self._country (str) or ISO3 (str): "JPN"
                        - Prefecture (str): '-' (country level), 'Entering' or province names
                        - Confirmed (int): the number of confirmed cases
                        - Fatal (int): the number of fatal cases
                        - Recovered (int): the number of recovered cases
                        - Tests (int): the number of tested persons
                        - Moderate (int): the number of cases who requires hospitalization but not severe
                        - Severe (int): the number of severe cases
                        - Vaccinations (int): cumulative number of vaccinations
                        - Vaccinations_boosters (int): cumulative number of booster vaccinations
                        - Vaccinated_once (int): cumulative number of people who received at least one vaccine dose
                        - Vaccinated_full (int): cumulative number of people who received all doses prescrived by the protocol
                - data_layers (list[str]): [self._country (str) or "ISO3", "Prefecture"]
                - citations (list[str]): citation of "CovsirPhy project - COVID-19 Dataset in Japan"
        """
        country = self._country or self.ISO3
        prefecture = "Prefecture"
        GITHUB_URL = "https://raw.githubusercontent.com"
        URL_C = f"{GITHUB_URL}/lisphilar/covid19-sir/master/data/japan/covid_jpn_total.csv"
        URL_P = f"{GITHUB_URL}/lisphilar/covid19-sir/master/data/japan/covid_jpn_prefecture.csv"
        # Country-level data
        c_col_dict = {
            "Date": self.DATE, "Location": "Location",
            "Positive": self.C, "Fatal": self.F, "Discharged": self.R, "Tested": self.TESTS,
            "Hosp_require": "Hosp_require", "Hosp_severe": self.SEVERE,
            "Vaccinated_1st": "Vaccinated_1st", "Vaccinated_2nd": "Vaccinated_2nd", "Vaccinated_3rd": "Vaccinated_3rd",
        }
        c_df = self.read_csv(URL_C, use_cols=c_col_dict, date="Date", date_format="%Y-%m-%d")
        c_df = c_df.groupby(self.DATE).sum().reset_index()
        c_df[prefecture] = self.NA
        # Prefecture-level data
        p_col_dict = {
            "Date": self.DATE, "Prefecture": prefecture,
            "Positive": self.C, "Fatal": self.F, "Discharged": self.R, "Tested": self.TESTS,
            "Hosp_require": "Hosp_require", "Hosp_severe": self.SEVERE,
        }
        p_df = self.read_csv(URL_P, use_cols=p_col_dict, date="Date", date_format="%Y-%m-%d")
        # Concatenate datasets
        df = pd.concat([c_df, p_df], axis=1, ignore_index=True, sort=True)
        df[country] = "JPN"
        df[self.MODERATE] = df["Hosp_require"] - df[self.SEVERE]
        df[self.V_ONCE] = df["Vaccinated_1st"].cumsum()
        df[self.V_FULL] = df["Vaccinated_2nd"].cumsum()
        df[self.VAC_BOOSTERS] = df["Vaccinated_3rd"].cumsum()
        df[self.VAC] = df[[self.V_ONCE, self.V_FULL, self.VAC_BOOSTERS]].sum(axis=1)
        columns = [
            self.DATE, country, prefecture, self.C, self.F, self.R, self.TESTS,
            self.MODERATE, self.SEVERE, self.VAC, self.VAC_BOOSTERS, self.V_ONCE, self.V_FULL,
        ]
        df = df.loc[:, columns]
        # Citation
        citation = "Hirokazu Takaya (2020-2022), COVID-19 dataset in Japan, GitHub repository, " \
            "https://github.com/lisphilar/covid19-sir/data/japan"
        return {"data": df, "data_layers": [country, prefecture], "citations": citation}
