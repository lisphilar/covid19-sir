#!/usr/bin/env python
# -*- coding: utf-8 -*-


from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
from covsirphy.util.term import Term
from covsirphy._deprecated.db_cs_japan import _CSJapan
from covsirphy._deprecated.db_covid19dh import _COVID19dh
from covsirphy._deprecated.db_owid import _OWID
from covsirphy._deprecated.db_google import _GoogleOpenData


class _Recommended(Term):
    """
    Download datasets from remote servers.

    Args:
        directory (str or pathlib.Path): directory to save downloaded datasets
        update_interval (int): update interval of downloading dataset
        country (str or None): country name of datasets to downloaded when @update_interval is an integer
        file_dict (dict[str, str] ot None): filename of downloaded CSV files,
            "covid19dh": COVID-19 Data Hub,
            "owid": Our World In Data,
            "google: COVID-19 Open Data by Google Cloud Platform,
            "wbdata": World Bank Open Data,
            "japan": COVID-19 Dataset in Japan.
        verbose (int): level of verbosity when downloading

    Note:
        If @verbose is 0, no descriptions will be shown.
        If @verbose is 1 or larger, URL and database name will be shown.
        If @verbose is 2, detailed citation list will be show, if available.
    """

    def __init__(self, update_interval, country, file_dict, verbose):
        # Update interval of the downloaded files
        self._update_interval = self._ensure_natural_int(update_interval, name="update_interval", include_zero=True)
        # Country names for automated downloading: None (all countries) or list[str]
        self._iso3_code = None if country is None else self._to_iso3(country)[0]
        # Dictionary of filenames to save remote datasets
        self._file_dict = file_dict.copy()
        # Verbosity
        self._verbose = self._ensure_natural_int(verbose, name="verbose", include_zero=True)
        # Column names to identify records
        self._id_cols = [self.ISO3, self.PROVINCE, self.DATE]

    @staticmethod
    def _ensure_natural_int(target, name="number", include_zero=False, none_ok=False):
        """
        Ensure a natural (non-negative) number.

        Args:
            target (int or float or str or None): value to ensure
            name (str): argument name of the value
            include_zero (bool): include 0 or not
            none_ok (bool): None value can be applied or not.

        Returns:
            int: as-is the target

        Note:
            When @target is None and @none_ok is True, None will be returned.
            If the value is a natural number and the type was float or string,
            it will be converted to an integer.
        """
        if target is None and none_ok:
            return None
        s = f"@{name} must be a natural number, but {target} was applied"
        try:
            number = int(target)
        except TypeError as e:
            raise TypeError(f"{s} and not converted to integer.") from e
        if number != target:
            raise ValueError(f"{s}. |{target} - {number}| > 0")
        min_value = 0 if include_zero else 1
        if number < min_value:
            raise ValueError(f"{s}. This value is under {min_value}")
        return number

    def retrieve(self):
        """
        Retrieve datasets from remote servers.

        Returns:
            tuple(pandas/DataFrame, dict[str, str]):
                pandas.DataFrame:
                    Index
                        reset index
                    Columns
                        - Date (str): column name for dates
                        - Country (str): country names (top level administration)
                        - Province (str): province names (2nd level administration)
                        - ISO3 (str): ISO3 codes
                        - Confirmed (numpy.float64): the number of confirmed cases
                        - Fatal (numpy.float64): the number of fatal cases
                        - Recovered (numpy.float64): the number of recovered cases
                        - Population (numpy.int64): population values
                        - Tests (numpy.float64): the number of tests
                        - Product (numpy.float64): vaccine product names
                        - Vaccinations (numpy.float64): cumulative number of vaccinations
                        - Vaccinations_boosters (numpy.float64): cumulative number of booster vaccinations
                        - Vaccinated_once (numpy.float64): cumulative number of people who received at least one vaccine dose
                        - Vaccinated_full (numpy.float64): cumulative number of people who received all doses prescribed by the protocol
                        - School_closing
                        - Workplace_closing
                        - Cancel_events
                        - Gatherings_restrictions
                        - Transport_closing
                        - Stay_home_restrictions
                        - Internal_movement_restrictions
                        - International_movement_restrictions
                        - Information_campaigns
                        - Testing_policy
                        - Contact_tracing
                        - Stringency_index
                        - Mobility_grocery_and_pharmacy: % to baseline in visits (grocery markets, pharmacies etc.)
                        - Mobility_parks: % to baseline in visits (parks etc.)
                        - Mobility_transit_stations: % to baseline in visits (public transport hubs etc.)
                        - Mobility_retail_and_recreation: % to baseline in visits (restaurant, museums etc.)
                        - Mobility_residential: % to baseline in visits (places of residence)
                        - Mobility_workplaces: % to baseline in visits (places of work)


        """
        variables = [
            self.COUNTRY, self.C, self.F, self.R, self.N, self.TESTS,
            self.PRODUCT, self.VAC, self.VAC_BOOSTERS, self.V_ONCE, self.V_FULL,
            *_COVID19dh.OXCGRT_VARS, *_GoogleOpenData.MOBILITY_VARS,
        ]
        all_columns = [*self._id_cols, *variables]
        df = pd.DataFrame(columns=all_columns, index=self._id_cols)
        citation_dict = dict.fromkeys(variables, [])
        # COVID-19 Dataset in Japan
        if self._iso3_code is None or self._iso3_code == "JPN":
            japan_filename = self._file_dict["japan"]
            df, citation_dict = self._add_remote(df, _CSJapan, japan_filename, citation_dict)
        # COVID19 Data Hub
        dh_filename = self._file_dict["covid19dh"]
        df, citation_dict = self._add_remote(df, _COVID19dh, dh_filename, citation_dict)
        # Our World In Data
        owid_filename = self._file_dict["owid"]
        df, citation_dict = self._add_remote(df, _OWID, owid_filename, citation_dict)
        # COVID-19 Open Data by Google Cloud Platform
        google_filename = self._file_dict["google"]
        df, citation_dict = self._add_remote(df, _GoogleOpenData, google_filename, citation_dict)
        # Select country data
        df = df.reset_index()
        if self._iso3_code is not None:
            df = df.loc[df[self.ISO3] == self._iso3_code]
        return (df, citation_dict)

    @staticmethod
    def _last_updated_local(path):
        """
        Return the date last updated of local file/directory.

        Args:
            path (str or pathlibPath): name of the file/directory

        Returns:
            (datetime.datetime): time last updated (UTC)
        """
        m_time = Path(path).stat().st_mtime
        date = datetime.fromtimestamp(m_time)
        return date.astimezone(timezone.utc).replace(tzinfo=None)

    def _download_necessity(self, filename):
        """
        Return whether we need to get the data from remote servers or not,
        comparing the last update of the files.

        Args:
            filename (str): filename of the local file

        Returns:
            (bool): whether we need to get the data from remote servers or not

        Note:
            If the last updated date is unknown, returns True.
            If @update_interval (of _Recommended) hours have passed and the remote file was updated, return True.
        """
        if not Path(filename).exists():
            return True
        date_local = self._last_updated_local(filename)
        time_limit = date_local + timedelta(hours=self._update_interval)
        return datetime.now() > time_limit

    def _add_remote(self, current_df, remote_handler, filename, citation_dict):
        """
        Update null elements of the current database with values in the same date/area in remote dataset.

        Args:
            current_df (pandas.DataFrame): the current database (index: Date, Country, Province)
            remote_handler (covsirphy.loading.db_base._RemoteDatabase): remote database handler class object
            filename (str): filename to save the dataframe retrieved from remote server
            citation_dict (dict[str, list[str]]): dictionary of citation for each variable (column)

        Returns:
            tuple(pandas.DataFrame, list[str]):
                updated database and citations
        """
        df = self._set_date_location(current_df)
        # Get the remote dataset
        force = self._download_necessity(filename)
        handler = remote_handler(filename, self._iso3_code)
        remote_df = self._set_date_location(handler.to_dataframe(force=force, verbose=self._verbose))
        # Update the current database
        df = df.combine_first(remote_df)
        # Update citations
        cite_dict = {k: [*v, handler.CITATION] if k in remote_df else v for (k, v) in citation_dict.items()}
        return (df, cite_dict)

    def _set_date_location(self, data):
        """
        Set date column and location columns.

        Args:
            data (pandas.DataFrame): dataframe to update (itself will be updated)

        Returns:
            pandas.DataFrame:
                Index
                    Date, ISO3, Province
                Columns
                    as-is
        """
        df = data.reset_index().drop("index", axis=1, errors="ignore")
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        df[self.ISO3] = df[self.ISO3].fillna(self.NA)
        if self.COUNTRY in df:
            df[self.COUNTRY] = df[self.COUNTRY].fillna(self.NA)
        df[self.PROVINCE] = df[self.PROVINCE].fillna(self.NA)
        return df.set_index(self._id_cols)

    def covid19dh_citation(self):
        """
        Return the list of primary sources of COVID-19 Data Hub.

        Returns:
            str: the list of primary sources of COVID-19 Data Hub
        """
        dh_handler = _COVID19dh(filename=self._file_dict["covid19dh"], iso3=self._iso3_code)
        return dh_handler.primary()
