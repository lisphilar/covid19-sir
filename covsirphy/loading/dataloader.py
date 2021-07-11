#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timezone, timedelta
from pathlib import Path
import warnings
import pandas as pd
from covsirphy.util.argument import find_args
from covsirphy.util.error import deprecate, DBLockedError, NotDBLockedError, UnExpectedValueError
from covsirphy.util.term import Term
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.cleaning.japan_data import JapanData
from covsirphy.cleaning.oxcgrt import OxCGRTData
from covsirphy.cleaning.population import PopulationData
from covsirphy.cleaning.pyramid import PopulationPyramidData
from covsirphy.cleaning.linelist import LinelistData
from covsirphy.cleaning.pcr_data import PCRData
from covsirphy.cleaning.vaccine_data import VaccineData
from covsirphy.loading.db_covid19dh import _COVID19dh
from covsirphy.loading.db_owid import _OWID


class DataLoader(Term):
    """
    Load/download datasets and perform data cleaning.

    Args:
        directory (str or pathlib.Path): directory to save downloaded datasets
        update_interval (int or None): update interval of downloaded or None (avoid downloading)
        basename_dict (dict[str, str]): basename of downloaded CSV files,
            "covid19dh": COVID-19 Data Hub (default: covid19dh.csv),
            "owid": Our World In Data (default: ourworldindata.csv),
            "wbdata": World Bank Open Data (default: wbdata_population_pyramid.csv),
            "japan": COVID-19 Dataset in Japan (default: covid_japan.csv).
        verbose (int): level of verbosity when downloading

    Note:
        If @update_interval (not None) hours have passed since the last update of downloaded datasets,
        the dawnloaded datasets will be updated automatically.
        When we do not use datasets of remote servers, set @update_interval as None.

    Note:
        If @verbose is 0, no discription will be shown.
        If @verbose is 1 or larger, URL and database name will be shown.
        If @verbose is 2, detailed citation list will be show, if available.
    """

    def __init__(self, directory="input", update_interval=12, basename_dict=None, verbose=1):
        # Directory
        try:
            self.dir_path = Path(directory)
        except TypeError:
            raise TypeError(f"@directory should be a path-like object, but {directory} was applied.")
        self.update_interval = self._ensure_natural_int(
            update_interval, name="update_interval", include_zero=True, none_ok=True)
        # Create the directory if not exist
        self.dir_path.mkdir(parents=True, exist_ok=True)
        # Dictionary of filenames to save remote datasets
        filename_dict = {
            "covid19dh": "covid19dh.csv",
            "owid": "ourworldindata.csv",
            "wbdata_pyramid": "wbdata_population_pyramid.csv",
            "japan": "covid_japan.csv",
        }
        filename_dict.update(
            {k: self.dir_path.joinpath((basename_dict or {}).get(k, v)) for (k, v) in filename_dict.items()}
        )
        self._filename_dict = filename_dict.copy()
        # Verbosity
        self._verbose = self._ensure_natural_int(verbose, name="verbose", include_zero=True)
        # Column names to indentify records
        self._id_cols = [self.COUNTRY, self.PROVINCE, self.DATE]
        # Datasets retrieved from local files
        self._local_df = pd.DataFrame()
        self._local_citations = []
        # Locked database
        self._locked_df = pd.DataFrame(columns=self._id_cols)
        self._locked_citation_dict = {}
        # COVID-19 Data Hub
        self._covid19dh_primary = []
        # COVID-19 dataset in Japan
        self._japan_data = None
        # Explain changes
        dep_pcf_file = self.dir_path.joinpath("ourworldindata_pcr.csv")
        dep_vac_file = self.dir_path.joinpath("ourworldindata_vaccine.csv")
        for filepath in [dep_pcf_file, dep_vac_file]:
            if filepath.exists() and filepath not in self._filename_dict.values():
                warnings.warn(
                    "The following file might be created with covsirphy.DataLoader, "
                    "but not used at this version as default. Please delete it.\n"
                    f"{filepath}"
                    "\n\nAt this version, ourworldindata.csv will be created.\n\n",
                    DeprecationWarning,
                    stacklevel=2
                )

    @property
    def local(self):
        """
        pandas.DataFrame: local dataset
        """
        return self._local_df

    @property
    def locked(self):
        """
        pandas.DataFrame: locked dataset
        """
        self._ensure_lock_status(lock_expected=True)
        return self._locked_df

    def _ensure_lock_status(self, lock_expected):
        """
        Check whether the local database has been locked or not.

        Args:
            lock_expected (bool): whether the local database is expected to be locked or not

        Raises:
            NotDBLockedError: @lock_expected is True, but the database has NOT been locked
            DBLockedError: @lock_expected is False, but the database has been locked
        """
        if lock_expected and self._locked_df.empty:
            raise NotDBLockedError(name="the local database")
        if not lock_expected and not self._locked_df.empty:
            raise DBLockedError(name="the local database")

    def read_csv(self, filename, citation=None, dayfirst=False, how_combine="replace", **kwargs):
        """
        Read dataset saved in a CSV file and include it local database.

        Args:
            filename (str or pathlib.Path): path/URL of the CSV file
            citation (str or None): citation of the CSV file or None (basename of the CSV file)
            dayfirst (bool): whether date format is DD/MM or not
            how_combine (str): how to combine datasets when we call this method multiple times
                - 'replace': replace registered dataset with the new data
                - 'concat': concat datasets with pandas.concat()
                - 'merge': merge datasets with pandas.DataFrame.merge()
                - 'update': update the current dataset with pandas.DataFrame.update()
            kwargs: keyword arguments of pandas.read_csv()
                and pandas.concat()/pandas.DataFrame.merge()/pandas.DataFrame.update().

        Raises:
            UnExpectedValueError: un-expected value was applied as @how_combine

        Returns:
            covsirphy.DataLoader: self

        Note:
            Please refer to https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
            for the keyword arguments of pandas.read_csv().

        Note:
            Please refer to https://pandas.pydata.org/docs/reference/api/pandas.concat.html
            for the keyword arguments of pandas.concat().
            Note that we always use 'ignore_index=True' and 'sort=True'.

        Note:
            Please refer to https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html
            for the keyword arguments of pandas.DataFrame.merge().

        Note:
            Please refer to https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.update.html
            for the keyword arguments of pandas.DataFrame.update().
        """
        self._ensure_lock_status(lock_expected=False)
        df = pd.read_csv(filename, dayfirst=dayfirst, **find_args(pd.read_csv, **kwargs))
        if self._local_df.empty or how_combine == "replace":
            self._local_df = df.copy()
        elif how_combine == "concat":
            self._local_df = pd.concat(
                [self._local_df, df], ignore_index=True, sort=True, **find_args(pd.concat, **kwargs))
        elif how_combine == "merge":
            self._local_df = self._local_df.merge(df, **find_args(pd.merge, **kwargs))
        elif how_combine == "update":
            self._local_df.update(df, **find_args(df.update, **kwargs))
        else:
            raise UnExpectedValueError(
                "how_combine", how_combine, candidates=["replace", "concat", "merge", "update"])
        self._local_citations.append(str(citation or Path(filename).name))
        return self

    def assign(self, **kwargs):
        """
        Assgn new columns to the dataset retrieved from local files.

        Args:
            kwargs: keyword arguments of pandas.DataFrame.assign().

        Returns:
            covsirphy.DataLoader: self

        Note:
            Please refer to https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.assign.html
            for the keyword arguments of pandas.DataFrame.assign().
        """
        self._ensure_lock_status(lock_expected=False)
        self._local_df = self._local_df.assign(**kwargs)
        return self

    def lock(self, date, country, province, iso3=None,
             confirmed=None, fatal=None, recovered=None, population=None, tests=None,
             product=None, vaccinations=None, vaccinated_once=None, vaccinated_full=None):
        """
        Lock the local database, specifying columns which has date and area information.

        Args:
            date (str): column name for dates
            country (str): country names (top level administration)
            procvince (str): province names (2nd level administration)
            iso3 (str or None): ISO3 codes
            confirmed (str or None): the number of confirmed cases
            fatal (str or None): the number of fatal cases
            recovered (str or None): the number of recovered cases
            population (str or None): population values
            tests (str or None): the number of tests
            product (str or None): vaccine product names
            vaccinations (str or None): cumulative number of vaccinations
            vaccinated_once (str or None): cumulative number of people who received at least one vaccine dose
            vaccinated_full (str or None): cumulative number of people who received all doses prescrived by the protocol

        Returns:
            covsirphy.DataLoader: self
        """
        self._ensure_lock_status(lock_expected=False)
        variables = [
            self.ISO3, self.C, self.F, self.R, self.N, self.TESTS,
            self.PRODUCT, self.VAC, self.V_ONCE, self.V_FULL,
        ]
        id_dict = {date: self.DATE, country: self.COUNTRY, province: self.PROVINCE}
        rename_dict = {
            **id_dict, iso3: self.ISO3,
            confirmed: self.C, fatal: self.F, recovered: self.R, population: self.N,
            tests: self.TESTS, product: self.PRODUCT, vaccinations: self.VAC,
            vaccinated_once: self.V_ONCE, vaccinated_full: self.V_FULL,
        }
        # Local database
        df = self._local_df.copy()
        if df.empty:
            citation_dict = dict.fromkeys(variables, [])
        else:
            self._ensure_dataframe(df, name="local database", columns=list(id_dict.keys()))
            df = df.rename(columns=rename_dict)
            df[self.DATE] = pd.to_datetime(df[self.DATE])
            citation_dict = {v: self._local_citations if v in df else [] for v in variables}
        df = df.reindex(columns=[*self._id_cols, *variables])
        df = df.pivot_table(
            values=variables, index=self.DATE, columns=[self.COUNTRY, self.PROVINCE], aggfunc="first")
        df = df.resample("D").first().ffill().bfill()
        df = df.stack().stack().reset_index()
        # With Remote datasets
        if self.update_interval is not None:
            df = df.set_index(self._id_cols)
            # COVID19 Data Hub
            dh_filename = self._filename_dict["covid19dh"]
            df, citation_dict, dh_handler = self._add_remote(df, _COVID19dh, dh_filename, citation_dict)
            self._covid19dh_primary = dh_handler.primary
            # Our World In Data
            owid_filename = self._filename_dict["owid"]
            df, citation_dict, _ = self._add_remote(df, _OWID, owid_filename, citation_dict)
            df = df.reset_index()
        # Complete database lock
        df = df.reindex(columns=[*self._id_cols, *variables])
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        df[self.COUNTRY] = df[self.COUNTRY].fillna(self.UNKNOWN)
        df[self.PROVINCE] = df[self.PROVINCE].fillna(self.UNKNOWN)
        df[self.ISO3] = df[self.ISO3].fillna(self.UNKNOWN)
        self._locked_df = df.reindex(columns=[*self._id_cols, *variables])
        self._locked_citation_dict = citation_dict.copy()
        return self

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
            If @self.update_interval hours have passed and the remote file was updated, return True.
        """
        if not Path(filename).exists():
            return True
        date_local = self._last_updated_local(filename)
        time_limit = date_local + timedelta(hours=self.update_interval)
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
            tuple(pandas.DataFrame, list[str], _RemoteDatabase):
                updated database, citations and the handler
        """
        df = current_df.reset_index()
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        df[self.COUNTRY] = df[self.COUNTRY].fillna(self.UNKNOWN)
        df[self.PROVINCE] = df[self.PROVINCE].fillna(self.UNKNOWN)
        df = df.set_index(self._id_cols)
        cite_dict = citation_dict.copy()
        # Get the remote dataset
        force = self._download_necessity(filename)
        handler = remote_handler(filename)
        remote_df = handler.to_dataframe(force=force, verbose=self._verbose)
        remote_df[self.DATE] = pd.to_datetime(remote_df[self.DATE])
        remote_df[self.COUNTRY] = remote_df[self.COUNTRY].fillna(self.UNKNOWN)
        remote_df[self.PROVINCE] = remote_df[self.PROVINCE].fillna(self.UNKNOWN)
        remote_df = remote_df.set_index(self._id_cols)
        # Update the current database
        df = df.replace(0, None).combine_first(remote_df).reset_index().set_index(self._id_cols)
        # Update citations
        cite_dict = {k: [*v, handler.CITATION] if k in remote_df else v for (k, v) in cite_dict.items()}
        return (df, cite_dict, handler)

    def _read_dep(self, basename=None, basename_owid=None, local_file=None, verbose=None):
        """
        Read deprecated keyword arguments.

        Args:
            basename (str or None): basename of the file to save the data
            basename_owid (str or None): basename of the file to save the data of "Our World In Data"
            local_file (str or None): if not None, load the data from this file
            verbose (int): level of verbosity
        """
        if basename is not None:
            raise TypeError("@basename argument was deprecated. Please use DataLoader(bansename_dict)")
        if basename_owid is not None:
            raise TypeError("@basename_owid argument was deprecated. Please use DataLoader(bansename_dict)")
        if local_file is not None:
            raise TypeError("local_file argument was deprecated. Please use DataLoader.read_csv().")
        if verbose is not None:
            warnings.warn(
                "verbose argument was deprecated. Please use DataLoader(verbose).",
                DeprecationWarning, stacklevel=2)
            self._verbose = self._ensure_natural_int(verbose, name="verbose")

    def _auto_lock(self, variables):
        """
        Automatic database lock before using database.

        Args:
            variables (list[str] or None): variables to check citations

        Returns:
            tuple(pandas.DataFrame, dict[str, list[str]]):
                - locked database
                - citation list of the variables
        """
        # Database lock
        try:
            self._ensure_lock_status(lock_expected=True)
        except NotDBLockedError:
            self.lock(*self._id_cols)
        # Citation list
        if variables is None:
            return (self._locked_df, [])
        citation_dict = self._locked_citation_dict.copy()
        citations = [c for (v, line) in citation_dict.items() for c in line if v in variables]
        return (self._locked_df, sorted(set(citations), key=citations.index))

    @property
    def covid19dh_citation(self):
        """
        Return the list of primary sources of COVID-19 Data Hub.
        """
        self._auto_lock(variables=None)
        return self._covid19dh_primary

    def jhu(self, **kwargs):
        """
        Load the dataset regarding the number of cases using local CSV file or COVID-19 Data Hub.

        Args:
            kwargs: all keyword arguments will be ignored

        Returns:
            covsirphy.JHUData: dataset regarding the number of cases
        """
        self._read_dep(**kwargs)
        df, citations = self._auto_lock(variables=[*JHUData.REQUIRED_COLS, *JHUData.OPTINAL_COLS])
        jhu_data = JHUData(data=df, citation="\n".join(citations))
        if self.update_interval is None:
            return jhu_data
        return jhu_data.replace(self.japan())

    def population(self, **kwargs):
        """
        Load the dataset regarding population values using local CSV file or COVID-19 Data Hub.

        Args:
            kwargs: all keyword arguments will be ignored

        Returns:
            covsirphy.PopulationData: dataset regarding population values
        """
        self._read_dep(**kwargs)
        df, citations = self._auto_lock(variables=PopulationData.RAW_COLS)
        return PopulationData(data=df, citation="\n".join(citations))

    def oxcgrt(self, **kwargs):
        """
        Load the dataset regarding OxCGRT indicators using local CSV file or COVID-19 Data Hub.

        Args:
            kwargs: all keyword arguments will be ignored

        Returns:
            covsirphy.JHUData: dataset regarding OxCGRT data
        """
        self._read_dep(**kwargs)
        df, citations = self._auto_lock(variables=OxCGRTData.RAW_COLS)
        return OxCGRTData(data=df, citation="\n".join(citations))

    def japan(self, **kwargs):
        """
        Load the dataset of the number of cases in Japan.
        https://github.com/lisphilar/covid19-sir/tree/master/data

        Args:
            kwargs: all keyword arguments will be ignored

        Returns:
            covsirphy.CountryData: dataset at country level in Japan
        """
        self._read_dep(**kwargs)
        filename = self._filename_dict["japan"]
        if self._japan_data is None:
            force = self._download_necessity(filename=filename)
            self._japan_data = JapanData(filename=filename, force=force, verbose=self._verbose)
        return self._japan_data

    @deprecate("DataLoader.linelist()", version="2.21.0-theta")
    def linelist(self, basename="linelist.csv", verbose=1):
        """
        Load linelist of case reports.
        https://github.com/beoutbreakprepared/nCoV2019

        Args:
            basename (str): basename of the file to save the data
            verbose (int): level of verbosity

        Returns:
            covsirphy.LinelistData: linelist data
        """
        filename = self.dir_path.joinpath(basename)
        force = self._download_necessity(filename=filename)
        return LinelistData(filename=filename, force=force, verbose=verbose)

    def pcr(self, **kwargs):
        """
        Load the dataset regarding the number of tests and confirmed cases,
        using local CSV file or COVID-19 Data Hub.

        Args:
            kwargs: all keyword arguments will be ignored

        Returns:
            covsirphy.PCRData: dataset regarding the number of tests and confirmed cases
        """
        self._read_dep(**kwargs)
        df, citations = self._auto_lock(variables=PCRData.RAW_COLS)
        pcr_data = PCRData(data=df, citation="\n".join(citations))
        if self.update_interval is None:
            return pcr_data
        # Update with Japan data
        pcr_data.replace(self.japan())
        return pcr_data

    def vaccine(self, **kwargs):
        """
        Load the dataset regarding vaccination.
        https://github.com/owid/covid-19-data/tree/master/public/data
        https://ourworldindata.org/coronavirus

        Args:
            kwargs: all keyword arguments will be ignored

        Returns:
            covsirphy.VaccineData: dataset regarding vaccines
        """
        self._read_dep(**kwargs)
        df, citations = self._auto_lock(variables=VaccineData.RAW_COLS)
        return VaccineData(data=df.dropna(), citation="\n".join(citations))

    def pyramid(self, **kwargs):
        """
        Load the dataset regarding population pyramid.
        World Bank Group (2020), World Bank Open Data, https://data.worldbank.org/

        Args:
            kwargs: all keyword arguments will be ignored

        Returns:
            covsirphy.PopulationPyramidData: dataset regarding population pyramid
        """
        self._read_dep(**kwargs)
        filename = self._filename_dict["wbdata_pyramid"]
        return PopulationPyramidData(filename=filename, force=False, verbose=self._verbose)

    def collect(self):
        """
        Collect data for scenario analysis and return them as a dictionary.

        Returns:
            dict(str, object):
                - jhu_data (covsirphy.JHUData)
                - extras (list[covsirphy.CleaningBase]):
                    - covsirphy.OXCGRTData
                    - covsirphy.PCRData
                    - covsirphy.VaccineData
        """
        return {
            "jhu_data": self.jhu(),
            "extras": [self.oxcgrt(), self.pcr(), self.vaccine()]
        }
