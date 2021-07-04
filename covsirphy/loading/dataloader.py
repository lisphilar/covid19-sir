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


class DataLoader(Term):
    """
    Load/download datasets and perform data cleaning.

    Args:
        directory (str or pathlib.Path): directory to save downloaded datasets
        update_interval (int or None): update interval of downloaded or None (avoid downloading)
        basename_dict (dict[str, str]): basename of downloaded CSV files,
            "covid19dh": COVID-19 Data Hub (default: covid19dh.csv),
            "owid_pcr": Our World In Data (the number of tests, default: ourworldindata_pcr.csv),
            "owid_vaccine": Our World In Data (vaccination, default: ourworldindata_vaccine.csv),
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
            "owid_pcr": "ourworldindata_pcr.csv",
            "owid_vaccine": "ourworldindata_vaccine.csv",
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
        self._id_cols = [self.DATE, self.COUNTRY, self.PROVINCE]
        # Datasets retrieved from local files
        self._local_df = pd.DataFrame()
        self._locked_df = pd.DataFrame(columns=self._id_cols)
        self._local_citations = []
        # COVID-19 Data Hub
        self._covid19dh_df = pd.DataFrame()
        self._covid19dh_citation = ""
        # COVID-19 dataset in Japan
        self._japan_data = None

    def _ensure_lock_status(self, lock_expected):
        """
        Check whether the local database has been locked or not.

        Args:
            lock_expected (bool): whether the local database is expected to be locked or not

        Raises:
            NotDBLockedError: @lock_expected is True, but the non-empty database has NOT been locked
            DBLockedError: @lock_expected is False, but the database has been locked
        """
        if lock_expected and (not self._local_df.empty) and self._locked_df.empty:
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

    def local(self, locked=False):
        """
        Return the local dataset.

        Args:
            locked (bool): whether the returned dataset is from locked locked database or unlocked

        Returns:
            pandas.DataFrame: dataset from locked or unlocked local database
                Index
                    reset index
                Columns
                    If locked,
                    - Date (pandas.Timestamp): dates
                    - Country (object): country names
                    - Province (object): province names
        """
        if locked:
            self._ensure_lock_status(lock_expected=True)
            return self._locked_df
        return self._local_df

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

    def lock(self, date, country, province, **kwargs):
        """
        Lock the local database, specifying columns which has date and area information.

        Args:
            date (str): column name for dates
            country (str): column name for country names (top level administration)
            procvince (str): column name for province names (2nd level administration)
            kwargs: keyword arguments of variable names


        Returns:
            covsirphy.DataLoader: self

        Note:
            Values will be grouped by @date, @country and @province.
            Total values will be used for each group.

        Note:
            For keyword names (column names with CovsirPhy terms) of kwargs, upper/lower case insensitive.

        Note:
        As keywords of kwargs, we ca use
            "confirmed": the number of confirmed cases,
            "fatal": the number of fatal cases,
            "recovered": the number of recovered cases,
            "population": population values,
            "tests": the number of tests,
            "iso3": ISO3 codes,
            "product": vaccine product names,
            "vaccinations": cumulative number of vaccinations,
            "vaccinated_once": cumulative number of people who received at least one vaccine dose,
            "vaccinated_full": cumulative number of people who received all doses prescrived by the protocol.
        """
        self._ensure_lock_status(lock_expected=False)
        df = self._local_df.copy()
        # Date/Country/Province
        id_dict = {date: self.DATE, country: self.COUNTRY, province: self.PROVINCE}
        self._ensure_dataframe(df, name="local database", columns=list(id_dict.keys()))
        df = df.rename(columns=id_dict)
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        # Variables
        variables = [
            self.C, self.F, self.R, self.N, self.TESTS, self.ISO3,
            self.PRODUCT, self.VAC, self.V_ONCE, self.V_FULL,
        ]
        df = df.rename(
            columns={v: k.capitalize().replace("Iso3", self.ISO3) for (k, v) in kwargs.items()})
        # Complete database lock
        df = df.reindex(columns=[*list(id_dict.values()), *variables])
        self._locked_df = df.groupby(list(id_dict.values()), as_index=False).sum()
        return self

    @ staticmethod
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

    def _covid19dh(self):
        """
        Return the datasets of COVID-19 Data Hub as a dataframe.

        Args:
            basename (str): basename of CSV file to save records
            verbose (int): level of verbosity

        Note:
            If @verbose is 2, detailed citation list will be shown when downloading.
            If @verbose is 1, how to show the list will be explained.

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date: observation date
                    - Country: country/region name
                    - Province: province/prefecture/state name
                    - Confirmed: the number of confirmed cases
                    - Infected: the number of currently infected cases
                    - Fatal: the number of fatal cases
                    - Recovered: the number of recovered cases
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
        """
        filename = self._filename_dict["covid19dh"]
        if self._covid19dh_df.empty:
            force = self._download_necessity(filename)
            handler = _COVID19dh(filename)
            self._covid19dh_df = handler.to_dataframe(force=force, verbose=self._verbose)
            self._covid19dh_citation = handler.CITATION
        return self._covid19dh_df

    @ property
    def covid19dh_citation(self):
        """
        Return the list of primary sources of COVID-19 Data Hub.
        """
        if not self._covid19dh_citation:
            self._covid19dh()
        return self._covid19dh_citation

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
                "verbose argument was deprecated. Please use DataLoader(verbose).", DeprecationWarning)
            self._verbose = self._ensure_natural_int(verbose, name="verbose")

    def jhu(self, **kwargs):
        """
        Load the dataset regarding the number of cases using local CSV file or COVID-19 Data Hub.

        Args:
            kwargs: all keyword arguments will be ignored

        Returns:
            covsirphy.JHUData: dataset regarding the number of cases
        """
        self._read_dep(**kwargs)
        # Only local data
        locked_df = self.local(locked=True)
        if self.update_interval is None:
            return JHUData(data=locked_df, citation="\n".join(self._local_citations))
        # Use remote data
        datahub_df = self._covid19dh().set_index(self._id_cols)
        locked_df = locked_df.set_index(self._id_cols)
        locked_df.update(datahub_df, overwrite=True)
        df = pd.concat([datahub_df, locked_df], axis=0, sort=True)
        df = df.reset_index().drop_duplicates(subset=self._id_cols, keep="last", ignore_index=True)
        # Citation
        citations = [*self._local_citations, self._covid19dh_citation]
        # Create JHUData instance
        jhu_data = JHUData(data=df, citation="\n".join(citations))
        jhu_data.replace(self.japan())
        return jhu_data

    def population(self, **kwargs):
        """
        Load the dataset regarding population values using local CSV file or COVID-19 Data Hub.

        Args:
            kwargs: all keyword arguments will be ignored

        Returns:
            covsirphy.PopulationData: dataset regarding population values
        """
        self._read_dep(**kwargs)
        # Only local data
        locked_df = self.local(locked=True)
        if self.update_interval is None:
            return PopulationData(data=locked_df, citation="\n".join(self._local_citations))
        # Use remote data
        datahub_df = self._covid19dh().set_index(self._id_cols)
        locked_df = locked_df.set_index(self._id_cols)
        locked_df.update(datahub_df, overwrite=True)
        df = pd.concat([datahub_df, locked_df], axis=0, sort=True)
        df = df.reset_index().drop_duplicates(subset=self._id_cols, keep="last", ignore_index=True)
        # Citation
        citations = [*self._local_citations, self._covid19dh_citation]
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
        # Only local data
        locked_df = self.local(locked=True)
        if self.update_interval is None:
            return OxCGRTData(data=locked_df, citation="\n".join(self._local_citations))
        # Use remote data
        datahub_df = self._covid19dh().set_index(self._id_cols)
        locked_df = locked_df.set_index(self._id_cols)
        locked_df.update(datahub_df, overwrite=True)
        df = pd.concat([datahub_df, locked_df], axis=0, sort=True)
        df = df.reset_index().drop_duplicates(subset=self._id_cols, keep="last", ignore_index=True)
        # Citation
        citations = [*self._local_citations, self._covid19dh_citation]
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
        # Only local data
        locked_df = self.local(locked=True)
        if self.update_interval is None:
            return PCRData(data=locked_df, citation="\n".join(self._local_citations))
        # Use remote data
        datahub_df = self._covid19dh().set_index(self._id_cols)
        locked_df = locked_df.set_index(self._id_cols)
        locked_df.update(datahub_df, overwrite=True)
        df = pd.concat([datahub_df, locked_df], axis=0, sort=True)
        df = df.reset_index().drop_duplicates(subset=self._id_cols, keep="last", ignore_index=True)
        # Citation
        citations = [*self._local_citations, self._covid19dh_citation]
        # Create PCRData instance
        pcr_data = PCRData(data=df, citation="\n".join(citations))
        pcr_data.replace(self.japan())
        # Update the values using "Our World In Data" dataset
        owid_filename = self._filename_dict["owid_pcr"]
        owid_force = self._download_necessity(filename=owid_filename)
        pcr_data.use_ourworldindata(filename=owid_filename, force=owid_force)
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
        filename = self._filename_dict["owid_vaccine"]
        force = self._download_necessity(filename=filename)
        return VaccineData(filename=filename, force=force, verbose=self._verbose)

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
