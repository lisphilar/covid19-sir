#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
from covsirphy.util.argument import find_args
from covsirphy.util.error import DBLockedError, NotDBLockedError, UnExpectedValueError
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
        update_interval (int or None): update interval of downloaded datasets or None (only use local files)

    Note:
        GitHub datasets will be always updated because headers of GET response
        does not have 'Last-Modified' keys.
        If @update_interval hours have passed since the last update of downloaded datasets,
        updating will be forced when updating is not prevented by the methods.
    """

    def __init__(self, directory="input", update_interval=12):
        # Directory
        try:
            self.dir_path = Path(directory)
        except TypeError:
            raise TypeError(f"@directory should be a path-like object, but {directory} was applied.")
        self.update_interval = self._ensure_natural_int(
            update_interval, name="update_interval", include_zero=True, none_ok=True)
        # Create the directory if not exist
        self.dir_path.mkdir(parents=True, exist_ok=True)
        # Datasets retrieved from local files
        self._local_df = pd.DataFrame()
        self._locked_df = pd.DataFrame()
        # COVID-19 Data Hub
        self._covid19dh_df = pd.DataFrame()
        self._covid19dh_citation = ""

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

    def read_csv(self, filename, how_combine="replace", dayfirst=False, **kwargs):
        """
        Read dataset saved in a CSV file and include it local database.

        Args:
            filename (str or pathlib.Path): path/URL of the CSV file
            how_combine (str): how to combine datasets when we call this method multiple times
                - 'replace': replace registered dataset with the new data
                - 'concat': concat datasets with pandas.concat()
                - 'merge': merge datasets with pandas.DataFrame.merge()
                - 'update': update the current dataset with pandas.DataFrame.update()
            dayfirst (bool): whether date format is DD/MM or not
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

    def lock(self, date, country, province):
        """
        Lock the local database, specifying columns which has date and area information.

        Args:
            date (str): column name for dates
            country (str): column name for country names (top level administration)
            procvince (str): column name for province names (2nd level administration)

        Returns:
            covsirphy.DataLoader: self

        Note:
            Values will be grouped by @date, @country and @province.
            Total values will be used for each group.
        """
        self._ensure_lock_status(lock_expected=False)
        df = self._local_df.copy()
        col_dict = {date: self.DATE, country: self.COUNTRY, province: self.PROVINCE}
        self._ensure_dataframe(df, name="local database", columns=list(col_dict.keys()))
        df = df.rename(columns=col_dict)
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        self._locked_df = df.groupby(list(col_dict.values()), as_index=False).sum()
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

    def _covid19dh(self, name, basename="covid19dh.csv", verbose=True):
        """
        Return the datasets of COVID-19 Data Hub.

        Args:
            name (str): name of dataset, "jhu", "population" or "oxcgrt"
            basename (str): basename of CSV file to save records
            verbose (int): level of verbosity

        Note:
            If @verbose is 2, detailed citation list will be shown when downloading.
            If @verbose is 1, how to show the list will be explained.
            Citation of COVID-19 Data Hub will be set as JHUData.citation etc.

        Returns:
            covsirphy.CleaningBase: the dataset
        """
        obj_dict = {
            "jhu": JHUData,
            "population": PopulationData,
            "oxcgrt": OxCGRTData,
            "pcr": PCRData,
        }
        filename, force = self.dir_path.joinpath(basename), False
        if self._covid19dh_df.empty:
            force = self._download_necessity(filename)
            handler = _COVID19dh(filename)
            self._covid19dh_df = handler.to_dataframe(force=force, verbose=verbose)
            self._covid19dh_citation = handler.CITATION
        return obj_dict[name](data=self._covid19dh_df, citation=self._covid19dh_citation)

    @property
    def covid19dh_citation(self):
        """
        Return the list of primary sources of COVID-19 Data Hub.
        """
        if not self._covid19dh_citation:
            self._covid19dh(name="jhu", verbose=0)
        return self._covid19dh_citation

    def jhu(self, basename="covid19dh.csv", local_file=None, verbose=1):
        """
        Load the dataset regarding the number of cases using local CSV file or COVID-19 Data Hub.

        Args:
            basename (str or None): basename of the file to save the data
            local_file (str or None): if not None, load the data from this file
            verbose (int): level of verbosity

        Note:
            If @verbose is 2, detailed citation list will be shown when downloading.
            If @verbose is 1, how to show the list will be explained.
            Citation of COVID-19 Data Hub will be set as JHUData.citation.

        Returns:
            covsirphy.JHUData: dataset regarding the number of cases
        """
        if local_file is not None:
            return JHUData(filename=local_file)
        # Retrieve JHU data from COVID-19 Data Hub
        jhu_data = self._covid19dh(
            name="jhu", basename=basename, verbose=verbose)
        # Replace Japan dataset with the government-announced data
        japan_data = self.japan()
        jhu_data.replace(japan_data)
        return jhu_data

    def population(self, basename="covid19dh.csv", local_file=None, verbose=1):
        """
        Load the dataset regarding population values using local CSV file or COVID-19 Data Hub.

        Args:
            basename (str or None): basename of the file to save the data
            local_file (str or None): if not None, load the data from this file
            verbose (int): level of verbosity

        Note:
            If @verbose is 2, detailed citation list will be shown when downloading.
            If @verbose is 1, how to show the list will be explained.
            Citation of COVID-19 Data Hub will be set as PopulationData.citation.

        Returns:
            covsirphy.PopulationData: dataset regarding population values
        """
        if local_file is not None:
            return PopulationData(filename=local_file)
        return self._covid19dh(name="population", basename=basename, verbose=verbose)

    def oxcgrt(self, basename="covid19dh.csv", local_file=None, verbose=1):
        """
        Load the dataset regarding OxCGRT indicators using local CSV file or COVID-19 Data Hub.

        Args:
            basename (str or None): basename of the file to save the data
            local_file (str or None): if not None, load the data from this file
            verbose (int): level of verbosity

        Note:
            If @verbose is 2, detailed citation list will be shown when downloading.
            If @verbose is 1, how to show the list will be explained.
            Citation of COVID-19 Data Hub will be set as OxCGRTData.citation.

        Returns:
            covsirphy.JHUData: dataset regarding OxCGRT data
        """
        if local_file is not None:
            return OxCGRTData(filename=local_file)
        return self._covid19dh(name="oxcgrt", basename=basename, verbose=verbose)

    def japan(self, basename="covid_japan.csv", local_file=None, verbose=1):
        """
        Load the dataset of the number of cases in Japan.
        https://github.com/lisphilar/covid19-sir/tree/master/data

        Args:
            basename (str): basename of the file to save the data
            local_file (str or None): if not None, load the data from this file
            verbose (int): level of verbosity

        Returns:
            covsirphy.CountryData: dataset at country level in Japan
        """
        filename = local_file or self.dir_path.joinpath(basename)
        force = self._download_necessity(filename=filename)
        return JapanData(filename=filename, force=force, verbose=verbose)

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

    def pcr(self, basename="covid19dh.csv", local_file=None,
            basename_owid="ourworldindata_pcr.csv", verbose=1):
        """
        Load the dataset regarding the number of tests and confirmed cases,
        using local CSV file or COVID-19 Data Hub.

        Args:
            basename (str or None): basename of the file to save "COVID-19 Data Hub" data
            local_file (str or None): if not None, load the data from this file
            basename_owid (str): basename of the file to save "Our World In Data" data
            verbose (int): level of verbosity

        Note:
            If @verbose is 2, detailed citation list will be shown when downloading.
            If @verbose is 1, how to show the list will be explained.
            Citation of COVID-19 Data Hub will be set as JHUData.citation.

        Returns:
            covsirphy.PCRData: dataset regarding the number of tests and confirmed cases
        """
        if local_file is not None:
            return PCRData(filename=local_file)
        # Retrieve JHU data from COVID-19 Data Hub
        pcr_data = self._covid19dh(
            name="pcr", basename=basename, verbose=verbose)
        # Update the values using "Our World In Data" dataset
        owid_filename = self.dir_path.joinpath(basename_owid)
        owid_force = self._download_necessity(filename=owid_filename)
        pcr_data.use_ourworldindata(filename=owid_filename, force=owid_force)
        # Replace Japan dataset with the government-announced data
        japan_data = self.japan()
        pcr_data.replace(japan_data)
        return pcr_data

    def vaccine(self, basename="ourworldindata_vaccine.csv", verbose=1):
        """
        Load the dataset regarding vaccination.
        https://github.com/owid/covid-19-data/tree/master/public/data
        https://ourworldindata.org/coronavirus

        Args:
            basename (str): basename of the file to save the data
            verbose (int): level of verbosity

        Returns:
            covsirphy.VaccineData: dataset regarding vaccines
        """
        filename = self.dir_path.joinpath(basename)
        force = self._download_necessity(filename=filename)
        return VaccineData(filename=filename, force=force, verbose=verbose)

    def pyramid(self, basename="wbdata_population_pyramid.csv", verbose=1):
        """
        Load the dataset regarding population pyramid.
        World Bank Group (2020), World Bank Open Data, https://data.worldbank.org/

        Args:
            basename (str): basename of the file to save the data
            verbose (int): level of verbosity

        Returns:
            covsirphy.PopulationPyramidData: dataset regarding population pyramid
        """
        filename = self.dir_path.joinpath(basename)
        return PopulationPyramidData(filename=filename, force=False, verbose=verbose)

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
