#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timezone, timedelta
from pathlib import Path
from dask import dataframe as dd
import pandas as pd
from covsirphy.util.file import save_dataframe
from covsirphy.cleaning.term import Term
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.cleaning.country_data import CountryData
from covsirphy.cleaning.oxcgrt import OxCGRTData
from covsirphy.cleaning.population import PopulationData
from covsirphy.cleaning.covid19datahub import COVID19DataHub
from covsirphy.cleaning.linelist import LinelistData


class DataLoader(Term):
    """
    Download the dataset and perform data cleaning.

    Args:
        directory (str or pathlib.Path): directory to save the downloaded datasets
        update_interval (int): update interval of the local datasets

    Notes:
        GitHub datasets will be always updated because headers of GET response
        does not have 'Last-Modified' keys.
        If @update_interval hours have passed since the last update of local datasets,
        updating will be forced when updating is not prevented by the methods.

    Examples:
        >>> # Setup
        >>> import covsirphy as cs
        >>> data_loader = cs.DataLoader("input")
        >>> # JHU data: the number of cases
        >>> jhu_data = data_loader.jhu()
        >>> print(jhu_data.citation)
        ...
        >>> print(type(jhu_data.cleaned()))
        <class 'pandas.core.frame.DataFrame'>
        >>> # The number of cases in Japan
        >>> jpn_data = data_loader.japan()
        >>> print(jpn_data.citation)
        ...
        >>> print(type(jpn_data.cleaned()))
        <class 'pandas.core.frame.DataFrame'>
        >>> # Population values
        >>> population_data = data_loader.population()
        >>> print(population_data.citation)
        ...
        >>> print(type(population_data.cleaned()))
        <class 'pandas.core.frame.DataFrame'>
        >>> # OxCGRT: Government responses
        >>> oxcgrt_data = data_loader.oxcgrt()
        >>> print(oxcgrt_data.citation)
        ...
        >>> print(type(oxcgrt_data.cleaned()))
        <class 'pandas.core.frame.DataFrame'>
        >>> # Citation list of COVID-19 Data Hub
        >>> print(data_loader.covid19dh_citation)
        ...
    """
    GITHUB_URL = "https://raw.githubusercontent.com"

    def __init__(self, directory="input", update_interval=12):
        # Directory
        try:
            self.dir_path = Path(directory)
        except TypeError:
            raise TypeError(
                f"@directory should be a path-like object, but {directory} was applied.")
        self.update_interval = self.ensure_natural_int(
            update_interval, name="update_interval", include_zero=True)
        # Create the directory if not exist
        self.dir_path.mkdir(parents=True, exist_ok=True)
        # COVID-19 Data Hub
        self.covid19dh = None

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

        Notes:
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

        Notes:
            If @verbose is 2, detailed citation list will be shown when downloading.
            If @verbose is 1, how to show the list will be explained.
            Citation of COVID-19 Data Hub will be set as JHUData.citation etc.

        Returns:
            covsirphy.CleaningBase: the dataset
        """
        filename, force = self.dir_path.joinpath(basename), False
        if self.covid19dh is None:
            self.covid19dh = COVID19DataHub(filename=filename)
            force = self._download_necessity(filename)
        return self.covid19dh.load(name=name, force=force, verbose=verbose)

    @ property
    def covid19dh_citation(self):
        """
        Return the list of primary sources of COVID-19 Data Hub.
        """
        if self.covid19dh is None:
            self._covid19dh(name="jhu", verbose=0)
        return self.covid19dh.primary

    def jhu(self, basename="covid19dh.csv", local_file=None, verbose=1):
        """
        Load the dataset regarding the number of cases using local CSV file or COVID-19 Data Hub.

        Args:
            basename (str or None): basename of the file to save the data
            local_file (str or None): if not None, load the data from this file
            verbose (int): level of verbosity

        Notes:
            If @verbose is 2, detailed citation list will be shown when downloading.
            If @verbose is 1, how to show the list will be explained.
            Citation of COVID-19 Data Hub will be set as JHUData.citation.

        Returns:
            covsirphy.JHUData: dataset regarding the number of cases
        """
        if local_file is not None:
            return JHUData(filename=local_file)
        return self._covid19dh(name="jhu", basename=basename, verbose=verbose)

    def population(self, basename="covid19dh.csv", local_file=None, verbose=1):
        """
        Load the dataset regarding population values using local CSV file or COVID-19 Data Hub.

        Args:
            basename (str or None): basename of the file to save the data
            local_file (str or None): if not None, load the data from this file
            verbose (int): level of verbosity

        Notes:
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
        Load the dataset regarding OxCGRT data using local CSV file or COVID-19 Data Hub.

        Args:
            basename (str or None): basename of the file to save the data
            local_file (str or None): if not None, load the data from this file
            verbose (int): level of verbosity

        Notes:
            If @verbose is 2, detailed citation list will be shown when downloading.
            If @verbose is 1, how to show the list will be explained.
            Citation of COVID-19 Data Hub will be set as OxCGRTData.citation.

        Returns:
            covsirphy.JHUData: dataset regarding OxCGRT data
        """
        if local_file is not None:
            return OxCGRTData(filename=local_file)
        return self._covid19dh(name="oxcgrt", basename=basename, verbose=verbose)

    def japan(self, basename="covid_jpn_total.csv", local_file=None):
        """
        Load the dataset of the number of cases in Japan.
        https://github.com/lisphilar/covid19-sir/tree/master/data

        Args:
            basename (str): basename of the file to save the data
            local_file (str or None): if not None, load the data from this file

        Returns:
            covsirphy.CountryData: dataset at country level in Japan
        """
        filename = local_file or self.dir_path.joinpath(basename)
        if self._download_necessity(filename=filename):
            url = f"{self.GITHUB_URL}/lisphilar/covid19-sir/master/data/japan/covid_jpn_total.csv"
            try:
                df = dd.read_csv(url, blocksize=None).compute()
            except (ValueError, FileNotFoundError):
                df = pd.read_csv(url)
            save_dataframe(df, filename=filename, index=False)
        japan_data = CountryData(filename=filename, country="Japan")
        japan_data.citation = "Lisphilar (2020), COVID-19 dataset in Japan, GitHub repository, " \
            "https://github.com/lisphilar/covid19-sir/data/japan"
        japan_data.set_variables(
            date="Date",
            confirmed="Positive",
            fatal="Fatal",
            recovered="Discharged",
            province=None)
        return japan_data

    def linelist(self, basename="linelist.csv", verbose=1):
        """
        Load linelist of case reports.
        https://github.com/beoutbreakprepared/nCoV2019

        Args:
            basename (str): basename of the file to save the data
            verbose (int): level of verbosity

        Returns:
            covsirphy.CountryData: dataset at country level in Japan
        """
        filename = self.dir_path.joinpath(basename)
        force = self._download_necessity(filename=filename)
        return LinelistData(filename=filename, force=force, verbose=verbose)
