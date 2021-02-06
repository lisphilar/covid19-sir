#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timezone, timedelta
from pathlib import Path
from covsirphy.util.term import Term
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.cleaning.japan_data import JapanData
from covsirphy.cleaning.oxcgrt import OxCGRTData
from covsirphy.cleaning.population import PopulationData
from covsirphy.cleaning.pyramid import PopulationPyramidData
from covsirphy.cleaning.covid19datahub import COVID19DataHub
from covsirphy.cleaning.linelist import LinelistData
from covsirphy.cleaning.pcr_data import PCRData
from covsirphy.cleaning.vaccine_data import VaccineData


class DataLoader(Term):
    """
    Download the dataset and perform data cleaning.

    Args:
        directory (str or pathlib.Path): directory to save the downloaded datasets
        update_interval (int): update interval of the local datasets

    Note:
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
        self.update_interval = self._ensure_natural_int(
            update_interval, name="update_interval", include_zero=True)
        # Create the directory if not exist
        self.dir_path.mkdir(parents=True, exist_ok=True)
        # COVID-19 Data Hub
        self.covid19dh = None
        # Cache instances
        self._linelist_data = None

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
        filename, force = self.dir_path.joinpath(basename), False
        if self.covid19dh is None:
            self.covid19dh = COVID19DataHub(filename=filename)
            force = self._download_necessity(filename)
        return self.covid19dh.load(name=name, force=force, verbose=verbose)

    @property
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
        Load the dataset regarding OxCGRT data using local CSV file or COVID-19 Data Hub.

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
            covsirphy.CountryData: dataset at country level in Japan
        """
        if self._linelist_data is None:
            filename = self.dir_path.joinpath(basename)
            force = self._download_necessity(filename=filename)
            self._linelist_data = LinelistData(
                filename=filename, force=force, verbose=verbose)
        return self._linelist_data

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
