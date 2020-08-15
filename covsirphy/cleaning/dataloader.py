#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime, timezone, timedelta
import covid19dh
from dask import dataframe as dd
import pandas as pd
import requests
from covsirphy.cleaning.cbase import CleaningBase
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.cleaning.country_data import CountryData
from covsirphy.cleaning.oxcgrt import OxCGRTData
from covsirphy.cleaning.population import PopulationData
from covsirphy.cleaning.term import Term


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
        >>> import covsirphy as cs
        >>> data_loader = cs.DataLoader("input")
        >>> jhu_data = data_loader.jhu()
        >>> print(jhu_data.citation)
        ...
        >>> print(type(jhu_data.cleaned()))
        <class 'pandas.core.frame.DataFrame'>
        >>> jpn_data = data_loader.japan()
        >>> print(jpn_data.citation)
        ...
        >>> print(type(jpn_data.cleaned()))
        <class 'pandas.core.frame.DataFrame'>
        >>> population_data = data_loader.population()
        >>> print(population_data.citation)
        ...
        >>> print(type(population_data.cleaned()))
        <class 'pandas.core.frame.DataFrame'>
        >>> oxcgrt_data = data_loader.oxcgrt()
        >>> print(oxcgrt_data.citation)
        ...
        >>> print(type(oxcgrt_data.cleaned()))
        <class 'pandas.core.frame.DataFrame'>
        >>> print(data_loader.covid19dh_citation)
        ...
    """

    def __init__(self, directory="input", update_interval=12):
        if not isinstance(directory, (str, Path)):
            raise TypeError(
                f"@directory must be a string or a path but {directory} was applied."
            )
        self.dir_path = Path(directory)
        self.update_interval = self.ensure_natural_int(
            update_interval, name="update_interval", include_zero=True
        )
        # Create the directory if not exist
        self.dir_path.mkdir(parents=True, exist_ok=True)
        # COVID-19 Data Hub
        self._covid19dh_citation = None
        self._covid19dh_basename = "covid19dh.csv"
        self._covid19dh_citation_secondary = '(Secondary source)' \
            ' Guidotti, E., Ardia, D., (2020), "COVID-19 Data Hub",' \
            ' Working paper, doi: 10.13140/RG.2.2.11649.81763.' \
            '\nWe can get Citation list of primary sources with DataLoader(...).covid19dh_citation'
        # JHU dataset: the number of cases
        self.jhu_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master" \
            "/csse_covid_19_data/csse_covid_19_time_series"
        self.jhu_citation = self._covid19dh_citation_secondary
        self.jhu_date_col = "ObservationDate"
        self.jhu_p_col = "Province/State"
        self.jhu_c_col = "Country/Region"
        # The number of cases in Japan
        self.japan_cases_url = "https://raw.githubusercontent.com/lisphilar/covid19-sir/master/data/japan"
        self.japan_cases_citation = "Lisphilar (2020), COVID-19 dataset in Japan, GitHub repository, " \
            "https://github.com/lisphilar/covid19-sir/data/japan"
        # Population dataset: THE WORLD BANK
        self.population_url = "http://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?"
        self.population_citation = self._covid19dh_citation_secondary
        # OxCGRT dataset: Oxford Covid-19 Government Response Tracker
        self.oxcgrt_url = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data"
        self.oxcgrt_citation = self._covid19dh_citation_secondary
        # Dictionary of datasets
        self.dataset_dict = {
            "JHU": {
                "class": JHUData, "url": self.jhu_url, "citation": self.jhu_citation
            },
            "Japan_cases": {
                "class": CountryData, "url": self.japan_cases_url, "citation": self.japan_cases_citation
            },
            "Population": {
                "class": PopulationData, "url": self.population_url, "citation": self.population_citation
            },
            "OxCGRT": {
                "class": OxCGRTData, "url": self.oxcgrt_url, "citation": self.oxcgrt_citation
            },
        }

    @staticmethod
    def _get_raw(url, is_json=False):
        """
        Get raw dataset from the URL.

        Args:
            url (str): URL to get raw dataset
            is_json (bool): if True, parse the response as Json data.

        Returns:
            (pandas.DataFrame): raw dataset
        """
        if is_json:
            r = requests.get(url)
            try:
                json_data = r.json()[1]
            except Exception:
                raise TypeError(
                    f"Unknown data format was used in Web API {url}")
            df = pd.json_normalize(json_data)
            return df
        df = dd.read_csv(url).compute()
        return df

    def _resolve_filename(self, basename):
        """
        Return the absolute path of the file in the @self.dir_path directory.

        Args:
            basename (str): basename of the file, like covid_19_data.csv

        Returns:
            (str): absolute path of the file
        """
        file_path = self.dir_path / basename
        return str(file_path.resolve())

    def _save(self, dataframe, filename):
        """
        Save the dataframe to the local environment.

        Args:
            dataframe (pandas.DataFrame): dataframe to save
            filename (str or None): filename to save

        Notes:
            CSV file will be created in @self.dirpath directory.
            If @self.dirpath is None, the dataframe will not be saved
        """
        df = self.ensure_dataframe(dataframe, name="dataframe")
        if filename is None:
            return None
        try:
            df.to_csv(filename, index=False)
        except OSError:
            pass
        return filename

    def _last_updated_remote(self, url):
        """
        Return the date last updated of remote file/directory.

        Args:
            url (str): URL

        Returns:
            (pandas.Timestamp or None): time last updated (UTC)

        Notes:
            If "Last-Modified" key is not in the header, returns None.
            If failed in connection with remote direcotry, returns None.
        """
        try:
            response = requests.get(url)
        except requests.ConnectionError:
            return False
        try:
            date_str = response.headers["Last-Modified"]
        except KeyError:
            return None
        return pd.to_datetime(date_str).tz_convert(None)

    def _last_updated_local(self, path):
        """
        Return the date last updated of local file/directory.

        Args:
            path (str or pathlibPath): name of the file/directory

        Returns:
            (datetime.datetime): time last updated (UTC)
        """
        path = Path(path)
        m_time = path.stat().st_mtime
        date = datetime.fromtimestamp(m_time)
        date = date.astimezone(timezone.utc).replace(tzinfo=None)
        return date

    def _create_dataset(self, data_key, filename, set_citation=True, **kwargs):
        """
        Return dataset class with citation.

        Args:
            data_key (str): key of self.dataset_dict
            filename (str): filename of the local dataset
            set_citation (str): if True, set citation
            kwargs: keyword arguments of @data_class

        Returns:
            (covsirphy.JHUData): the dataset

        Notes:
            ".citation" attribute will returns the citation
        """
        # Get information from self.dataset_dict
        target_dict = self.dataset_dict[data_key]
        data_class = target_dict["class"]
        citation = target_dict["citation"]
        # Validate the data class
        data_class = self.ensure_subclass(data_class, CleaningBase)
        # Create instance and set citation
        data_instance = data_class(filename=filename, **kwargs)
        if set_citation:
            data_instance.citation = citation
        return data_instance

    def _needs_pull(self, filename, url=None):
        """
        Return whether we need to get the data from remote servers or not,
        comparing the last update of the files.

        Args:
            filename (str): filename of the local file
            url (str): URL of the remote server

        Returns:
            (bool): whether we need to get the data from remote servers or not

        Notes:
            If the last updated date is unknown, returns True.
            If @self.update_interval hours have passed and the remote file was updated, return True.
            If @url is None, the modified date of the remote server will not be referenced.
        """
        if filename is None or (not Path(filename).exists()):
            return True
        date_local = self._last_updated_local(filename)
        time_limit = date_local + timedelta(hours=self.update_interval)
        if datetime.now() > time_limit:
            if url is None:
                return True
            date_remote = self._last_updated_remote(url)
            if date_remote is None or date_remote > date_local:
                return True
        return False

    def jhu(self, basename=None, local_file=None, verbose=True):
        """
        Load JHU dataset (the number of cases).
        https://github.com/CSSEGISandData/COVID-19/

        Args:
            basename (str or None): basename of the file to save the data
            local_file (str or None): if not None, load the data from this file
            verbose (bool): if True, detailed citation list will be shown when downloading

        Notes:
            Regardless the value of @local_file, the data will be save in the directory.
            If @basename is None, the value of DataLoader.DH_BASENAME will be used.

        Returns:
            (covsirphy.JHUData): JHU dataset
        """
        filename = self._covid19dh_filename(basename, local_file)
        if local_file is not None:
            if Path(local_file).exists():
                jhu_data = self._create_dataset(
                    "JHU", local_file, set_citation=False
                )
                self._save(jhu_data.raw, filename)
                return jhu_data
            raise FileNotFoundError(f"{local_file} does not exist.")
        # Create th dataset using the data of COVID-19 Data Hub
        self.covid19dh(verbose=verbose)
        return self._create_dataset("JHU", filename)

    def japan(self, basename="covid_jpn_total.csv", local_file=None):
        """
        Load the datset of the number of cases in Japan.
        https://github.com/lisphilar/covid19-sir/tree/master/data

        Args:
            basename (str): basename of the file to save the data
            local_file (str or None): if not None, load the data from this file

        Notes:
            Regardless the value of @local_file, the data will be save in the directory.

        Returns:
            (covsirphy.CountryData): dataset at country level
        """
        filename = self._resolve_filename(basename)
        if local_file is not None:
            if Path(local_file).exists():
                japan_data = self._create_dataset_japan_cases(
                    local_file, set_citation=False)
                df = japan_data.cleaned()
                if set(df[self.COUNTRY].unique()) != set(["Japan"]):
                    raise TypeError(
                        f"{local_file} does not have Japan dataset."
                    )
                self._save(japan_data.raw, filename)
                return japan_data
            raise FileNotFoundError(f"{local_file} does not exist.")
        if not self._needs_pull(filename, self.japan_cases_url):
            return self._create_dataset_japan_cases(filename)
        # Retrieve the raw data
        df = self._japan_cases_get()
        # Save the dataset and return dataset
        self._save(df, filename)
        return self._create_dataset_japan_cases(filename)

    def _japan_cases_get(self):
        """
        Get the raw data from the following repository.
        https://github.com/lisphilar/covid19-sir/tree/master/data/japan

        Returns:
            (pandas.DataFrame) : the raw data
               Index:
                    reset index
                Columns:
                    as-is the repository
        """
        url = f"{self.japan_cases_url}/covid_jpn_total.csv"
        return self._get_raw(url)

    def _create_dataset_japan_cases(self, filename, set_citation=True):
        """
        Create a dataset for Japan with a local file.

        Args:
            filename (str): filename of the local file
            set_citation (str): if True, set citation

        Returns:
            (covsirphy.CountryData): dataset at country level
        """
        country_data = self._create_dataset(
            "Japan_cases", filename, country="Japan",
            set_citation=set_citation
        )
        country_data.set_variables(
            date="Date",
            confirmed="Positive",
            fatal="Fatal",
            recovered="Discharged",
            province=None
        )
        return country_data

    def population(self, basename=None, local_file=None, verbose=True):
        """
        Load Population dataset from THE WORLD BANK, Population, total.
        https://data.worldbank.org/indicator/SP.POP.TOTL

        Args:
            basename (str or None): basename of the file to save the data
            local_file (str or None): if not None, load the data from this file
            verbose (bool): if True, detailed citation list will be shown when downloading

        Notes:
            Regardless the value of @local_file, the data will be save in the directory.
            If @basename is None, the value of DataLoader.DH_BASENAME will be used.

        Returns:
            (covsirphy.Population): Population dataset
        """
        filename = self._covid19dh_filename(basename, local_file)
        if local_file is not None:
            if Path(local_file).exists():
                population_data = self._create_dataset(
                    "Population", local_file, set_citation=False)
                self._save(population_data.raw, filename)
                return population_data
            raise FileNotFoundError(f"{local_file} does not exist.")
        if not self._needs_pull(filename, self.population_url):
            return self._create_dataset("Population", filename)
        # Create th dataset using the data of COVID-19 Data Hub
        self.covid19dh(verbose=verbose)
        return self._create_dataset("Population", filename)

    def oxcgrt(self, basename=None, local_file=None, verbose=True):
        """
        Load OxCGRT dataset.
        https://github.com/OxCGRT/covid-policy-tracker

        Args:
            basename (str or None): basename of the file to save the data
            local_file (str or None): if not None, load the data from this file
            verbose (bool): if True, detailed citation list will be shown when downloading

        Notes:
            Regardless the value of @local_file, the data will be save in the directory.
            If @basename is None, the value of DataLoader.DH_BASENAME will be used.

        Returns:
            (covsirphy.OxCGRTData): OxCGRT dataset
        """
        filename = self._covid19dh_filename(basename, local_file)
        if local_file is not None:
            if Path(local_file).exists():
                oxcgrt_data = self._create_dataset(
                    "OxCGRT", local_file, set_citation=False)
                self._save(oxcgrt_data.raw, filename)
                return oxcgrt_data
            raise FileNotFoundError(f"{local_file} does not exist.")
        # Create th dataset using the data of COVID-19 Data Hub
        self.covid19dh(verbose=verbose)
        return self._create_dataset("OxCGRT", filename)

    def covid19dh(self, basename=None, verbose=True):
        """
        Load the dataset of COVID-19 Data Hub.
        https://covid19datahub.io/

        Args:
            basename (str or None): basename of the file to save the data
            verbose (bool): if True, detailed citation list will be shown when downloading

        Notes:
            Regardless the value of @local_file, the data will be save in the directory.
            If @basename is None, the value of DataLoader.DH_BASENAME will be used.

        Returns:
            (pandas.DataFrame)
        """
        basename = basename or self._covid19dh_basename
        filename = self._resolve_filename(basename)
        # Use the file saved in the directory
        if Path(filename).exists() and not self._needs_pull(filename):
            df = dd.read_csv(
                filename, dtype={"Province/State": "object"}
            ).compute()
            return df
        # Use the dataset of remote server
        df = self._covid19dh_get(verbose=verbose)
        self._save(df, filename)
        return df

    def _covid19dh_get(self, verbose=True):
        """
        Retrieve datasets from COVID-19 Data Hub.
        Level 1 (country) and level2 (province) will be used and combined to a dataframe.

        Args:
            verbose (bool): if True, detailed citation list will be shown when downloading
        """
        # Country level
        if verbose:
            print(
                "Retrieving datasets from COVID-19 Data Hub: https://covid19datahub.io/")
        c_df, p_df = self._covid19dh_retrieve()
        # Change column names and select columns to use
        # All columns: https://covid19datahub.io/articles/doc/data.html
        col_dict = {
            "date": "ObservationDate",
            "confirmed": "Confirmed",
            "recovered": "Recovered",
            "deaths": "Deaths",
            "population": "Population",
            "iso_alpha_3": "ISO3",
            "administrative_area_level_2": "Province/State",
            "administrative_area_level_1": "Country/Region",
        }
        columns = list(col_dict.values()) + OxCGRTData.OXCGRT_VARIABLES_RAW
        # Merge the datasets
        c_df = c_df.rename(col_dict, axis=1).loc[:, columns]
        p_df = p_df.rename(col_dict, axis=1).loc[:, columns]
        df = pd.concat([c_df, p_df], axis=0, ignore_index=True)
        if verbose:
            print("\nDetailed citaition list:")
            print(self._covid19dh_citation)
            print("\n\n")
        return df

    def _covid19dh_filename(self, basename, local_file):
        """
        Return the filename to save the dataset of COVID-19 Data Hub.

        Args:
            basename (str or None): basename of the file to save the data
            local_file (str or None): if not None, load the data from this file

        Returns:
            (str): absolute path of the file
        """
        if local_file is not None:
            basename = Path(local_file).name
        else:
            basename = basename or self._covid19dh_basename
        return self._resolve_filename(basename)

    @property
    def covid19dh_citation(self):
        """
        Return the citation list.
        """
        if self._covid19dh_citation is None:
            self._covid19dh_retrieve()
        return self._covid19dh_citation

    def _covid19dh_retrieve(self):
        """
        Retrieve dataset and citation list from COVID-19 Data Hub.
        Citation list will be saved to self.

        Returns:
            (tuple):
                (pandas.DataFrame): dataset at country level
                (pandas.DataFrame): dataset at province level
        """
        c_df = covid19dh.covid19(country=None, level=1, verbose=False)
        c_citations = covid19dh.cite(c_df)
        # For some countries, province-level data is included
        p_df = covid19dh.covid19(country=None, level=2, verbose=False)
        p_citations = covid19dh.cite(p_df)
        # Citation
        citations = list(dict.fromkeys(c_citations + p_citations))
        self._covid19dh_citation = "\n".join(citations)
        return (c_df, p_df)
