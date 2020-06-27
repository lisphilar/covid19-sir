#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime, timezone, timedelta
from dask import dataframe as dd
import pandas as pd
import requests
from covsirphy.cleaning.cbase import CleaningBase
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.cleaning.country_data import CountryData
from covsirphy.cleaning.oxcgrt import OxCGRTData
from covsirphy.cleaning.population import PopulationData
from covsirphy.cleaning.word import Word


class DataLoader(Word):
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
    """

    def __init__(self, directory, update_interval=12):
        if not isinstance(directory, (str, Path)):
            raise TypeError(
                f"@directory must be a string or a path but {directory} was applied."
            )
        self.dir_path = Path(directory)
        self.update_interval = self.validate_natural_int(
            update_interval, name="update_interval", include_zero=True
        )
        # Create the directory if not exist
        self.dir_path.mkdir(parents=True, exist_ok=True)
        # JHU dataset: the number of cases
        self.jhu_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master" \
            "/csse_covid_19_data/csse_covid_19_time_series"
        self.jhu_citation = "COVID-19 Data Repository" \
            " by the Center for Systems Science and Engineering (CSSE)" \
            " at Johns Hopkins University (2020)," \
            " GitHub repository," \
            " https://github.com/CSSEGISandData/COVID-19"
        self.jhu_date_col = "ObservationDate"
        self.jhu_p_col = "Province/State"
        self.jhu_c_col = "Country/Region"
        # The number of cases in Japan
        self.japan_cases_url = "https://raw.githubusercontent.com/lisphilar/covid19-sir/master/data/japan"
        self.japan_cases_citation = "Lisphilar (2020), COVID-19 dataset in Japan, GitHub repository, " \
            "https://github.com/lisphilar/covid19-sir/data/japan"
        # Population dataset: THE WORLD BANK
        self.population_url = "http://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?"
        self.population_citation = "The World Bank Group (2020), THE WORLD BANK, Population, total," \
            " https://data.worldbank.org/indicator/SP.POP.TOTL," \
            " licensed under CC BY-4.0."
        # OxCGRT dataset: Oxford Covid-19 Government Response Tracker
        self.oxcgrt_url = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data"
        self.oxcgrt_citation = "Thomas Hale, Sam Webster, Anna Petherick, Toby Phillips, and Beatriz Kira." \
            " (2020). Oxford COVID-19 Government Response Tracker. Blavatnik School of Government," \
            " https://github.com/OxCGRT/covid-policy-tracker," \
            " licensed under CC BY-4.0"
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
        filename = str(file_path.resolve())
        return filename

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
        df = self.validate_dataframe(dataframe, name="dataframe")
        if filename is None:
            return None
        df.to_csv(filename, index=False)
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
        date = pd.to_datetime(date_str).tz_convert(None)
        return date

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

    def _create_dataset(self, data_key, filename, **kwargs):
        """
        Return dataset class with citation.

        Args:
            data_key (str): key of self.dataset_dict
            filename (str): filename of the local dataset
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
        data_class = self.validate_subclass(data_class, CleaningBase)
        # Create instance and set citation
        data_instance = data_class(filename=filename, **kwargs)
        data_instance.citation = citation
        return data_instance

    def _needs_pull(self, filename, url):
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
            IF @self.update_interval hours have passed and the remote file was updated, return True.
        """
        if filename is None or (not Path(filename).exists()):
            return True
        date_local = self._last_updated_local(filename)
        time_limit = date_local + timedelta(hours=self.update_interval)
        if datetime.now() > time_limit:
            date_remote = self._last_updated_remote(url)
            if date_remote is None or date_remote > date_local:
                return True
        return False

    def jhu(self, basename="covid_19_data.csv", local_file=None):
        """
        Load JHU dataset (the number of cases).
        https://github.com/CSSEGISandData/COVID-19/

        Args:
            basename (str): basename of the file to save the data
            local_file (str or None): if not None, load the data from this file

        Notes:
            Regardless the value of @local_file, the data will be save in the directory.

        Returns:
            (covsirphy.JHUData): JHU dataset
        """
        filename = self._resolve_filename(basename)
        if local_file is not None:
            if Path(local_file).exists():
                jhu_data = self._create_dataset("JHU", local_file)
                self._save(jhu_data.raw, filename)
                return jhu_data
            raise FileNotFoundError(f"{local_file} does not exist.")
        if not self._needs_pull(filename, self.jhu_url):
            return self._create_dataset("JHU", filename)
        # Retrieve and combine the raw data
        df = self._jhu_get()
        # Save the dataset and return dataset
        self._save(df, filename)
        return self._create_dataset("JHU", filename)

    def _jhu_get(self):
        """
        Get the raw data of the variable from JHU repository.

        Args:
            variable (str): confirmed, deaths or recovered

        Returns:
            (pandas.DataFrame) : JHU data with all variables to use
               Index:
                    reset index
                Columns:
                    - SNo
                    - Province/State
                    - Country/Region
                    - Updated
                    - Confirmed
                    - Deaths
                    - Recovered
        """
        # Retrieve and combine the raw data
        df = self._jhu_get_separately("confirmed")
        deaths_df = self._jhu_get_separately("deaths")
        recovered_df = self._jhu_get_separately("recovered")
        df = pd.merge(df, deaths_df.loc[:, ["SNo", "Deaths"]], on="SNo")
        df = pd.merge(df, recovered_df.loc[:, ["SNo", "Recovered"]], on="SNo")
        # Columns will match that of Kaggle dataset
        date_stamps = pd.to_datetime(df[self.jhu_date_col])
        df[self.jhu_date_col] = date_stamps.dt.strftime("%m/%d/%Y")
        df[self.jhu_p_col] = df[self.jhu_p_col].fillna(str())
        updated_col = "Last Update"
        df[updated_col] = date_stamps.dt.strftime("%m/%d/%Y %H:%M")
        key_cols = [self.jhu_date_col, self.jhu_p_col, self.jhu_c_col]
        df = df.loc[
            :, ["SNo", *key_cols, updated_col, "Confirmed", "Deaths", "Recovered"]
        ]
        return df

    def _jhu_get_separately(self, variable):
        """
        Get the raw data of the variable from JHU repository.

        Args:
            variable (str): confirmed, deaths or recovered

        Returns:
            (pandas.DataFrame): data of the variable
                Index:
                    reset index
                Columns:
                    - Province/State
                    - Country/Region
                    - @variable
                    - SNo
        """
        # Retrieve the data
        url = f"{self.jhu_url}/time_series_covid19_{variable}_global.csv"
        df = self._get_raw(url)
        # Arrange the data
        df = df.drop(["Lat", "Long"], axis=1)
        df = df.set_index([self.jhu_c_col, self.jhu_p_col]).stack()
        df = df.reset_index()
        df.columns = [
            *df.columns.tolist()[:2], self.jhu_date_col, variable.capitalize()
        ]
        df["SNo"] = df.index + 1
        return df

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
                japan_data = self._create_dataset_japan_cases(local_file)
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
        df = self._get_raw(url)
        return df

    def _create_dataset_japan_cases(self, filename):
        """
        Create a dataset for Japan with a local file.

        Args:
            filename (str): filename of the local file

        Returns:
            (covsirphy.CountryData): dataset at country level
        """
        country_data = self._create_dataset(
            "Japan_cases", filename, country="Japan")
        country_data.set_variables(
            date="Date",
            confirmed="Positive",
            fatal="Fatal",
            recovered="Discharged",
            province=None
        )
        return country_data

    def population(self, basename="locations_population.csv", local_file=None):
        """
        Load Population dataset from THE WORLD BANK, Population, total.
        https://data.worldbank.org/indicator/SP.POP.TOTL

        Args:
            basename (str): basename of the file to save the data
            local_file (str or None): if not None, load the data from this file

        Notes:
            Regardless the value of @local_file, the data will be save in the directory.

        Returns:
            (covsirphy.Population): Population dataset
        """
        filename = self._resolve_filename(basename)
        if local_file is not None:
            if Path(local_file).exists():
                population_data = self._create_dataset(
                    "Population", local_file)
                self._save(population_data.raw, filename)
                return population_data
            raise FileNotFoundError(f"{local_file} does not exist.")
        if not self._needs_pull(filename, self.population_url):
            return self._create_dataset("Population", filename)
        # Retrieve the raw data
        df = self._population_get()
        # Save the dataset and return dataset
        self._save(df, filename)
        return self._create_dataset("Population", filename)

    def _population_get(self):
        """
        Get the raw data using the following Web API.
        http://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json&mrv=1&per_page=1000

        Returns:
            (pandas.DataFrame) : the raw data
               Index:
                    reset index
                Columns:
                    ISO3 (str): ISO3 code
                    Province.State (str): "-"
                    Country.Region (str): country name
                    Population (int): population value
                    Provenance (str): "https://data.worldbank.org/indicator/SP.POP.TOTL"
        """
        url = f"{self.population_url}format=json&mrv=1&per_page=1000"
        df = self._get_raw(url, is_json=True)
        # Change columns names
        df = df.rename(
            {
                "countryiso3code": self.ISO3,
                "country.value": "Country.Region",
                "value": "Population"
            },
            axis=1
        )
        df["Province.State"] = "-"
        df["Provenance"] = "https://data.worldbank.org/indicator/SP.POP.TOTL"
        df = df.loc[~df["Population"].isna(), :].reset_index()
        df = df.loc[
            :, [self.ISO3, "Province.State", "Country.Region", "Population", "Provenance"]
        ]
        return df

    def oxcgrt(self, basename="OxCGRT_latest.csv", local_file=None):
        """
        Load OxCGRT dataset.
        https://github.com/OxCGRT/covid-policy-tracker

        Args:
            basename (str): basename of the file to save the data
            local_file (str or None): if not None, load the data from this file

        Notes:
            Regardless the value of @local_file, the data will be save in the directory.

        Returns:
            (covsirphy.OxCGRTData): OxCGRT dataset
        """
        filename = self._resolve_filename(basename)
        if local_file is not None:
            if Path(local_file).exists():
                oxcgrt_data = self._create_dataset("OxCGRT", local_file)
                self._save(oxcgrt_data.raw, filename)
                return oxcgrt_data
            raise FileNotFoundError(f"{local_file} does not exist.")
        if not self._needs_pull(filename, self.oxcgrt_url):
            return self._create_dataset("OxCGRT", filename)
        # Retrieve the raw data
        df = self._oxcgrt_get()
        # Save the dataset and return dataset
        self._save(df, filename)
        return self._create_dataset("OxCGRT", filename)

    def _oxcgrt_get(self):
        """
        Get the raw data from the following repository.
        https://github.com/OxCGRT/covid-policy-tracker

        Returns:
            (pandas.DataFrame) : the raw data
               Index:
                    reset index
                Columns:
                    as-is the repository
        """
        url = f"{self.oxcgrt_url}/OxCGRT_latest.csv"
        df = self._get_raw(url)
        return df
