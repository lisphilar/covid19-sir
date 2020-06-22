#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
from dask import dataframe as dd
import pandas as pd
import requests
from covsirphy.cleaning.cbase import CleaningBase
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.cleaning.country_data import CountryData
from covsirphy.cleaning.word import Word


class DataLoader(Word):
    """
    Download the dataset and perform data cleaning.

    Args:
        directory <str/pathlib.Path>: directory to save the downloaded datasets

    Notes:
        If @directory is None, the files will not be saved in local environment.

    Examples:
        >>> import covsirphy as cs
        >>> data_loader = cs.DataLoader("input")
        >>> jhu_data = data_loader.jhu()
        >>> print(jhu_data.citation)
        >>> print(type(jhu_data.cleaned()))
        <class 'pandas.core.frame.DataFrame'>
        >>> jpn_data = data_loader.japan()
        >>> print(jpn_data.citation)
        >>> print(type(jpn_data.cleaned()))
        <class 'pandas.core.frame.DataFrame'>
    """

    def __init__(self, directory):
        self.dir_path = None if directory is None else Path(directory)
        # Create the directory if not exist
        if self.dir_path is not None:
            self.dir_path.mkdir(parents=True, exist_ok=True)
        # JHU data
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
        # Dictionary of datasets
        self.dataset_dict = {
            "JHU": {"class": JHUData, "url": self.jhu_url, "citation": self.jhu_citation},
            "Japan_cases": {
                "class": CountryData, "url": self.japan_cases_url, "citation": self.japan_cases_citation
            },
        }

    @staticmethod
    def _get_raw(url):
        """
        Get raw dataset from the URL.

        Args:
            url <str>: URL to get raw dataset

        Returns:
            <pandas.DataFrame>: raw dataset
        """
        df = dd.read_csv(url).compute()
        return df

    def _resolve_filename(self, basename):
        """
        Return the absolute path of the file in the @self.dir_path directory.

        Args:
            basename <str>: basename of the file, like covid_19_data.csv

        Returns:
            <str/None>: absolute path of the file

        Notes:
            If @self.dirpath is None, return None
        """
        if self.dir_path is None:
            return None
        file_path = self.dir_path / basename
        filename = str(file_path.resolve())
        return filename

    def _save(self, dataframe, filename):
        """
        Save the dataframe to the local environment.

        Args:
            dataframe <pandas.DataFrame>: dataframe to save
            filename <str/None>: filename to save

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
            url <str>: URL

        Returns:
            <pandas.Timestamp>: date last updated
        """
        response = requests.get(url)
        date = pd.to_datetime(response.headers["Date"]).tz_convert(None)
        return date

    def _last_updated_local(self, path):
        """
        Return the date last updated of local file/directory.

        Args:
            path <str/pathlibPath>: name of the file/directory

        Returns:
            <datetime.datetime>: date last updated
        """
        path = Path(path)
        m_time = path.stat().st_mtime
        date = datetime.fromtimestamp(m_time)
        return date

    def _create_dataset(self, data_key, filename, **kwargs):
        """
        Return dataset class with citation.

        Args:
            data_key <str>: key of self.dataset_dict
            filename <str>: filename of the local dataset
            kwargs: keyword arguments of @data_class

        Returns:
            <covsirphy.cleaning.jhu_data.JHUData>: the dataset

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
            filename <str>: filename of the local file
            url <str>: URL of the remote server

        Returns:
            <bool>: whether we need to get the data from remote servers or not
        """
        if filename is None or (not Path(filename).exists()):
            return True
        updated_local = self._last_updated_local(filename)
        updated_remote = self._last_updated_remote(url)
        if updated_local < updated_remote:
            return True
        return False

    def jhu(self, basename="covid_19_data.csv", local_file=None):
        """
        Load JHU dataset (the number of cases).
        https://github.com/CSSEGISandData/COVID-19/

        Args:
            basename <str>: basename of the file to save the data
            local_file <str/None>: if not None, load the data from this file

        Notes:
            Regardless the value of @local_file, the data will be save in the directory.

        Returns:
            <covsirphy.cleaning.jhu_data.JHUData>: JHU dataset
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
            variable <str>: confirmed, deaths or recovered

        Returns:
            <pandas.DataFrame> : JHU data with all variables to use
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
        last_updated_remote = self._last_updated_remote(self.jhu_url)
        df[updated_col] = last_updated_remote.strftime("%m/%d/%Y %H:%M")
        key_cols = [self.jhu_date_col, self.jhu_p_col, self.jhu_c_col]
        df = df.loc[
            :, ["SNo", *key_cols, updated_col, "Confirmed", "Deaths", "Recovered"]
        ]
        return df

    def _jhu_get_separately(self, variable):
        """
        Get the raw data of the variable from JHU repository.

        Args:
            variable <str>: confirmed, deaths or recovered

        Returns:
            <pandas.DataFrame>: data of the variable
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
            basename <str>: basename of the file to save the data
            local_file <str/None>: if not None, load the data from this file

        Notes:
            Regardless the value of @local_file, the data will be save in the directory.

        Returns:
            <covsirphy.cleaning.country_data.CountryData>: dataset at country level
        """
        filename = self._resolve_filename(basename)
        if local_file is not None:
            if Path(local_file).exists():
                country_data = self._create_dataset_japan_cases(local_file)
                self._save(country_data.raw, filename)
                return country_data
            raise FileNotFoundError(f"{local_file} does not exist.")
        if not self._needs_pull(filename, self.japan_cases_url):
            return self._create_dataset_japan_cases(filename)
        # Retrieve and combine the raw data
        df = self._japan_cases_get()
        # Save the dataset and return dataset
        self._save(df, filename)
        return self._create_dataset_japan_cases(filename)

    def _japan_cases_get(self):
        """
        Get the raw data from the following repository.
        https://github.com/lisphilar/covid19-sir/tree/master/data/japan

        Args:
            variable <str>: confirmed, deaths or recovered

        Returns:
            <pandas.DataFrame> : the raw data
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
            filename <str>: filename of the local file

        Returns:
            <covsirphy.cleaning.country_data.CountryData>: dataset at country level
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
