#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
from dask import dataframe as dd
import pandas as pd
import requests
from covsirphy.cleaning.word import Word
from covsirphy.cleaning.jhu_data import JHUData


class DataLoader(Word):
    """
    Download the dataset and perform data cleaning.

    Args:
        directory <str/pathlib.Path>: directory to save the downloaded datasets

    Notes:
        If @directory is None, the files will not be saved in local environment.

    Examples:
        >>> data_loader = DataLoader("../input")
        >>> jhu_data = data_loader.jhu()
        >>> print(jhu_data.citation)
        >>> print(type(jhu_data.cleaned()))
        <class 'pandas.core.frame.DataFrame'>
    """
    # JHU dataset
    JHU_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master" \
              "/csse_covid_19_data/csse_covid_19_time_series"
    JHU_CITATION = "COVID-19 Data Repository" \
        " by the Center for Systems Science and Engineering (CSSE)" \
        " at Johns Hopkins University (2020)," \
        " GitHub repository," \
        " https://github.com/CSSEGISandData/COVID-19"

    def __init__(self, directory):
        self.dir_path = None if directory is None else Path(directory)
        # Create the directory if not exist
        if self.dir_path is not None:
            self.dir_path.mkdir(parents=True, exist_ok=True)
        # JHU data
        self.jhu_date_col = "ObservationDate"
        self.jhu_p_col = "Province/State"
        self.jhu_c_col = "Country/Region"

    def jhu(self):
        """
        Load JHU dataset (the number of cases).
        https://github.com/CSSEGISandData/COVID-19/

        Returns:
            <covsirphy.cleaning.jhu_data.JHUData>: JHU dataset
        """
        last_updated_remote = self._last_updated_remote(self.JHU_URL)
        filename = self._resolve_filename("covid_19_data.csv")
        if filename is not None and Path(filename).exists():
            last_updated_local = self._last_updated_local(filename)
            if last_updated_local > last_updated_remote:
                return self._jhu(filename)
        # Get the raw data
        df = self._jhu_get("confirmed")
        # Combine the data
        deaths_df = self._jhu_get("deaths").loc[:, ["SNo", "Deaths"]]
        recovered_df = self._jhu_get("recovered").loc[:, ["SNo", "Recovered"]]
        df = pd.merge(df, deaths_df, on="SNo")
        df = pd.merge(df, recovered_df, on="SNo")
        # Arrange the data
        date_stamps = pd.to_datetime(df[self.jhu_date_col])
        df[self.jhu_date_col] = date_stamps.dt.strftime("%m/%d/%Y")
        df[self.jhu_p_col] = df[self.jhu_p_col].fillna(str())
        updated_col = "Last Update"
        df[updated_col] = last_updated_remote.strftime("%m/%d/%Y %H:%M")
        key_cols = [self.jhu_date_col, self.jhu_p_col, self.jhu_c_col]
        df = df.loc[
            :, ["SNo", *key_cols, updated_col, "Confirmed", "Deaths", "Recovered"]
        ]
        # Save the dataset and return dataset
        self._save(df, filename)
        return self._jhu(filename)

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

    def _jhu_get(self, variable):
        """
        Get the data from JHU repository.

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
        url = f"{self.JHU_URL}/time_series_covid19_{variable}_global.csv"
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

    def _jhu(self, filename):
        """
        Return JHU dataset.

        Args:
            filename <str>: filename of the local dataset

        Returns:
            <covsirphy.cleaning.jhu_data.JHUData>: the dataset
        """
        jhu_data = JHUData(filename)
        jhu_data.citation = self.JHU_CITATION
        return jhu_data
