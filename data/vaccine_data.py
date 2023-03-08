from pathlib import Path
import warnings
import covsirphy as cs
import pandas as pd


class VaccineData:
    """Retrieve the latest vaccine data in Japan from the following primary sources and merge them with the CSV files.
    
    - https://www.kantei.go.jp/jp/headline/kansensho/vaccine.html
    - https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/vaccine_sesshujisseki.html
    """
    URL = "https://www.kantei.go.jp/jp/content/vaccination_data5.xlsx"

    def _load(self, first_removed, **kwargs):
        """Load a sheet of the Excel file of the source and return it as a dataframe without NA rows, sorted with date index.
        
        Args:
            first_removed (bool): whether the first line of the dataframe should be removed or not
            **kwargs: keyword arguments of pandas.read_excel() except for @io and @comment
        
        Returns:
            pandas.DataFrame: the dataframe

        Note:
            https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html
        """
        df = pd.read_excel(self.URL, comment="注", **kwargs)
        if first_removed:
            df = df.drop(index=df.index[0])
        df.index = pd.to_datetime(df.index, errors="coerce")
        return df.dropna(axis="index", how="all").sort_index().convert_dtypes()

    def load_total(self):
        """Get the cumulative number of vaccinated persons.
        
        Returns:
            pandas.DataFrame: the dataframe
                Index: Date (pandas.DatetimeIndex)
                Columns:
                    1st (Int64):  the number of vaccinated persons with the 3rd dose on the date
                    2nd (Int64):  the number of vaccinated persons with the 4th dose on the date
                    3rd (Int64):  the number of vaccinated persons with the 3rd dose on the date
                    4th (Int64):  the number of vaccinated persons with the 4th dose on the date
                    5th (Int64):  the number of vaccinated persons with the 5th dose on the date
        """
        initial_df = self._load_total_1st_2nd()
        others_df = self._load_total_except_1st_2nd()
        df = initial_df.join(others_df, how="outer")
        df = df.ffill().fillna(0)
        df.index.name = "Date"
        df.columns = [f"Vaccinated_{col}" for col in df.columns]
        return df.convert_dtypes()

    def _load_total_except_1st_2nd(self):
        """Get the cumulative number of vaccinated persons (>= 3rd shot).
        
        Returns:
            pandas.DataFrame: the dataframe
                Index: pandas.DatetimeIndex
                Columns:
                    3rd (Int64):  the number of vaccinated persons with the 3rd dose on the date
                    4th (Int64):  the number of vaccinated persons with the 4th dose on the date
                    5th (Int64):  the number of vaccinated persons with the 5th dose on the date
        """
        df = self._load(
            first_removed=True,
            sheet_name="総接種回数", index_col=0, skiprows=28, 
            usecols=["Unnamed: 0", "3回目接種", "4回目接種", "5回目接種"])
        df.columns = ["3rd", "4th", "5th"]
        return df

    def _load_total_1st_2nd(self):
        """Get the cumulative number of vaccinated persons (= 1st, 2nd shot).
        
        Returns:
            pandas.DataFrame: the dataframe
                Index: pandas.DatetimeIndex
                Columns:
                    1st (Int64):  the number of vaccinated persons with the 3rd dose on the date
                    2nd (Int64):  the number of vaccinated persons with the 4th dose on the date
        """
        vrs_df = self._load_1st_2nd_vrs()
        med_df = self._load_1st_2nd_medical()
        wrk_df = self._load_1st_2nd_work()
        dup_df = self._load_1st_2nd_dup()
        df = vrs_df.join(med_df, how="outer").join(wrk_df, how="outer").join(dup_df, how="outer")
        df = df.ffill().fillna(0)
        df["1st"] = df["1st_VRS"] + df["1st_medical"] + df["1st_work"] - df["1st_dup"]
        df["2nd"] = df["2nd_VRS"] + df["2nd_medical"] + df["2nd_work"] - df["2nd_dup"]
        return df.loc[:, ["1st", "2nd"]].convert_dtypes()

    def _load_1st_2nd_vrs(self):
        """Get the cumulative number of vaccinated persons (= 1st, 2nd shot), recorded with VRS (Vaccination Record System).
        
        Returns:
            pandas.DataFrame: the dataframe
                Index: pandas.DatetimeIndex
                Columns:
                    1st_VRS (Int64):  the number of vaccinated persons with the 3rd dose on the date (recorded with VRS)
                    2nd_VRS (Int64):  the number of vaccinated persons with the 4th dose on the date (recorded with VRS)
        """
        df = self._load(
            first_removed=True,
            sheet_name="初回接種_一般接種", index_col=0, skiprows=2, header=[0, 1, 2])
        df = df.loc[:, "すべて"].T.groupby(level=0).sum().T.cumsum()
        df.columns = ["1st_VRS", "2nd_VRS"]
        return df.convert_dtypes()

    def _load_1st_2nd_medical(self):
        """Get the cumulative number of vaccinated persons (= 1st, 2nd shot), only for medical staffs.
        
        Returns:
            pandas.DataFrame: the dataframe
                Index: pandas.DatetimeIndex
                Columns:
                    1st_medical (Int64):  the number of vaccinated persons with the 3rd dose on the date (medical staffs)
                    2nd_medical (Int64):  the number of vaccinated persons with the 4th dose on the date (medical staffs)
        """
        df = self._load(
            first_removed=True,
            sheet_name="初回接種_医療従事者等", index_col=0, skiprows=2, header=[0, 1])
        df = df.loc[:, [" 内１回目", " 内２回目"]].T.groupby(level=0).sum().T.cumsum()
        df.columns = ["1st_medical", "2nd_medical"]
        # https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/vaccine_sesshujisseki.html
        df["1st_medical"] += 1101698
        df["2nd_medical"] += 490819
        return df.convert_dtypes()

    def _load_1st_2nd_work(self):
        """Get the cumulative number of vaccinated persons (= 1st, 2nd shot), only recoded in workspaces.
        
        Returns:
            pandas.DataFrame: the dataframe
                Index: pandas.DatetimeIndex
                Columns:
                    1st_work (Int64):  the number of vaccinated persons with the 3rd dose on the date (workspaces)
                    2nd_work (Int64):  the number of vaccinated persons with the 4th dose on the date (workspaces)
        """
        df = self._load(
            first_removed=False,
            sheet_name="初回接種_職域接種", index_col=2, skiprows=2, header=[0, 1])
        df = df.loc[:, [" 内１回目", " 内２回目"]].T.groupby(level=0).sum().T
        df.columns = ["1st_work", "2nd_work"]
        return df.convert_dtypes()

    def _load_1st_2nd_dup(self):
        """Get the cumulative number of vaccinated persons (= 1st, 2nd shot), duplicates of VRS and the other system (V-SYS).
        
        Returns:
            pandas.DataFrame: the dataframe
                Index: pandas.DatetimeIndex
                Columns:
                    1st_dup (Int64):  the number of vaccinated persons with the 3rd dose on the date (duplicates)
                    2nd_dup (Int64):  the number of vaccinated persons with the 4th dose on the date (duplicates)
        """
        warnings.filterwarnings("ignore", category=FutureWarning)
        df = self._load(
            first_removed=False,
            sheet_name="初回接種_重複", index_col=2, skiprows=2, header=[0, 1])
        df = df.loc[:, [" 内１回目", " 内２回目"]].T.groupby(level=0).sum().T
        df = df.resample("D").max()
        df.columns = ["1st_dup", "2nd_dup"]
        return df.convert_dtypes()


class JapanData:
    """Merge new vaccine data with the current dataset.
    
    Args:
        directory (str or pathlib.Path): path of the directory which has covid_jpn_total.csv.
    """
    def __init__(self, directory):
        self._total_csv = Path(directory).joinpath("covid_jpn_total.csv")
        self._total_df = pd.read_csv(self._total_csv)

    def uppdate_vaccine_data(self):
        """Update vaccine data of covid_jpn_total.csv.
        
        Returns:
            pandas.DataFrame:
                Index: reset index
                Columns:
                    Date, Location
                    Positive, Tested, Symptomatic, Asymptomatic, Sym-unknown,
                    Hosp_require, Hosp_mild, Hosp_severe, Hosp_unknown, Hosp_waiting, Discharged, Fatal,
                    Vaccinated_1st, Vaccinated_2nd, Vaccinated_3rd, Vaccinated_4th, Vaccinated_5th
        """
        v = VaccineData()
        v_df = v.load_total().reset_index()
        v_df["Location"] = "Domestic"
        df = self._total_df.set_index(["Date", "Location"])
        df.update(v_df.set_index(["Date", "Location"]))
        df = df.reset_index()
        for col in df.columns:
            if col.startswith("Unnamed:"):
                df = df.drop(col, axis=1)
        df.to_csv(self._total_csv, index=False)
        return df


def main():
    japan_directory = Path(__file__).resolve().parent / "japan"
    j = JapanData(directory=japan_directory)
    updated_df = j.uppdate_vaccine_data()
    df = updated_df.loc[updated_df["Location"] == "Domestic"]
    print(df.tail())
    # Postive
    cs.line_plot(
        df[["Positive"]],
        title="Japan: Postive",
        filename=japan_directory / "positive.jpg",
    )
    # Hosp_severe, Fatal
    cs.line_plot(
        df[["Hosp_severe", "Fatal"]],
        title="Japan: Hosp_severe, Fatal",
        filename=japan_directory / "severe_fatal.jpg",
    )
    # Vaccinated
    cs.line_plot(
        df[["Vaccinated_1st", "Vaccinated_2nd", "Vaccinated_3rd", "Vaccinated_4th", "Vaccinated_5th"]],
        title="Japan: Vaccinated",
        filename=japan_directory / "vaccinated.jpg",
    )


if __name__ == "__main__":
    main()
