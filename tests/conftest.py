from pathlib import Path
import warnings

warnings.simplefilter("ignore", FutureWarning)
import pandas as pd
import pytest


@pytest.fixture(scope="function")
def imgfile():
    dirpath = Path("input")
    dirpath.mkdir(exist_ok=True)
    filepath = dirpath.joinpath("test.jpg")
    yield str(filepath)
    filepath.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def c_df():
    GITHUB_URL = "https://raw.githubusercontent.com"
    URL_C = f"{GITHUB_URL}/lisphilar/covid19-sir/master/data/japan/covid_jpn_total.csv"
    path = Path("data", "japan", "covid_jpn_total.csv")
    filepath = path if path.exists() else URL_C
    df = pd.read_csv(filepath, dayfirst=False).rename(columns={"Date": "date"})
    df = df[["date", "Positive", "Tested", "Discharged", "Fatal"]].groupby("date", as_index=False).sum()
    df.insert(0, "Country", "Japan")
    return df


@pytest.fixture(scope="session")
def p_df():
    GITHUB_URL = "https://raw.githubusercontent.com"
    URL_P = f"{GITHUB_URL}/lisphilar/covid19-sir/master/data/japan/covid_jpn_prefecture.csv"
    path = Path("data", "japan", "covid_jpn_prefecture.csv")
    filepath = path if path.exists() else URL_P
    df = pd.read_csv(filepath, dayfirst=False).rename(columns={"Date": "date"})
    df.insert(0, "Country", "Japan")
    return df[["Country", "Prefecture", "date", "Positive", "Tested", "Discharged", "Fatal"]]


@pytest.fixture(scope="session")
def japan_df(c_df, p_df):
    df = pd.concat([c_df, p_df], axis=0)
    assert set(df.columns) == {"Country", "Prefecture", "date", "Positive", "Tested", "Discharged", "Fatal"}
    return df
