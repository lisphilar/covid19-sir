import pandas as pd
import pytest
from covsirphy import GIS, NotRegisteredError, SubsetNotFoundError


def test_all(c_df, p_df):
    system = GIS(layers=["Country", "Province"], country="Country", date="Date")
    with pytest.raises(NotRegisteredError):
        system.all(variables=["Positive"])
    system.register(data=c_df, layers=["Country"], date="date", citations="Country-level")
    system.register(data=p_df, layers=["Country", "Prefecture"], date="date", citations="Prefecture-level")
    all_df = system.all(variables=["Positive"])
    assert all_df.columns.tolist() == ["Country", "Province", "Date", "Positive"]
    assert set(system.citations()) == {"Country-level", "Prefecture-level"}


@pytest.mark.parametrize(
    "geo, end_date, length",
    [
        (None, "31Dec2021", 365),
        ((None,), "31Dec2021", 365),
        ("Japan", "31Dec2020", 0),
        ("Japan", "31Dec2021", 365 * 47),
        (("Japan",), "31Dec2021", 365 * 47),
        ((["Japan", "UK"],), "31Dec2021", 365 * 47),
        ("UK", "31Dec2021", 0),
    ]
)
def test_layer(c_df, p_df, geo, end_date, length):
    system = GIS(layers=["Country", "Province"], country="Country", date="Date")
    with pytest.raises(NotRegisteredError):
        system.layer(geo=geo, start_date="01Jan2021", end_date=end_date, variables=None)
    system.register(data=c_df, layers=["Country"], date="date", citations="Country-level")
    system.register(data=p_df, layers=["Country", "Prefecture"], date="date", citations="Prefecture-level")
    if length == 0:
        with pytest.raises(NotRegisteredError):
            system.layer(geo=geo, start_date="01Jan2021", end_date=end_date, variables=None)
    else:
        df = system.layer(geo=geo, start_date="01Jan2021", end_date=end_date, variables=None)
        assert set(df.columns) == {"Country", "Province", "Date", "Positive", "Tested", "Discharged", "Fatal"}
        assert len(df) == length


@pytest.mark.parametrize("geo, on", [(None, "01Jan2022"), ("Japan", None), ("Japan", "01Feb2022")])
def test_to_geo_pandas_choropleth(c_df, p_df, geo, on, imgfile):
    with pytest.raises(ValueError):
        GIS(layers=["Province"], country="Country").to_geopandas()
    with pytest.raises(ValueError):
        GIS(layers=["Country", "Province"], country=None).to_geopandas()
    system = GIS(layers=["Country", "Province"], country="Country", date="Date")
    system.register(data=c_df, layers=["Country"], date="date", citations="Country-level")
    system.register(data=p_df, layers=["Country", "Prefecture"], date="date", citations="Prefecture-level")
    gdf = system.to_geopandas(geo=geo, on=on)
    if geo is None:
        assert "Province" not in gdf.columns
        assert gdf["Country"].unique() == ["JPN"]
    else:
        assert "Country" not in gdf.columns
        assert gdf["Province"].nunique() == 47
    if on is None:
        assert gdf["Date"].max() == system.layer(geo=geo)["Date"].max()
    else:
        assert pd.to_datetime(gdf["Date"].unique()) == [pd.to_datetime(on)]
    # Choropleth map
    system.choropleth(geo=geo, on=on, variable="Positive", filename=imgfile)


@pytest.mark.parametrize(
    "geo, end_date, length",
    [
        (None, "31Dec2021", 365),
        ((None,), "31Dec2021", 365),
        ("Japan", "31Dec2020", 0),
        ("Japan", "31Dec2021", 365),
        (("Japan",), "31Dec2021", 365),
        ((["Japan", "UK"],), "31Dec2021", 365),
        ("UK", "31Dec2021", 0),
    ]
)
def test_subset(c_df, p_df, geo, end_date, length):
    system = GIS(layers=["Country", "Province"], country="Country", date="Date")
    with pytest.raises(NotRegisteredError):
        system.subset(geo=geo, start_date="01Jan2021", end_date=end_date, variables=None)
    system.register(data=c_df, layers=["Country"], date="date", citations="Country-level")
    system.register(data=p_df, layers=["Country", "Prefecture"], date="date", citations="Prefecture-level")
    if length == 0:
        with pytest.raises(SubsetNotFoundError):
            system.subset(geo=geo, start_date="01Jan2021", end_date=end_date, variables=None)
    else:
        df = system.subset(geo=geo, start_date="01Jan2021", end_date=end_date, variables=None)
        assert set(df.columns) == {"Date", "Positive", "Tested", "Discharged", "Fatal"}
        assert len(df) == length


@pytest.mark.parametrize(
    "geo, answer",
    [
        (None, "the world"),
        ((None,), "the world"),
        ("Japan", "Japan"),
        (("Japan",), "Japan"),
        ((["Japan", "UK"],), "Japan_UK"),
        (("Japan", "Tokyo"), "Tokyo/Japan"),
    ]
)
def test_area_name(geo, answer):
    assert GIS.area_name(geo) == answer
