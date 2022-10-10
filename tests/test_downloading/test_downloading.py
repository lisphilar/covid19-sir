import pytest
from covsirphy import DataDownloader, SubsetNotFoundError, Term


@pytest.mark.parametrize(
    "country, province",
    [
        ("Japan", None),
        ("USA", None),
        ("USA", "Alabama"),
        ("UK", None),
        (None, None),
    ]
)
def test_download(country, province):
    dl = DataDownloader()
    dl.layer(country=country, province=province, databases=["japan", "covid19dh", "owid"])
    assert dl.citations()


@pytest.mark.parametrize(
    "country, province",
    [
        ("UK", "London Region"),
    ]
)
def test_download_error(country, province):
    dl = DataDownloader()
    with pytest.raises(SubsetNotFoundError):
        dl.layer(country=country, province=province, databases=["covid19dh"])


def test_download_wpp():
    dl = DataDownloader()
    c_df = dl.layer(databases=["wpp"])
    assert Term.N in c_df
    with pytest.raises(SubsetNotFoundError):
        dl.layer(databases=["wpp"], country="Japan")
    with pytest.raises(SubsetNotFoundError):
        dl.layer(databases=["wpp"], country="Japan", province="Tokyo")
