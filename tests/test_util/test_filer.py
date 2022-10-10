import pytest
from covsirphy import Filer


def test_filing():
    with pytest.raises(ValueError):
        Filer("output", numbering="xxx")
    filer = Filer(directory=["output", "A"], prefix="jpn", suffix=None, numbering="01")
    filer = Filer(directory=("output", "A", "B"), prefix="jpn", suffix=None, numbering="01")
    filer = Filer(directory="output", prefix="jpn", suffix=None, numbering="01")
    # Create filenames
    filer.png("records")
    filer.jpg("records")
    filer.json("records")
    filer.geojson("records")
    filer.csv("records", index=True)
    # Check files
    assert len(filer.files(ext=None)) == 5
    assert len(filer.files(ext="png")) == 1
    assert len(filer.files(ext="jpg")) == 1
    assert len(filer.files(ext="json")) == 1
    assert len(filer.files(ext="geojson")) == 1
    assert len(filer.files(ext="csv")) == 1
