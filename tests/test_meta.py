import covsirphy as cs


def test_version():
    assert isinstance(cs.__version__, str)
    assert isinstance(cs.get_version(), str)


def test_citation():
    assert isinstance(cs.__citation__, str)
    assert isinstance(cs.get_citation(), str)
