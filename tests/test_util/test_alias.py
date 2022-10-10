from covsirphy import Alias


def test_alias():
    alias = Alias()
    alias.update(name="alias0", target=0)
    alias.update(name="alias1", target=1)
    assert alias.find(name="alias1", default=-1) == 1
    assert alias.find(name="alias2", default=-1) == -1
    assert alias.find(name=["alias1"], default=-1) == -1
    assert alias.all() == {"alias0": 0, "alias1": 1}


def test_for_variable():
    alias = Alias.for_variables()
    assert alias.find("C") == [Alias.C]
