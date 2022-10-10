from covsirphy import StopWatch


def test_stopwatch():
    stop_watch = StopWatch()
    assert isinstance(stop_watch.stop_show(), str)
