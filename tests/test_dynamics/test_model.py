import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
import pytest
from covsirphy import ODEModel, SIRModel, SIRDModel, SIRFModel, SEWIRFModel
from covsirphy import Term, Validator, NotNoneError, UnExpectedNoneError


@pytest.fixture(scope="module", params=[SIRModel, SIRDModel, SIRFModel, SEWIRFModel])
def model_class(request):
    return request.param


def test_not_implemented():
    model = ODEModel.from_sample()
    with pytest.raises(NotImplementedError):
        model.solve()
    with pytest.raises(NotImplementedError):
        model.transform(data=pd.DataFrame())
    with pytest.raises(NotImplementedError):
        model.inverse_transform(data=pd.DataFrame)
    with pytest.raises(NotImplementedError):
        model.r0()
    with pytest.raises(NotImplementedError):
        model.dimensional_parameters()
    with pytest.raises(NotImplementedError):
        model._param_quantile(data=pd.DataFrame(), q=0.5)
    with pytest.raises(NotImplementedError):
        model.sr(data=pd.DataFrame())


def test_special(model_class):
    model = model_class.from_sample()
    assert str(model) == model._NAME == model_class.name()
    assert model == model_class(**model.settings())
    assert model == eval(repr(model))
    assert model.definitions() == model_class.definitions()


@pytest.mark.parametrize("tau", [720, 1440])
def test_solve(model_class, tau):
    model = model_class.from_sample(tau=tau)
    df = model.solve()
    assert df.iloc[0].to_dict() == model.settings()["initial_dict"]
    Validator(df, name="analytical solution").dataframe(time_index=True, columns=model_class._VARIABLES)
    assert df.index.name == Term.DATE


@pytest.mark.parametrize("tau", [720, 1440, None])
def test_transform(model_class, tau):
    model = model_class.from_sample(tau=tau or 360)
    start_date = model.settings()["date_range"][0]
    solved_df = model.solve()
    with pytest.raises(NotNoneError):
        model_class.inverse_transform(solved_df.reset_index(drop=True), tau=None, start_date=start_date)
    with pytest.raises(UnExpectedNoneError):
        model_class.inverse_transform(solved_df.reset_index(drop=True), tau=tau or 360, start_date=None)
    actual_df = model_class.inverse_transform(solved_df, tau=tau, start_date=None if tau is None else start_date)
    trans_df = model_class.transform(actual_df, tau=tau)
    if issubclass(model_class, SEWIRFModel):
        return
    assert_frame_equal(model_class.transform(actual_df), solved_df)
    assert_frame_equal(trans_df.reset_index(drop=True), solved_df.reset_index(drop=True))


def test_r0(model_class):
    model = model_class.from_sample()
    assert model.r0() > 0
    with pytest.raises(ZeroDivisionError):
        _dict = model.settings()
        _dict.update(param_dict={param: 0 for param in _dict["param_dict"].keys()})
        model_class(**_dict).r0()


def test_dimensional_parameters(model_class):
    model = model_class.from_sample()
    assert model.dimensional_parameters()
    with pytest.raises(ZeroDivisionError):
        _dict = model.settings()
        _dict.update(param_dict={param: 0 for param in _dict["param_dict"].keys()})
        model_class(**_dict).dimensional_parameters()


@pytest.mark.parametrize("tau", [720, 1440])
def test_from_data(model_class, tau):
    sample_model = model_class.from_sample(tau=tau)
    sample_df = sample_model.solve()
    sample_dict = sample_model.settings()
    trans_df = model_class.inverse_transform(sample_df).reset_index()
    model = model_class.from_data(data=trans_df, param_dict=sample_dict["param_dict"], tau=tau, digits=None)
    assert model.settings(with_estimation=True)["estimation_dict"]["method"] == "not_performed"
    solved_df = model.solve()
    if issubclass(model_class, SEWIRFModel):
        return
    assert_index_equal(solved_df.index, sample_df.index)
    assert_series_equal(solved_df.iloc[0], sample_df.iloc[0])


@pytest.mark.parametrize("tau", [720, 1440])
@pytest.mark.parametrize("q", [0.5])
def test_from_data_with_quantile(model_class, tau, q):
    if issubclass(model_class, SEWIRFModel):
        with pytest.raises(NotImplementedError):
            model_class.from_data_with_quantile()
        return
    sample_model = model_class.from_sample(tau=tau)
    sample_df = sample_model.solve()
    trans_df = model_class.inverse_transform(sample_df).reset_index()
    model = model_class.from_data_with_quantile(data=trans_df, tau=tau, q=q, digits=2)
    assert model.settings(with_estimation=True)["estimation_dict"]["method"] == "with_quantile"
    solved_df = model.solve()
    assert_index_equal(solved_df.index, sample_df.index)
    assert_series_equal(solved_df.iloc[0], sample_df.iloc[0])


@pytest.mark.parametrize("tau", [720, 1440])
@pytest.mark.parametrize("metric", ["RMSLE"])
def test_from_data_with_optimization(model_class, tau, metric):
    if issubclass(model_class, SEWIRFModel):
        with pytest.raises(NotImplementedError):
            model_class.from_data_with_optimization()
        return
    sample_model = model_class.from_sample(tau=tau)
    sample_df = sample_model.solve()
    trans_df = model_class.inverse_transform(sample_df).reset_index()
    model = model_class.from_data_with_optimization(data=trans_df, tau=tau, metric=metric, digits=4)
    assert model.settings(with_estimation=True)["estimation_dict"]["method"] == "with_optimization"
    solved_df = model.solve()
    assert_index_equal(solved_df.index, sample_df.index)
    assert_series_equal(solved_df.iloc[0], sample_df.iloc[0])


@pytest.mark.parametrize("tau", [720])
@pytest.mark.parametrize("metric", ["RMSLE"])
def test_from_data_with_optimization_with_infinity(model_class, tau, metric):
    if issubclass(model_class, SEWIRFModel):
        return
    sample_model = model_class.from_sample(tau=tau)
    sample_df = sample_model.solve()
    trans_df = model_class.inverse_transform(sample_df).reset_index()
    trans_df[Term.R] = 0
    model_class.from_data_with_optimization(data=trans_df, tau=tau, metric=metric, digits=4)


def test_sr(model_class):
    if issubclass(model_class, SEWIRFModel):
        return
    sample_model = model_class.from_sample()
    record_df = model_class.inverse_transform(sample_model.solve())
    sr_df = sample_model.sr(record_df)
    sr_df.corr()
    assert set(sr_df.reset_index().columns) == {model_class.DATE, model_class._logS, model_class._r}
