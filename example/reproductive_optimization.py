#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import covsirphy as cs


def main():
    # Create output directory in example directory
    code_path = Path(__file__)
    output_dir = code_path.with_name("output").joinpath(code_path.stem)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Setting
    eg_population = 1_000_000
    eg_tau = 1440
    start_date = "22Jan2020"
    model = cs.SIRF
    set_param_dict = {
        "theta": 0.002, "kappa": 0.005, "rho": 0.2, "sigma": 0.075
    }
    set_param_dict_2 = {
        "theta": 0.002, "kappa": 0.005, "rho": 0.1, "sigma": 0.075
    }
    y0_dict = {
        "Susceptible": 999_000, "Infected": 1000, "Recovered": 0, "Fatal": 0
    }
    # Dataset for S-R trend analysis
    simulator = cs.ODESimulator(country="Trend", province=model.NAME)
    simulator.add(
        model=model, step_n=30, population=eg_population,
        param_dict=set_param_dict, y0_dict=y0_dict
    )
    simulator.add(
        model=model, step_n=30, population=eg_population,
        param_dict=set_param_dict_2
    )
    simulator.run()
    dim_df = simulator.dim(tau=eg_tau, start_date=start_date)
    cs.line_plot(
        dim_df.set_index("Date"),
        title=f"{model.NAME}: Example data for S-R trend analysis",
        h=eg_population,
        y_integer=True,
        filename=output_dir.joinpath("trend_data.png")
    )
    restored_df = model.restore(dim_df)
    # 1st trial of S-R trend analysis
    change_finder = cs.ChangeFinder(restored_df, eg_population, "Trend")
    change_finder.run(n_points=1, n_jobs=1, seed=0)
    change_finder.show(filename=output_dir.joinpath("trend_1st.png"))
    # 2nd trial of S-R trend analysis
    change_finder = cs.ChangeFinder(restored_df, eg_population, "Trend")
    change_finder.run(n_points=1, n_jobs=1, seed=0)
    change_finder.show(filename=output_dir.joinpath("trend_2nd.png"))
    # Dataset for parameter estimation
    simulator = cs.ODESimulator(country="Param", province=model.NAME)
    simulator.add(
        model=model, step_n=150, population=eg_population,
        param_dict=set_param_dict, y0_dict=y0_dict
    )
    simulator.run()
    dim_df = simulator.dim(tau=eg_tau, start_date=start_date)
    cs.line_plot(
        dim_df.set_index("Date"),
        title=f"{model.NAME}: Example data for parameter estimation",
        h=eg_population,
        y_integer=True,
        filename=output_dir.joinpath("estimate_data.png")
    )
    # 1st trial of parameter estimation
    estimator = cs.Estimator(
        clean_df=dim_df, model=model, population=eg_population,
        country="Param", province=model.NAME, tau=eg_tau
    )
    estimator.run(n_jobs=1, seed=0)
    estimator.history(filename=output_dir.joinpath("estimate_history_1st.png"))
    # 2nd trial of parameter estimation
    estimator = cs.Estimator(
        clean_df=dim_df, model=model, population=eg_population,
        country="Param", province=model.NAME, tau=eg_tau
    )
    estimator.run(n_jobs=1, seed=0)
    estimator.history(filename=output_dir.joinpath("estimate_history_2nd.png"))


if __name__ == "__main__":
    main()
