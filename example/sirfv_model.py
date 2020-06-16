#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
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
    model = cs.SIRFV
    set_param_dict = {
        "theta": 0.002, "kappa": 0.005, "rho": 0.2, "sigma": 0.075,
        "omega": 0.001
    }
    y0_dict = {
        "Susceptible": 999_000, "Infected": 1000, "Recovered": 0, "Fatal": 0,
        "Vaccinated": 0
    }
    # Simulation
    simulator = cs.ODESimulator(country="Example", province=model.NAME)
    simulator.add(
        model=model, step_n=180, population=eg_population,
        param_dict=set_param_dict, y0_dict=y0_dict
    )
    simulator.run()
    # Non-dimensional
    nondim_df = simulator.non_dim()
    nondim_df.to_csv(output_dir.joinpath(
        f"{model.NAME}_non_dim.csv"), index=False
    )
    cs.line_plot(
        nondim_df.set_index("t"),
        title=f"{model.NAME}: Example data (non-dimensional)",
        ylabel=str(),
        h=1.0,
        filename=output_dir.joinpath(f"{model.NAME}_non_dim.png")
    )
    # Dimensional
    dim_df = simulator.dim(tau=eg_tau, start_date=start_date)
    dim_df.to_csv(output_dir.joinpath("dim.csv"), index=False)
    cs.line_plot(
        dim_df.set_index("Date"),
        title=f"{model.NAME}: Example data (dimensional)",
        h=eg_population,
        y_integer=True,
        filename=output_dir.joinpath(f"{model.NAME}_dim.png")
    )
    # Hyperparameter estimation of example data
    estimator = cs.Estimator(
        clean_df=dim_df, model=model, population=eg_population,
        country="Example", province=model.NAME, tau=eg_tau
    )
    estimator.run()
    estimated_df = estimator.summary(name=model.NAME)
    estimated_df.loc["Setted"] = pd.Series(
        {**set_param_dict, "tau": eg_tau}
    )
    estimated_df["tau"] = estimated_df["tau"].astype(np.int64)
    estimated_df.to_csv(
        output_dir.joinpath(f"{model.NAME}_estimate_parameter.csv"), index=True
    )
    # Show the history of optimization
    estimator.history(filename=output_dir.joinpath(
        f"{model.NAME}_estimate_history.png")
    )
    # Show the accuracy as a figure
    estimator.accuracy(filename=output_dir.joinpath(
        f"{model.NAME}_estimate_accuracy.png")
    )


if __name__ == "__main__":
    main()
