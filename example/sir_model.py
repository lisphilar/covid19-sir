#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import covsirphy as cs


def main():
    warnings.simplefilter("error")
    # Create output directory in example directory
    code_path = Path(__file__)
    output_dir = code_path.with_name("output").joinpath(code_path.stem)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Setting
    eg_tau = 1440
    eg_population = 1_000_000
    model = cs.SIR
    eg_population = model.EXAMPLE["population"]
    set_param_dict = model.EXAMPLE["param_dict"]
    # Simulation
    example_data = cs.ExampleData(tau=eg_tau)
    example_data.add(model)
    # Non-dimensional
    nondim_df = example_data.non_dim(model)
    nondim_df.to_csv(output_dir.joinpath(
        f"{model.NAME}_non_dim.csv"), index=False)
    cs.line_plot(
        nondim_df.set_index("t"),
        title=f"{model.NAME}: Example data (non-dimensional)",
        ylabel=str(),
        h=1.0,
        filename=output_dir.joinpath(f"{model.NAME}_non_dim.png")
    )
    # Dimensional
    dim_df = example_data.cleaned()
    dim_df.to_csv(output_dir.joinpath(f"{model.NAME}_dim.csv"), index=False)
    cs.line_plot(
        dim_df.set_index("Date").drop("Confirmed", axis=1),
        title=f"{model.NAME}: Example data (dimensional)",
        h=eg_population,
        y_integer=True,
        filename=output_dir.joinpath(f"{model.NAME}_dim.png")
    )
    # Hyperparameter estimation of example data
    estimator = cs.Estimator(
        example_data.subset(model), model=model, population=eg_population,
        country=model.NAME, province=None, tau=eg_tau
    )
    estimator.run()
    estimated_df = pd.DataFrame.from_dict(
        {model.NAME: estimator.to_dict()}, orient="index")
    estimated_df = estimated_df.append(
        pd.Series({**set_param_dict, "tau": eg_tau}, name="set")
    )
    estimated_df["tau"] = estimated_df["tau"].astype(np.int64)
    estimated_df.to_csv(
        output_dir.joinpath("estimate_parameter.csv"), index=True
    )
    # Show the history of optimization
    estimator.history(filename=output_dir.joinpath("estimate_history.png"))
    # Show the accuracy as a figure
    estimator.accuracy(filename=output_dir.joinpath("estimate_accuracy.png"))


if __name__ == "__main__":
    main()
