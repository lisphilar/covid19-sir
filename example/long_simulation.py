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
    y0_dict = {
        "Susceptible": 999_000, "Infected": 1000, "Recovered": 0, "Fatal": 0
    }
    # Simulation
    simulator = cs.ODESimulator(country="Example", province=model.NAME)
    simulator.add(
        model=cs.SIRF, step_n=1000, population=eg_population,
        param_dict=set_param_dict, y0_dict=y0_dict
    )
    simulator.run()
    # Non-dimensional
    nondim_df = simulator.non_dim()
    nondim_df.to_csv(output_dir.joinpath("non_dim.csv"), index=False)
    cs.line_plot(
        nondim_df.set_index("t"),
        title=f"{model.NAME}: Example data (non-dimensional)",
        ylabel=str(),
        h=1.0,
        filename=output_dir.joinpath("non_dim_long.png")
    )
    # Dimensional
    dim_df = simulator.dim(tau=eg_tau, start_date=start_date)
    dim_df.to_csv(output_dir.joinpath("dim.csv"), index=False)
    cs.line_plot(
        dim_df.set_index("Date"),
        title=f"{model.NAME}: Example data (dimensional)",
        h=eg_population,
        y_integer=True,
        filename=output_dir.joinpath("dim_long.png")
    )


if __name__ == "__main__":
    main()
