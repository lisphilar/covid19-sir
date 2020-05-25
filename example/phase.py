#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from covsirphy import ChangeFinder
from .dataset import main as dat_main
from .population import main as pop_main


def main():
    # Create output directory in example directory
    code_path = Path(__file__)
    output_dir = code_path.with_name("output").joinpath(code_path.stem)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Read JHU dataset
    ncov_df = dat_main()
    # Read population dataset
    pop = pop_main()
    ita_population = pop.value(country="Italy")
    # Show S-R trend
    ita_change = ChangeFinder(
        ncov_df, ita_population, country="Italy"
    )
    ita_change.run(n_points=0)
    ita_change.show(filename=output_dir.joinpath("ita_trend.png"))
    # Find change points
    ita_change = ChangeFinder(
        ncov_df, ita_population, country="Italy"
    )
    ita_change.run(n_points=4)
    ita_change.show(filename=output_dir.joinpath("ita_change_points.png"))


if __name__ == "__main__":
    main()
