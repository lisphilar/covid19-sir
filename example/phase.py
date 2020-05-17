#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from covsirphy import NondimData, SIRF
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
    population = pop.value(country="Italy")
    # Analyzable dataset
    nondim_data = NondimData(ncov_df, country="Italy")
    train_df = nondim_data.make(
        model=SIRF, population=population,
        start_date="01May2020", end_date=None
    )
    train_df.to_csv(output_dir.joinpath("train.csv"), index=True)


if __name__ == "__main__":
    main()
