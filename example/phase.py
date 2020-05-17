#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from covsirphy import Estimator, SIRF
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
    # Non-dimentional dataset
    estimator = Estimator(
        ncov_df, country="Italy", province=None,
        model=SIRF, population=population,
        start_date="28Mar2020", end_date="04Apr2020"
    )
    estimator.train_df.to_csv(output_dir.joinpath("train.csv"), index=True)


if __name__ == "__main__":
    main()
