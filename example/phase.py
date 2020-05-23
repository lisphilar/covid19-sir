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
    print(ncov_df)
    print(population)


if __name__ == "__main__":
    main()
