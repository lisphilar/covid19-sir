#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
# from covsirphy import Estimator, SIRF
from covsirphy import Trend
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
    ita_population = pop.value(country="Japan")
    # Trend analysis
    ita_trend = Trend(
        ncov_df, ita_population, country="Japan"
    )
    ita_trend.analyse()
    ita_trend.rmsle()
    ita_trend.show(filename=output_dir.joinpath("ita_trend.jpg"))


if __name__ == "__main__":
    main()
