#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from covsirphy import JHUData


def main():
    # Create output directory in example directory
    code_path = Path(__file__)
    output_dir = code_path.with_name("output").joinpath(code_path.stem)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Read JHU dataset
    jhu_file = "input/covid_19_data.csv"
    jhu_data = JHUData(jhu_file)
    # Show the cleaned data as a CSV file
    ncov_df = jhu_data.cleaned()
    ncov_df.to_csv(output_dir.joinpath("cleaned.csv"), index=False)


if __name__ == "__main__":
    main()
