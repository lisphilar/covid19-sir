#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from covsirphy import CountryData


def main():
    # Create output directory in example directory
    code_path = Path(__file__)
    output_dir = code_path.with_name("output").joinpath(code_path.stem)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Read Japan dataset
    jpn_file = "input/covid_jpn_total.csv"
    jpn_data = CountryData(jpn_file, country="Japan")
    jpn_data.set_variables(
        date="Date",
        confirmed="Positive",
        fatal="Fatal",
        recovered="Discharged",
        province=None
    )
    # Show the cleaned data as a CSV file
    jpn_df = jpn_data.cleaned()
    jpn_df.to_csv(output_dir.joinpath("cleaned.csv"), index=False)


if __name__ == "__main__":
    main()
