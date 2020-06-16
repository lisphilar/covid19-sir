#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import covsirphy as cs


def main():
    # Create output directory in example directory
    code_path = Path(__file__)
    output_dir = code_path.with_name("output").joinpath(code_path.stem)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Read OxCGRT data
    oxcgrt_file = "input/oxcgrt/OxCGRT_latest.csv"
    oxcgrt_data = cs.OxCGRTData(oxcgrt_file)
    oxcgrt_df = oxcgrt_data.cleaned()
    oxcgrt_df.to_csv(
        output_dir.joinpath("oxcgrt_cleaned.csv"), index=False
    )
    oxcgrt_data.subset(iso3="JPN").to_csv(
        output_dir.joinpath("oxcgrt_cleaned_jpn.csv"), index=False
    )


if __name__ == "__main__":
    main()
