#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings
import covsirphy as cs


def main():
    warnings.simplefilter("error")
    # Create output directory in example directory
    code_path = Path(__file__)
    input_dir = code_path.parent.with_name("input")
    output_dir = code_path.with_name("output").joinpath(code_path.stem)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Create data loader instance
    data_loader = cs.DataLoader(input_dir)
    # Load Linelist of case reports
    linelist_data = data_loader.linelist()
    print(linelist_data.citation)
    linelist_data.cleaned().to_csv(
        output_dir.joinpath("linelist_cleaned.csv"), index=False)
    # Subset by area
    linelist_data.subset("Japan").to_csv(
        output_dir.joinpath("linelist_japan.csv"), index=False)
    # Global closed records (only recovered)
    linelist_data.closed(outcome="Recovered").to_csv(
        output_dir.joinpath("linelist_global_recovered.csv"), index=False)


if __name__ == "__main__":
    main()
