#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import covsirphy as cs


def main():
    # Create output directory in example directory
    code_path = Path(__file__)
    input_dir = code_path.parent.with_name("input")
    output_dir = code_path.with_name("output").joinpath(code_path.stem)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Read JHU dataset
    data_loader = cs.DataLoader(input_dir)
    jhu_data = data_loader.jhu()
    print(jhu_data.citation)
    ncov_df = jhu_data.cleaned()
    ncov_df.to_csv(output_dir.joinpath("jhu_cleaned.csv"), index=False)


if __name__ == "__main__":
    main()
