#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings
import covsirphy as cs


def main():
    warnings.simplefilter("error")
    # Create output directory in example directory
    code_path = Path(__file__)
    output_dir = code_path.with_name("output").joinpath(code_path.stem)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Setting
    validator = cs.ModelValidator(n_trials=8, seed=1)
    # Execute validation with default setting (60 sec, 0.98, 1.02)
    validator.run(cs.SIR)
    validator.run(cs.SIRD)
    validator.run(cs.SIRF)
    print(validator.summary())
    save_df(
        validator.summary(), name="default.csv", country="summary", use_index=False)
    # Execute validation with long timeout and default allowance setting (0.98, 1.02)
    validator.run(cs.SIR, timeout=90)
    validator.run(cs.SIRD, timeout=90)
    validator.run(cs.SIRF, timeout=90)
    print(validator.summary())
    save_df(
        validator.summary(), name="long.csv", country="summary", use_index=False)
    # Execute validation with long timeout and restricted allowance
    validator.run(cs.SIR, timeout=90, allowance=(0.99, 1.01))
    validator.run(cs.SIRD, timeout=90, allowance=(0.99, 1.01))
    validator.run(cs.SIRF, timeout=90, allowance=(0.99, 1.01))
    print(validator.summary())
    save_df(
        validator.summary(), name="long_restricted.csv", country="summary", use_index=False)


def save_df(df, name, output_dir, country, use_index=True):
    """
    Save the dataframe as a CSV file.

    Args:
        df (pandas.DataFrame): dataframe
        name (str): name of the dataframe
        output_dir (pathlib.Path): path of the directory to save the figure
        country (str): country name or abbr
        use_index (bool): if True, include index
    """
    df.to_csv(output_dir.joinpath(f"{name}.csv"), index=use_index)


if __name__ == "__main__":
    main()
