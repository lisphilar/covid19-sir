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
    models = [cs.SIR, cs.SIRD, cs.SIRF]
    # Execute validation with default setting (60 sec, 0.98, 1.02)
    validation(models, "default", output_dir)
    # Execute validation with long timeout and restricted allowance
    validation(
        models, "long_restricted", output_dir, timeout=90, allowance=(0.99, 1.01))


def validation(models, name, output_dir, **kwargs):
    """
    Perform model validation.

    Args:
        models (list[covsirphy.ModelBase]): ODE models
        name (str): name of the dataframe
        output_dir (pathlib.Path): path of the directory to save the figure
        kwargs: keyword arguments of covsirphy.ModelValidator.run()
    """
    validator = cs.ModelValidator(n_trials=8, seed=1)
    for model in models:
        validator.run(model, **kwargs)
    df = validator.summary()
    print(df)
    save_df(df, name, output_dir, "summary", use_index=False)


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
