#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
When you use this file from the top directory of the repository with poetry, please run
cd example; poetry run ./model_validation.py; cd ../
"""


from pathlib import Path
import covsirphy as cs


def main():
    # This script works with version >= 2.17.0-eta
    print(cs.get_version())
    # Create output directory in example directory
    code_path = Path(__file__)
    output_dir = code_path.with_name("output").joinpath(code_path.stem)
    output_dir.mkdir(exist_ok=True, parents=True)
    filer = cs.Filer(output_dir, numbering="01")
    # Setting
    models = [cs.SIR, cs.SIRD, cs.SIRF]
    # Execute validation
    validator = cs.ModelValidator(n_trials=8, seed=1)
    for model in models:
        validator.run(model)
    validator.summary().to_csv(**filer.csv("summary", index=False))


if __name__ == "__main__":
    main()
