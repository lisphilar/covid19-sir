#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
When you use this file from the top directory of the repository with poetry, please run
cd example; poetry run ./model_validation.py; cd ../
"""

import os
import sys
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
except Exception:
    pass
from pathlib import Path
import covsirphy as cs


def main():
    # This script works with version >= 2.18.0-alpha
    print(cs.get_version())
    # Create output directory in example directory
    code_path = Path(__file__)
    output_dir = code_path.with_name("output").joinpath(code_path.stem)
    output_dir.mkdir(exist_ok=True, parents=True)
    filer = cs.Filer(output_dir, numbering="01")
    # Setting
    models = [cs.SIR, cs.SIRD, cs.SIRF]
    step_numbers = list(range(3, 10))
    # Execute validation
    for step_n in step_numbers:
        validator = cs.ModelValidator(tau=1440, n_trials=8, step_n=step_n, seed=2)
        for model in models:
            validator.run(model)
        validator.summary().to_csv(**filer.csv(f"summary_{step_n}-points", index=False))


if __name__ == "__main__":
    main()
