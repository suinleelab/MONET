import argparse
import glob
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

# from MONET.datamodules.components.base_dataset import BaseDataset
from MONET.utils.io import convert_dict, load_hdf5, load_pkl, save_to_hdf5, save_to_pkl

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="save_as_binary.py",
        description="save as binary",
        epilog="",
    )

    parser.add_argument("-i", "--input", type=str, help="input path", required=True)

    parser.add_argument("-o", "--output", type=str, help="output path", required=True)

    parser.add_argument("-f", "--field", type=str, help="field", required=True)

    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    path_input = Path(args.input)
    path_output = Path(args.output)
    field = args.field

    print("Loading data...")
    if path_input.suffix == ".hdf5":
        data_dict = load_hdf5(path_input=path_input, field=field, verbose=True)
    elif path_input.suffix == ".pkl":
        data_dict = load_pkl(path_input=path_input, field=field, verbose=True)
    else:
        raise ValueError(f"Unknown file type: {path_input.suffix}")

    print("Converting to binary...")
    for key, value in data_dict.items():
        print(f"ex. Converting {key}: {value}...")
        break
    data_dict = convert_dict(data_dict, method="path_to_binary")

    if path_output.suffix == ".hdf5":
        print("Saving to hdf5...")
        save_to_hdf5(data_dict=data_dict, path_output=path_output, field=field, overwrite=False)
    elif path_output.suffix == ".pkl":
        print("Saving to pkl...")
        save_to_pkl(
            data_dict=data_dict,
            path_output=path_output,
            field=field,
            overwrite=False,
        )
    else:
        raise ValueError(f"Unsupported file extension {path_output.suffix}")
