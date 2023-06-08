import argparse
import glob
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm

# from MONET.datamodules.components.base_dataset import BaseDataset
from MONET.utils.io import merge_hdf5
from MONET.utils.loader import custom_collate_per_batch

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="merge_files.py",
        description="merge files",
        epilog="",
    )

    parser.add_argument("-i", "--input", type=str, nargs="+", help="input path", required=True)

    parser.add_argument("-o", "--output", type=str, help="output path", required=True)

    parser.add_argument("-f", "--field", type=str, help="field", required=False)

    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    path_input_list = [Path(path_input) for path_input in args.input]
    path_output = Path(args.output)
    field = args.field

    print("Loading data...")

    if path_input_list[0].suffix == ".hdf5" and path_output.suffix == ".hdf5":
        # data_dict = load_hdf5(path_input=path_input, field=field, verbose=True)
        print("Merging hdf5 files...")
        assert all(
            [path_input.suffix == ".hdf5" for path_input in path_input_list]
        ), "All input files must be hdf5."
        merge_hdf5(
            path_input_list=path_input_list,
            path_output=path_output,
            field=field,
            overwrite=False,
        )
    if path_input_list[0].suffix == ".pt" and path_output.suffix == ".pt":
        loaded_list = []
        for path_input in tqdm.tqdm(path_input_list):
            loaded_list.append(torch.load(path_input))
        loaded_concat = custom_collate_per_batch(loaded_list)
        if field is not None:
            loaded_concat = {
                field: loaded_concat[field],
                "metadata": loaded_concat["metadata"],
            }
        torch.save(loaded_concat, path_output)

    else:
        raise ValueError(f"Unknown file type: {path_input_list[0].suffix}")
