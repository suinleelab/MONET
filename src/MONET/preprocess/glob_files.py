import argparse
import glob
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

# from MONET.datamodules.components.base_dataset import BaseDataset
from MONET.utils.io import convert_dict, save_to_hdf5, save_to_pkl


def path_key_mapper(path, path_input_dir, style="slash_to_underscore"):

    if style == "slash_to_underscore":
        # key = "_".join(path.replace("data/textbook/pdf_extracted_new/", "/").split("/"))
        key = "_".join(path.replace(path_input_dir, "/").strip("/").split("/"))
        # "_".join(path.replace(str(path_input_dir), "/").strip("/").split("/"))
    else:
        raise ValueError(f"style {style} not supported")

    return key


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="build_hdf5.py",
        description="build hdf5 file from multiple files",
        epilog="",
    )

    parser.add_argument("-i", "--input", type=str, help="input directory", required=True)

    parser.add_argument("-o", "--output", type=str, help="output path", required=True)

    parser.add_argument("-f", "--field", type=str, help="field", required=True)

    parser.add_argument(
        "-s",
        "--style",
        help="style of key mapping",
        required=True,
    )

    parser.add_argument("-b", "--binary", help="binary", action="store_true")

    parser.add_argument(
        "-e",
        "--extension",
        help="extract files with these extensions",
        required=True,
    )

    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    path_input_dir = Path(args.input)
    path_output = Path(args.output)
    field = args.field
    style = args.style
    binary = args.binary
    include_extension = args.extension

    include_extension = include_extension.split(",")

    print("Globbing files...")
    path_list = glob.glob(str(path_input_dir / "**/*"), recursive=True)
    path_list = [path for path in path_list if os.path.isfile(path)]
    print(f"Number of files: {len(path_list)}")
    path_extension_list = [os.path.splitext(path)[1] for path in path_list]
    print(f"Extensions: {np.unique(path_extension_list, return_counts=True)}")
    path_filtered_list = [
        path for path in path_list if os.path.splitext(path)[1].lower() in include_extension
    ]
    print(f"Number of files with extension {include_extension}: {len(path_filtered_list)}")

    data_dict = OrderedDict()

    for idx, path in enumerate(path_filtered_list):
        # with open(path, "rb") as f:
        #     file_data = f.read()
        # key = path_key_mapper(str(path), str(path_input_dir), style=style)
        # data_dict[key] = file_data
        # with open(path, "rb") as f:
        #     file_data = f.read()
        key = path_key_mapper(str(path), str(path_input_dir), style=style)
        if idx == 0:
            print(f"Mapped keys. e.g. {path} -> {key}")
        data_dict[key] = path

    if binary:
        print("Converting to binary...")
        data_dict = convert_dict(data_dict, method="path_to_binary")
    print("Length of data_dict: ", len(data_dict))
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
