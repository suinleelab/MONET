# import pyrootutils
# import os
# os.chdir(pyrootutils.find_root())

import argparse

# import sys
# sys.path.append(str(pyrootutils.find_root()/"src"))
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from MONET.utils.io import filter_hdf5, load_pkl, save_to_pkl

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="filter_cluster.py",
        description="filter samples using clustering algorithms",
        epilog="",
    )

    # parser_filter = sub_parsers.add_parser("filter", help="filter")
    parser.add_argument("-i", "--input", type=str, help="input path", required=True)
    parser.add_argument("--label-file", type=str, help="label path", required=True)

    parser.add_argument("-o", "--output", type=str, help="output path", required=True)

    parser.add_argument(
        "--exclude-label",
        nargs="+",
        help="extract files with labels",
        required=True,
    )

    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    path_input = Path(args.input)
    label_file = Path(args.label_file)
    path_output = Path(args.output)
    exclude_label = args.exclude_label

    print("Loading label...")
    kmeans_label = pd.read_csv(
        label_file, index_col=0, dtype={"label": "str"}, header=0, names=["label"]
    )
    kmeans_label["include"] = True
    print(kmeans_label.value_counts())

    print("Finding samples to exclude...")
    for label in exclude_label:
        num_included_before = sum(kmeans_label["include"])
        if "_" in label:
            kmeans_label.loc[kmeans_label["label"] == label, "include"] = False
        else:
            kmeans_label.loc[kmeans_label["label"].str.startswith(label + "_"), "include"] = False
        num_included_after = sum(kmeans_label["include"])

        print(
            f"Excluding {label}... Num of samples: {num_included_before} - {num_included_before - num_included_after} = {num_included_after}"
        )

    print("Loading dataset...")
    key_list = kmeans_label.index[kmeans_label["include"].values].tolist()

    if path_output.suffix == ".hdf5":
        print("filtering to hdf5...")
        filter_hdf5(
            path_input=path_input,
            path_output=path_output,
            key_list=key_list,
            field="images",
            overwrite=False,
        )
    elif path_output.suffix == ".pkl":
        print("filtering to pkl...")
        data_dict = load_pkl(path_input=path_input, field="images", verbose=True)
        new_data_dict = OrderedDict()
        for key in key_list:
            new_data_dict[key] = data_dict[key]
        save_to_pkl(new_data_dict, path_output, field="images", overwrite=False)
    else:
        raise ValueError("output file must be hdf5 or pkl")
