# import pyrootutils
# import os
# os.chdir(pyrootutils.find_root())

import argparse

# import sys
# sys.path.append(str(pyrootutils.find_root()/"src"))
import os
import shutil
import tarfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.cluster import KMeans

from MONET.datamodules.components.base_dataset import BaseDataset
from MONET.utils.io import load_hdf5, load_pkl
from MONET.utils.plotting import stack_images

if __name__ == "__main__":
    # python src/MONET/preprocess/filter_clusters.py --step cluster --input data/textbook/pdf_extracted_temp.hdf5 --output data/textbook/images_clustering --device cuda:2

    parser = argparse.ArgumentParser(
        prog="filter_cluster.py",
        description="filter samples using clustering algorithms",
        epilog="",
    )

    parser.add_argument("-i", "--input", type=str, help="input path", required=True)
    parser.add_argument(
        "--input-featurized-file", type=str, help="input featurized path", required=True
    )
    parser.add_argument(
        "--reference-featurized-file",
        type=str,
        help="reference featurized path",
        required=True,
    )
    parser.add_argument("-o", "--output", type=str, help="output path", required=True)
    parser.add_argument("-f", "--feature-to-use", type=str, help="feature to use", required=True)
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    path_input = Path(args.input)
    input_featurized_file = Path(args.input_featurized_file)
    reference_featurized_file = Path(args.reference_featurized_file)
    path_output_dir = Path(args.output)

    feature_to_use = args.feature_to_use

    if not os.path.exists(path_output_dir):  # make subfolder if needed
        os.mkdir(path_output_dir)
        print(f"Created output directory: {path_output_dir}")

    print("Loading featurized data...")
    input_featurized = torch.load(input_featurized_file)
    reference_featurized = torch.load(reference_featurized_file)

    print("Initializing dataset...")
    if path_input.suffix == ".hdf5":
        data_dict = load_hdf5(path_input=path_input, field="images", verbose=True)
    elif path_input.suffix == ".pkl":
        data_dict = load_pkl(path_input=path_input, field="images", verbose=True)
    else:
        raise ValueError(f"Unknown file type: {path_input.suffix}")

    dataset = BaseDataset(
        image_path_or_binary_dict=data_dict,
        n_px=224,
        norm_mean=(0.48145466, 0.4578275, 0.40821073),
        norm_std=(0.26862954, 0.26130258, 0.27577711),
        augment=False,
        metadata_all=input_featurized["metadata"],
    )

    # metadata_all = pd.read_csv(str(path_input) + ".tsv", sep="\t", index_col=0)
    # assert (
    #     metadata_all.index == loader_applied["metadata"].index
    # ).all(), "Metadata mismatch"
    # dataset = initialize_hdf5_image_dataset(
    #     path_hdf5=str(path_input) + ".hdf5",
    #     metadata_all=metadata_all,
    #     config={
    #         "n_px": 224,
    #         "norm_mean": (0.48145466, 0.4578275, 0.40821073),
    #         "norm_std": (0.26862954, 0.26130258, 0.27577711),
    #         "augment": False,
    #     },
    #     verbose=True,
    # )

    # if feature_to_use == "resnet":
    #     input_feature = input_featurized["resnet_feature"]
    #     reference_feature = reference_featurized["resnet_feature"]
    # elif feature_to_use == "clip":
    #     input_feature = input_featurized["clip_feature"]
    #     reference_feature = reference_featurized["clip_feature"]
    # elif feature_to_use == "efficientnet":
    #     input_feature = input_featurized["efficientnet_feature"]
    #     reference_feature = reference_featurized["efficientnet_feature"]
    # else:
    #     raise ValueError(f"Unknown feature: {feature_to_use}")

    # input_feature = input_feature.numpy()
    # reference_feature = reference_feature.numpy()

    # value = np.array(
    #     [
    #         np.square(x - reference_feature).sum(axis=1).mean(axis=0)
    #         for x in tqdm.tqdm(input_feature)
    #     ]
    # )

    # value = 1 - value
    # value_normalized = (value - np.min(value)) / (np.max(value) - np.min(value)) * 100
    # for idx in tqdm.tqdm(np.arange(100)[::-1]):
    #     idx_true = (value_normalized > idx) & (value_normalized <= (idx + 1))
    #     input_featurized["metadata"].index
    #     sample_id_list = np.random.RandomState(42).choice(
    #         input_featurized["metadata"].index[idx_true],
    #         size=min(50, sum(idx_true)),
    #         replace=False,
    #     )
    #     image_list = [
    #         dataset.getitem(dataset.sample_id_to_idx(sample_id))["image"]
    #         for sample_id in sample_id_list
    #     ]
    #     stack_images(image_list, path=path_output_dir / f"{idx:02d}.jpg")

    if feature_to_use == "resnet":
        input_feature = input_featurized["resnet_feature"]
        reference_feature = reference_featurized["resnet_feature"]
    elif feature_to_use == "clip":
        input_feature = input_featurized["clip_feature"]
        reference_feature = reference_featurized["clip_feature"]
    elif feature_to_use == "efficientnet":
        input_feature = input_featurized["efficientnet_feature"]
        reference_feature = reference_featurized["efficientnet_feature"]
    else:
        raise ValueError(f"Unknown feature: {feature_to_use}")

    input_feature /= input_feature.norm(dim=-1, keepdim=True)
    reference_feature /= reference_feature.norm(dim=-1, keepdim=True)
    input_feature = input_feature.numpy()
    reference_feature = reference_feature.numpy()
    value = (input_feature @ reference_feature.T).mean(axis=1)

    value_normalized = (value - np.min(value)) / (np.max(value) - np.min(value)) * 100
    for idx in tqdm.tqdm(np.arange(100)[::-1]):
        idx_true = (value_normalized > idx) & (value_normalized <= (idx + 1))

        sample_id_list = np.random.RandomState(42).choice(
            input_featurized["metadata"].index[idx_true],
            size=min(50, sum(idx_true)),
            replace=False,
        )
        image_list = [
            dataset.getitem(dataset.sample_id_to_idx(sample_id))["image"]
            for sample_id in sample_id_list
        ]
        stack_images(image_list, path=path_output_dir / f"{idx:02d}.jpg")

    print("Compressing output directory to tar.gz file...")
    print(f"Output path: {str(path_output_dir) + '.tar.gz'}")

    with tarfile.open(str(path_output_dir) + ".tar.gz", "w:gz") as tar:
        tar.add(path_output_dir, arcname=os.path.basename(path_output_dir))
