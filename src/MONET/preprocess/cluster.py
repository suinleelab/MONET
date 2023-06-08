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
from sklearn.decomposition import PCA
from tqdm.contrib.concurrent import process_map

from MONET.datamodules.components.base_dataset import BaseDataset
from MONET.utils.io import load_hdf5, load_pkl
from MONET.utils.plotting import stack_images


def run_kmeans(features, n_clusters_upper=20, n_clusters_lower=5):
    kmeans_upper = KMeans(n_clusters=n_clusters_upper, random_state=42, n_init="auto").fit(
        features
    )
    kmeans_label_upper = pd.Series(
        index=range(len(kmeans_upper.labels_)), data=kmeans_upper.labels_
    )
    kmeans_label_upper = kmeans_label_upper.map(lambda x: f"{x:02d}")
    kmeans_label_lower = pd.Series(index=range(len(kmeans_upper.labels_)), data=np.nan)
    for label_upper in kmeans_label_upper.unique():
        if sum(kmeans_label_upper == label_upper) < n_clusters_lower:
            kmeans_label_lower[kmeans_label_upper == label_upper] = [
                f"{label_upper}_00" for _ in range(sum(kmeans_label_upper == label_upper))
            ]
        else:
            kmeans_lower = KMeans(n_clusters=n_clusters_lower, random_state=42, n_init="auto").fit(
                features[kmeans_label_upper == label_upper]
            )
            # kmeans_label_100[kmeans_label_20 == label] = f"{label}_" + pd.Series(
            #     kmeans_small.labels_
            # )
            kmeans_label_lower[kmeans_label_upper == label_upper] = [
                f"{label_upper:s}_{i:02d}" for i in kmeans_lower.labels_
            ]
    return kmeans_label_upper, kmeans_label_lower, kmeans_upper, kmeans_lower


if __name__ == "__main__":
    # python src/MONET/preprocess/filter_clusters.py --step cluster --input data/textbook/pdf_extracted_temp.hdf5 --output data/textbook/images_clustering --device cuda:2

    parser = argparse.ArgumentParser(
        prog="filter_cluster.py",
        description="filter samples using clustering algorithms",
        epilog="",
    )

    parser.add_argument("-i", "--input", type=str, help="input path", required=True)
    parser.add_argument("--featurized-file", type=str, help="input path", required=True)
    parser.add_argument("--pca", help="pca", action="store_true")
    parser.add_argument("-o", "--output", type=str, help="output path", required=True)
    parser.add_argument("-f", "--feature-to-use", type=str, help="feature to use", required=True)
    parser.add_argument(
        "-n1",
        "--n-clusters-upper",
        type=int,
        help="number of clusters (upper)",
        required=False,
        default=20,
    )

    parser.add_argument(
        "-n2",
        "--n-clusters-lower",
        type=int,
        help="number of clusters (lower)",
        required=False,
        default=5,
    )

    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    path_input = Path(args.input)
    featurized_file = Path(args.featurized_file)
    path_output_dir = Path(args.output)

    n_clusters_upper = args.n_clusters_upper
    n_clusters_lower = args.n_clusters_lower
    feature_to_use = args.feature_to_use
    use_pca = args.pca

    if not os.path.exists(path_output_dir):  # make subfolder if needed
        os.mkdir(path_output_dir)
        print(f"Created output directory: {path_output_dir}")

    print("Loading dataset...")
    if path_input.suffix == ".hdf5":
        data_dict = load_hdf5(path_input=path_input, field="images", verbose=True)
    elif path_input.suffix == ".pkl":
        data_dict = load_pkl(path_input=path_input, field="images", verbose=True)
    else:
        raise ValueError(f"Unknown file type: {path_input.suffix}")

    print("Loading featurized data...")
    featurized = torch.load(featurized_file)

    featurized_key_list = featurized["metadata"].index.tolist()
    assert set(data_dict.keys()).issubset(featurized_key_list), "Metadata mismatch"

    if not set(data_dict.keys()) == set(featurized_key_list):
        print("Selecting matching index from loaded features...")

        # dframe['b'].progress_map(example)

        def get_idx(idx):
            return featurized_key_list.index(idx)

        key_idx_mapping = process_map(
            get_idx,
            list(data_dict.keys()),
            max_workers=24,
            chunksize=1000,
        )
        print(key_idx_mapping)

        for key in tqdm.tqdm(featurized.keys()):
            print("Ordering", key)
            if isinstance(featurized[key], torch.Tensor):
                featurized_shape = featurized[key].shape
                featurized[key] = featurized[key][key_idx_mapping]
                print(f"featurized[{key}].shape: {featurized_shape} -> {featurized[key].shape}")
            elif isinstance(featurized[key], pd.DataFrame):
                featurized_shape = featurized[key].shape
                featurized[key] = featurized[key].iloc[key_idx_mapping]
                print(f"featurized[{key}].shape: {featurized_shape} -> {featurized[key].shape}")
            else:
                raise ValueError(f"Unknown type: {type(featurized[key])}")

    print("Initializing dataset...")

    dataset = BaseDataset(
        image_path_or_binary_dict=data_dict,
        n_px=224,
        norm_mean=(0.48145466, 0.4578275, 0.40821073),
        norm_std=(0.26862954, 0.26130258, 0.27577711),
        augment=False,
        metadata_all=featurized["metadata"],
    )

    if feature_to_use == "resnet":
        feature = featurized["resnet_feature"]
    elif feature_to_use == "clip":
        feature = featurized["clip_feature"]
    elif feature_to_use == "efficientnet":
        feature = featurized["efficientnet_feature"]
    else:
        raise ValueError(f"Unknown feature: {feature_to_use}")

    if use_pca:
        print("Applying PCA...")
        pca = PCA(n_components=50, svd_solver="auto")
        print("Fitting PCA...")
        if feature.shape[0] > 100000:
            pca.fit(
                feature[
                    np.random.RandomState(42).choice(
                        np.arange(len(feature)),
                        size=10000,
                        replace=False,
                    )
                ]
            )
        else:
            pca.fit(feature)

        print("Transforming feature...")
        feature_new = pca.transform(feature)
        print("Explained variance ratio:", pca.explained_variance_ratio_)
        print(f"feature shape (before PCA): {feature.shape}")
        print(f"feature shape (after PCA): {feature_new.shape}")
        feature = feature_new

    print("Running kmeans...")
    kmeans_label_upper, kmeans_label_lower, kmeans_upper, kmeans_lower = run_kmeans(
        features=feature,
        n_clusters_upper=n_clusters_upper,
        n_clusters_lower=n_clusters_lower,
    )
    kmeans_label_lower.index = featurized["metadata"].index
    kmeans_label_upper.index = featurized["metadata"].index

    print(kmeans_label_upper.value_counts())
    print(kmeans_label_lower.value_counts())

    kmeans_label_upper.to_csv(path_output_dir / "kmeans_label_upper.csv", header=True, index=True)
    kmeans_label_lower.to_csv(path_output_dir / "kmeans_label_lower.csv", header=True, index=True)

    print("Saving upper kmeans model")
    for label_idx, label_upper in enumerate(
        tqdm.tqdm(kmeans_label_upper.value_counts().sort_values(ascending=False).index.tolist())
    ):
        # create output directory
        if os.path.exists(path_output_dir / label_upper):
            shutil.rmtree(path_output_dir / label_upper)
            # shutil.rmtree(path_output_dir / label_upper)
            # os.rmdir(path_output_dir / label_upper)
        os.mkdir(path_output_dir / label_upper)
        print(f"Created output directory: {path_output_dir/label_upper}")
        # sample index with label
        sample_id_list = np.random.RandomState(42).choice(
            kmeans_label_upper[kmeans_label_upper == label_upper].index,
            size=min(50, sum(kmeans_label_upper == label_upper)),
            replace=False,
        )

        image_list = [
            dataset.getitem(dataset.sample_id_to_idx(sample_id))["image"]
            for sample_id in sample_id_list
        ]

        stack_images(image_list, path=path_output_dir / label_upper / f"{label_upper:s}.jpg")

    print("Saving lower kmeans model")
    for label_idx, label_lower in enumerate(
        tqdm.tqdm(kmeans_label_lower.value_counts().sort_values(ascending=False).index.tolist())
    ):
        label_upper = label_lower.split("_")[0]
        # sample index with label
        sample_id_list = np.random.RandomState(42).choice(
            kmeans_label_lower[kmeans_label_lower == label_lower].index,
            size=min(50, sum(kmeans_label_lower == label_lower)),
            replace=False,
        )
        image_list = [
            dataset.getitem(dataset.sample_id_to_idx(sample_id))["image"]
            for sample_id in sample_id_list
        ]
        stack_images(image_list, path=path_output_dir / label_upper / f"{label_lower:s}.jpg")

    # def make_tarfile(output_filename, source_dir):
    print("Compressing output directory to tar.gz file...")
    print(f"Output path: {str(path_output_dir) + '.tar.gz'}")

    with tarfile.open(str(path_output_dir) + ".tar.gz", "w:gz") as tar:
        tar.add(path_output_dir, arcname=os.path.basename(path_output_dir))
