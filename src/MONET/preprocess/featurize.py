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
import torch
import torchvision
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

import clip
from MONET.datamodules.components.base_dataset import BaseDataset
from MONET.utils.io import load_hdf5, load_pkl
from MONET.utils.loader import (
    custom_collate,
    custom_collate_per_key,
    dataloader_apply_func,
)


def get_layer_feature(model, feature_layer_name, image):
    # image = self.normalize(self.toTensor(img)).unsqueeze(0).to(self.device)
    # embedding = torch.zeros(image.shape[0], num_features, 1, 1).to(image.device)
    feature_layer = model._modules.get(feature_layer_name)

    embedding = []

    def copyData(module, input, output):
        embedding.append(output.data)

    h = feature_layer.register_forward_hook(copyData)
    out = model(image.to(image.device))
    h.remove()
    embedding = embedding[0]
    assert embedding.shape[0] == image.shape[0], f"{embedding.shape[0]} != {image.shape[0]}"
    assert embedding.shape[2] == 1, f"{embedding.shape[2]} != 1"
    assert embedding.shape[2] == 1, f"{embedding.shape[3]} != 1"
    return embedding[:, :, 0, 0]


if __name__ == "__main__":
    # python src/MONET/preprocess/filter_clusters.py --step cluster --input data/textbook/pdf_extracted_temp.hdf5 --output data/textbook/images_clustering --device cuda:2

    parser = argparse.ArgumentParser(
        prog="filter_cluster.py",
        description="filter samples using clustering algorithms",
        epilog="",
    )
    # sub_parsers = parser.add_subparsers(dest="cmd")
    # parser_featurize = sub_parsers.add_parser("featurize", help="featurize")
    parser.add_argument("-i", "--input", type=str, help="input path", required=True)
    parser.add_argument("-o", "--output", type=str, help="output path", required=True)
    parser.add_argument(
        "-d", "--device", type=str, help="device", required=False, default="cuda:0"
    )

    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    path_input = Path(args.input)
    path_output = Path(args.output)
    device = args.device

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
        metadata_all=pd.DataFrame(index=data_dict.keys()),
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        collate_fn=custom_collate,
        num_workers=4,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    resnet = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    ).to(device)
    resnet.eval()

    efficientnet = torchvision.models.efficientnet_v2_s(
        weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    ).to(device)
    efficientnet.eval()

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()

    def batch_func(batch):
        with torch.no_grad():
            resnet_feature = get_layer_feature(resnet, "avgpool", batch["image"].to(device))
            efficientnet_feature = get_layer_feature(
                efficientnet, "avgpool", batch["image"].to(device)
            )
            # resnet_feature /= resnet_feature.norm(dim=-1, keepdim=True)
            clip_feature = clip_model.encode_image(batch["image"].to(device))
            clip_feature /= clip_feature.norm(dim=-1, keepdim=True)

        return {
            "resnet_feature": resnet_feature.detach().cpu(),
            "clip_feature": clip_feature.detach().cpu(),
            "efficientnet_feature": efficientnet_feature.detach().cpu(),
            "metadata": batch["metadata"],
        }

    print("Featurizing...")
    loader_applied = dataloader_apply_func(
        dataloader=loader,
        func=batch_func,
        collate_fn=custom_collate_per_key,
    )

    print(f"Saving to {path_output}")
    torch.save(
        loader_applied,
        path_output,
    )
