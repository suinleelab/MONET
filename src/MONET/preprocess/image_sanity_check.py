import argparse
import io
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torchvision.transforms as T
import tqdm
from PIL import Image
from torch import nn

import clip
from MONET.datamodules.components.base_dataset import BaseDataset
from MONET.utils.io import load_hdf5, load_pkl, save_to_hdf5, save_to_pkl


def convert_image_to_rgb(image):
    return image.convert("RGB")


def sanity_check_image(
    data_dict,
    n_px=224,
    norm_mean=(0.48145466, 0.4578275, 0.40821073),
    norm_std=(0.26862954, 0.26130258, 0.27577711),
):

    # transforms = T.Compose(
    #     [
    #         T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
    #         T.CenterCrop(n_px),
    #         convert_image_to_rgb,
    #         T.ToTensor(),
    #         T.Normalize(norm_mean, norm_std),
    #     ]
    # )

    transforms = T.Compose(
        [
            T.RandomResizedCrop(
                size=n_px,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.33),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomVerticalFlip(p=0.5),
            T.RandomHorizontalFlip(p=0.3),
            # T.RandomApply(nn.ModuleList([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1)]), p=0.5),
            convert_image_to_rgb,
            T.RandomApply(
                nn.ModuleList(
                    [
                        T.ColorJitter(
                            brightness=0.2,
                            contrast=0.2,
                            saturation=0.1,
                            hue=0.0,
                        )
                    ]
                ),
                p=1.0,
            ),
            # convert_image_to_rgb,
            T.ToTensor(),
            T.Normalize(norm_mean, norm_std),
        ]
    )

    success_key_list = []
    failure_key_list = []
    pbar = tqdm.tqdm(list(data_dict.keys()))
    for key in pbar:
        try:
            if isinstance(data_dict[key], bytes):
                image = Image.open(io.BytesIO(data_dict[key]))
            elif isinstance(data_dict[key], str):
                image = Image.open(data_dict[key])
            else:
                raise ValueError(f"unknown type {type(data_dict[key])}")
            image = transforms(image)
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            if isinstance(data_dict[key], bytes):
                print(e, key)
            elif isinstance(data_dict[key], str):
                print(e, key, data_dict[key])
            failure_key_list.append(key)
        else:
            success_key_list.append(key)
        finally:
            pbar.set_postfix_str(
                f"success: {len(success_key_list)}, failure: {len(failure_key_list)}"
            )
    return success_key_list, failure_key_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="image_sanity_check.py",
        description="image sanity check",
        epilog="",
    )

    parser.add_argument("-i", "--input", type=str, help="input directory", required=True)

    parser.add_argument("-o", "--output", type=str, help="output path", required=True)

    parser.add_argument("-f", "--field", type=str, help="field", required=True)

    parser.add_argument("--relative-path", help="binary", action="store_true", required=False)
    # parser.add_argument(
    #     "-d", "--device", type=str, help="device", required=False, default="cuda:0"
    # )

    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    path_input = Path(args.input)
    path_output = Path(args.output)
    field = args.field
    relative_path = args.relative_path
    # device = args.device

    clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
    clip_model.eval()

    print("loading data...")
    if path_input.suffix == ".hdf5":
        data_dict = load_hdf5(path_input, field, verbose=True)
        data_dict_ = data_dict
    elif path_input.suffix == ".pkl":
        data_dict = load_pkl(path_input, field, verbose=True)
        if relative_path:
            data_dict_ = OrderedDict()
            for key, value in data_dict.items():
                data_dict_[key] = str(path_input.parent / path_input.stem / value)
        else:
            data_dict_ = data_dict

    assert isinstance(data_dict, OrderedDict)

    success_key_list, failure_key_list = sanity_check_image(data_dict_)
    print("success", len(success_key_list))
    print("failure", len(failure_key_list))
    data_dict_success = OrderedDict()
    for key in success_key_list:
        data_dict_success[key] = data_dict[key]

    data_dict_failure = OrderedDict()
    for key in failure_key_list:
        data_dict_failure[key] = data_dict[key]

    # print("new data dict", len(new_data_dict))

    if path_output.suffix == ".hdf5":
        # data_dict = load_hdf5(path_input, field)
        save_to_hdf5(
            data_dict=data_dict_success,
            path_output=path_output,
            field="images",
            key_list=None,
            overwrite=False,
        )
        save_to_hdf5(
            data_dict=data_dict_failure,
            path_output=path_output.with_suffix(".failure.pkl"),
            field="images",
            key_list=None,
            overwrite=False,
        )
    elif path_output.suffix == ".pkl":
        save_to_pkl(
            data_dict=data_dict_success,
            path_output=path_output,
            field="images",
            key_list=None,
            overwrite=False,
        )
        save_to_pkl(
            data_dict=data_dict_failure,
            path_output=path_output.with_suffix(".failure.pkl"),
            field="images",
            key_list=None,
            overwrite=False,
        )
