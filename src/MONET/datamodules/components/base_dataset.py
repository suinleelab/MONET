import io
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn

from MONET.utils.loader import custom_collate

# from MONET.utils.io import save_to_hdf5


def convert_image_to_rgb(image):
    return image.convert("RGB")


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_path_or_binary_dict,
        n_px,
        norm_mean,
        norm_std,
        augment,
        metadata_all,
        generate_prompt_token=None,
        static_data=None,
        return_label=None,
        integrity_level="strong",
    ):
        """Generic dataset class for MONET.

        Args:
            image_path_or_binary_dict (_type_): _description_
            n_px (_type_): _description_
            norm_mean (_type_): _description_
            norm_std (_type_): _description_
            augment (_type_): _description_
            metadata_all (_type_): _description_
            generate_prompt_token (_type_, optional): _description_. Defaults to None.
            static_data (_type_, optional): _description_. Defaults to None.
            return_label (_type_, optional): _description_. Defaults to None.
            integrity_level (str, optional): _description_. Defaults to "strong".

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """
        if integrity_level == "strong":
            assert set(image_path_or_binary_dict.keys()) == set(
                metadata_all.index
            ), "Mismatch between image and metadata indices"
        elif integrity_level == "weak":
            assert set(metadata_all.index).issubset(
                set(image_path_or_binary_dict.keys())
            ), "Metadata not in image"
        else:
            raise ValueError(f"integrity_level {integrity_level} not supported")
        if isinstance(image_path_or_binary_dict, OrderedDict):
            pass
        elif isinstance(image_path_or_binary_dict, dict):
            image_path_or_binary_dict = OrderedDict(image_path_or_binary_dict)
        else:
            raise ValueError("image_path_or_binary_dict must be a dict or OrderedDict")

        if isinstance(image_path_or_binary_dict[list(image_path_or_binary_dict.keys())[0]], str):
            image_path_dict = image_path_or_binary_dict
            image_binary_dict = None
        elif isinstance(
            image_path_or_binary_dict[list(image_path_or_binary_dict.keys())[0]], bytes
        ):
            image_binary_dict = image_path_or_binary_dict
            image_path_dict = None
        else:
            raise ValueError("image_path_or_binary_dict must be a list of str or bytes")

        self.image_path_dict = image_path_dict
        self.image_binary_dict = image_binary_dict
        self.metadata_all = metadata_all
        self.n_px = n_px
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.augment = augment
        self.generate_prompt_token = generate_prompt_token
        self.static_data = static_data
        self.return_label = return_label

        # define transformation
        if self.augment:
            self.tranforms_beforetensor = T.Compose(
                [
                    T.RandomResizedCrop(
                        size=self.n_px,
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
                ]
            )

            self.transforms_aftertensor = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(self.norm_mean, self.norm_std),
                ]
            )
        else:
            self.tranforms_beforetensor = T.Compose(
                [
                    T.Resize(self.n_px, interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(self.n_px),
                    convert_image_to_rgb,
                ]
            )

            self.transforms_aftertensor = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(self.norm_mean, self.norm_std),
                ]
            )

    def __len__(self):
        return len(self.metadata_all)

    def __getitem__(self, idx):
        item = self.getitem(idx)
        ret = {
            "image": item["image_transformed_aftertensor"],
            "metadata": item["metadata"],
        }
        if self.return_label is not None:
            label = item["metadata"][self.return_label].values.astype(float)
            if label.shape[0] == 1:
                assert len(label.shape) == 1, "return_label must be a single column"
                label = label[0]
            ret["label"] = label

        if "prompt_tokenized" in item.keys():
            ret["text"] = item["prompt_tokenized"][0]
        return ret

    def sample_id_to_idx(self, key):
        return self.metadata_all.index.tolist().index(key)

    def getitem(self, idx):
        """This function is for debugging purposes. It returns the image before and after tensor
        transformation.

        Args:
            idx (int): _description_

        Returns:
            dict: _description_
        """

        # get metadata
        metadata = self.metadata_all.iloc[idx]

        if self.image_path_dict is not None:
            image = Image.open(self.image_path_dict[metadata.name])
        else:
            image = Image.open(io.BytesIO(self.image_binary_dict[metadata.name]))
        image_transformed_beforetensor = self.tranforms_beforetensor(image)
        image_transformed_aftertensor = self.transforms_aftertensor(image_transformed_beforetensor)

        ret = {
            "image": image,
            "image_transformed_beforetensor": image_transformed_beforetensor,
            "image_transformed_aftertensor": image_transformed_aftertensor,
            "metadata": metadata,
        }

        if self.generate_prompt_token is not None:
            prompt, prompt_tokenized = self.generate_prompt_token(self, idx)
            ret["prompt"] = prompt
            ret["prompt_tokenized"] = prompt_tokenized
        return ret


if __name__ == "__main__":
    # python src/MONET/datamodules/components/base_dataset.py

    base_dataset = BaseDataset(
        image_path_or_binary_dict=[
            "data/fitzpatrick17k/images/dc1d348f4b3e4df8f7e5bf07a7264be1.jpg",
            "data/fitzpatrick17k/images/db871411820979160cb098e6453399ce.jpg",
        ],
        n_px=224,
        norm_mean=(0.48145466, 0.4578275, 0.40821073),
        norm_std=(0.26862954, 0.26130258, 0.27577711),
        augment=False,
        metadata_all=pd.DataFrame(index=[1, 2]),
    )

    print(base_dataset[0])
    print(base_dataset[1])
    loader = torch.utils.data.DataLoader(
        base_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate
    )
    print(next(iter(loader)))
    print(next(iter(loader)))
