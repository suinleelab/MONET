import os
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split

from MONET.datamodules.components.base_dataset import BaseDataset
from MONET.utils.io import load_pkl
from MONET.utils.static import (
    ddi_map,
    derm7pt_diagnosis_labels,
    fitzpatrick17k_disease_label,
    fitzpatrick17k_ninelabel,
    fitzpatrick17k_threelabel,
    ham10k_dx_labels,
    skincon_cols,
)
from MONET.utils.text_processing import (
    generate_prompt_token_from_caption,
    generate_prompt_token_from_concept,
)


def setup_textbook(data_dir, n_px, norm_mean, norm_std, split_seed):
    # load image
    image_dict = load_pkl(
        Path(data_dir) / "final_image.pkl",
        field="images",
        verbose=True,
    )
    for k, v in image_dict.items():
        image_dict[k] = str(Path(data_dir) / "final_image" / v)

    # load text
    text_df = pd.read_csv(Path(data_dir) / "final_text.csv")
    text_df = text_df[~text_df.duplicated(subset=["image_key", "text_formatted"])]

    text_df = text_df[
        ~text_df["text_formatted"].isnull()
    ]  # maybe have to change to do this before

    text_df = text_df.set_index("image_key")

    # check if indices match
    assert text_df.index.isin(
        image_dict.keys()
    ).all(), "Mismatch between text and image indices"

    # split train/val/test
    train_idx, val_idx = train_test_split(
        text_df.index, test_size=0.2, random_state=split_seed
    )

    # set train dataset
    data_train = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=True,
        metadata_all=text_df.loc[
            train_idx, ["image_pdf_name", "image_page_num", "image_xref"]
        ],
        integrity_level="weak",
        # return_label=skincon_cols,
    )
    # add text data
    data_train.text_data = text_df.loc[train_idx]
    data_train.generate_prompt_token = partial(
        generate_prompt_token_from_caption,
        caption_col="text_formatted",
        use_random=False,
    )

    # set val dataset
    data_val = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[
            val_idx, ["image_pdf_name", "image_page_num", "image_xref"]
        ],
        integrity_level="weak",
        # return_label=skincon_cols,
    )

    # add text data
    data_val.text_data = text_df.loc[val_idx]
    data_val.generate_prompt_token = partial(
        generate_prompt_token_from_caption,
        caption_col="text_formatted",
        use_random=False,
    )

    # set text data
    data_test = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[
            val_idx, ["image_pdf_name", "image_page_num", "image_xref"]
        ],
        integrity_level="weak",
        # return_label=skincon_cols,
    )

    # add text data
    data_test.text_data = text_df.loc[val_idx]
    data_test.generate_prompt_token = partial(
        generate_prompt_token_from_caption,
        caption_col="text_formatted",
        use_random=False,
    )

    # set text data
    data_all = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[:, ["image_pdf_name", "image_page_num", "image_xref"]],
        integrity_level="weak",
        # return_label=skincon_cols,
    )

    # add text data
    data_all.text_data = text_df.loc[:]

    data_all.generate_prompt_token = partial(
        generate_prompt_token_from_caption,
        caption_col="text_formatted",
        use_random=False,
    )
    return data_train, data_val, data_test, data_all


def setup_pubmed(data_dir, n_px, norm_mean, norm_std, split_seed):
    image_dict = load_pkl(
        Path(data_dir) / "final_image.pkl",
        field="images",
        verbose=True,
    )
    image_dict_ = OrderedDict()
    for k, v in image_dict.items():
        image_dict_[os.path.splitext(k)[0]] = str(
            Path(data_dir) / "final_image" / v
        )  # v
    image_dict = image_dict_

    # load text
    text_df = pd.read_pickle(Path(data_dir) / "final_text.pkl")

    # check if indices match
    assert text_df.index.isin(
        image_dict.keys()
    ).all(), "Mismatch between text and image indices"

    # split train/val/test
    train_idx, val_idx = train_test_split(
        text_df.index, test_size=0.2, random_state=split_seed
    )

    # set train dataset
    data_train = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=True,
        metadata_all=text_df.loc[train_idx, ["article_id", "href"]],
        integrity_level="weak",
    )
    # add text data
    data_train.text_data = text_df.loc[train_idx]
    data_train.generate_prompt_token = partial(
        generate_prompt_token_from_caption,
        caption_col="caption_final",
        use_random=False,
    )

    # set val dataset
    data_val = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[val_idx, ["article_id", "href"]],
        integrity_level="weak",
    )

    # add text data
    data_val.text_data = text_df.loc[val_idx]
    data_val.generate_prompt_token = partial(
        generate_prompt_token_from_caption,
        caption_col="caption_final",
        use_random=False,
    )

    # set text data
    data_test = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[val_idx, ["article_id", "href"]],
        integrity_level="weak",
    )

    # add text data
    data_test.text_data = text_df.loc[val_idx]

    data_test.generate_prompt_token = partial(
        generate_prompt_token_from_caption,
        caption_col="caption_final",
        use_random=False,
    )

    # set text data
    data_all = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[:, ["article_id", "href"]],
        integrity_level="weak",
    )

    # add text data
    data_all.text_data = text_df.loc[:]

    data_all.generate_prompt_token = partial(
        generate_prompt_token_from_caption,
        caption_col="caption_final",
        use_random=False,
    )
    return data_train, data_val, data_test, data_all


def setup_fitzpatrick17k(
    data_dir,
    n_px,
    norm_mean,
    norm_std,
    split_seed,
    skincon_only=False,
    clean_only=False,
    label_type=None,
    no_duplicates=False,
    pseudo_label=False,
):
    image_dict = load_pkl(
        Path(data_dir) / "final_image.pkl",
        field="images",
        verbose=True,
    )

    image_dict_ = OrderedDict()
    for k, v in image_dict.items():
        image_dict_[k] = str(Path(data_dir) / "final_image" / v)  # v
    image_dict = image_dict_
    # load text
    if pseudo_label:
        text_df = pd.read_pickle(
            Path(data_dir) / "final_metadata_all_pseudo.pkl"
        ).set_index("ImageID")
    else:
        text_df = pd.read_pickle(Path(data_dir) / "final_metadata_all.pkl").set_index(
            "ImageID"
        )
    if skincon_only:
        text_df = text_df.dropna(subset=["skincon_Nodule"])
    if clean_only:
        clean_indices = pd.read_csv(
            Path(data_dir) / "final_clean_images.txt", header=None
        )[0]
        text_df = text_df[text_df["md5hash"].isin(clean_indices)]
    if no_duplicates:
        dup_idx = pd.read_csv(Path(data_dir) / "final_dupcheck.csv", index_col=0)
        dup_idx = [
            row[~row.isnull()].values.tolist() for idx, row in dup_idx.iterrows()
        ]
        # print(dup_idx)
        # print(text_df)
        text_df = text_df.drop([j for i in dup_idx for j in i[1:]], axis="index")
        # print(text_df)

    if label_type == "skincon":
        return_label = skincon_cols
    elif label_type == "disease":
        # 114 nodes
        cols = []
        series_list = []
        for disease in fitzpatrick17k_disease_label:
            cols.append("label_" + disease)
            series_list.append(text_df["label"].map(lambda x: 1 if disease in x else 0))
        text_df = pd.concat(
            [text_df, pd.concat(series_list, axis=1, keys=cols)], axis=1
        )
        return_label = cols
    elif label_type == "ninelabel":
        # 9 nodes
        # cols = []
        # series_list = []
        # for label in fitzpatrick17k_ninelabel:
        #     cols.append("ninelabel_" + label)
        #     series_list.append(
        #         text_df["nine_partition_label"].map(lambda x: 1 if x == label else 0)
        #     )
        # text_df = pd.concat(
        #     [text_df, pd.concat(series_list, axis=1, keys=cols)], axis=1
        # )
        text_df["nine_partition_label_indices"] = text_df["nine_partition_label"].map(
            lambda x: fitzpatrick17k_ninelabel.index(x)
        )
        return_label = ["nine_partition_label_indices"]
    elif label_type == "threelabel":
        # 9 nodes
        # cols = []
        # series_list = []
        # for label in fitzpatrick17k_ninelabel:
        #     cols.append("ninelabel_" + label)
        #     series_list.append(
        #         text_df["nine_partition_label"].map(lambda x: 1 if x == label else 0)
        #     )
        # text_df = pd.concat(
        #     [text_df, pd.concat(series_list, axis=1, keys=cols)], axis=1
        # )
        text_df["three_partition_label_indices"] = text_df["three_partition_label"].map(
            lambda x: fitzpatrick17k_threelabel.index(x)
        )
        return_label = ["three_partition_label_indices"]
    elif label_type == "malignant":
        print(text_df.columns)
        text_df["malignant_indices"] = text_df["malignant"].astype(int)
        return_label = ["malignant_indices"]
    else:
        return_label = None

    concept_prompt_dict = {}
    for concept_col in skincon_cols:
        concept_prompt_dict[concept_col] = generate_prompt_token_from_concept(
            concept_col[8:], use_random=True
        )

    # check if indices match
    assert text_df.index.isin(
        image_dict.keys()
    ).all(), "Mismatch between text and image indices"
    # print(text_df.index)
    # print(image_dict.keys())

    # split train/val/test
    train_idx, val_idx = train_test_split(
        text_df.index, test_size=0.2, random_state=split_seed
    )

    data_train = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=True,
        metadata_all=text_df.loc[train_idx, :],
        integrity_level="weak",
        return_label=return_label,
    )
    data_train.concept_prompt_dict = concept_prompt_dict

    data_val = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[val_idx, :],
        integrity_level="weak",
        return_label=return_label,
    )
    data_val.concept_prompt_dict = concept_prompt_dict

    data_test = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[val_idx, :],
        integrity_level="weak",
        return_label=return_label,
    )
    data_test.concept_prompt_dict = concept_prompt_dict

    data_all = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[:, :],
        integrity_level="weak",
        return_label=return_label,
    )
    data_all.concept_prompt_dict = concept_prompt_dict

    return data_train, data_val, data_test, data_all


def setup_ddi(
    data_dir,
    n_px,
    norm_mean,
    norm_std,
    split_seed,
    skincon_only=False,
    label_type=None,
):
    image_dict = load_pkl(
        Path(data_dir) / "final_image.pkl",
        field="images",
        verbose=True,
    )
    image_dict_ = OrderedDict()
    for k, v in image_dict.items():
        image_dict_[k] = str(Path(data_dir) / "final_image" / v)  # v
    image_dict = image_dict_

    # load text
    text_df = pd.read_pickle(Path(data_dir) / "final_metadata_all.pkl").set_index(
        "DDI_file"
    )

    if skincon_only:
        text_df = text_df.dropna(subset=["skincon_Nodule"])

    if label_type == "skincon":
        return_label = skincon_cols
    elif label_type == "disease":
        return_label = ["label"]
    else:
        return_label = None

    concept_prompt_dict = {}
    for concept_col in skincon_cols:
        concept_prompt_dict[concept_col] = generate_prompt_token_from_concept(
            concept_col[8:], use_random=True
        )

    # check if indices match
    assert text_df.index.isin(
        image_dict.keys()
    ).all(), "Mismatch between text and image indices"

    # split train/val/test
    train_idx, val_idx = train_test_split(
        text_df.index, test_size=0.2, random_state=split_seed
    )

    data_train = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=True,
        metadata_all=text_df.loc[train_idx, :],
        integrity_level="weak",
        return_label=return_label,
    )
    data_train.concept_prompt_dict = concept_prompt_dict

    data_val = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[val_idx, :],
        integrity_level="weak",
        return_label=return_label,
    )
    data_val.concept_prompt_dict = concept_prompt_dict

    data_test = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[val_idx, :],
        integrity_level="weak",
        return_label=return_label,
    )
    data_test.concept_prompt_dict = concept_prompt_dict

    data_all = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[:, :],
        integrity_level="weak",
        return_label=return_label,
    )

    data_all.concept_prompt_dict = concept_prompt_dict

    return data_train, data_val, data_test, data_all


def setup_fitzddi(
    data_dir_fitz,
    data_dir_ddi,
    n_px,
    norm_mean,
    norm_std,
    split_seed,
    skincon_only=False,
    clean_only=False,
    melanoma_only=False,
    no_duplicates=False,
    label_type=None,
):
    # load fitz
    image_dict_fitz = load_pkl(
        Path(data_dir_fitz) / "final_image.pkl",
        field="images",
        verbose=True,
    )
    image_dict_fitz_ = OrderedDict()
    for k, v in image_dict_fitz.items():
        image_dict_fitz_[k] = str(Path(data_dir_fitz) / "final_image" / v)  # v
    image_dict_fitz = image_dict_fitz_

    text_df_fitz = pd.read_pickle(
        Path(data_dir_fitz) / "final_metadata_all.pkl"
    ).set_index("ImageID")
    text_df_fitz["source"] = "fitz"

    # load ddi
    image_dict_ddi = load_pkl(
        Path(data_dir_ddi) / "final_image.pkl",
        field="images",
        verbose=True,
    )
    image_dict_ddi_ = OrderedDict()
    for k, v in image_dict_ddi.items():
        image_dict_ddi_[k] = str(Path(data_dir_ddi) / "final_image" / v)  # v
    image_dict_ddi = image_dict_ddi_

    text_df_ddi = pd.read_pickle(
        Path(data_dir_ddi) / "final_metadata_all.pkl"
    ).set_index("DDI_file")
    text_df_ddi["source"] = "ddi"

    assert image_dict_fitz.keys().isdisjoint(
        image_dict_ddi.keys()
    ), "image_dict keys overlap"

    # combine image_dict_fitz and image_dict_ddi
    image_dict = OrderedDict()
    image_dict.update(image_dict_fitz)
    image_dict.update(image_dict_ddi)

    # filter text_df_fitz
    if clean_only:
        fitz_clean_indices = pd.read_csv(
            Path(data_dir_fitz) / "final_clean_images.txt", header=None
        )[0]
        text_df_fitz = text_df_fitz[text_df_fitz["md5hash"].isin(fitz_clean_indices)]

    if no_duplicates:
        dup_idx = pd.read_csv(Path(data_dir_fitz) / "final_dupcheck.csv", index_col=0)
        dup_idx = [
            row[~row.isnull()].values.tolist() for idx, row in dup_idx.iterrows()
        ]
        # print(dup_idx)
        print(text_df_fitz)
        text_df_fitz = text_df_fitz.drop(
            [j for i in dup_idx for j in i[1:]], axis="index"
        )
        print(text_df_fitz)

    if melanoma_only:
        text_df_fitz = text_df_fitz[
            (text_df_fitz["nine_partition_label"] == "malignant melanoma")
            | (text_df_fitz["nine_partition_label"] == "benign melanocyte")
            | (text_df_fitz["label"] == "seborrheic keratosis")
            | (text_df_fitz["label"] == "dermatofibroma")
        ]

    # filter text_df_ddi
    if melanoma_only:
        text_df_ddi = text_df_ddi[
            text_df_ddi["disease"].map(lambda x: x in ddi_map.keys())
        ]

    # set melanoma label
    text_df_fitz["is_melanoma"] = text_df_fitz["nine_partition_label"].map(
        lambda x: 1 if x == "malignant melanoma" else 0
    )
    text_df_ddi["is_melanoma"] = text_df_ddi["disease"].map(
        lambda x: 1 if x == "melanoma" else 0
    )

    text_df = pd.concat([text_df_fitz, text_df_ddi], axis=0)

    if skincon_only:
        text_df = text_df.dropna(subset=["skincon_Nodule"])

    if label_type == "melanoma":
        return_label = ["is_melanoma"]
    else:
        return_label = None

    concept_prompt_dict = {}
    for concept_col in skincon_cols:
        concept_prompt_dict[concept_col] = generate_prompt_token_from_concept(
            concept_col[8:], use_random=True
        )

    # check if indices match
    assert text_df.index.isin(
        image_dict.keys()
    ).all(), "Mismatch between text and image indices"
    # print(text_df.index)
    # print(image_dict.keys())

    # split train/val/test
    train_idx, val_idx = train_test_split(
        text_df.index, test_size=0.2, random_state=split_seed
    )

    data_train = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=True,
        metadata_all=text_df.loc[train_idx, :],
        integrity_level="weak",
        return_label=return_label,
    )
    data_train.concept_prompt_dict = concept_prompt_dict

    data_val = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[val_idx, :],
        integrity_level="weak",
        return_label=return_label,
    )
    data_val.concept_prompt_dict = concept_prompt_dict

    data_test = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[val_idx, :],
        integrity_level="weak",
        return_label=return_label,
    )
    data_test.concept_prompt_dict = concept_prompt_dict

    data_all = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[:, :],
        integrity_level="weak",
        return_label=return_label,
    )
    data_all.concept_prompt_dict = concept_prompt_dict

    return data_train, data_val, data_test, data_all


def setup_ham10k(
    data_dir,
    n_px,
    norm_mean,
    norm_std,
    split_seed,
    label_type=None,
):
    image_dict = load_pkl(
        Path(data_dir) / "final_image.pkl",
        field="images",
        verbose=True,
    )
    image_dict_ = OrderedDict()
    for k, v in image_dict.items():
        image_dict_[k.replace(".jpg", "")] = str(
            Path(data_dir) / "final_image" / v
        )  # v
    image_dict = image_dict_

    # load text
    text_df = pd.read_csv(Path(data_dir) / "final_metadata_all.csv").set_index(
        "image_id"
    )

    if label_type == "skincon":
        return_label = skincon_cols
    elif label_type == "disease":
        return_label = ["label"]
    else:
        return_label = None

    if label_type == "threelabel":
        # 9 nodes
        # cols = []
        # series_list = []
        # for label in fitzpatrick17k_ninelabel:
        #     cols.append("ninelabel_" + label)
        #     series_list.append(
        #         text_df["nine_partition_label"].map(lambda x: 1 if x == label else 0)
        #     )
        # text_df = pd.concat(
        #     [text_df, pd.concat(series_list, axis=1, keys=cols)], axis=1
        # )
        text_df["dx_indices"] = text_df["dx"].map(lambda x: ham10k_dx_labels.index(x))
        return_label = ["dx_indices"]
    else:
        return_label = None

    concept_prompt_dict = {}
    for concept_col in skincon_cols:
        concept_prompt_dict[concept_col] = generate_prompt_token_from_concept(
            concept_col[8:], use_random=True
        )

    # check if indices match
    assert text_df.index.isin(
        image_dict.keys()
    ).all(), "Mismatch between text and image indices"

    # split train/val/test
    train_idx, val_idx = train_test_split(
        text_df["lesion_id"].values, test_size=0.2, random_state=split_seed
    )
    train_idx = text_df[text_df["lesion_id"].isin(train_idx)].index
    val_idx = text_df[text_df["lesion_id"].isin(val_idx)].index

    data_train = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=True,
        metadata_all=text_df.loc[train_idx, :],
        integrity_level="weak",
        return_label=return_label,
    )
    data_train.concept_prompt_dict = concept_prompt_dict

    data_val = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[val_idx, :],
        integrity_level="weak",
        return_label=return_label,
    )
    data_val.concept_prompt_dict = concept_prompt_dict

    data_test = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[val_idx, :],
        integrity_level="weak",
        return_label=return_label,
    )
    data_test.concept_prompt_dict = concept_prompt_dict

    data_all = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[:, :],
        integrity_level="weak",
        return_label=return_label,
    )

    data_all.concept_prompt_dict = concept_prompt_dict

    return data_train, data_val, data_test, data_all


def setup_derm7pt(
    data_dir,
    n_px,
    norm_mean,
    norm_std,
    split_seed,
    derm_or_clinic="all",
    label_type=None,
):
    image_dict = load_pkl(
        Path(data_dir) / "final_image.pkl",
        field="images",
        verbose=True,
    )
    image_dict_ = OrderedDict()
    for k, v in image_dict.items():
        image_dict_[k] = str(Path(data_dir) / "final_image" / v)  # v
    image_dict = image_dict_

    # load text
    text_df = pd.read_csv(Path(data_dir) / "final_metadata_all.csv")
    text_df_clinic = text_df.copy()
    text_df_clinic["image_type"] = "clinic"
    text_df_clinic = text_df_clinic.rename(columns={"clinic": "path"}).drop(
        columns=["derm"]
    )
    text_df_derm = text_df.copy()
    text_df_derm["image_type"] = "derm"
    text_df_derm = text_df_derm.rename(columns={"derm": "path"}).drop(
        columns=["clinic"]
    )

    text_df = pd.concat([text_df_clinic, text_df_derm]).set_index("path")
    text_df.index = text_df.index.str.replace("/", "_")

    if label_type == "diagnosis":
        text_df["diagnosis_indices"] = text_df["diagnosis"].map(
            lambda x: derm7pt_diagnosis_labels.index(x)
        )
        return_label = ["diagnosis_indices"]
    else:
        return_label = None

    concept_prompt_dict = {}
    for concept_col in skincon_cols:
        concept_prompt_dict[concept_col] = generate_prompt_token_from_concept(
            concept_col[8:], use_random=True
        )

    # check if indices match
    assert text_df.index.isin(
        image_dict.keys()
    ).all(), "Mismatch between text and image indices"

    # split train/val/test
    train_idx, val_idx = train_test_split(
        text_df["case_num"].unique(), test_size=0.2, random_state=split_seed
    )
    if derm_or_clinic == "derm":
        train_idx = text_df[
            text_df["case_num"].isin(train_idx) & (text_df["image_type"] == "derm")
        ].index
        val_idx = text_df[
            text_df["case_num"].isin(val_idx) & (text_df["image_type"] == "derm")
        ].index
        all_idx = text_df[text_df["image_type"] == "derm"].index
    elif derm_or_clinic == "clinic":
        train_idx = text_df[
            text_df["case_num"].isin(train_idx) & (text_df["image_type"] == "clinic")
        ].index
        val_idx = text_df[
            text_df["case_num"].isin(val_idx) & (text_df["image_type"] == "clinic")
        ].index
        all_idx = text_df[text_df["image_type"] == "clinic"].index
    elif derm_or_clinic == "all":
        train_idx = text_df[text_df["case_num"].isin(train_idx)].index
        val_idx = text_df[text_df["case_num"].isin(val_idx)].index
        all_idx = text_df.index

    data_train = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=True,
        metadata_all=text_df.loc[train_idx, :],
        integrity_level="weak",
        return_label=return_label,
    )
    data_train.concept_prompt_dict = concept_prompt_dict

    data_val = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[val_idx, :],
        integrity_level="weak",
        return_label=return_label,
    )
    data_val.concept_prompt_dict = concept_prompt_dict

    data_test = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[val_idx, :],
        integrity_level="weak",
        return_label=return_label,
    )
    data_test.concept_prompt_dict = concept_prompt_dict

    data_all = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[all_idx, :],
        integrity_level="weak",
        return_label=return_label,
    )

    data_all.concept_prompt_dict = concept_prompt_dict

    return data_train, data_val, data_test, data_all


def setup_isic(
    data_dir,
    n_px,
    norm_mean,
    norm_std,
    split_seed,
    label_type=None,
):
    image_dict = load_pkl(
        Path(data_dir) / "final_image.pkl",
        field="images",
        verbose=True,
    )

    image_dict_ = OrderedDict()
    for k, v in image_dict.items():
        image_dict_[k.replace(".JPG", "")] = str(
            Path(data_dir) / "final_image" / v
        )  # v
    image_dict = image_dict_

    # load text
    text_df = pd.read_csv(Path(data_dir) / "final_metadata_all.csv").set_index(
        "isic_id"
    )
    text_df = text_df[~(text_df["image_type"] == "overview")]

    return_label = None

    concept_prompt_dict = {}
    # for concept_col in skincon_cols:
    #     concept_prompt_dict[concept_col] = generate_prompt_token_from_concept(
    #         concept_col[8:], use_random=True
    #     )

    # check if indices match
    assert text_df.index.isin(
        image_dict.keys()
    ).all(), "Mismatch between text and image indices"

    # split train/val/test
    train_idx, val_idx = train_test_split(
        text_df.index, test_size=0.2, random_state=split_seed
    )

    data_train = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=True,
        metadata_all=text_df.loc[train_idx, :],
        integrity_level="weak",
        return_label=return_label,
    )
    data_train.concept_prompt_dict = concept_prompt_dict

    data_val = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[val_idx, :],
        integrity_level="weak",
        return_label=return_label,
    )
    data_val.concept_prompt_dict = concept_prompt_dict

    data_test = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[val_idx, :],
        integrity_level="weak",
        return_label=return_label,
    )
    data_test.concept_prompt_dict = concept_prompt_dict

    data_all = BaseDataset(
        image_path_or_binary_dict=image_dict,
        n_px=n_px,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
        metadata_all=text_df.loc[:, :],
        integrity_level="weak",
        return_label=return_label,
    )

    data_all.concept_prompt_dict = concept_prompt_dict

    return data_train, data_val, data_test, data_all


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)

    # cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "pubmed.yaml")
    # cfg.data_dir = str(root / "data" / "pubmed_final")
    # cfg.data_dir = "/sdata/chanwkim/dermatology_datasets/pubmed/"

    # dm = hydra.utils.instantiate(cfg)
    # dm.setup()

    data_train, data_val, data_test, data_all = setup_isic(
        Path("/sdata/chanwkim/dermatology_datasets/isic"),
        n_px=224,
        norm_mean=(0, 0, 0),
        norm_std=(1, 1, 1),
        split_seed=42,
    )

    # data_train, data_val, data_test, data_all = setup_derm7pt(
    #     Path("/sdata/chanwkim/dermatology_datasets/derm7pt"),
    #     224,
    #     (0, 0, 0),
    #     (1, 1, 1),
    #     split_seed=42,
    #     derm_or_clinic="derm",
    # )

    # data_train, data_val, data_test, data_all = setup_ham10k(
    #     Path("/sdata/chanwkim/dermatology_datasets/ham10k"),
    #     224,
    #     (0, 0, 0),
    #     (1, 1, 1),
    #     split_seed=42,
    # )

    # data_train, data_val, data_test, data_all = setup_ddi(
    #     Path("/sdata/chanwkim/dermatology_datasets/ddi"),
    #     224,
    #     (0, 0, 0),
    #     (1, 1, 1),
    # )

    # data_train, data_val, data_test, data_all = setup_fitzpatrick17k(
    #     Path("/sdata/chanwkim/dermatology_datasets/fitzpatrick17k"),
    #     224,
    #     (0, 0, 0),
    #     (1, 1, 1),
    # )
    # /sdata/chanwkim/dermatology_datasets/fitzpatrick17k/metadata_all.csv
    # python src/MONET/preprocess/save_as_path.py \
    # --input data/fitzpatrick17k/images.compact.uncorrupted.pkl \
    # --field images \
    # --output /sdata/chanwkim/dermatology_datasets/fitzpatrick17k/final_image
