import os
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

import clip
from MONET.datamodules.components.base_dataset import BaseDataset
from MONET.datamodules.setup_dataset import (
    setup_ddi,
    setup_derm7pt,
    setup_fitzddi,
    setup_fitzddiderm7pt,
    setup_fitzpatrick17k,
    setup_ham10k,
    setup_isic,
    setup_pubmed,
    setup_textbook,
)
from MONET.utils.io import load_hdf5, load_pkl
from MONET.utils.loader import custom_collate


class MultiplexDatamodule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        n_px: int = 224,
        norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        batch_size_train: int = 64,
        batch_size_val: int = 64,
        batch_size_test: int = 64,
        dataset_name_train: str = "pubmed_train",
        dataset_name_val: str = "pubmed_val",
        dataset_name_test: str = "pubmed_test",
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        split_seed=42,
    ):
        dataset_name_train = [i.split("=") for i in dataset_name_train.split(",")]
        dataset_name_val = [i.split("=") for i in dataset_name_val.split(",")]
        dataset_name_test = [i.split("=") for i in dataset_name_test.split(",")]

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset_save = {}
            dataset_name_unique = np.unique(
                [name for name, split in self.hparams.dataset_name_train]
                + [name for name, split in self.hparams.dataset_name_val]
                + [name for name, split in self.hparams.dataset_name_test]
            )

            for dataset_name in dataset_name_unique:
                if dataset_name == "pubmed":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_pubmed(
                        data_dir=Path(self.hparams.data_dir) / "pubmed",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )
                if dataset_name == "textbook":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_textbook(
                        data_dir=Path(self.hparams.data_dir) / "textbook",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )
                if dataset_name == "fitzpatrick17k":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_fitzpatrick17k(
                        data_dir=Path(self.hparams.data_dir) / "fitzpatrick17k",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )

                if dataset_name == "fitzpatrick17k_pseudo_skinon":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_fitzpatrick17k(
                        data_dir=Path(self.hparams.data_dir) / "fitzpatrick17k",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        skincon_only=True,
                        label_type="skincon",
                        split_seed=self.hparams.split_seed,
                        pseudo_label=True,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )
                if dataset_name == "fitzpatrick17k_skincon":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_fitzpatrick17k(
                        data_dir=Path(self.hparams.data_dir) / "fitzpatrick17k",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        skincon_only=True,
                        label_type="skincon",
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )

                if dataset_name == "fitzpatrick17k_clean_skincon":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_fitzpatrick17k(
                        data_dir=Path(self.hparams.data_dir) / "fitzpatrick17k",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        skincon_only=True,
                        clean_only=True,
                        label_type="skincon",
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )

                if dataset_name == "fitzpatrick17k_threelabel":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_fitzpatrick17k(
                        data_dir=Path(self.hparams.data_dir) / "fitzpatrick17k",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        skincon_only=False,
                        label_type="threelabel",
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )

                if dataset_name == "fitzpatrick17k_clean_threelabel":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_fitzpatrick17k(
                        data_dir=Path(self.hparams.data_dir) / "fitzpatrick17k",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        skincon_only=False,
                        clean_only=True,
                        label_type="threelabel",
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )

                if dataset_name == "fitzpatrick17k_clean_threelabel_nodup":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_fitzpatrick17k(
                        data_dir=Path(self.hparams.data_dir) / "fitzpatrick17k",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        skincon_only=False,
                        clean_only=True,
                        label_type="threelabel",
                        no_duplicates=True,
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )

                if dataset_name == "fitzpatrick17k_disease":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_fitzpatrick17k(
                        data_dir=Path(self.hparams.data_dir) / "fitzpatrick17k",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        skincon_only=False,
                        label_type="disease",
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )

                if dataset_name == "fitzpatrick17k_clean_disease":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_fitzpatrick17k(
                        data_dir=Path(self.hparams.data_dir) / "fitzpatrick17k",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        skincon_only=False,
                        clean_only=True,
                        label_type="disease",
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )

                if dataset_name == "fitzpatrick17k_clean_ninelabel":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_fitzpatrick17k(
                        data_dir=Path(self.hparams.data_dir) / "fitzpatrick17k",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        skincon_only=False,
                        clean_only=True,
                        label_type="ninelabel",
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )

                if dataset_name == "fitzpatrick17k_ninelabel":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_fitzpatrick17k(
                        data_dir=Path(self.hparams.data_dir) / "fitzpatrick17k",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        skincon_only=False,
                        clean_only=False,
                        label_type="ninelabel",
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )

                if dataset_name == "clinical_fd":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_fitzddi(
                        data_dir_fitz=Path(self.hparams.data_dir) / "fitzpatrick17k",
                        data_dir_ddi=Path(self.hparams.data_dir) / "ddi",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        skincon_only=False,
                        clean_only=False,
                        melanoma_only=False,
                        label_type="melanoma",
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )

                if dataset_name == "clinical_fd_clean":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_fitzddi(
                        data_dir_fitz=Path(self.hparams.data_dir) / "fitzpatrick17k",
                        data_dir_ddi=Path(self.hparams.data_dir) / "ddi",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        skincon_only=False,
                        clean_only=True,
                        melanoma_only=False,
                        label_type="melanoma",
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )

                if dataset_name == "clinical_fd_clean_nodup":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_fitzddi(
                        data_dir_fitz=Path(self.hparams.data_dir) / "fitzpatrick17k",
                        data_dir_ddi=Path(self.hparams.data_dir) / "ddi",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        skincon_only=False,
                        clean_only=True,
                        melanoma_only=False,
                        no_duplicates=True,
                        label_type="melanoma",
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )

                if dataset_name == "clinical_fd_clean_nodup_nooverlap":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_fitzddi(
                        data_dir_fitz=Path(self.hparams.data_dir) / "fitzpatrick17k",
                        data_dir_ddi=Path(self.hparams.data_dir) / "ddi",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        skincon_only=False,
                        clean_only=True,
                        melanoma_only=False,
                        no_duplicates=True,
                        no_training_overlap=True,
                        label_type="melanoma",
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )
                if dataset_name == "clinical_fitzddiderm7pt_clean_nodup_nooverlap":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_fitzddiderm7pt(
                        data_dir_fitz=Path(self.hparams.data_dir) / "fitzpatrick17k",
                        data_dir_ddi=Path(self.hparams.data_dir) / "ddi",
                        data_dir_derm7pt=Path(self.hparams.data_dir) / "derm7pt",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        skincon_only=False,
                        clean_only=True,
                        melanoma_only=False,
                        no_duplicates=True,
                        no_training_overlap=True,
                        label_type="melanoma",
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )

                if dataset_name == "ddi":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_ddi(
                        data_dir=Path(self.hparams.data_dir) / "ddi",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )
                if dataset_name == "ddiskincon":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_ddi(
                        data_dir=Path(self.hparams.data_dir) / "ddi",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        skincon_only=True,
                        label_type="skincon",
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )
                if dataset_name == "ddidisease":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_ddi(
                        data_dir=Path(self.hparams.data_dir) / "ddi",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        skincon_only=False,
                        label_type="disease",
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )

                if dataset_name == "ham10k":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_ham10k(
                        data_dir=Path(self.hparams.data_dir) / "ham10k",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )

                if dataset_name == "derm7pt":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_derm7pt(
                        data_dir=Path(self.hparams.data_dir) / "derm7pt",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )

                if dataset_name == "derm7pt_derm_nodup":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_derm7pt(
                        data_dir=Path(self.hparams.data_dir) / "derm7pt",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        split_seed=self.hparams.split_seed,
                        derm_or_clinic="derm",
                        no_duplicates=True,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )

                if dataset_name == "derm7pt_clinical_nodup":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_derm7pt(
                        data_dir=Path(self.hparams.data_dir) / "derm7pt",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        split_seed=self.hparams.split_seed,
                        derm_or_clinic="clinic",
                        no_duplicates=True,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )

                if dataset_name == "isic":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_isic(
                        data_dir=Path(self.hparams.data_dir) / "isic",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        split_seed=self.hparams.split_seed,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )

                if dataset_name == "isic_nodup_nooverlap":
                    (
                        data_train,
                        data_val,
                        data_test,
                        data_all,
                    ) = setup_isic(
                        data_dir=Path(self.hparams.data_dir) / "isic",
                        n_px=self.hparams.n_px,
                        norm_mean=self.hparams.norm_mean,
                        norm_std=self.hparams.norm_std,
                        split_seed=self.hparams.split_seed,
                        no_duplicates=True,
                        no_training_overlap=True,
                    )
                    dataset_save[dataset_name] = {
                        "train": data_train,
                        "val": data_val,
                        "test": data_test,
                        "all": data_all,
                    }
                    print(
                        f"Loaded {dataset_name} dataset. train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)} all: {len(data_all)}"
                    )

            if len(self.hparams.dataset_name_train) == 1:
                self.data_train = dataset_save[self.hparams.dataset_name_train[0][0]][
                    self.hparams.dataset_name_train[0][1]
                ]
            else:
                self.data_train = ConcatDataset(
                    [
                        dataset_save[name][split]
                        for name, split in self.hparams.dataset_name_train
                    ]
                )
            if len(self.hparams.dataset_name_val) == 1:
                self.data_val = dataset_save[self.hparams.dataset_name_val[0][0]][
                    self.hparams.dataset_name_val[0][1]
                ]
            else:
                self.data_val = ConcatDataset(
                    [
                        dataset_save[name][split]
                        for name, split in self.hparams.dataset_name_val
                    ]
                )

            if len(self.hparams.dataset_name_test) == 1:
                self.data_test = dataset_save[self.hparams.dataset_name_test[0][0]][
                    self.hparams.dataset_name_test[0][1]
                ]
            else:
                self.data_test = ConcatDataset(
                    [
                        dataset_save[name][split]
                        for name, split in self.hparams.dataset_name_test
                    ]
                )

            # load image

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size_train,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=True,
            collate_fn=custom_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size_val,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
            collate_fn=custom_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size_test,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
            collate_fn=custom_collate,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "multiplex.yaml")
    # cfg.data_dir = str(root / "data" / "pubmed_final")
    cfg.data_dir = "/sdata/chanwkim/dermatology_datasets/"
    cfg.dataset_name_train = "fitzpatrick17kcleandisease_train"
    cfg.split_seed = 42
    dm = hydra.utils.instantiate(cfg)
    dm.setup()

    loader_val = dm.val_dataloader()
    a = next(iter(loader_val))
    # print(a)
    # target_gpus = [0, 1, 2, 3]
    # dim = 0
    # import torch
    # from torch.nn.parallel._functions import Gather, Scatter

    # def scatter_map(obj):
    #     if isinstance(obj, torch.Tensor):
    #         print("tensor")
    #         return Scatter.apply(target_gpus, None, dim, obj)
    #     # if _is_namedtuple(obj):
    #     # return [type(obj)(*args) for args in zip(*map(scatter_map, obj))]
    #     if isinstance(obj, tuple) and len(obj) > 0:
    #         return list(zip(*map(scatter_map, obj)))
    #     if isinstance(obj, list) and len(obj) > 0:
    #         print("list", obj)
    #         return [list(i) for i in zip(*map(scatter_map, obj))]
    #     if isinstance(obj, dict) and len(obj) > 0:
    #         return [type(obj)(i) for i in zip(*map(scatter_map, obj.items()))]
    #     return [obj for targets in target_gpus]

    # b = scatter_map(a)

    import tqdm

    from MONET.utils.plotting import stack_images

    print(dm)
    print(len(dm.data_train), len(dm.data_val), len(dm.data_test))

    sample_id_list = np.random.RandomState(42).choice(
        dm.data_train.metadata_all.index,
        size=100,
        replace=False,
    )
    image_list = []
    text_list = []
    for sample_id in tqdm.tqdm(sample_id_list):
        # print("check", dm.data_train.generate_prompt_token)
        data = dm.data_train.getitem(dm.data_train.sample_id_to_idx(sample_id))
        image_list.append(data["image"])
        text_list.append(data["prompt"])

    stack_images(image_list=image_list, text_list=text_list, path="textbook.jpg")
