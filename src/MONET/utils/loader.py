import pandas as pd
import torch
import tqdm


def custom_collate(batch):
    """Custom collate function for the dataloader.

    Args:
        batch (list): list of dictionaries, each dictionary is a batch of data

    Returns:
        dict: dictionary of collated data
    """
    # def collate_helper(elems):
    #     if isinstance(elems[0], pd.Series):
    #         return pd.concat(elems, axis=1).T
    #     else:
    #         return torch.utils.data.dataloader.default_collate(elems)

    # return {key: collate_helper([d[key] for d in batch]) for key in batch[0]}

    ret = {}
    for key in batch[0]:
        if isinstance(batch[0][key], pd.Series):
            # ret[key] = pd.concat(batch_all[key], axis=1).T
            try:
                ret[key] = pd.concat([d[key] for d in batch], axis=1).T
            except RuntimeError:
                raise RuntimeError(f"Error while concatenating {key}")
            # ret[key] = [d[key] for d in batch]
        else:
            try:
                ret[key] = torch.utils.data.dataloader.default_collate([d[key] for d in batch])
            except RuntimeError:
                raise RuntimeError(f"Error while concatenating {key}")
    # print("collated", ret.keys)
    # dsdsd

    return ret


def custom_collate_per_key(batch_all):
    """Custom collate function batched outputs.

    Args:
        batch_all (dict): dictionary of lists of objects, each dictionary is a batch of data
    Returns:
        dict: dictionary of collated data
    """

    ret = {}
    for key in batch_all:
        if isinstance(batch_all[key][0], pd.DataFrame):
            ret[key] = pd.concat(batch_all[key], axis=0)
        elif isinstance(batch_all[key][0], torch.Tensor):
            ret[key] = torch.concat(batch_all[key], axis=0)
        else:
            print(f"Collating {key}...")
            ret[key] = torch.utils.data.dataloader.default_collate(
                [elem for batch in tqdm.tqdm(batch_all[key]) for elem in batch]
            )

    return ret


def custom_collate_per_batch(batch_all):
    """Custom collate function batched outputs.

    Args:
        batch_all (dict): dictionary of lists of objects, each dictionary is a batch of data
    Returns:
        dict: dictionary of collated data
    """

    ret = {}
    for key in batch_all[0]:
        if isinstance(batch_all[0][key], pd.DataFrame):
            ret[key] = pd.concat([batch[key] for batch in batch_all], axis=1)
        elif isinstance(batch_all[0][key], torch.Tensor):
            ret[key] = torch.concat([batch["resnet_feature"] for batch in batch_all], axis=0)
        else:
            print(f"Collating {key}...")
            ret[key] = torch.utils.data.dataloader.default_collate(
                [elem for batch in tqdm.tqdm(batch_all[key]) for elem in batch]
            )

    return ret


def dataloader_apply_func(dataloader, func, collate_fn=custom_collate_per_key, verbose=True):
    """Apply a function to a dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): torch dataloader
        func (function): function to apply to each batch
        collate_fn (function, optional): collate function. Defaults to custom_collate_batch.

    Returns:
        dict: dictionary of outputs
    """
    func_out_dict = {}
    if verbose:
        for batch in tqdm.tqdm(dataloader):
            for key, func_out in func(batch).items():
                func_out_dict.setdefault(key, []).append(func_out)
    else:
        for batch in dataloader:
            for key, func_out in func(batch).items():
                func_out_dict.setdefault(key, []).append(func_out)
    return collate_fn(func_out_dict)
