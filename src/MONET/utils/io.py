import copy
import io
import os
import pickle
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import tqdm


def load_file_as_binary(path_input):
    if isinstance(path_input, str):
        path_input = Path(path_input)
    with open(path_input, "rb") as f:
        file_data = f.read()
    return file_data


def convert_dict(data_dict, method="path_to_binary", inplace=False):
    if inplace is False:
        data_dict = copy.deepcopy(data_dict)
    for key in tqdm.tqdm(data_dict.keys()):
        if method == "path_to_binary":
            file_data = load_file_as_binary(data_dict[key])
            data_dict[key] = file_data
        else:
            raise ValueError(f"Method {method} not implemented")

    return data_dict


def save_to_pkl(data_dict, path_output, field, key_list=None, overwrite=False):
    if os.path.exists(path_output) and not overwrite:
        raise ValueError(f"File {path_output} already exists. Set overwrite=True to overwrite.")
    if isinstance(path_output, str):
        path_output = Path(path_output)
    assert path_output.suffix == ".pkl", "path_output must be a pickle file"

    with open(path_output, "wb") as f:
        if key_list is None:
            key_list = data_dict.keys()
        # pickle.dump({key: data_dict[key] for key in key_list}, f)
        new_data_dict = OrderedDict()
        for key in tqdm.tqdm(key_list):
            new_data_dict[key] = data_dict[key]
        pickle.dump({field: new_data_dict}, f)
        # pickle.dump({field: OrderedDict({key: data_dict[key] for key in key_list})}, f)


def save_to_hdf5(data_dict, path_output, field, key_list=None, overwrite=False):
    if os.path.exists(path_output) and not overwrite:
        raise ValueError(f"File {path_output} already exists. Set overwrite=True to overwrite.")
    if isinstance(path_output, str):
        path_output = Path(path_output)
    assert path_output.suffix == ".hdf5", "path_output must be a hdf5 file"

    with h5py.File(str(path_output), "a" if overwrite else "w") as h5f:
        group = h5f.create_group(field)

        if key_list is None:
            key_list = data_dict.keys()

        for key in tqdm.tqdm(key_list):
            if isinstance(data_dict[key], bytes):
                group.create_dataset(key, data=np.asarray(data_dict[key]))
            elif isinstance(data_dict[key], io.BytesIO):
                group.create_dataset(key, data=np.asarray(data_dict[key].getvalue()))
            # elif isinstance(data_dict[key], str):
            #     group.create_dataset(key, data=data_dict[key].encode("utf-8"))
            # elif isinstance(data_dict[key], np.ndarray):
            #     group.create_dataset(key, data=data_dict[key])
            # elif isinstance(data_dict[key], torch.Tensor):
            #     group.create_dataset(key, data=data_dict[key].numpy())
            # elif isinstance(data_dict[key], Path):
            #     with open(data_dict[key], "rb") as f:
            #         file_data = f.read()
            #     group.create_dataset(key, data=np.asarray(file_data))
            else:
                raise ValueError(f"Unknown type {type(data_dict[key])}")


def merge_hdf5(path_input_list, path_output, field, overwrite=False):
    if isinstance(path_input_list[0], str):
        path_input_list = [Path(path_input) for path_input in path_input_list]
    if isinstance(path_output, str):
        path_output = Path(path_output)
    assert all(
        [path_input.suffix == ".hdf5" for path_input in path_input_list]
    ), "path_input must be a hdf5 file"
    assert path_output.suffix == ".hdf5", "path_output must be a hdf5 file"

    if os.path.exists(path_output) and not overwrite:
        raise ValueError(f"File {path_output} already exists. Set overwrite=True to overwrite.")

    with h5py.File(str(path_output), "w") as h5f_out:
        group_out = h5f_out.create_group(field)
        for path_input in path_input_list:
            print(f"Processing {path_input} ...")
            success_list = []
            failure_list = []
            with h5py.File(str(path_input), "r") as h5f_in:
                group_in = h5f_in[field]
                pbar = tqdm.tqdm(group_in.keys())
                for key in pbar:
                    try:
                        group_out.create_dataset(key, data=group_in[key])
                    except ValueError as e:
                        # print(f"encountered error while processing {key}")
                        # print(e)
                        failure_list.append(key)
                    else:
                        success_list.append(key)
                    pbar.set_postfix({"success": len(success_list), "failure": len(failure_list)})


def filter_hdf5(path_input, path_output, field, key_list, overwrite=False):
    if isinstance(path_input, str):
        path_input = Path(path_input)
    if isinstance(path_output, str):
        path_output = Path(path_output)
    assert path_input.suffix == ".hdf5", "path_input must be a hdf5 file"
    assert path_output.suffix == ".hdf5", "path_output must be a hdf5 file"

    if os.path.exists(path_output) and not overwrite:
        raise ValueError(f"File {path_output} already exists. Set overwrite=True to overwrite.")

    with h5py.File(str(path_input), "r") as h5f_in:
        with h5py.File(str(path_output), "w") as h5f_out:
            group_in = h5f_in[field]
            group_out = h5f_out.create_group(field)
            for key in tqdm.tqdm(key_list):
                group_out.create_dataset(key, data=group_in[key])


def get_hdf5_key(path_input, field):
    if isinstance(path_input, str):
        path_input = Path(path_input)
    with h5py.File(path_input, "r") as h5f:
        key_list = list(h5f[field].keys())
    return key_list


def load_pkl(path_input, field, key_list=None, verbose=False):
    """Load a pickle file.

    Args:
        path_input (str): path to pickle file
        field (str): field to load

    Returns:
        dict: dictionary of loaded data
    """
    if isinstance(path_input, str):
        path_input = Path(path_input)
    assert path_input.suffix == ".pkl", "path_input must be a pickle file"

    data_dict = OrderedDict()

    with open(path_input, "rb") as f:
        pickle_loaded = pickle.load(f)

        if key_list is None:
            key_list = pickle_loaded[field].keys()
            if verbose:
                print("Loading pickle file without filtering...")
        else:
            if verbose:
                print("Loading pickle file with filtering...")

        for key in tqdm.tqdm(key_list):
            data_dict[key] = pickle_loaded[field][key]

    return data_dict


def load_hdf5(path_input, field, key_list=None, verbose=False):
    """Load a hdf5 file.

    Args:
        path_input (str): path to hdf5 file
        field (str): field to load

    Returns:
        dict: dictionary of loaded data
    """
    if isinstance(path_input, str):
        path_input = Path(path_input)
    assert path_input.suffix == ".hdf5", "path_input must be a hdf5 file"

    data_dict = OrderedDict()

    with h5py.File(path_input, "r") as h5f:
        if key_list is None:
            key_list = h5f[field].keys()
            if verbose:
                print("Loading hdf5 file without filtering...")
        else:
            if verbose:
                print("Loading hdf5 file with filtering...")

        for key in tqdm.tqdm(key_list):
            data_dict[key] = h5f[field][key][()]
    return data_dict
