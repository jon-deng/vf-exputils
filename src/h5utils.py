import os

from typing import Mapping

import numpy as np
import h5py


def dict_to_h5(dat: Mapping[str, np.ndarray], f: h5py.File):
    """
    Write contents of a dictionary to an hdf5 file

    Parameters
    ----------
    dat : dict
    f : h5py.File

    Returns
    -------
    h5py.File
    """
    for key, val in dat.items():
        # If `val` is also a dictionary, we need to use a recursive call
        if isinstance(val, dict):
            dict_to_h5(f.require_group(key), val)
        else:
            if key in f:
                f[key][:] = val
            else:
                f[key] = val
    return f


def h5_to_dict(f, dat):
    """
    Read contents of an hdf5 file into a dictionary

    Parameters
    ----------
    f : h5py.File
    dat : dict

    Returns
    -------
    dict
    """
    dset_names = all_h5dset_names(f)

    dat_ = {name: index_general_dset(f[name]) for name in dset_names}
    dat.update(dat_)
    return dat


def trans_h5file_as_path(argnum, mode):
    """
    Return a decorator that converts a file object argument to a path argument
    """

    def _trans_h5file_as_path(function):
        def _function(*args, **kwargs):
            assert argnum <= len(args)

            with h5py.File(args[argnum], mode) as f:
                trans_args = tuple(
                    [f if ii == argnum else arg for ii, arg in enumerate(args)]
                )
                return function(*trans_args, **kwargs)

        return _function


def all_h5dset_names(f):
    names = []

    def append_dset_name(name):
        if isinstance(f[name], h5py.Dataset):
            names.append(name)

    f.visit(append_dset_name)
    return names


def index_general_dset(dset):
    if dset.shape == ():
        return dset[()]
    else:
        return dset[:]
