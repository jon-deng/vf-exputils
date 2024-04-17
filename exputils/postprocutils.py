"""
Post-processing utilities

Contains commonly used functions used for post-processing results
"""

from typing import Callable, Mapping, Optional, Union, List, Container
import os.path as path
import multiprocessing as mp
import functools as ft

import h5py
import numpy as np
from tqdm import tqdm

from femvf.models.transient.base import BaseTransientModel
from femvf import statefile as sf

from . import h5utils


def postprocess(
    out_file: h5py.Group,
    in_files: List[str],
    model: BaseTransientModel,
    result_to_proc: Mapping[str, Callable[[sf.StateFile], np.ndarray]],
    overwrite_results: Optional[Container[str]] = None,
    use_tqdm: Optional[bool] = True,
    num_proc: Optional[int] = 1,
):
    """
    Postprocess all supplied quantities from a list of files

    Parameters
    ----------
    out_file :
        The file to write post-processing results to
    in_file_paths : List[str]
        A list of filepaths to post process
    model :
        The transient model being post-processed
    result_to_proc :
        A mapping of result names to functions that post-process the result
    overwrite_results :
        A list of result names to force overwrite in `out_file`
    use_tqdm :
        Whether to display a progress bar from `tqdm`
    """
    # Convert `out_file` to a `h5py.File` instance
    if not isinstance(out_file, h5py.Group):
        raise TypeError(
            f"`out_file` must be instance of `h5py.Group` not {type(out_file)}"
        )

    def get_model(in_file):
        return model

    def get_result_to_proc(model):
        return result_to_proc

    result_names = list(result_to_proc.keys())

    for in_file in in_files:
        in_fname = path.splitext(in_file.split('/')[-1])[0]
        postprocess_parallel(
            out_file.require_group(in_fname),
            in_file,
            get_model,
            get_result_to_proc,
            overwrite_results=overwrite_results,
            use_tqdm=use_tqdm,
            num_proc=1,
        )


def postprocess_parallel(
    out_file: h5py.File,
    in_file: str,
    get_model: Callable[[str], BaseTransientModel],
    get_result_to_proc: Callable[
        [BaseTransientModel, str], Mapping[str, Callable[[sf.StateFile], np.ndarray]]
    ],
    overwrite_results: Optional[Container[str]] = None,
    use_tqdm: Optional[bool] = True,
    num_proc: Optional[int] = 1,
):
    """
    Postprocess all supplied quantities from a list of files

    Parameters
    ----------
    out_file :
        The file to write post-processing results to
    in_file_paths : List[str]
        A list of filepaths to post process
    get_model :
        A function that returns an appropriate model object from the input file
    get_result_to_proc :
        A function that returns a mapping of result names to functions that
        post-process that result
    overwrite_results :
        A list of result names to force overwrite in `out_file`
    use_tqdm :
        Whether to display a progress bar from `tqdm`
    num_proc :
        Number of processors to use
    """

    if overwrite_results is None:
        overwrite_results = {}

    # Load the model and mapping from result names to post-processing functions
    # Then specify results to post-process if they don't already exist or
    # if they are overwritten
    model = get_model(in_file)
    result_to_proc = get_result_to_proc(model)
    result_names = list(result_to_proc.keys())
    result_names = [
        name
        for name in result_names
        if name in overwrite_results or name not in out_file
    ]
    if use_tqdm:
        result_names = tqdm(result_names)

    # Post-process each result in parallel/serial
    if num_proc == 1:
        with sf.StateFile(model, in_file, mode='r') as f:
            results = [result_to_proc[result_name](f) for result_name in result_names]
    else:
        _proc = ft.partial(proc, get_model, in_file, get_result_to_proc)
        with mp.Pool(num_proc) as pool:
            results = pool.map(_proc, result_names)

    # Write out post-processed results
    case_to_signal_update = {
        f'{result_name}': result for result_name, result in zip(result_names, results)
    }
    h5utils.dict_to_h5(case_to_signal_update, out_file)


def proc(get_model, in_file, get_result_to_proc, result_name):
    """
    Function to process a single result from an input file
    """
    model = get_model(in_file)
    with sf.StateFile(model, in_file, mode='r') as f:
        return get_result_to_proc(model)[result_name](f)
