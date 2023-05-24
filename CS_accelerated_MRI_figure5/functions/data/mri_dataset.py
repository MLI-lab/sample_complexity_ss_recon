"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
import pickle
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
import yaml


def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.

    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.

    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.

    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


class SliceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        dataset: str,
        path_to_dataset: str,
        path_to_sensmaps: str,
        provide_senmaps: bool,
        #path_to_max_vals: str,
        #use_SENSE_targets: bool,
        challenge: str,
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = True,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
    ):
        """
        Args:
            dataset: Path to a file that contains a list of volumes/slices in the dataset.
            path_to_dataset: Path to a all the volumes/slices in the dataset.
            path_to_sensmaps: Path to a all the sensmaps. One sensmap for each slice
            provide_senmaps: Load sensmaps or not
            challenge: "singlecoil" or "multicoil" depending on which challenge
                to use.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
        """
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.dataset_cache_file = Path(dataset_cache_file)
        self.path_to_sensmaps = path_to_sensmaps
        self.provide_senmaps = provide_senmaps
        #self.path_to_max_vals = path_to_max_vals
        #self.use_SENSE_targets = use_SENSE_targets

        self.transform = transform
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.examples = []

        # Load the dataset cache if it exists and we want to use it.
        # The dataset cache is a dictionary with one entry for every train, val or test set
        # for which a cache has already been created. One entry contains a list of tuples, 
        # where each tuple consists of (filename, slice_ind, meta_data).
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # Check if the dataset is in the cache.
        # If yes, use that cache as list of data examples with corresponding meta data, 
        # if not, then generate the list of data examples and also the meta data.
        if dataset in dataset_cache.keys() and use_dataset_cache:
            logging.info(f"For dataset {dataset} using dataset cache from {self.dataset_cache_file}.")
            self.examples = dataset_cache[dataset]
        else:
            with open(dataset, 'r') as stream:
                # files contains a list of dictionaries. Every dictionary contains an entry fname,
                # which can contain a path prefix like multicoil_val, and optionally a slice number.               
                files = yaml.safe_load(stream)
            # Go through all files and add them to the data examples.
            # If no slice number is given, all slices are added to the dataset.
            #print(files)
            for file in files:
                metadata, num_slices = self._retrieve_metadata(path_to_dataset + file['path'])
                if file['slice'] is not None:
                    self.examples += [
                        (path_to_dataset + file['path'], file['slice'], metadata, file['filename'])
                    ]
                else:
                    self.examples += [
                        (path_to_dataset + file['path'], slice_ind, metadata, file['filename']) for slice_ind in range(num_slices)
                    ]
            if use_dataset_cache:
                dataset_cache[dataset] = self.examples
                logging.info(f"For dataset {dataset} saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as f:
                    pickle.dump(dataset_cache, f)
        
        
        if num_cols:
            self.examples = [
                ex
                for ex in self.examples
                if ex[2]["encoding_size"][1] in num_cols  # type: ignore
            ]

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

        metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            "recon_size": recon_size,
        }

        return metadata, num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int): 
        filepath, dataslice, metadata, filename = self.examples[i]

        if self.provide_senmaps:
            smap_fname = filename + '_smaps_slice' + str(dataslice) + '.h5'
            with h5py.File(self.path_to_sensmaps + smap_fname, "r") as hf:
                sens_maps = hf["sens_maps"][()] #np.array of shape coils,height,width with complex valued entries
        else:
            sens_maps = None

        
            

        with h5py.File(filepath, "r") as hf:
            kspace = hf["kspace"][dataslice]

            #mask = np.asarray(hf["mask"]) if "mask" in hf else None

            target = hf[self.recons_key][dataslice] if self.recons_key in hf else None

            attrs = dict(hf.attrs)
            attrs.update(metadata)

        sample = self.transform(kspace, sens_maps, target, attrs, filename, dataslice)

        return sample
