"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import contextlib
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch


@contextlib.contextmanager
def temp_seed(rng: np.random, seed: Optional[Union[int, Tuple[int, ...]]]):
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class MaskFunc:
    """
    An object for GRAPPA-style sampling masks.

    This creates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.

    It creates one mask for the input and one mask for the target.
    """

    def __init__(self, self_sup: bool, center_fraction: float, acceleration: float, acceleration_total: Optional[float]):
        """
        Args:
            self_sup: If False the target mask is all ones. If True the target mask is also undersampled
            center_fractions: Fraction of low-frequency columns to be retained both in input and target.
            accelerations: Amount of under-sampling for the input
            acceleration_total: Required if self_sup=True. Determines how much measurements are available for the split into input and target masks
        """

        self.self_sup = self_sup
        self.center_fraction = center_fraction #cent
        self.acceleration = acceleration #p
        self.acceleration_total = acceleration_total #mu
        self.rng = np.random.RandomState()  # pylint: disable=no-member

        if self_sup and acceleration_total==None:
            raise ValueError("For self-supervised training or validation acceleration_total has to be defined.")

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> torch.Tensor:
        raise NotImplementedError


class n2nMaskFunc(MaskFunc):
    """
    n2nMaskFunc creates a sub-sampling mask of a given shape.
    It returns a mask for the training input and a mask for the training target.
    """

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None, fix_selfsup_inputtarget_split: Optional[bool] = True
    ) -> torch.Tensor:
        """
        Create the mask.

        Args:
            shape: The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last
                dimension.
            seed: Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same
                shape. The random state is reset afterwards.
            fix_selfsup_inputtarget_split: Only important for self-sup training. 
                If it is False the input/target split is random.

        Returns:
            input_mask: Input mask of the specified shape.
            target_mask:  Target mask is all ones in the supervised case, but ones and zeros in the self-supervised case.
            weighted_target_mask: Only important for self-supervised training. Must be used to scale the random non-center 
                at the output and target before computing the training loss (and validation losses in the k-space).
                If supervised training weighted_target_mask is just ones same as target_mask.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            n = shape[-2]
            nu = self.center_fraction
            p = 1/self.acceleration

            mask_shape = [1 for _ in shape]
            mask_shape[-2] = n

            if self.self_sup:
                mu = 1/self.acceleration_total
                q = (mu-p+nu-mu*nu)/(1-p)

                # 1. Determine the set S_low consisting of the indices of the nu*n many center frequencies which are always sampled
                size_low = int(round(n*nu))
                pad = (n - size_low + 1) // 2
                # set of indices of all lines in kspace
                S_all = np.arange(n)
                S_low = S_all[pad : pad + size_low]

                # 1.1 Determine S_mu_high, i.e, S_mu without S_low, so only the random high frequencies
                # set of indices of all high frequencies
                S_high = np.hstack((S_all[: pad],S_all[pad + size_low :]))
                S_mu_size_high = int(round((mu-nu)*n))

                S_p_size_high = int(round((p-nu)*n))
                
                #### Depending on whether the input/target split is fixed or re-sampled, the order of sampling needs to be adapted
                # This is so that validation during training samples the same input mask as during testing
                # Recall that during testing selfsup=False, hence S_mu_high is not sampled.
                if fix_selfsup_inputtarget_split:
                    # If split is fixed, first sample S_p_high and then additional lines for S_mu_high
                    # such that the set S_p_high is the same as if we would sample for selfsup=False
                    S_p_high = self.rng.choice(S_high, size=S_p_size_high, replace=False, p=None) 
                    S_mu_size_high_remainding = S_mu_size_high - S_p_size_high

                    S_high_remainding = np.array(list(set(S_high)-set(S_p_high)))
                    S_q_high = self.rng.choice(S_high_remainding, size=S_mu_size_high_remainding, replace=False, p=None) 

                else:
                    # If split is random, first sample S_mu_high such that this set is always fixed.
                    S_mu_high = self.rng.choice(S_high, size=S_mu_size_high, replace=False, p=None)

                    # 2. From S_mu_high sample the set S_p_high of size (p-nu)n
                    S_p_high = np.random.choice(S_mu_high, size=S_p_size_high, replace=False, p=None)

                    # 3. All other indices in S_mu_high add to the set S_q_high
                    S_q_high = np.array(list(set(S_mu_high)-set(S_p_high)))

                # 4. Determine the size of the overlap between S_p_high and S_q_high, sample this many indices from S_p_high and add them to S_q_high
                overlap_size_high = int(round(( (p-nu) / (1-nu) ) * ( (q-nu) / (1-nu) ) *(n-n*nu)))
                S_overlap = S_p_high[0:overlap_size_high]
                S_q_high = np.concatenate([S_q_high,S_overlap])

                # 5. Define the final input and target masks by setting entries to zero or to one for S_p=S_low+S_p_high and S_q=S_low+S_q_high
                input_mask = np.zeros(n)
                input_mask[S_low] = 1.0
                input_mask[S_p_high] = 1.0
                input_mask = torch.from_numpy(input_mask.reshape(*mask_shape).astype(np.float32))

                target_mask = np.zeros(n)
                target_mask[S_low] = 1.0
                target_mask[S_q_high] = 1.0
                target_mask = torch.from_numpy(target_mask.reshape(*mask_shape).astype(np.float32))

                # Create a version of the target mask where the random entries are weighted
                weight_on_random_lines = np.sqrt((1-nu)/(q-nu))
                target_mask_weighted = np.zeros(n)
                target_mask_weighted[S_low] = 1.0
                target_mask_weighted[S_q_high] = weight_on_random_lines
                target_mask_weighted = torch.from_numpy(target_mask_weighted.reshape(*mask_shape).astype(np.float32))

            else:
                # In the supervised case this just creates random input mask with fixed center lines, same as random_mask
                # The target mask is all ones
                target_mask = torch.ones(mask_shape,dtype=torch.float32)
                target_mask_weighted = target_mask.clone()

                size_low = int(round(n*nu))
                p_size_high = int(round(n*p)) - size_low

                pad = (n - size_low + 1) // 2

                # set of indices of all lines in kspace
                S_all = np.arange(n)
                # set of indices of all high frequencies
                S_high = np.hstack((S_all[: pad],S_all[pad + size_low :]))
                # set of indices of high frequencies in the input
                # recall that even using rng here, there can be no seed depending on hp_exp['use_mask_seed_for_training']
                S_p_high = self.rng.choice(S_high, size=p_size_high, replace=False, p=None)

                input_mask = np.zeros(n)
                input_mask[pad : pad + size_low] = 1.0
                input_mask[S_p_high] = 1.0
                input_mask = torch.from_numpy(input_mask.reshape(*mask_shape).astype(np.float32))


        return input_mask, target_mask, target_mask_weighted


def create_mask_for_mask_type(
    mask_type_str: str,
    self_sup: bool,
    center_fraction: float,
    acceleration: float,
    acceleration_total: Optional[float],
) -> MaskFunc:
    """
    Creates a mask of the specified type.

    Args:
        center_fractions: What fraction of the center of k-space to include.
        accelerations: What accelerations to apply.
    """
    if mask_type_str == "n2n":
        return n2nMaskFunc(self_sup, center_fraction, acceleration, acceleration_total)
    else:
        raise Exception(f"{mask_type_str} not supported")