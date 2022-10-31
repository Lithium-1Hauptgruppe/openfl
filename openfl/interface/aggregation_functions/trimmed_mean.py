# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Trimmed mean module."""

import numpy as np
import torch

from openfl.interface.aggregation_functions.core import AggregationFunction
from openfl.utilities import LocalTensor

class TrimmedMean(AggregationFunction):
    """Trimmed mean aggregation."""

    def __init__(self, k):
        """
            Args:
                k(int): Number of outer examples trimmed before aggregation
        """
        self.k = k
    def call(self, local_tensors,  *_) -> np.ndarray:
        """Aggregate tensors.

        Args:
            local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
            db_iterator: iterator over history of all tensors. Columns:
                - 'tensor_name': name of the tensor.
                    Examples for `torch.nn.Module`s: 'conv1.weight', 'fc2.bias'.
                - 'round': 0-based number of round corresponding to this tensor.
                - 'tags': tuple of tensor tags. Tags that can appear:
                    - 'model' indicates that the tensor is a model parameter.
                    - 'trained' indicates that tensor is a part of a training result.
                        These tensors are passed to the aggregator node after local learning.
                    - 'aggregated' indicates that tensor is a result of aggregation.
                        These tensors are sent to collaborators for the next round.
                    - 'delta' indicates that value is a difference between rounds
                        for a specific tensor.
                    also one of the tags is a collaborator name
                    if it corresponds to a result of a local task.

                - 'nparray': value of the tensor.
            tensor_name: name of the tensor
            fl_round: round number
            tags: tuple of tags for this tensor
            k: number of outer examples removed before aggregation
        Returns:
            np.ndarray: aggregated tensor
        """
        tensors = torch.tensor([x.tensor for x in local_tensors])
        tensors2, weights = zip(*[(x.tensor, x.weight) for x in local_tensors])
        #n = len(local_tensors)
        #x = np.mean(np.sort(tensors, axis=-1)[self.k:n-self.k], axis=0)
        #return x

        n = len(local_tensors)
        k = self.k
        # trim k biggest and smallest values of gradients
        sorted, _ = torch.sort(tensors, dim=-1)
        reduced = sorted[k:(n - k)]
        x = torch.mean(reduced, dim=0)
        return x
