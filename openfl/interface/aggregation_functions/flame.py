# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Trimmed mean module."""

import numpy as np
import torch
import torch.nn.functional as F
import hdbscan

from openfl.interface.aggregation_functions.core import AggregationFunction
from openfl.utilities import LocalTensor

class Flame(AggregationFunction):
    """Trimmed mean aggregation."""

    def __init__(self, epsilon=1000, delta=0.001):
        """
            Args:
                k(int): Number of outer examples trimmed before aggregation
        """
        self.epsilon = epsilon
        self.delta = delta

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

        """
        Based on the description in https://arxiv.org/abs/2101.02281
        gradients: list of gradients.
        net: model parameters.
        lr: learning rate.
        f: number of malicious clients. The first f clients are malicious.
        byz: attack type.
        device: computation device.
        epsilon: parameter for differential privacy
        delta: parameter for differential privacy
        """
        tensors = torch.tensor([x.tensor for x in local_tensors])
        tensors = [torch.reshape(ten, (-1,)) for ten in tensors]
        n = len(local_tensors)
        delta = self.delta
        epsilon = self.epsilon

        # compute pairwise cosine distances
        cos_dist = torch.zeros((n, n), dtype=torch.double)
        for i in range(n):
            for j in range(i + 1, n):
                d = 1 - F.cosine_similarity(tensors[i], tensors[j], dim=0, eps=1e-9)
                cos_dist[i, j], cos_dist[j, i] = d, d

        # clustering of gradients
        np_cos_dist = cos_dist.cpu().numpy()
        clusterer = hdbscan.HDBSCAN(metric='precomputed', min_samples=1, min_cluster_size=(n // 2) + 1,
                                    cluster_selection_epsilon=0.0, allow_single_cluster=True).fit(np_cos_dist)

        # compute clipping bound
        euclid_dist = []
        for grad in tensors:
            euclid_dist.append(torch.norm(grad, p=2))

        clipping_bound, _ = torch.median(torch.stack(euclid_dist).reshape((-1, 1)), dim=0)

        # gradient clipping
        clipped_gradients = []
        for i in range(n):
            if clusterer.labels_[i] == 0:
                gamma = clipping_bound / euclid_dist[i]
                clipped_gradients.append(tensors[i] * torch.min(torch.ones((1,)), gamma))

        # aggregation
        global_update = torch.mean(torch.stack(clipped_gradients, dim=1), dim=-1)

        # adaptive noise
        std = (clipping_bound * np.sqrt(2 * np.log(1.25 / delta)) / epsilon) ** 2
        global_update += torch.normal(mean=0, std=std.item(), size=tuple(global_update.size()))
        size = local_tensors[0].tensor.shape
        global_update = torch.reshape(global_update, size)

        return global_update