
"""Min-Max attack module."""

import numpy as np
import torch

from .base_attacker import BaseAttacker


class MinMaxAttacker(BaseAttacker):
    """Min-Max attack module."""

    def call(self, local_tensors, *_) -> np.ndarray:
        """Local model poisoning attack from https://par.nsf.gov/servlets/purl/10286354
    The implementation is based of their repository (https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning)
    but refactored for clarity.

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
            f: number of malcious devices
        Returns:
            np.ndarray: attacked tensor
        """
        f = self.f
        tensors = local_tensors
        tensors = [torch.reshape(ten, (-1,1)) for ten in tensors]
        catv = torch.cat(tensors, dim=1)
        grad_mean = torch.mean(catv, dim=1)
        deviation = grad_mean / torch.norm(grad_mean, p=2)  # decided to use unit_vec distance which was their default
        # but they also had the option to use sign and standard deviation
        gamma = torch.Tensor([50.0]).float()
        threshold_diff = 1e-5
        gamma_fail = gamma
        gamma_succ = 0

        distances = []
        for update in tensors:
            distance = torch.norm(catv - update, dim=1, p=2) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        max_distance = torch.max(distances)  # determine max distance left side of optimization
        del distances

        # finding optimal gamma according to algorithm 1
        while torch.abs(gamma_succ - gamma) > threshold_diff:
            mal_update = (grad_mean - gamma * deviation)
            distance = torch.norm(catv - mal_update[:, None], dim=1, p=2) ** 2
            max_d = torch.max(distance)

            if max_d <= max_distance:
                gamma_succ = gamma
                gamma = gamma + gamma_fail / 2
            else:
                gamma = gamma - gamma_fail / 2

            gamma_fail = gamma_fail / 2
        mal_update = (grad_mean - gamma_succ * deviation)

        for i in range(f):
            tensors[i] = mal_update[:, None]

        return tensors