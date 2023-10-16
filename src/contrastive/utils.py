"""

"""
import logging
import math
import random
from collections import defaultdict
from typing import Any, Callable, Dict, KeysView, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


Self = TypeVar("Self")


def temperature_calibration(
    logits: Tensor,
    labels: Tensor,
    lower: float = 0.2,
    upper: float = 5.0,
    eps: float = 0.0001,
) -> float:
    """
    :param logits: the logits predicted by the model
    :param labels: ground truth labels
    :param lower: lower bound for the search
    :param upper: upper bound for the search
    :param eps: minimum change necessary to continue optimization
    """
    logits = torch.FloatTensor(logits)
    labels = torch.LongTensor(labels)
    t_guess = torch.FloatTensor([0.5 * (lower + upper)]).requires_grad_()

    while upper - lower > eps:
        if torch.autograd.grad(F.cross_entropy(logits / t_guess, labels), t_guess)[0] > 0:
            upper = 0.5 * (lower + upper)
        else:
            lower = 0.5 * (lower + upper)
        t_guess = t_guess * 0 + 0.5 * (lower + upper)

    t = min(
        [lower, 0.5 * (lower + upper), upper],
        key=lambda x: float(F.cross_entropy(logits / x, labels)),
    )

    return t


def calc_openness(n_train, n_test, n_target):
    """
    :param n_train: number of classes for training
    :param n_test: total number of classes used during testing
    :param n_target: number of classes for classification during testing

    :return: Openness of the problem
    """
    frac = 2 * n_train / (n_test + n_target)
    return 1 - math.sqrt(frac)


#######################################
# Helpers for labels
#######################################
def is_known(labels) -> Union[bool, Tensor]:
    """
    :returns: True, if label :math:`>= 0`
    """
    return labels >= 0


def is_unknown(labels) -> Union[bool, Tensor]:
    """
    :returns: True, if label :math:`< 0`
    """
    return labels < 0


def contains_known_and_unknown(labels) -> Union[bool, Tensor]:
    """
    :return: true if the labels contain *IN* and *OOD* classes
    """
    return contains_known(labels) and contains_unknown(labels)


def contains_known(labels) -> Union[bool, Tensor]:
    """
    :return: true if the labels contains any *IN* labels
    """
    return is_known(labels).any()


def contains_unknown(labels) -> Union[bool, Tensor]:
    """
    :return: true if the labels contains any *OOD* labels
    """
    return is_unknown(labels).any()


#######################################
# Distance functions etc.
#######################################
def estimate_class_centers(embedding: Tensor, target: Tensor, num_centers: int = None) -> Tensor:
    """
    Estimates class centers from the given embeddings and labels, using mean as estimator.

    TODO: the loop can prob. be replaced
    """
    batch_classes = torch.unique(target).long().to(embedding.device)
    if num_centers is None:
        num_centers = torch.max(target) + 1
    centers = torch.zeros((num_centers, embedding.shape[1]), device=embedding.device)
    for clazz in batch_classes:
        centers[clazz] = embedding[target == clazz].mean(dim=0)
    return centers


def torch_get_distances(centers, embeddings):
    """
    TODO: this can be done more efficiently
    """
    n_instances = embeddings.shape[0]
    n_centers = centers.shape[0]
    distances = torch.empty((n_instances, n_centers)).to(embeddings.device)
    for clazz in torch.arange(n_centers):
        distances[:, clazz] = torch.norm(embeddings - centers[clazz], dim=1, p=2)
    return distances


def pairwise_distances(x, y=None) -> Tensor:
    """
    Calculate pairwise distance by quadratic expansion.

    :param x: is a Nxd matrix
    :param y:  Mxd matrix
    """
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.Inf)


class TensorBuffer(object):
    """
    Used to buffer tensors
    """

    def __init__(self, device="cpu"):
        """

        :param device: device used to store buffers. Default is *cpu*.
        """
        self._buffer: Dict[Any, Tensor] = defaultdict(list)
        self.device = device

    def is_empty(self) -> bool:
        """
        Returns true if this buffer does not hold any tensors.
        """
        return len(self._buffer) == 0

    def append(self: Self, key, value: Tensor) -> Self:
        """
        Appends a tensor to the buffer.

        :param key: tensor identifier
        :param value: tensor
        """
        if not isinstance(value, Tensor):
            raise ValueError(f"Can not handle value type {type(value)}")

        value = value.detach().to(self.device)
        self._buffer[key].append(value)
        return self

    def __contains__(self, elem) -> bool:
        return elem in self._buffer

    def __getitem__(self, item) -> Tensor:
        return self.get(item)

    def sample(self, key) -> Tensor:
        """
        Samples a random tensor from the buffer

        :param key: tensor identifier
        :return: random tensor
        """
        index = torch.randint(0, len(self._buffer[key]), size=(1,))
        return self._buffer[key][index]

    def keys(self) -> KeysView:
        return self._buffer.keys()

    def get(self, key) -> Tensor:
        """
        Retrieves tensor from the buffer

        :param key: tensor identifier
        :return: concatenated tensor
        """
        if key not in self._buffer:
            raise KeyError(key)

        v = torch.cat(self._buffer[key])
        return v

    def clear(self: Self) -> Self:
        """
        Clears the buffer
        """
        log.debug("Clearing buffer")
        self._buffer.clear()
        return self

    def save(self: Self, path) -> Self:
        """
        Save buffer to disk

        :return: self
        """
        d = {k: self.get(k).cpu() for k in self._buffer.keys()}
        torch.save(d, path)
        return self


def apply_reduction(tensor: Tensor, reduction: str) -> Tensor:
    """
    Apply specific reduction to a tensor
    """
    if reduction == "mean":
        return tensor.mean()
    elif reduction == "sum":
        return tensor.sum()
    elif reduction is None or reduction == "none":
        return tensor
    else:
        raise ValueError


def fix_random_seed(seed: int = 12345) -> None:
    """
    Set all random seeds.

    :param seed: seed to set
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

# This extract feature function considers the logits
# def extract_features(
#     data_loader: DataLoader, model: Callable[[Tensor], Tensor], device: Optional[str]
# ) -> Tuple[Tensor, Tensor]:
#     """
#     Helper to extract outputs from model. Ignores OOD inputs.

#     :param data_loader: dataset to extract from
#     :param model: neural network to pass inputs to
#     :param device: device used for calculations
#     :return: Tuple with outputs and labels
#     """
#     # TODO: add option to buffer to GPU
#     buffer = TensorBuffer()
#     with torch.no_grad():
#         for batch in data_loader:
#             x, y = batch
#             x = x.to(device)
#             y = y.to(device)
#             known = is_known(y)
#             if known.any():
#                 output = model(x[known])
#                 logits = output.logits  # Access the logits attribute
#                 z = logits.view(known.sum(), -1)  # flatten

#                 buffer.append("embedding", z)
#                 buffer.append("label", y[known])
                
#         if buffer.is_empty():
#             raise ValueError("No IN instances in loader")

#     z = buffer.get("embedding")
#     y = buffer.get("label")
#     return z, y


# # This extract feature function considers the average embedding 
# def extract_features(
#     data_loader: DataLoader, model: Callable[[Tensor], Tensor], device: Optional[str]
# ) -> Tuple[Tensor, Tensor]:
#     """
#     Helper to extract outputs from model. Ignores OOD inputs.

#     :param data_loader: dataset to extract from
#     :param model: neural network to pass inputs to
#     :param device: device used for calculations
#     :return: Tuple with outputs and labels
#     """
#     # TODO: add option to buffer to GPU
#     buffer = TensorBuffer()
#     with torch.no_grad():
#         for batch in data_loader:
#             x, y, mask = batch
#             x = x.to(device)
#             y = y.to(device)
#             mask = mask.to(device)
#             #print((x.size()))
#             #print((y.size()))
#             known = is_known(y)
#             if known.any():
                
#                 output = model(x[known])
#                 hidden_state = output.hidden_states 
#                 layer_embeds = hidden_state[-1] # layer_embeds is of size torch.Size([50, 256, 768]) when batch size is 50
#                 z = torch.div(layer_embeds.sum(dim=1),mask.sum(dim=1,keepdim=True))
#                 z = z.view(known.sum(), -1) # z  s of size torch.Size([50, 768]) when batch size is 50
                
#                 buffer.append("embedding", z)
#                 buffer.append("label", y[known])
                
#         if buffer.is_empty():
#             raise ValueError("No IN instances in loader")

#     z = buffer.get("embedding")
#     y = buffer.get("label")
#     return z, y

# This extract feature function considers the CLS token alone 
def extract_features(
    data_loader: DataLoader, model: Callable[[Tensor], Tensor], device: Optional[str]
) -> Tuple[Tensor, Tensor]:
    """
    Helper to extract outputs from model. Ignores OOD inputs.

    :param data_loader: dataset to extract from
    :param model: neural network to pass inputs to
    :param device: device used for calculations
    :return: Tuple with outputs and labels
    """
    # TODO: add option to buffer to GPU
    buffer = TensorBuffer()
    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            known = is_known(y)
            if known.any():
                
                output = model(x[known])
                hidden_state = output.hidden_states 
                last_hidden_layer = hidden_state[-1] # last_hidden_layer is of size torch.Size([50, 256, 768]) when batch size is 50
                # Taking the CLS token 
                z = last_hidden_layer[:,0,:] # z is of size torch.Size([50, 768]) when batch size is 50
                
                buffer.append("embedding", z)
                buffer.append("label", y[known])
                
        if buffer.is_empty():
            raise ValueError("No IN instances in loader")

    z = buffer.get("embedding")
    y = buffer.get("label")
    return z, y