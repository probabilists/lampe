r"""Datasets and data loaders."""

__all__ = ["JointLoader", "JointDataset", "H5Dataset"]

import h5py
import numpy as np
import torch

from numpy import ndarray as Array
from pathlib import Path
from torch import Size, Tensor
from torch.distributions import Distribution
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm
from typing import Callable, Iterable, Iterator, Tuple, Union


class IterableJointDataset(IterableDataset):
    r"""Creates an iterable dataset of batched pairs :math:`(\theta, x)`."""

    def __init__(
        self,
        prior: Distribution,
        simulator: Callable,
        batch_shape: Size = (),
        numpy: bool = False,
    ):
        super().__init__()

        self.prior = prior
        self.simulator = simulator
        self.batch_shape = batch_shape
        self.numpy = numpy

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        while True:
            theta = self.prior.sample(self.batch_shape)

            if self.numpy:
                x = self.simulator(theta.detach().cpu().numpy().astype(np.double))
                x = torch.from_numpy(x).to(theta)
            else:
                x = self.simulator(theta)

            yield theta, x


class JointLoader(DataLoader):
    r"""Creates an infinite data loader of batched pairs :math:`(\theta, x)` generated
    by a prior distribution :math:`p(\theta)` and a simulator.

    The simulator is a stochastic function taking (a vector of) parameters
    :math:`\theta`, in the form of a NumPy array or a PyTorch tensor, as input and
    returning an observation :math:`x` as output, which implicitly defines a
    likelihood distribution :math:`p(x | \theta)`. Together with the prior, they form
    a joint distribution :math:`p(\theta, x) = p(\theta) p(x | \theta)` from which
    pairs :math:`(\theta, x)` are independently drawn.

    Arguments:
        prior: A prior distribution :math:`p(\theta)`.
        simulator: A callable simulator.
        batch_size: The batch size of the generated pairs.
        vectorized: Whether the simulator accepts batched inputs or not.
        numpy: Whether the simulator requires NumPy or PyTorch inputs.
        kwargs: Keyword arguments passed to :class:`torch.utils.data.DataLoader`.

    Example:
        >>> loader = JointLoader(prior, simulator, numpy=True, num_workers=4)
        >>> for theta, x in loader:
        ...     theta, x = theta.cuda(), x.cuda()
        ...     something(theta, x)
    """

    def __init__(
        self,
        prior: Distribution,
        simulator: Callable,
        batch_size: int = 2**8,  # 256
        vectorized: bool = False,
        numpy: bool = False,
        **kwargs,
    ):
        super().__init__(
            IterableJointDataset(
                prior,
                simulator,
                batch_shape=(batch_size,) if vectorized else (),
                numpy=numpy,
            ),
            batch_size=None if vectorized else batch_size,
            **kwargs,
        )


class JointDataset(Dataset):
    r"""Creates an in-memory dataset of pairs :math:`(\theta, x)`.

    :class:`JointDataset` supports indexing and slicing, but also implements a custom
    :meth:`__iter__` method which supports batching and shuffling.

    Arguments:
        theta: A tensor of parameters :math:`\theta`.
        x: A tensor of observations :math:`x`.
        batch_size: The size of the batches.
        shuffle: Whether the pairs are shuffled or not when iterating.

    Example:
        >>> dataset = JointDataset(theta, x, batch_size=256, shuffle=True)
        >>> theta, x = dataset[42:69]
        >>> theta.shape
        torch.Size([27, 5])
        >>> for theta, x in dataset:
        ...     theta, x = theta.cuda(), x.cuda()
        ...     something(theta, x)
    """

    def __init__(
        self,
        theta: Tensor,
        x: Tensor,
        batch_size: int = None,
        shuffle: bool = False,
    ):
        super().__init__()

        assert len(theta) == len(x)

        self.theta = torch.as_tensor(theta)
        self.x = torch.as_tensor(x)

        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.theta)

    def __getitem__(self, i: Union[int, slice]) -> Tuple[Tensor, Tensor]:
        return self.theta[i], self.x[i]

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        if self.shuffle:
            order = torch.randperm(len(self))

            if self.batch_size is None:
                return (self[i] for i in order)
            else:
                return (self[i] for i in order.split(self.batch_size))
        else:
            if self.batch_size is None:
                return zip(self.theta, self.x)
            else:
                return zip(
                    self.theta.split(self.batch_size),
                    self.x.split(self.batch_size),
                )


class H5Dataset(IterableDataset):
    r"""Creates an iterable dataset of pairs :math:`(\theta, x)` from an HDF5 file.

    As it can be slow to load pairs from disk one by one, :class:`H5Dataset` implements
    a custom :meth:`__iter__` method that loads several contiguous chunks of pairs at
    once and shuffles their concatenation before yielding the pairs.

    :class:`H5Dataset` also implements the :meth:`__len__` and :meth:`__getitem__`
    methods for convenience.

    Important:
        When using :class:`H5Dataset` with :class:`torch.utils.data.DataLoader`, it is
        recommended to disable the loader's automatic batching and provide the batch
        size to the dataset, as it significantly improves loading performances.
        In fact, unless using several workers or memory pinning, it is not even
        necessary to wrap the dataset in a :class:`torch.utils.data.DataLoader`.

    Arguments:
        file: An HDF5 file containing pairs :math:`(\theta, x)`.
        batch_size: The size of the batches.
        chunk_size: The size of the contiguous chunks.
        chunk_step: The number of chunks loaded at once.
        shuffle: Whether the pairs are shuffled or not when iterating.

    Example:
        >>> dataset = H5Dataset('data.h5', batch_size=256, shuffle=True)
        >>> theta, x = dataset[0]
        >>> theta
        tensor([-0.1215, -1.3641,  0.7233, -1.2150, -1.9263])
        >>> for theta, x in dataset:
        ...     theta, x = theta.cuda(), x.cuda()
        ...     something(theta, x)
    """

    def __init__(
        self,
        file: Union[str, Path],
        batch_size: int = None,
        chunk_size: int = 2**8,  # 256
        chunk_step: str = 2**8,  # 256
        shuffle: bool = False,
    ):
        super().__init__()

        self.file = h5py.File(file, mode="r")

        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.chunk_step = chunk_step
        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.file["theta"])

    def __getitem__(self, i: Union[int, slice]) -> Tuple[Tensor, Tensor]:
        theta, x = self.file["theta"][i], self.file["x"][i]

        return torch.from_numpy(theta), torch.from_numpy(x)

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        chunks = torch.tensor([
            (i, i + self.chunk_size) for i in range(0, len(self), self.chunk_size)
        ])

        if self.shuffle:
            order = torch.randperm(len(chunks))
            chunks = chunks[order]

        for slices in chunks.split(self.chunk_step):
            # Merge contiguous slices
            slices = sorted(slices.tolist())
            stack = []

            for s in slices:
                if stack and stack[-1][-1] == s[0]:
                    stack[-1][-1] = s[-1]
                else:
                    stack.append(s)

            # Load
            theta = np.concatenate([self.file["theta"][i:j] for i, j in stack])
            x = np.concatenate([self.file["x"][i:j] for i, j in stack])

            theta, x = torch.from_numpy(theta), torch.from_numpy(x)

            # Shuffle
            if self.shuffle:
                order = torch.randperm(len(theta))
                theta, x = theta[order], x[order]

            # Batch
            if self.batch_size is None:
                yield from zip(theta, x)
            else:
                yield from zip(
                    theta.split(self.batch_size),
                    x.split(self.batch_size),
                )

    def to_memory(self) -> JointDataset:
        r"""Loads all pairs in memory and returns them as a :class:`JointDataset`.

        Example:
            >>> dataset = H5Dataset('data.h5').to_memory()
        """

        return JointDataset(
            self.file["theta"][:],
            self.file["x"][:],
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

    @staticmethod
    def store(
        pairs: Iterable[Tuple[Array, Array]],
        file: Union[str, Path],
        size: int,
        overwrite: bool = False,
        dtype: np.dtype = np.float32,
        **meta,
    ) -> None:
        r"""Creates an HDF5 file containing pairs :math:`(\theta, x)`.

        The sets of parameters :math:`\theta` are stored in a collection named
        :py:`'theta'` and the observations in a collection named :py:`'x'`.

        Arguments:
            pairs: An iterable over batched pairs :math:`(\theta, x)`.
            file: An HDF5 filename to store pairs in.
            size: The number of pairs to store.
            overwrite: Whether to overwrite existing files or not. If :py:`False`
                and the file already exists, the function raises an error.
            dtype: The data type to store pairs in.
            meta: Metadata to store in the file.

        Example:
            >>> loader = JointLoader(prior, simulator, batch_size=16)
            >>> H5Dataset.store(loader, 'data.h5', 4096)
            100%|██████████| 4096/4096 [01:35<00:00, 42.69pair/s]
        """

        # Pairs
        pairs = iter(pairs)

        # File
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(file, "w" if overwrite else "w-") as f:
            ## Attributes
            f.attrs.update(meta)

            ## Datasets
            theta, x = map(np.asarray, next(pairs))

            f.create_dataset("theta", (size,) + theta.shape[1:], dtype=dtype)
            f.create_dataset("x", (size,) + x.shape[1:], dtype=dtype)

            ## Store
            with tqdm(total=size, unit="pair") as tq:
                i = 0

                while True:
                    j = min(i + theta.shape[0], size)

                    f["theta"][i:j] = theta[: j - i]
                    f["x"][i:j] = x[: j - i]

                    tq.update(j - i)

                    if j < size:
                        i = j

                        try:
                            theta, x = map(np.asarray, next(pairs))
                        except StopIteration:
                            break
                    else:
                        break
