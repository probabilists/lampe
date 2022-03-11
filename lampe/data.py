r"""Datasets and data loaders"""

import h5py
import numpy as np
import random
import torch
import torch.utils.data as data

from numpy.typing import ArrayLike
from pathlib import Path
from torch import Tensor
from torch.distributions import Distribution
from tqdm import tqdm
from typing import *


class IterableSimulatorDataset(data.IterableDataset):
    r"""Iterable dataset of (theta, x) batches"""

    def __init__(
        self,
        prior: Distribution,
        simulator: Callable,
        batch_shape: torch.Size = (),
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
                x = self.simulator(theta.detach().cpu().numpy().astype(np.float64))
                x = torch.from_numpy(x).to(theta)
            else:
                x = self.simulator(theta)

            yield theta, x


class SimulatorLoader(data.DataLoader):
    r"""Iterable data loader of (theta, x) batches"""

    def __init__(
        self,
        prior: Distribution,
        simulator: Callable,
        batch_size: int = 2**10,  # 1024
        batched: bool = False,
        numpy: bool = False,
        rng: torch.Generator = None,
        **kwargs,
    ):
        dataset = IterableSimulatorDataset(
            prior,
            simulator,
            batch_shape=(batch_size,) if batched else (),
            numpy=numpy,
        )

        super().__init__(
            dataset,
            batch_size=None if batched else batch_size,
            worker_init_fn=self.worker_init,
            generator=rng,
            **kwargs,
        )

    @staticmethod
    def worker_init(*args) -> None:
        seed = torch.initial_seed() % 2**32
        np.random.seed(seed)
        random.seed(seed)


class H5Loader(data.Dataset):
    r"""Data loader of (theta, x) pairs saved in HDF5 files"""

    def __init__(
        self,
        *filenames,
        batch_size: int = 2**10,  # 1024
        chunk_size: int = 2**12,  # 4096
        group_size: str = 2**4,  # 16
        pin_memory: bool = False,
        shuffle: bool = True,
        seed: int = None,
    ):
        super().__init__()

        self.fs = list(map(h5py.File, filenames))

        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.group_size = group_size

        self.pin_memory = pin_memory

        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __del__(self) -> None:
        for f in self.fs:
            f.close()

    def __len__(self) -> int:
        return sum(len(f['x']) for f in self.fs)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        idx = idx % len(self)

        for f in self.fs:
            if idx < len(f['x']):
                break
            idx = idx - len(f['x'])

        if 'theta' in f:
            theta = torch.from_numpy(f['theta'][idx])
        else:
            theta = None

        x = torch.from_numpy(f['x'][idx])

        return theta, x

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        chunks = [
            (i, j, j + self.chunk_size)
            for i, f in enumerate(self.fs)
            for j in range(0, len(f['x']), self.chunk_size)
        ]

        if self.shuffle:
            self.rng.shuffle(chunks)

        for l in range(0, len(chunks), self.group_size):
            slices = sorted(chunks[l:l+self.group_size])

            # Load
            theta = np.concatenate([self.fs[i]['theta'][j:k] for i, j, k in slices])
            x = np.concatenate([self.fs[i]['x'][j:k] for i, j, k in slices])

            # Shuffle
            if self.shuffle:
                order = self.rng.permutation(len(theta))
                theta, x = theta[order], x[order]

            # Tensor
            theta, x = torch.from_numpy(theta), torch.from_numpy(x)

            if self.pin_memory:
                theta, x = theta.pin_memory(), x.pin_memory()

            # Batches
            yield from zip(
                theta.split(self.batch_size),
                x.split(self.batch_size),
            )


def h5save(
    iterable: Iterable[Tuple[ArrayLike, ArrayLike]],
    filename: str,
    size: int,
    dtype: type = np.float32,
    **kwargs,
) -> None:
    r"""Saves (theta, x) batches to an HDF5 file"""

    # File
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filename, 'w') as f:
        ## Attributes
        for k, v in kwargs.items():
            f.attrs[k] = v

        ## Datasets
        theta, x = map(np.asarray, next(iter(iterable)))
        theta, x = theta[0], x[0]

        f.create_dataset('theta', (size,) + theta.shape, dtype=dtype)
        f.create_dataset('x', (size,) + x.shape, dtype=dtype)

        ## Samples
        with tqdm(total=size, unit='sample') as tq:
            i = 0

            for theta, x in iterable:
                j = min(i + theta.shape[0], size)

                f['theta'][i:j] = np.asarray(theta)[:j-i]
                f['x'][i:j] = np.asarray(x)[:j-i]

                tq.update(j - i)

                if j < size:
                    i = j
                else:
                    break
