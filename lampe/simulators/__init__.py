r"""Simulators and datasets"""

import h5py
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.utils.data as data

from functools import cached_property
from itertools import islice, count
from tqdm import tqdm

from torch import Tensor, BoolTensor
from typing import Callable

from .priors import Distribution


class Simulator(nn.Module):
    r"""Abstract simulator"""

    @cached_property
    def prior(self) -> Distribution:
        r""" p(theta) """

        return self.marginal_prior(...)

    def _apply(self, *args, **kwargs):  # -> self
        if 'prior' in self.__dict__:
            del self.__dict__['prior']

        return super()._apply(*args, **kwargs)

    def marginal_prior(self, mask: BoolTensor) -> Distribution:
        r""" p(theta_a) """

        raise NotImplementedError()

    @cached_property
    def labels(self) -> list[str]:  # parameters' labels
        theta_size = self.prior.event_shape.numel()
        labels = [f'$\\theta_{{{i + 1}}}$' for i in range(theta_size)]

        return labels

    def likelihood(self, theta: Tensor) -> Distribution:
        r""" p(x | theta) """

        raise NotImplementedError()

    @cached_property
    def tractable(self) -> bool:
        theta = self.prior.sample()

        try:
            lkh = self.likelihood(theta)
            return True
        except NotImplementedError:
            return False

    def log_prob(self, theta: Tensor, x: Tensor) -> Tensor:
        r""" log p(x | theta) """

        return self.likelihood(theta).log_prob(x)

    def sample(self, theta: Tensor, shape: torch.Size = ()) -> Tensor:
        r""" x ~ p(x | theta) """

        return self.likelihood(theta).sample(shape)

    def forward(self, theta: Tensor) -> Tensor:
        return self.sample(theta)

    def joint(self, shape: torch.Size = ()) -> tuple[Tensor, Tensor]:
        r""" (theta, x) ~ p(theta) p(x | theta) """

        theta = self.prior.sample(shape)
        x = self.sample(theta)

        return theta, x


class IterableSimulator(data.IterableDataset):
    r"""Iterable simulator dataset"""

    def __init__(
        self,
        simulator: Simulator,
        batch_size: int = 2 ** 10,  # 1024
        length: int = None,
    ):
        super().__init__()

        self.simulator = simulator
        self.batch_size = batch_size
        self._len = length

    def __len__(self) -> int:
        return self._len

    def __iter__(self):  # -> tuple[Tensor, Tensor]
        counter = count() if len(self) is None else range(len(self))
        for _ in counter:
            yield self.simulator.joint((self.batch_size,))

    def loader(
        self,
        group_by: int = 2 ** 4,  # 16
        seed: int = None,
        **kwargs,
    ) -> data.DataLoader:
        r"""Iterable simulator data loader"""

        rng = torch.Generator()

        if seed is None:
            rng.seed()
        else:
            rng.manual_seed(seed)

        def collate(data: list[tuple[Tensor, ...]]) -> tuple[Tensor, ...]:
            return tuple(map(torch.cat, zip(*data)))

        def worker_init(n: int) -> None:
            seed = torch.initial_seed() % 2 ** 32
            np.random.seed(seed)
            random.seed(seed)

        return data.DataLoader(
            self,
            batch_size=group_by,
            collate_fn=collate,
            worker_init_fn=worker_init,
            generator=rng,
            **kwargs,
        )

    def save(
        self,
        file: str,
        samples: int = 2 ** 18,  # 262144
        chunk_size: int = 2 ** 14,  # 16384
        attrs: dict = {},
        **kwargs,
    ) -> None:
        r"""Save simulator samples to HDF5 file"""

        assert samples >= chunk_size >= self.batch_size

        chunk_size = chunk_size - chunk_size % self.batch_size
        samples = samples - samples % chunk_size

        # File
        if os.path.dirname(file):
            os.makedirs(os.path.dirname(file), exist_ok=True)

        with h5py.File(file, 'w') as f:
            ## Attributes
            for k, v in attrs.items():
                f.attrs[k] = v

            ## Data
            theta, x = map(np.asarray, self.simulator.joint())

            f.create_dataset(
                'theta',
                (samples,) + theta.shape,
                chunks=(chunk_size,) + theta.shape,
                dtype=theta.dtype,
            )

            f.create_dataset(
                'x',
                (samples,) + x.shape,
                chunks=(chunk_size,) + x.shape,
                dtype=x.dtype,
            )

            loader = self.loader(chunk_size // self.batch_size, **kwargs)
            loader = islice(loader, samples // chunk_size)

            with tqdm(total=samples) as tq:
                for i, (theta, x) in enumerate(loader):
                    i = i * chunk_size
                    f['theta'][i:i + chunk_size] = np.asarray(theta)
                    f['x'][i:i + chunk_size] = np.asarray(x)
                    tq.update(chunk_size)


class OfflineSimulator(data.Dataset):
    r"""Offline simulator dataset"""

    def __init__(
        self,
        files: list[str],  # H5
        batch_size: int = 2 ** 10,  # 1024
        group_by: str = 2 ** 4,  # 16
        hook: Callable = None,
        device: str = 'cpu',
        shuffle: bool = True,
        seed: int = None,
    ):
        super().__init__()

        self.fs = [h5py.File(f, 'r') for f in files]
        self.chunks = list({
            (i, s.start, s.stop)
            for i, f in enumerate(self.fs)
            for s, *_ in f['x'].iter_chunks()
        })

        self.batch_size = batch_size
        self.group_by = group_by
        self.hook = hook
        self.device = device

        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return sum(len(f['x']) for f in self.fs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        for f in self.fs:
            if idx < len(f['x']):
                break
            idx = idx - len(f['x'])

        if 'theta' in f:
            theta = torch.from_numpy(f['theta'][idx]).to(self.device)
        else:
            theta = None

        x = torch.from_numpy(f['x'][idx]).to(self.device)

        if self.hook is not None:
            theta, x = self.hook(theta, x)

        return theta, x

    def __iter__(self):  # -> tuple[Tensor, Tensor]
        if self.shuffle:
            self.rng.shuffle(self.chunks)

        for i in range(0, len(self.chunks), self.group_by):
            slices = sorted(self.chunks[i:i + self.group_by])

            # Load
            theta_chunk = np.concatenate([self.fs[j]['theta'][k:l] for j, k, l in slices])
            x_chunk = np.concatenate([self.fs[j]['x'][k:l] for j, k, l in slices])

            # Shuffle
            if self.shuffle:
                order = self.rng.permutation(len(x_chunk))
                theta_chunk, x_chunk = theta_chunk[order], x_chunk[order]

            # CUDA
            theta_chunk, x_chunk = torch.from_numpy(theta_chunk), torch.from_numpy(x_chunk)

            if self.device == 'cuda':
                theta_chunk, x_chunk = theta_chunk.pin_memory(), x_chunk.pin_memory()

            # Batches
            for theta, x in zip(
                theta_chunk.split(self.batch_size),
                x_chunk.split(self.batch_size),
            ):
                theta, x = theta.to(self.device), x.to(self.device)

                if self.hook is not None:
                    theta, x = self.hook(theta, x)

                yield theta, x
