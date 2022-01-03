r"""Simulators and datasets"""

import h5py
import numpy as np
import random
import torch
import torch.nn as nn
import torch.utils.data as data

from functools import cached_property
from itertools import count
from pathlib import Path
from tqdm import tqdm

from torch import Tensor, BoolTensor
from typing import Callable, Iterable, Union

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

    def iterable(
        self,
        batch_size: int = 2 ** 10,
        length: int = None,
        group_by: int = 1,
        seed: int = None,
        **kwargs,
    ) -> data.DataLoader:
        r"""Iterable over batched simulations"""

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
            IterableSimulator(self, batch_size, length),
            batch_size=group_by if group_by > 1 else None,
            collate_fn=collate if group_by > 1 else None,
            worker_init_fn=worker_init,
            generator=rng,
            **kwargs,
        )

    def save(
        self,
        filename: str,
        iterable: Iterable[tuple[Tensor, Tensor]] = None,
        samples: int = 2 ** 18,  # 262144
        chunk_size: int = 2 ** 14,  # 16384
        batch_size: int = 2 ** 10,  # 1024
        attrs: dict = {},
        **kwargs,
    ) -> None:
        r"""Save simulator samples to HDF5 file"""

        if iterable is None:
            iterable = self.iterable(batch_size, **kwargs)

        # File
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filename, 'w') as f:
            ## Attributes
            for k, v in attrs.items():
                f.attrs[k] = v

            ## Data
            theta, x = map(np.asarray, self.joint())

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

            with tqdm(total=samples) as tq:
                i = 0

                for theta, x in iterable:
                    j = min(i + theta.shape[0], samples)

                    f['theta'][i:j] = np.asarray(theta)[:j-i]
                    f['x'][i:j] = np.asarray(x)[:j-i]

                    tq.update(j - i)

                    if j < samples:
                        i = j
                    else:
                        break

    @staticmethod
    def load(*args, **kwargs) -> data.Dataset:
        return OfflineSimulator(*args, **kwargs)


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
        self.length = length

    def __iter__(self):  # -> tuple[Tensor, Tensor]
        counter = count() if self.length is None else range(self.length)
        for _ in counter:
            yield self.simulator.joint((self.batch_size,))


class OfflineSimulator(data.Dataset):
    r"""Offline simulator dataset"""

    def __init__(
        self,
        filenames: Union[str, list[str]],  # H5
        batch_size: int = 2 ** 10,  # 1024
        group_by: str = 2 ** 4,  # 16
        device: str = 'cpu',
        pin_memory: bool = False,
        shuffle: bool = True,
        seed: int = None,
    ):
        super().__init__()

        if type(filenames) is not list:
            filenames = [filenames]

        self.fs = [h5py.File(f, 'r') for f in filenames]
        self.chunks = list({
            (i, s.start, s.stop)
            for i, f in enumerate(self.fs)
            for s, *_ in f['x'].iter_chunks()
        })

        self.batch_size = batch_size
        self.group_by = group_by

        self.device = device
        self.pin_memory = pin_memory or device == 'cuda'

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
            theta = torch.from_numpy(f['theta'][idx])
        else:
            theta = None

        x = torch.from_numpy(f['x'][idx])

        return theta.to(self.device), x.to(self.device)

    def __iter__(self):  # -> tuple[Tensor, Tensor]
        if self.shuffle:
            self.rng.shuffle(self.chunks)

        for i in range(0, len(self.chunks), self.group_by):
            slices = sorted(self.chunks[i:i + self.group_by])

            # Load
            theta = np.concatenate([self.fs[j]['theta'][k:l] for j, k, l in slices])
            x = np.concatenate([self.fs[j]['x'][k:l] for j, k, l in slices])

            # Shuffle
            if self.shuffle:
                order = self.rng.permutation(len(theta))
                theta, x_chunk = theta[order], x[order]

            # CUDA
            theta, x = torch.from_numpy(theta), torch.from_numpy(x)

            if self.pin_memory:
                theta, x = theta.pin_memory(), x.pin_memory()

            # Batches
            for a, b in zip(
                theta.split(self.batch_size),
                x.split(self.batch_size),
            ):
                yield a.to(self.device), b.to(self.device)

