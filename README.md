![LAMPE's banner](https://raw.githubusercontent.com/probabilists/lampe/master/docs/images/banner.svg)

# LAMPE

LAMPE is a simulation-based inference (SBI) package that focuses on amortized estimation of posterior distributions, without relying on explicit likelihood functions; hence the name *Likelihood-free AMortized Posterior Estimation* (LAMPE). The package provides [PyTorch](https://pytorch.org) implementations of modern amortized simulation-based inference algorithms like neural ratio estimation (NRE), neural posterior estimation (NPE) and more. Similar to PyTorch, the philosophy of LAMPE is to avoid obfuscation and expose all components, from network architecture to optimizer, to the user such that they are free to modify or replace anything they like.

As part of the inference pipeline, `lampe` provides components to efficiently [store and load data](lampe/data.py) from disk, [diagnose predictions](lampe/diagnostics/) and [display results](lampe/plots.py) graphically.

> [!IMPORTANT]
> In an effort to unite communities, the development of LAMPE has stopped in favor of the [sbi](https://github.com/sbi-dev/sbi) project. The `sbi` package already supports many of `lampe`'s features, and you are welcome to [submit issues](https://github.com/sbi-dev/sbi/issues) and PRs for the features you would like to be ported to `sbi`.

## Installation

The `lampe` package is available on [PyPI](https://pypi.org/project/lampe), which means it is installable via `pip`.

```
pip install lampe
```

Alternatively, if you need the latest features, you can install it from the repository.

```
pip install git+https://github.com/probabilists/lampe
```

## Documentation

The documentation is made with [Sphinx](https://www.sphinx-doc.org) and [Furo](https://github.com/pradyunsg/furo) and is hosted at [lampe.readthedocs.io](https://lampe.readthedocs.io).

## Contributing

If you have a question, an issue or would like to contribute, please read our [contributing guidelines](https://github.com/probabilists/lampe/blob/master/CONTRIBUTING.md).
