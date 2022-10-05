![LAMPE's banner](https://raw.githubusercontent.com/francois-rozet/lampe/master/sphinx/images/banner.svg)

# LAMPE

LAMPE is a simulation-based inference (SBI) package that focuses on amortized estimation of posterior distributions, without relying on explicit likelihood functions; hence the name *Likelihood-free AMortized Posterior Estimation* (LAMPE). The package provides [PyTorch](https://pytorch.org) implementations of modern amortized simulation-based inference algorithms like neural ratio estimation (NRE), neural posterior estimation (NPE) and more. Similar to PyTorch, the philosophy of LAMPE is to avoid obfuscation and expose all components, from network architecture to optimizer, to the user such that they are free to modify or replace anything they like.

As part of the inference pipeline, `lampe` provides components to efficiently [store and load data](lampe/data.py) from disk, [diagnose predictions](lampe/diagnostics.py) and [display results](lampe/plots.py) graphically.

## Installation

The `lampe` package is available on [PyPI](https://pypi.org/project/lampe), which means it is installable via `pip`.

```
pip install lampe
```

Alternatively, if you need the latest features, you can install it from the repository.

```
pip install git+https://github.com/francois-rozet/lampe
```

## Documentation

The documentation is made with [Sphinx](https://www.sphinx-doc.org) and [Furo](https://github.com/pradyunsg/furo) and is hosted at [francois-rozet.github.io/lampe](https://francois-rozet.github.io/lampe).

## Contributing

If you have a question, an issue or would like to contribute, please read our [contributing guidelines](CONTRIBUTING.md).
