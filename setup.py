#!/usr/bin/env python

import setuptools
import lampe

with open('README.md', 'r') as f:
    readme = f.read()

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='lampe',
    version=lampe.__version__,
    description='Likelihood-free AMortized Posterior Estimation with PyTorch',
    long_description=readme,
    long_description_content_type='text/markdown',
    keywords='posterior parameter amortized inference torch pytorch',
    author='François Rozet',
    author_email='francois.rozet@outlook.com',
    url='https://github.com/francois-rozet/lampe',
    install_requires=required,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
