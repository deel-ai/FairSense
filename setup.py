# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import setuptools

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

install_requires = [
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "seaborn",
    "tqdm",
]

dev_requires = [
    "setuptools",
    "pre-commit",
    "tox",
    "black",
    "flake8",
    "flake8-black",
    "pytest",
    "pylint",
    "bump2version",
]

docs_requires = [
    "mkdocs",
    "mkdocs-material",
    "mkdocstrings[python]",
    "mknotebooks",
    "ipython",
]

setuptools.setup(
    name="fairsense",
    version="0.0.1b",
    author=", ".join(["Thibaut BOISSIN", "Alexandre LANGLADE"]),
    author_email=", ".join(
        [
            "thibaut.boissin@irt-saintexupery.com",
            "alexandre.langlade@irt-saintexupery.com",
        ]
    ),
    description="This library allow to compute global sensitivity indices in the context of fairness measurements.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deel-ai/FairSense",
    packages=setuptools.find_namespace_packages(include=["deel.*"]),
    install_requires=install_requires,
    license="MIT",
    extras_require={
        "dev": dev_requires,
        "docs": docs_requires,
        "all": install_requires + dev_requires + docs_requires,
    },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
