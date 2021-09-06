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

dev_requires = [
    "tox",
    "black",
    "flake8",
    "flake8-black",
    "numpy",
]

docs_requires = [
    "sphinx",
    "recommonmark",
    "sphinx_rtd_theme",
    "sphinx_markdown_builder",
    "sphinxcontrib_katex",
    "ipython",  # required for Pygments
]

setuptools.setup(
    name="libfairness",
    version="0.0.1",
    author=", ".join(["Alexandre LANGLADE", "Thibaut BOISSIN"]),
    author_email=", ".join(
        [
            "alexandre.langlade@irt-saintexupery.com",
            "thibaut.boissin@irt-saintexupery.com",
        ]
    ),
    description="todo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="todo",
    packages=setuptools.find_namespace_packages(include=["libfairness.*"]),
    install_requires=["numpy", "pandas", "matplotlib", "scikit-learn", "seaborn"],
    license="MIT",
    extras_require={"dev": dev_requires, "docs": docs_requires},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
