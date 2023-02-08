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

with open(path.join(this_directory, "requirements.txt"), encoding="utf-8") as f:
    install_requires = f.read().split()

with open(path.join(this_directory, "requirements_dev.txt"), encoding="utf-8") as f:
    dev_requires = f.read().split()

with open(path.join(this_directory, "requirements_docs.txt"), encoding="utf-8") as f:
    docs_requires = f.read().split()

setuptools.setup(
    name="fairsense",
    version="0.0.1",
    author=", ".join(["Thibaut BOISSIN", "Alexandre LANGLADE"]),
    author_email=", ".join(
        [
            "thibaut.boissin@irt-saintexupery.com",
            "alexandre.langlade@irt-saintexupery.com",
        ]
    ),
    description="todo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="todo",
    packages=setuptools.find_namespace_packages(include=["fairsense.*"]),
    install_requires=install_requires,
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
