<!-- Banner section -->
<div align="center">
    <img src="./assets/banner_dark.png#only-dark" width="75%" alt="lib banner" align="center" />
    <img src="./assets/banner_light.png#only-light" width="75%" alt="lib banner" align="center" />
</div>

<!-- Badge section -->
<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.7, 3.8, 3.9, 3.10-efefef">
    </a>
    <a href="https://github.com/deel-ai/FairSense/actions/workflows/python-lints.yml">
        <img alt="PyLint" src="https://github.com/deel-ai/FairSense/actions/workflows/python-lints.yml/badge.svg">
    </a>
    <a href="https://github.com/deel-ai/FairSense/actions/workflows/python-tests.yml">
        <img alt="Tox" src="https://github.com/deel-ai/FairSense/actions/workflows/python-tests.yml/badge.svg">
    </a>
    <a href="https://github.com/deel-ai/FairSense/actions/workflows/python-publish.yml">
        <img alt="Pypi" src="https://github.com/deel-ai/FairSense/actions/workflows/python-publish.yml/badge.svg">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
    <br>
    <a href="https://deel-ai.github.io/FairSense/"><strong>Explore FairSense docs</strong></a>
</div>
<br>

<!-- Short description of your library -->


This library allow to compute global sensitivity indices in the context of fairness measurements.
The paper `Fairness seen as Global Sensitivity Analysis` bridges the gap between 
global sensitivity analysis (GSA) and fairness. It states that for each sensitivity 
analysis, there is a fairness measure, and vice-versa.

    @misc{https://doi.org/10.48550/arxiv.2103.04613,
      doi = {10.48550/ARXIV.2103.04613},  
      url = {https://arxiv.org/abs/2103.04613},  
      author = {Bénesse, Clément and Gamboa, Fabrice and Loubes, Jean-Michel and Boissin, Thibaut},
      keywords = {Statistics Theory (math.ST), Methodology (stat.ME), FOS: Mathematics, FOS: Mathematics, FOS: Computer and information sciences, FOS: Computer and information sciences},
      title = {Fairness seen as Global Sensitivity Analysis},

This library is a toolbox which ease the computation of fairness and GSA indices.

👉 The problem
----------

Each index has it's characteristics: some can be applied on continuous variables and 
some cannot. Some can handle regression problems and some handle classification 
problems. Some can handle variable groups and some cannot. Finally some can only be
applied on the predictions of a model while some can be applied on the error made by
the model.

The objective is then to provide a tool to investigate the fairness of an ML problem
by computing the GSA indices while avoiding the aforementioned issues.

🚀 The strategy
------------

The library allows to formulate a fairness problem which is stated as following:

- a dataset describing the training distribution
- a model which can be a function or a machine learning model
- a fairness objective which indicate what should be studied : one can study the
 intrinsic bias of a dataset, or the bias of the model or the bias of the model's
  errors

These elements are encapsulated in an object called `IndicesInput`.

Then it becomes possible to compute GSA indices (in a interchangeable way) using the
functions provided in `fairsense.indices`.

These functions output `IndicesOutput` objects that encapsulate the values of the
indices. These results can finally be visualized with the functions available in the 
`fairsense.visualization` module.


💻 install fairsense
-------------------

### ‍for users

```bash
pip install fairsense
```

### for developpers

After cloning the repository
```bash
pip install -e .[dev]
```

to clean code, at the root of the lib:
```bash
black .
```

### for docs

```bash
pip install -e .[docs]
```

build rst files, in the docs folder:
```bash
sphinx-apidoc ..\libfairness -o source
```
the generate html docs:
```bash
make html
```
Warning: the library must be installed to generate the doc.

## 👍 Contributing

Feel free to propose your ideas or come and contribute with us on the Libname toolbox! We have a specific document where we describe in a simple way how to make your first pull request: [just here](CONTRIBUTING.md).

## 👀 See Also

More from the DEEL project:

- [Xplique](https://github.com/deel-ai/xplique) a Python library exclusively dedicated to explaining neural networks.
- [deel-lip](https://github.com/deel-ai/deel-lip) a Python library for training k-Lipschitz neural networks on TF.
- [Influenciae](https://github.com/deel-ai/influenciae) Python toolkit dedicated to computing influence values for the discovery of potentially problematic samples in a dataset.
- [deel-torchlip](https://github.com/deel-ai/deel-torchlip) a Python library for training k-Lipschitz neural networks on PyTorch.
- [DEEL White paper](https://arxiv.org/abs/2103.10529) a summary of the DEEL team on the challenges of certifiable AI and the role of data quality, representativity and explainability for this purpose.

## 🙏 Acknowledgments

<img align="right" src="https://share.deel.ai/apps/theming/image/logo?useSvg=1&v=10#only-dark" width="25%" alt="DEEL Logo" />
<img align="right" src="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png#only-light" width="25%" alt="DEEL Logo" />
This project received funding from the French ”Investing for the Future – PIA3” program within the Artificial and Natural Intelligence Toulouse Institute (ANITI). The authors gratefully acknowledge the support of the <a href="https://www.deel.ai/"> DEEL </a> project.

## 🗞️ Citation

If you use fairsense as part of your workflow in a scientific publication, please
 consider citing the 🗞️ [our paper](https://www.youtube.com/watch?v=dQw4w9WgXcQ):

```
    @misc{https://doi.org/10.48550/arxiv.2103.04613,
      doi = {10.48550/ARXIV.2103.04613},  
      url = {https://arxiv.org/abs/2103.04613},  
      author = {Bénesse, Clément and Gamboa, Fabrice and Loubes, Jean-Michel and Boissin, Thibaut},
      keywords = {Statistics Theory (math.ST), Methodology (stat.ME), FOS: Mathematics, FOS: Mathematics, FOS: Computer and information sciences, FOS: Computer and information sciences},
      title = {Fairness seen as Global Sensitivity Analysis},
```

## 📝 License

The package is released under <a href="https://choosealicense.com/licenses/mit"> MIT license</a>.
