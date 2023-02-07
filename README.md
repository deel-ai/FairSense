<img src="docs/source/fairsense.png" alt="logo fairsense" style
="width
:500px;"/>

FairSense
===========

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

The problem
----------

Each index has it's characteristics: some can be applied on continuous variables and 
some cannot. Some can handle regression problems and some handle classification 
problems. Some can handle variable groups and some cannot. Finally some can only be
applied on the predictions of a model while some can be applied on the error made by
the model.

The objective is then to provide a tool to investigate the fairness of an ML problem
by computing the GSA indices while avoiding the aforementioned issues.

The strategy
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


install fairsense
-------------------

###for users

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
