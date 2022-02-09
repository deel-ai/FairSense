libfairness
===========

This lib does a lot of amazing things !


install libfairness
-------------------

###for users

```bash
pip install .
```

or 

```bash
pip install git+https://forge.deel.ai/Fair/global_sensitivity_analysis_fairness
.git@refactoring
```

### for developpers

```bash
pip install -e .[dev]
```

to clean code, at the root of the lib:
```bash
black .
```

to run tests TODO:
```bash
tox -e py37
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

-----------------------------------

checklist des choses a faire avant de mettre en ligne:
- confidence intervals [x]
- variables groups for indices [x]
- model wrapper for sklearn pt & tf []
- check tests coverage []
- headers and docstrings []
- remove private functions from documentation [x]
- jupyter notebook for demo []
- make a real README []
- check licences dependences []