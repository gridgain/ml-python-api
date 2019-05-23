# GridGain ML Python API

Detailed documentation can be found on [https://machine-learning-python-api.readthedocs.io/](https://machine-learning-python-api.readthedocs.io/) and in `docs` directory.

This instruction will help you to upload this package into PyPi:

* Package version in `setup.py` should be updated.
* The following commands should be called.

```
python3 setup.py sdist
twine upload dist/*
```
