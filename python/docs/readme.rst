..  Licensed to the Apache Software Foundation (ASF) under one or more
    contributor license agreements.  See the NOTICE file distributed with
    this work for additional information regarding copyright ownership.
    The ASF licenses this file to You under the Apache License, Version 2.0
    (the "License"); you may not use this file except in compliance with
    the License.  You may obtain a copy of the License at

..      http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

=================
Basic Information
=================

What is it
----------

This is GridGain ML client library, written in Python 3, abbreviated as *ggml*.

`GridGain`_ is a memory-centric distributed database, caching, and processing platform for transactional, analytical, and streaming workloads delivering in-memory speeds at petabyte scale.

GridGain ML client library provides user applications the ability to work with GridGain ML functionality using `Py4J`_ as an integration mechanism.

Prerequisites
-------------

- *Python 3.4* or above (3.6 is tested),
- IGNITE_HOME environment variable with path to Apache Ignite.
- Apache Ignite should includea *ml-python-api.jar* in *libs* folder. The *ml-python-api.jar* could be built using this repository.

Installation
------------

for end user
^^^^^^^^^^^^

If you want to use *ggml* in your project, you may install it from PyPI:

::

$ pip install ggml

for developer
^^^^^^^^^^^^^

If you want to run tests, examples or build documentation, clone the whole repository:

::

$ git clone git@github.com:gridgain/ml-python-api.git
$ cd python
$ pip install -e .

This will install the repository version of *ggml* into your environment in so-called “develop” or “editable” mode. You may read more about `editable installs`_ in the pip manual.

.. _GridGain: https://apacheignite.readme.io/docs/what-is-ignite
.. _Py4J: https://www.py4j.org/
.. _editable installs: https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs
