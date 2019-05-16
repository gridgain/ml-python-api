#
# Copyright 2019 GridGain Systems, Inc. and Contributors.
#
# Licensed under the GridGain Community Edition License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gridgain.com/products/software/community-edition/gridgain-community-edition-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np
from numbers import Number

from ..common import SupervisedTrainer
from ..common import Proxy
from ..common import Utils
from ..common import LearningEnvironmentBuilder

from ..common import gateway

from ..core import Cache

__evaluator = gateway.jvm.org.gridgain.ml.python.PythonEvaluator

def __evaluate_regression(cache, mdl, preprocessor=None):
    if not isinstance(cache, Cache):
        raise Exception("Unexpected type of cache (%s)." % type(cache))

    return __evaluator.evaluateRegression(cache.proxy, None, mdl.proxy, preprocessor)

def __evaluate_classification(cache, mdl, preprocessor=None):
    if not isinstance(cache, Cache):
        raise Exception("Unexpected type of cache (%s)." % type(cache))

    return __evaluator.evaluateClassification(cache.proxy, None, mdl.proxy, preprocessor)

def accuracy_score(cache, mdl, preprocessor=None):
    """Calculate accuracy score (classification metric).

    Parameters
    ----------
    cache : Cache or cache view (cache with filter).
    mdl : Model.
    preprocessor : Preprocessor.
    """
    classification_metrics = __evaluate_classification(cache, mdl, preprocessor)
    return classification_metrics.accuracy()

def balanced_accuracy_score(cache, mdl, preprocessor=None):
    """Calculate balanced accuracy score (classification metric).

    Parameters
    ----------
    cache : Cache or cache view (cache with filter).
    mdl : Model.
    preprocessor : Preprocessor.
    """
    classification_metrics = __evaluate_classification(cache, mdl, preprocessor)
    return classification_metrics.balancedAccuracy()

def precision_score(cache, mdl, preprocessor=None):
    """Calculate precision score (classification metric).

    Parameters
    ----------
    cache : Cache or cache view (cache with filter).
    mdl : Model.
    preprocessor : Preprocessor.
    """
    classification_metrics = __evaluate_classification(cache, mdl, preprocessor)
    return classification_metrics.precision()

def recall_score(cache, mdl, preprocessor=None):
    """Calculate recall score (classification metric).

    Parameters
    ----------
    cache : Cache or cache view (cache with filter).
    mdl : Model.
    preprocessor : Preprocessor.
    """
    classification_metrics = __evaluate_classification(cache, mdl, preprocessor)
    return classification_metrics.recall()

def f1_score(cache, mdl, preprocessor=None):
    """Calculate f1 score (classification metric).

    Parameters
    ----------
    cache : Cache or cache view (cache with filter).
    mdl : Model.
    preprocessor : Preprocessor.
    """
    classification_metrics = __evaluate_classification(cache, mdl, preprocessor)
    return classification_metrics.f1Score()

def mae_score(cache, mdl, preprocessor=None):
    """Calculate mean absolute error score (regression metric).

    Parameters
    ----------
    cache : Cache or cache view (cache with filter).
    mdl : Model.
    preprocessor : Preprocessor.
    """
    regression_metrics = __evaluate_regression(cache, mdl, preprocessor)
    return regression_metrics.mae()

def mse_score(cache, mdl, preprocessor=None):
    """Calculate mean squared error score (regression metric).

    Parameters
    ----------
    cache : Cache or cache view (cache with filter).
    mdl : Model.
    preprocessor : Preprocessor.
    """
    regression_metrics = __evaluate_regression(cache, mdl, preprocessor)
    return regression_metrics.mse()

def rss_score(cache, mdl, preprocessor=None):
    """Calculate residual sum of squares score (regression metric).

    Parameters
    ----------
    cache : Cache or cache view (cache with filter).
    mdl : Model.
    preprocessor : Preprocessor.
    """
    regression_metrics = __evaluate_regression(cache, mdl, preprocessor)
    return regression_metrics.rss()

def rmse_score(cache, mdl, preprocessor=None):
    """Calculate root mean squared error score (regression metric).

    Parameters
    ----------
    cache : Cache or cache view (cache with filter).
    mdl : Model.
    preprocessor : Preprocessor.
    """
    regression_metrics = __evaluate_regression(cache, mdl, preprocessor)
    return regression_metrics.rmse()
