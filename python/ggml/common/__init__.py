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
"""Common classes.
"""

from abc import abstractmethod

import os
import sys
import numpy as np
from py4j.java_gateway import JavaGateway

if 'IGNITE_HOME' in os.environ:
    ignite_home = os.environ['IGNITE_HOME']

    libs_jar = []
    for f in os.listdir(ignite_home + '/libs'):
        if f.endswith('.jar'):
            libs_jar.append(ignite_home + '/libs/' + f)    
        if os.path.isdir(ignite_home + '/libs/' + f):
            for fi in os.listdir(ignite_home + '/libs/' + f):
                if fi.endswith('.jar'):
                    libs_jar.append(ignite_home + '/libs/' + f + '/' + fi)

    optional_libs_jar = []
    for opt in os.listdir(ignite_home + '/libs/optional'):
        for f in os.listdir(ignite_home + '/libs/optional/' + opt):
            if f.endswith('.jar'):
                optional_libs_jar.append(ignite_home + '/libs/optional/' + opt + '/' + f)

    classpath = ':'.join(libs_jar + optional_libs_jar)

    gateway = JavaGateway.launch_gateway(classpath=classpath, die_on_exit=True)
else:
    # To build documentation.
    gateway = None

class Utils:
    """Util class.
    """
    def to_java_double_array(array):
        """Converts python array into java double array.
        """
        array = np.array(array)
        java_array = gateway.new_array(gateway.jvm.double, *array.shape)
        Utils.__to_java_double_array_backtrack(array, java_array)
        return java_array

    def from_java_double_array(java_array):
        """Converts java double array into python array.
        """
        array = np.zeros(len(java_array))
        for i in range(len(java_array)):
            array[i] = java_array[i]
        return array

    def __to_java_double_array_backtrack(array, java_array):
        if array.ndim == 0:
            raise Exception("Array is scalar [dim=%d]" % array.ndim)

        for i in range(array.shape[0]):
            if array.ndim == 1:
                if array[i] is not None:
                    java_array[i] = float(array[i])
                else:
                    java_array[i] = float('NaN')
            else:
                Utils.__to_java_double_array_backtrack(array[i], java_array[i])

class Proxy:
    """Proxy class for Java object.
    """
    def __init__(self, proxy):
        """Constructs a new instance of proxy class for Java object.
        """
        self.proxy = proxy

    def proxy_or_none(proxy):
        """Returns proxy of the given object or None of object is None itself.
        """
        if proxy:
            return proxy.proxy
        else:
            return None

class LearningEnvironmentBuilder(Proxy):
    """Learning environment builder.
    """
    def __init__(self):
        """Constructs a new instance of learning environemtn builder.
        """
        if gateway:
            java_proxy = gateway.jvm.org.apache.ignite.ml.environment.LearningEnvironmentBuilder.defaultBuilder()
            Proxy.__init__(self, java_proxy)

class Model(Proxy):
    """Model.
    """
    def __init__(self, proxy, accepts_matrix):
        """Constructs a new instance of regression model.

        Parameters
        ----------
        proxy : Proxy object that represents Java model.
        accept_matrix : Flag that identifies if model accepts matrix or vector.
        """
        self.accepts_matrix = accepts_matrix
        Proxy.__init__(self, proxy)

    def predict(self, X):
        """Predicts a result.

        Parameters
        ----------

        X : Features.
        """
        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)
        elif X.ndim > 2:
            raise Exception("X has unexpected dimension [dim=%d]" % X.ndim)

        # Check if model accepts multiple objects for inference.
        if self.accepts_matrix:
            java_array = Utils.to_java_double_array(X)
            java_matrix = gateway.jvm.org.apache.ignite.ml.math.primitives.matrix.impl.DenseMatrix(java_array)
            # Check if model is a single model or model-per-label.
            if isinstance(self.proxy, list):
                predictions = np.array([mdl.predict(java_matrix) for mdl in self.proxy])
            else:
                res = self.proxy.predict(java_matrix)
                rows = res.rowSize()
                cols = res.columnSize()
                predictions = np.zeros((rows, cols))
                for i in range(rows):
                    for j in range(cols):
                        predictions[i, j] = res.get(i, j)
        else:
            predictions = []
            for i in range(X.shape[0]):
                java_array = Utils.to_java_double_array(X[i])
                java_vector_utils = gateway.jvm.org.apache.ignite.ml.math.primitives.vector.VectorUtils
                # Check if model is a single model or model-per-label.
                if isinstance(self.proxy, list):
                    def parse_response(m):
                        res = m.predict(java_vector_utils.of(java_array))
                        # This if handles 'future' response.
                        if hasattr(res, 'get') and callable(res.get):
                            res = res.get()
                        return res
                    prediction = [parse_response(mdl) for mdl in self.proxy]
                else:
                    prediction = [self.proxy.predict(java_vector_utils.of(java_array))]
                predictions.append(prediction)
            predictions = np.array(predictions)

        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = np.hstack(predictions)

        return predictions

class SupervisedTrainer:
    """Supervised trainer.
    """
    @abstractmethod
    def fit(self, X, y=None):
        """Trains model based on data.

        Parameters
        ----------
        X : x.
        y : y.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def fit_on_cache(self, cache):
        """Trains model based on data.

        Parameters
        ----------
        cache : Apache Ignite cache.
        """
        raise Exception("Not implemented")

class UnsupervisedTrainer:
    """Unsupervised trainer.
    """
    @abstractmethod
    def fit(self, X):
        """Trains model based on data.

        Parameters
        ----------
        X : x.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def fit_on_cache(self, cache):
        """Trains model based on data.

        Parameters
        ----------
        cache : Apache Ignite cache.
        """
        raise Exception("Not implemented")
