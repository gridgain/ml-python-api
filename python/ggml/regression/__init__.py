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
"""Regression trainers.
"""

import numpy as np

from ..common import Model
from ..common import SupervisedTrainer
from ..common import Proxy
from ..common import Utils
from ..common import LearningEnvironmentBuilder

from ..core import Cache

from ..common import gateway

class RegressionTrainer(SupervisedTrainer, Proxy):
    """Regression.
    """
    def __init__(self, proxy, multiple_labels=False, accepts_matrix=False):
        """Constructs a new instance of regression trainer.
        """
        self.multiple_labels = multiple_labels
        self.accepts_matrix = accepts_matrix
        Proxy.__init__(self, proxy)

    def fit(self, X, y=None):
        if isinstance(X, Cache):
            if y is not None:
                raise Exception("Second argument (y) is unexpected in case the first parameters is cache.")
            return self.fit_on_cache(X)

        X = np.array(X)
        y = np.array(y)

        # Check dimensions: we expected to have 2-dim X and y arrays.
        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)
        elif X.ndim > 2:
            raise Exception("X has unexpected dimension [dim=%d]" % X.ndim)

        if y.ndim == 1:
            y = y.reshape(y.shape[0], 1)
        elif y.ndim > 2:
            raise Exception("y has unexpected dimension [dim=%d]" % y.ndim)

        X_java = Utils.to_java_double_array(X)

        # We have two types of models: first type can accept multiple labels, second can't.
        if self.multiple_labels:        
            y_java = Utils.to_java_double_array(y)
            java_model = self.proxy.fit(X_java, y_java, None)
            return Model(java_model, self.accepts_matrix)
        else:
            java_models = []
            # Here we need to prepare a model for each y column.
            for i in range(y.shape[1]):
                y_java = Utils.to_java_double_array(y[:,i])
                java_model = self.proxy.fit(X_java, y_java, None)
                java_models.append(java_model)
            return Model(java_models, self.accepts_matrix)

    def fit_on_cache(self, cache):
        if not isinstance(cache, Cache):
            raise Exception("Unexpected type of cache (%s)." % type(cache))

        java_model = self.proxy.fitOnCache(cache.proxy, cache.cache_filter, Proxy.proxy_or_none(cache.preprocessor))
        return Model(java_model, self.accepts_matrix)


class DecisionTreeRegressionTrainer(RegressionTrainer):
    """DecisionTree regression trainer.
    """
    def __init__(self, env_builder=LearningEnvironmentBuilder(),
                 max_deep=5,
                 min_impurity_decrease=0.0, compressor=None, use_index=True):
        """Constructs a new instance of DecisionTree regression trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        max_deep : Max deep.
        min_impurity_decrease : Min impurity decrease.
        compressor : Compressor.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.tree.DecisionTreeRegressionTrainer(max_deep, min_impurity_decrease, compressor)
        proxy.withEnvironmentBuilder(Proxy.proxy_or_none(env_builder))
        proxy.withUsingIdx(use_index)

        RegressionTrainer.__init__(self, gateway.jvm.org.gridgain.ml.python.PythonDatasetTrainer(proxy))

class KNNRegressionTrainer(RegressionTrainer):
    """KNN regression trainer.
    """
    def __init__(self, env_builder=LearningEnvironmentBuilder()):
        """Constructs a new instance of linear regression trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.knn.regression.KNNRegressionTrainer()
        proxy.withEnvironmentBuilder(Proxy.proxy_or_none(env_builder))

        RegressionTrainer.__init__(self, gateway.jvm.org.gridgain.ml.python.PythonDatasetTrainer(proxy))

class LinearRegressionTrainer(RegressionTrainer):
    """Linear regression trainer.
    """
    def __init__(self, env_builder=LearningEnvironmentBuilder()):
        """Constructs a new instance of linear regression trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.regressions.linear.LinearRegressionLSQRTrainer()
        proxy.withEnvironmentBuilder(Proxy.proxy_or_none(env_builder))

        RegressionTrainer.__init__(self, gateway.jvm.org.gridgain.ml.python.PythonDatasetTrainer(proxy))

class RandomForestRegressionTrainer(RegressionTrainer):
    """RandomForest classification trainer.
    """
    def __init__(self, features, env_builder=LearningEnvironmentBuilder(),
                 trees=1, sub_sample_size=1.0, max_depth=5,
                 min_impurity_delta=0.0, seed=None):
        """Constructs a new instance of RandomForest classification trainer.

        Parameters
        ----------
        features : Number of features.
        env_builder : Environment builder.
        trees : Number of trees.
        sub_sample_size : Sub sample size.
        max_depth : Max depth.
        min_impurity_delta : Min impurity delta.
        seed : Seed.
        """
        metas = gateway.jvm.java.util.ArrayList()
        for i in range(features):
            meta = gateway.jvm.org.apache.ignite.ml.dataset.feature.FeatureMeta(None, i, False)
            metas.add(meta)

        proxy = gateway.jvm.org.apache.ignite.ml.tree.randomforest.RandomForestClassifierTrainer(metas)
        proxy.withEnvironmentBuilder(env_builder.proxy)
        proxy.withAmountOfTrees(trees)
        proxy.withSubSampleSize(sub_sample_size)
        proxy.withMaxDepth(max_depth)
        proxy.withMinImpurityDelta(min_impurity_delta)
        if seed is not None:
            proxy.withSeed(seed)

        RegressionTrainer.__init__(self, gateway.jvm.org.gridgain.ml.python.PythonDatasetTrainer(proxy))

class MLPArchitecture(Proxy):
    """MLP architecture.
    """
    def __init__(self, input_size):
        """Constructs a new instance of MLP architecture.

        Parameters
        ----------
        input_size : Input size.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.nn.architecture.MLPArchitecture(input_size)
        Proxy.__init__(self, proxy)

    def with_layer(self, neurons, has_bias=True, activator='sigmoid'):
        """Add layer.

        Parameters
        ----------
        neurons : Number of neurons.
        has_bias : Has bias or not (default value is True).
        activator : Activation function ('sigmoid', 'relu' or 'linear', default value is 'sigmoid')
        """
        java_activator = None
        if activator == 'sigmoid':
            java_activator = gateway.jvm.org.apache.ignite.ml.nn.Activators.SIGMOID
        elif activator == 'relu':
            java_activator = gateway.jvm.org.apache.ignite.ml.nn.Activators.RELU
        elif activator == 'linear':
            java_activator = gateway.jvm.org.apache.ignite.ml.nn.Activators.LINEAR
        else:
            raise Exception("Unknown activator: %s" % activator)

        self.proxy = self.proxy.withAddedLayer(neurons, has_bias, java_activator)

        return self

class MLPRegressionTrainer(RegressionTrainer):
    """MLP regression trainer.
    """
    def __init__(self, arch, env_builder=LearningEnvironmentBuilder(), loss='mse',
                 learning_rate=0.1, max_iter=1000, batch_size=100, loc_iter=10, seed=None):
        """Constructs a new instance of MLP regression trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        arch : Architecture.
        loss : Loss function ('mse', 'log', 'l2', 'l1' or 'hinge', default value is 'mse').
        update_strategy : Update strategy.
        max_iter : Max number of iterations.
        batch_size : Batch size.
        loc_iter : Number of local iterations.
        seed : Seed.
        """
        java_loss = None
        if loss == 'mse':
            java_loss = gateway.jvm.org.apache.ignite.ml.optimization.LossFunctions.MSE
        elif loss == 'log':
            java_loss = gateway.jvm.org.apache.ignite.ml.optimization.LossFunctions.LOG
        elif loss == 'l2':
            java_loss = gateway.jvm.org.apache.ignite.ml.optimization.LossFunctions.L2
        elif loss == 'l1':
            java_loss = gateway.jvm.org.apache.ignite.ml.optimization.LossFunctions.L1
        elif loss == 'hinge':
            java_loss = gateway.jvm.org.apache.ignite.ml.optimization.LossFunctions.HINGE
        else:
            raise Exception('Unknown loss: %s' % loss)

        proxy = gateway.jvm.org.gridgain.ml.python.PythonMLPDatasetTrainer(arch.proxy, java_loss, learning_rate, max_iter, batch_size, loc_iter, seed)
        RegressionTrainer.__init__(self, proxy, True, True)
