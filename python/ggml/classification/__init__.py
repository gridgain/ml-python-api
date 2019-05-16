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
"""Classification trainers.
"""

import numpy as np

from ..common import Model
from ..common import SupervisedTrainer
from ..common import Proxy
from ..common import Utils
from ..common import LearningEnvironmentBuilder

from ..common import gateway
from ..core import Cache

class ClassificationTrainer(SupervisedTrainer, Proxy):
    """Classification trainer.
    """
    def __init__(self, proxy):
        """Constructs a new instance of classification trainer.
        """
        Proxy.__init__(self, proxy)

    def fit(self, X, y=None):
        if isinstance(X, Cache):
            if y is not None:
                raise Exception("Second argument (y) is unexpected in case the first parameters is cache.")
            return self.fit_on_cache(X)

        X_java = Utils.to_java_double_array(X)
        y_java = Utils.to_java_double_array(y)

        java_model = self.proxy.fit(X_java, y_java, None)

        return Model(java_model, False)

    def fit_on_cache(self, cache):
        if not isinstance(cache, Cache):
            raise Exception("Unexpected type of cache (%s)." % type(cache))

        java_model = self.proxy.fitOnCache(cache.proxy, cache.cache_filter, Proxy.proxy_or_none(cache.preprocessor))
        return Model(java_model, False)

class ANNClassificationTrainer(ClassificationTrainer):
    """ANN classification trainer.
    """
    def __init__(self, env_builder=LearningEnvironmentBuilder(), k=2,
                 max_iter=10, eps=1e-4, distance='euclidean'):
        """Constructs a new instance of ANN classification trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        k : Number of clusters.
        max_iter : Max number of iterations.
        eps : Epsilon, delta of convergence.
        distance : Distance measure ('euclidean', 'hamming', 'manhattan').
        """
        proxy = gateway.jvm.org.apache.ignite.ml.knn.ann.ANNClassificationTrainer()

        proxy.withEnvironmentBuilder(Proxy.proxy_or_none(env_builder))
        proxy.withK(k)
        proxy.withMaxIterations(max_iter)
        proxy.withEpsilon(eps)

        java_distance = None
        if distance == 'euclidean':
            java_distance = gateway.jvm.org.apache.ignite.ml.math.distances.EuclideanDistance()
        elif distance == 'hamming':
            java_distance = gateway.jvm.org.apache.ignite.ml.math.distances.HammingDistance()
        elif distance == 'manhattan':
            java_distance = gateway.jvm.org.apache.ignite.ml.math.distances.ManhattanDistance()
        elif distance:
            raise Exception("Unknown distance type: %s" % distance)
        proxy.withDistance(java_distance)

        ClassificationTrainer.__init__(self, gateway.jvm.org.gridgain.ml.python.PythonDatasetTrainer(proxy))

class DecisionTreeClassificationTrainer(ClassificationTrainer):
    """DecisionTree classification trainer.
    """
    def __init__(self, env_builder=LearningEnvironmentBuilder(), max_deep=5,
                 min_impurity_decrease=0.0, compressor=None, use_index=True):
        """Constructs a new instance of DecisionTree classification trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        max_deep : Max deep.
        min_impurity_decrease : Min impurity decrease.
        compressor : Compressor.
        use_index : Use index.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.tree.DecisionTreeClassificationTrainer(max_deep, min_impurity_decrease, compressor)

        proxy.withEnvironmentBuilder(Proxy.proxy_or_none(env_builder))
        proxy.withUseIndex(use_index)

        ClassificationTrainer.__init__(self, gateway.jvm.org.gridgain.ml.python.PythonDatasetTrainer(proxy))

class KNNClassificationTrainer(ClassificationTrainer):
    """KNN classification trainer.
    """
    def __init__(self, env_builder=LearningEnvironmentBuilder()):
        """Constructs a new instance of KNN classification trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.knn.classification.KNNClassificationTrainer()
        proxy.withEnvironmentBuilder(Proxy.proxy_or_none(env_builder))

        ClassificationTrainer.__init__(self, gateway.jvm.org.gridgain.ml.python.PythonDatasetTrainer(proxy))

class LogRegClassificationTrainer(ClassificationTrainer):
    """LogisticRegression classification trainer.
    """
    def __init__(self, env_builder=LearningEnvironmentBuilder(), max_iter=100,
                 batch_size=100, max_loc_iter=100, seed=1234):
        """Constructs a new instance of LogisticRegression classification trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        max_iter : Max number of iterations.
        batch_size : Batch size.
        max_loc_iter : Max number of local iterations.
        update_strategy : Update strategy.
        seed : Seed.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.regressions.logistic.LogisticRegressionSGDTrainer()

        proxy.withEnvironmentBuilder(Proxy.proxy_or_none(env_builder))

        proxy.withMaxIterations(max_iter)
        proxy.withBatchSize(batch_size)
        proxy.withLocIterations(max_loc_iter)
        if seed is not None:
            proxy.withSeed(seed)

        ClassificationTrainer.__init__(self, gateway.jvm.org.gridgain.ml.python.PythonDatasetTrainer(proxy))

class RandomForestClassificationTrainer(ClassificationTrainer):
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

        ClassificationTrainer.__init__(self, gateway.jvm.org.gridgain.ml.python.PythonDatasetTrainer(proxy))

class SVMClassificationTrainer(ClassificationTrainer):
    """SVM classification trainer.
    """
    def __init__(self, env_builder=LearningEnvironmentBuilder(), l=0.4, max_iter=200, max_local_iter=100, seed=1234):
        """Constructs a new instance of SVM classification trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        l : Lambda.
        max_iter : Max number of iterations.
        max_loc_iter : Max number of local iterations.
        seed : Seed.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.svm.SVMLinearClassificationTrainer()    
        proxy.withEnvironmentBuilder(Proxy.proxy_or_none(env_builder))
        proxy.withLambda(l)
        proxy.withAmountOfIterations(max_iter)
        proxy.withAmountOfLocIterations(max_local_iter)
        if seed is not None:
            proxy.withSeed(seed)

        ClassificationTrainer.__init__(self, gateway.jvm.org.gridgain.ml.python.PythonDatasetTrainer(proxy))

class MLPClassificationTrainer(ClassificationTrainer):
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
        RegressionTrainer.__init__(self, proxy, True)
