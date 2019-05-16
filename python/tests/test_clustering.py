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

import unittest

from sklearn.datasets import make_blobs

from ggml.clustering import KMeansClusteringTrainer
from ggml.clustering import GMMClusteringTrainer

class TestClustering(unittest.TestCase):

    def test_kmeans_clustering(self):
        x, y = self.__generate_dataset()
        trainer = KMeansClusteringTrainer()
        trainer.fit(x)

    def test_gmm_clustering(self):
        x, y = self.__generate_dataset()
        trainer = GMMClusteringTrainer()
        trainer.fit(x)

    def __generate_dataset(self):
        x, y = make_blobs(n_samples=2000, n_features=2, cluster_std=1.0, centers=[(-3, -3), (0, 0), (3, 3)])
        return (x, y)

if __name__ == '__main__':
    unittest.main()
