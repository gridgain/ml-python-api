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

from sklearn.metrics import r2_score
from sklearn.datasets import make_regression

from ggml.regression import LinearRegressionTrainer
from ggml.regression import DecisionTreeRegressionTrainer
from ggml.regression import KNNRegressionTrainer
from ggml.regression import RandomForestRegressionTrainer
from ggml.regression import MLPArchitecture
from ggml.regression import MLPRegressionTrainer

class TestRegressions(unittest.TestCase):

    def test_linear_regression(self):
        x_train, x_test, y_train, y_test = self.__generate_dataset()
        trainer = LinearRegressionTrainer()
        model = trainer.fit(x_train, y_train)
        self.assertTrue(r2_score(y_test, model.predict(x_test)) > 0.5)

    def test_decision_tree_regression(self):
        x_train, x_test, y_train, y_test = self.__generate_dataset()
        trainer = DecisionTreeRegressionTrainer(max_deep=100)
        model = trainer.fit(x_train, y_train)
        self.assertTrue(r2_score(y_test, model.predict(x_test)) > 0.5)

    def test_knn_regression(self):
        x_train, x_test, y_train, y_test = self.__generate_dataset()
        trainer = KNNRegressionTrainer()
        model = trainer.fit(x_train, y_train)
        self.assertTrue(r2_score(y_test, model.predict(x_test)) > 0.5)

    def test_random_forest_regression(self):
        x_train, x_test, y_train, y_test = self.__generate_dataset()
        trainer = RandomForestRegressionTrainer(20, trees=10, max_depth=100)
        model = trainer.fit(x_train, y_train)
        self.assertTrue(r2_score(y_test, model.predict(x_test)) > 0.5)

    def test_mlp_regression(self):
        x_train, x_test, y_train, y_test = self.__generate_dataset()
        trainer = MLPRegressionTrainer(MLPArchitecture(20).with_layer(1, activator='linear'))
        model = trainer.fit(x_train, y_train)
        self.assertTrue(r2_score(y_test, model.predict(x_test)) > 0.5)

    def __generate_dataset(self):
        x, y = make_regression(random_state=42, n_features=20, n_informative=20, n_samples=100)
        return (x, x, y, y)

if __name__ == '__main__':
    unittest.main()
