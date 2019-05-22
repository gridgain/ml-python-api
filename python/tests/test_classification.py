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

from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

from ggml.classification import DecisionTreeClassificationTrainer
from ggml.classification import ANNClassificationTrainer
from ggml.classification import KNNClassificationTrainer
from ggml.classification import LogRegClassificationTrainer
from ggml.classification import SVMClassificationTrainer
from ggml.classification import RandomForestClassificationTrainer

class TestClassification(unittest.TestCase):

    def test_decision_tree_classification(self):
        train_x, test_x, train_y, test_y = self.__generate_dataset()
        trainer = DecisionTreeClassificationTrainer()
        model = trainer.fit(train_x, train_y)
        self.assertTrue(accuracy_score(test_y, model.predict(test_x)) > 0.8)

    def test_ann_classification(self):
        train_x, test_x, train_y, test_y = self.__generate_dataset()
        trainer = ANNClassificationTrainer()
        model = trainer.fit(train_x, train_y)
        self.assertTrue(accuracy_score(test_y, model.predict(test_x)) > 0.5)

    def test_knn_classification(self):
        train_x, test_x, train_y, test_y = self.__generate_dataset()
        trainer = KNNClassificationTrainer()
        model = trainer.fit(train_x, train_y)
        self.assertTrue(accuracy_score(test_y, model.predict(test_x)) > 0.8)

    def test_log_reg_classification(self):
        train_x, test_x, train_y, test_y = self.__generate_dataset()
        trainer = LogRegClassificationTrainer()
        model = trainer.fit(train_x, train_y)
        self.assertTrue(accuracy_score(test_y, model.predict(test_x)) > 0.8)

    def test_svm_classification(self):
        train_x, test_x, train_y, test_y = self.__generate_dataset()
        trainer = SVMClassificationTrainer()
        model = trainer.fit(train_x, train_y)
        self.assertTrue(accuracy_score(test_y, model.predict(test_x)) > 0.7)

    def test_random_forest_classification(self):
        train_x, test_x, train_y, test_y = self.__generate_dataset()
        trainer = RandomForestClassificationTrainer(20)
        model = trainer.fit(train_x, train_y)
        self.assertTrue(accuracy_score(test_y, model.predict(test_x)) > 0.8)

    def __generate_dataset(self):
        x, y = make_classification(random_state=42, n_features=20, n_informative=10, n_samples=100)
        return (x, x, y, y)

if __name__ == '__main__':
    unittest.main()
