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

# Distributed inference using Ignite ML.
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ggml.core import Ignite
from ggml.inference import IgniteDistributedModel
from ggml.classification import DecisionTreeClassificationTrainer

x, y = make_classification()
x_train, x_test, y_train, y_test = train_test_split(x, y)

trainer = DecisionTreeClassificationTrainer()
model = trainer.fit(x_train, y_train)

with Ignite("example-ignite.xml") as ignite:
    with IgniteDistributedModel(ignite, model) as ignite_distr_mdl:
        print(accuracy_score(
            y_test, 
            ignite_distr_mdl.predict(x_test)
        ))

# Model storage using Ignite ML (local).
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from ggml.core import Ignite
from ggml.regression import LinearRegressionTrainer
from ggml.storage import save_model
from ggml.storage import read_model

x, y = make_regression()
x_train, x_test, y_train, y_test = train_test_split(x, y)

trainer = LinearRegressionTrainer()
model = trainer.fit(x_train, y_train)

with Ignite("example-ignite.xml") as ignite:
    save_model(model, 'test.mdl', ignite)
    model = read_model('test.mdl', ignite)

r2_score(y_test, model.predict(x_test))

# Model storage using Ignite ML (cache).
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from ggml.core import Ignite
from ggml.regression import LinearRegressionTrainer
from ggml.storage import save_model
from ggml.storage import read_model

x, y = make_regression()
x_train, x_test, y_train, y_test = train_test_split(x, y)

trainer = LinearRegressionTrainer()
model = trainer.fit(x_train, y_train)

with Ignite("example-ignite-ml.xml") as ignite:
    save_model(model, 'igfs:///test.mdl', ignite)
    model = read_model('igfs:///test.mdl', ignite)

r2_score(y_test, model.predict(x_test))
