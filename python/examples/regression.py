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

# Linear regression with Ignite ML (local).
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from ggml.regression import LinearRegressionTrainer

x, y = make_regression()
x_train, x_test, y_train, y_test = train_test_split(x, y)

trainer = LinearRegressionTrainer()
model = trainer.fit(x_train, y_train)

r2_score(y_test, model.predict(x_test))

# Linear regression with Ignite ML (cache).
import numpy as np
from sklearn.datasets import make_regression
from ggml.core import Ignite
from ggml.model_selection import train_test_split
from ggml.metrics import rmse_score
from ggml.regression import LinearRegressionTrainer

with Ignite("example-ignite.xml") as ignite:
    cache = ignite.create_cache("my-cache")
    for i, row in enumerate(np.column_stack(make_regression())):
        cache.put(i, row)

    train_cache, test_cache = train_test_split(cache)
    
    trainer = LinearRegressionTrainer()
    model = trainer.fit(train_cache)
    print(rmse_score(test_cache, model))

# Decision Tree regression with Ignite ML (local).
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from ggml.regression import DecisionTreeRegressionTrainer

x, y = make_regression()
x_train, x_test, y_train, y_test = train_test_split(x, y)

trainer = DecisionTreeRegressionTrainer()
model = trainer.fit(x_train, y_train)

r2_score(y_test, model.predict(x_test))

# Decision Tree regression with Ignite ML (cache).
import numpy as np
from sklearn.datasets import make_regression
from ggml.core import Ignite
from ggml.model_selection import train_test_split
from ggml.metrics import rmse_score
from ggml.regression import DecisionTreeRegressionTrainer

with Ignite("example-ignite.xml") as ignite:
    cache = ignite.create_cache("my-cache")
    for i, row in enumerate(np.column_stack(make_regression())):
        cache.put(i, row)

    train_cache, test_cache = train_test_split(cache)

    trainer = DecisionTreeRegressionTrainer()
    model = trainer.fit(train_cache)
    print(rmse_score(test_cache, model))

# KNN regression with Ignite ML (local).
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from ggml.regression import KNNRegressionTrainer

x, y = make_regression()
x_train, x_test, y_train, y_test = train_test_split(x, y)

trainer = KNNRegressionTrainer()
model = trainer.fit(x_train, y_train)

r2_score(y_test, model.predict(x_test))

# KNN regression with Ignite ML (cache).
import numpy as np
from sklearn.datasets import make_regression
from ggml.core import Ignite
from ggml.model_selection import train_test_split
from ggml.metrics import rmse_score
from ggml.regression import KNNRegressionTrainer

with Ignite("example-ignite.xml") as ignite:
    cache = ignite.create_cache("my-cache")
    for i, row in enumerate(np.column_stack(make_regression())):
        cache.put(i, row)

    train_cache, test_cache = train_test_split(cache)

    trainer = KNNRegressionTrainer()
    model = trainer.fit(train_cache)
    print(rmse_score(test_cache, model))

# Random Forest regression with Ignite ML (local).
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from ggml.regression import RandomForestRegressionTrainer

x, y = make_regression()
x_train, x_test, y_train, y_test = train_test_split(x, y)

trainer = RandomForestRegressionTrainer(features=100)
model = trainer.fit(x_train, y_train)

r2_score(y_test, model.predict(x_test))

# Random Forest regression with Ignite ML (cache).
import numpy as np
from sklearn.datasets import make_regression
from ggml.core import Ignite
from ggml.model_selection import train_test_split
from ggml.metrics import rmse_score
from ggml.regression import RandomForestRegressionTrainer

with Ignite("example-ignite.xml") as ignite:
    cache = ignite.create_cache("my-cache")
    for i, row in enumerate(np.column_stack(make_regression())):
        cache.put(i, row)

    train_cache, test_cache = train_test_split(cache)

    trainer = RandomForestRegressionTrainer(features=100)
    model = trainer.fit(train_cache)
    print(rmse_score(test_cache, model))

# MLP regression with Ignite ML (local).
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from ggml.regression import MLPArchitecture
from ggml.regression import MLPRegressionTrainer

x, y = make_regression()
x_train, x_test, y_train, y_test = train_test_split(x, y)

trainer = MLPRegressionTrainer(MLPArchitecture(input_size=100).with_layer(neurons=1, activator='linear'))
model = trainer.fit(x_train, y_train)

r2_score(y_test, model.predict(x_test))

# MLP regression with Ignite ML (cache).
import numpy as np
from sklearn.datasets import make_regression
from ggml.core import Ignite
from ggml.model_selection import train_test_split
from ggml.metrics import rmse_score
from ggml.regression import MLPArchitecture
from ggml.regression import MLPRegressionTrainer

with Ignite("example-ignite.xml") as ignite:
    cache = ignite.create_cache("my-cache")
    for i, row in enumerate(np.column_stack(make_regression())):
        cache.put(i, row)

    train_cache, test_cache = train_test_split(cache)

    trainer = MLPRegressionTrainer(MLPArchitecture(input_size=100).with_layer(neurons=1, activator='linear'))
    model = trainer.fit(train_cache)
    print(rmse_score(test_cache, model))
