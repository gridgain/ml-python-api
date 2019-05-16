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

# Test/Train splitting.
import numpy as np
from sklearn.datasets import make_classification
from ggml.core import Ignite
from ggml.model_selection import train_test_split

with Ignite("example-ignite.xml") as ignite:
    cache = ignite.create_cache("my-cache")
    for i, row in enumerate(np.column_stack(make_classification())):
        cache.put(i, row)

    train_cache, test_cache = train_test_split(cache)
    
    dataset_1 = test_cache.head()
    dataset_2 = train_cache.head()

# Cross Validation.
import numpy as np
from sklearn.datasets import make_classification
from ggml.core import Ignite
from ggml.classification import DecisionTreeClassificationTrainer
from ggml.model_selection import cross_val_score

with Ignite("example-ignite.xml") as ignite:
    cache = ignite.create_cache("my-cache", parts=1)
    for i, row in enumerate(np.column_stack(make_classification())):
        cache.put(i, row)

    trainer = DecisionTreeClassificationTrainer()
    score = cross_val_score(trainer, cache)
score
