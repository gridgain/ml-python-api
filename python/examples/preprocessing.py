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

# Normalization preprocessing with Ignite ML (local).
from sklearn.datasets import make_classification
from ggml.preprocessing import NormalizationTrainer

x, y = make_classification()
normalizer = NormalizationTrainer().fit(x)
normalizer.transform(x)

# Normalization preprocessing with Ignite ML (cache).
import numpy as np
from sklearn.datasets import make_classification
from ggml.core import Ignite
from ggml.preprocessing import NormalizationTrainer

with Ignite("example-ignite.xml") as ignite:
    cache = ignite.create_cache("my-cache")
    for i, row in enumerate(np.column_stack(make_classification())):
        cache.put(i, row)
    normalizer = NormalizationTrainer().fit(cache)
    cache_transformed = cache.transform(normalizer)
    head = cache_transformed.head()
head

# Binarization preprocessing with Ignite ML (local).
from sklearn.datasets import make_classification
from ggml.preprocessing import BinarizationTrainer

x, y = make_classification()
binarizer = BinarizationTrainer(threshold=0.5).fit([[]])
binarizer.transform(x)

# Binarization preprocessing with Ignite ML (cache).
import numpy as np
from sklearn.datasets import make_classification
from ggml.core import Ignite
from ggml.preprocessing import BinarizationTrainer

with Ignite("example-ignite.xml") as ignite:
    cache = ignite.create_cache("my-cache")
    for i, row in enumerate(np.column_stack(make_classification())):
        cache.put(i, row)
    binarizer = BinarizationTrainer().fit(cache)
    cache_transformed = cache.transform(binarizer)
    head = cache_transformed.head()
head

# Imputing preprocessing with Ignite ML (local).
from sklearn.datasets import make_classification
from ggml.preprocessing import ImputerTrainer

x = [[None, 1, 1], [2, None, 2]]
imputer = ImputerTrainer().fit(x)
imputer.transform(x)

# Imputing preprocessing with Ignite ML (cache).
import numpy as np
from sklearn.datasets import make_classification
from ggml.core import Ignite
from ggml.preprocessing import ImputerTrainer

with Ignite("example-ignite.xml") as ignite:
    cache = ignite.create_cache("my-cache")
    for i, row in enumerate([[None, 1, 1, 0], [2, None, 2, 0]]):
        cache.put(i, row)
    imputer = ImputerTrainer().fit(cache)
    cache_transformed = cache.transform(imputer)
    head = cache_transformed.head()
head

# One-Hot-Encoding preprocessing with Ignite ML (local).
from sklearn.datasets import make_classification
from ggml.preprocessing import EncoderTrainer

x = [[1, 2, 0], [2, 1, 0]]
encoder = EncoderTrainer(encoded_features=[0, 1]).fit(x)
encoder.transform(x)

# One-Hot-Encoding preprocessing with Ignite ML (cache).
import numpy as np
from sklearn.datasets import make_classification
from ggml.core import Ignite
from ggml.preprocessing import EncoderTrainer

with Ignite("example-ignite.xml") as ignite:
    cache = ignite.create_cache("my-cache")
    for i, row in enumerate([[1, 2, 0], [2, 1, 0]]):
        cache.put(i, row)
    encoder = EncoderTrainer(encoded_features=[0, 1]).fit(cache)
    cache_transformed = cache.transform(encoder)
    head = cache_transformed.head()
head

# MinMax scaling preprocessing with Ignite ML (local).
from sklearn.datasets import make_classification
from ggml.preprocessing import MinMaxScalerTrainer

x, y = make_classification()
scaler = MinMaxScalerTrainer().fit(x)
scaler.transform(x)

# MinMax scaling preprocessing with Ignite ML (cache).
import numpy as np
from sklearn.datasets import make_classification
from ggml.core import Ignite
from ggml.preprocessing import MinMaxScalerTrainer

with Ignite("example-ignite.xml") as ignite:
    cache = ignite.create_cache("my-cache")
    for i, row in enumerate(np.column_stack(make_classification())):
        cache.put(i, row)
    scaler = MinMaxScalerTrainer().fit(cache)
    cache_transformed = cache.transform(scaler)
    head = cache_transformed.head()
head

# MaxAbs scaling preprocessing with Ignite ML (local).
from sklearn.datasets import make_classification
from ggml.preprocessing import MaxAbsScalerTrainer

x, y = make_classification()
scaler = MaxAbsScalerTrainer().fit(x)
scaler.transform(x)

# MaxAbs scaling preprocessing with Ignite ML (cache).
import numpy as np
from sklearn.datasets import make_classification
from ggml.core import Ignite
from ggml.preprocessing import MaxAbsScalerTrainer

with Ignite("example-ignite.xml") as ignite:
    cache = ignite.create_cache("my-cache")
    for i, row in enumerate(np.column_stack(make_classification())):
        cache.put(i, row)
    scaler = MaxAbsScalerTrainer().fit(cache)
    cache_transformed = cache.transform(scaler)
    head = cache_transformed.head()
head
