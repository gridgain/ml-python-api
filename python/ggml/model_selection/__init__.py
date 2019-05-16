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

from ..core import Cache
from ..common import gateway
from ..common import Utils

def train_test_split(cache, test_size=0.25, train_size=0.75, random_state=None):
    """Splits given cache on two parts: test and train with given sizes.

    Parameters
    ----------
    cache : Ignite cache.
    test_size : Test size.
    train_size : Train size.
    random_state : Random state.
    """
    if not isinstance(cache, Cache):
        raise Exception("Unexpected type of cache (%s)." % type(cache))    

    split = gateway.jvm.org.apache.ignite.ml.selection.split.TrainTestDatasetSplitter().split(train_size, test_size)
    train_filter = split.getTrainFilter()
    test_filter = split.getTestFilter()
    return (cache.filter(train_filter), cache.filter(test_filter))

def cross_val_score(trainer, cache, cv=5, scoring='accuracy'):
    """Makes cross validation for given trainer, cache and scoring.

    Parameters
    ----------
    trainer : Trainer.
    cache : Cache.
    cv : Number of folds.
    scoring : Metric to be scored.
    """
    if not isinstance(cache, Cache):
        raise Exception("Unexpected type of cache (%s)." % type(cache))

    if scoring == 'accuracy':
        metric = gateway.jvm.org.apache.ignite.ml.selection.scoring.metric.classification.Accuracy()
    else:
        raise Exception("Unsupported type of scoring metric: %s" % scoring)

    res = gateway.jvm.org.gridgain.ml.python.PythonCrossValidation.score(trainer.proxy.getDelegate(), metric, cache.proxy, cv)

    return Utils.from_java_double_array(res)
