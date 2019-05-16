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
"""Ignite inference functionality.
"""

from ..common import Proxy
from ..common import Utils
from ..common import Model
from ..common import gateway
from copy import copy

class DistributedModel(Model):
    def __init__(self, ignite, reader, parser, instances=1, max_per_node=1):
        """Constructs a new instance of distributed model.

        Parameters
        ----------
        ignite : Ignite instance.
        reader : Model reader.
        parser : Model parser.
        mdl : Model.
        instances : Number of worker instances.
        max_per_node : Max number of worker per node.
        """
        super(DistributedModel, self).__init__(None, False)

        self.ignite = ignite
        self.reader = reader
        self.parser = parser
        self.instances = instances
        self.max_per_node = max_per_node

    def __enter__(self):
        self.proxy = [gateway.jvm.org.apache.ignite.ml.inference.builder.IgniteDistributedModelBuilder(
            self.ignite.ignite,
            self.instances,
            self.max_per_node
        ).build(r, self.parser) for r in self.reader]
        return self

    def __exit__(self, t, v, trace):
        if self.proxy is not None:
            for p in self.proxy:
                p.close()
            self.proxy = None
        return False

class XGBoostDistributedModel(DistributedModel):
    def __init__(self, ignite, mdl, instances=1, max_per_node=1):
        reader = [gateway.jvm.org.apache.ignite.ml.inference.reader.FileSystemModelReader(mdl)]
        parser = gateway.jvm.org.apache.ignite.ml.xgboost.parser.XGModelParser()
        
        super(XGBoostDistributedModel, self).__init__(ignite, reader, parser, instances, max_per_node)

    def predict(self, X):
        keys = gateway.jvm.java.util.HashMap()
        data = []

        idx = 0
        for key in X:
            keys[key] = idx
            idx = idx + 1
            data.append(X[key])

        java_array = Utils.to_java_double_array(data)
        java_vector_utils = gateway.jvm.org.apache.ignite.ml.math.primitives.vector.VectorUtils

        X = gateway.jvm.org.apache.ignite.ml.math.primitives.vector.impl.DelegatingNamedVector(java_vector_utils.of(java_array), keys)

        res = self.proxy[0].predict(X)
        # This if handles 'future' response.
        if hasattr(res, 'get') and callable(res.get):
            res = res.get()
        return res

class IgniteDistributedModel(DistributedModel):
    """Ignite distributed model.

    Parameters
    ----------
    ignite : Ignite instance.
    mdl : Model.
    instances : Number of instances.
    max_per_node : Max number of instance per node.
    """
    def __init__(self, ignite, mdl, instances=1, max_per_node=1):
        """Constructs a new instance of Ignite distributed model.

        Parameters
        ----------
        ignite : Ignite instance.
        reader : Model reader.
        parser : Model parser.
        instances : Number of worker instances.
        max_per_node : Max number of worker instances per ignite node.
        """
        if isinstance(mdl.proxy, list):
            reader = [gateway.jvm.org.apache.ignite.ml.inference.reader.InMemoryModelReader(p) for p in mdl.proxy]
        else:
            reader = [gateway.jvm.org.apache.ignite.ml.inference.reader.InMemoryModelReader(mdl.proxy)]

        parser = gateway.jvm.org.apache.ignite.ml.inference.parser.IgniteModelParser()

        super(IgniteDistributedModel, self).__init__(ignite, reader, parser, instances, max_per_node)
