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

from ..common import Model
from ..common import Proxy
from ..common import Utils
from ..common import gateway
from copy import copy
from numbers import Number

def save_model(mdl, dst, ignite=None):
    if not isinstance(mdl.proxy, list):
        gateway.jvm.org.gridgain.ml.python.PythonModelSaver.save(mdl.proxy, dst, ignite.ignite)

    if len(mdl.proxy) != 1:
        raise Exception("Models with multiple inner models are not supported")

    if mdl.accepts_matrix:
        raise Exception("Models that accept matrix are not supported")

    gateway.jvm.org.gridgain.ml.python.PythonModelSaver.save(mdl.proxy[0], dst, ignite.ignite)

def read_model(src, ignite=None):
    proxy = [gateway.jvm.org.gridgain.ml.python.PythonModelSaver.read(src, ignite.ignite)]

    return Model(proxy, False)
