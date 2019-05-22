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
"""Ignite cache API.
"""

from ..common import Proxy
from ..common import Utils

from ..common import gateway

import pandas as pd
import numpy as np

class Ignite:
 
    def __init__(self, cfg=None):
        """
        Constructs a new instance of Ignite that is required to work with Cache, IGFS storage and
        distributed inference.

        :param cfg: Path to Apache Ignite configuration file.
        """
        self.cfg = cfg
   
    def get_cache(self, name):
        """
        Returns existing Apache Ignite Cache by name. This module is built with assumption that Ignite
        Cache contains integer keys and double[] values.

        :param name: Name of the Apache Ignite cache.
        """
        if self.ignite is None:
            raise Exception("Use Ignite() inside with.. as.. command.")
        java_cache = self.ignite.cache(name)
        return Cache(java_cache)

    def create_cache(self, name, excl_neighbors=False, parts=10):
        """
        Creates a new Apache Ignite Cache using specified name and configuration. This module is built with
        assumption that Ignite Cache contains integer keys and double[] values.

        :param name: Name of the Apache Ignite cache,
        :param excl_neighbors: (optional, False by default) exclude neighbours,
        :param parts: (optional, 10 by default) number of partitions.
        """
        if self.ignite is None:
            raise Exception("Use Ignite() inside with.. as.. command.")
        affinity = gateway.jvm.org.apache.ignite.cache.affinity.rendezvous.RendezvousAffinityFunction(excl_neighbors, parts)
        cc = gateway.jvm.org.apache.ignite.configuration.CacheConfiguration()
        cc.setName(name)
        cc.setAffinity(affinity)
        java_cache = self.ignite.createCache(cc)
        return Cache(java_cache)

    def __enter__(self):
        if self.cfg is not None:
            self.ignite = gateway.jvm.org.apache.ignite.Ignition.start(self.cfg)
        else:
            self.ignite = gateway.jvm.org.apache.ignite.Ignition.start()
        return self

    def __exit__(self, t, v, trace):
        if self.ignite is not None:
            self.ignite.close()

class Cache(Proxy):
    """Internal constructor that creates a wrapper of Apache Ignite cache. User is expected to use Ignite
    object to create cache instead of this constructor.
    """
    def __init__(self, proxy, cache_filter=None, preprocessor=None):
        """
        Constructs a wrapper of Apache Ignite cache. It's internal method, user is expected to use Ignite
        methods to create or get cache.

        :param proxy: Py4J proxy that represents Apache Ignite Cache,
        :param cache_filter: Py4J proxy that represents filter,
        :param preprocessor: Py4J proxy that represents preprocessor.
        """
        Proxy.__init__(self, proxy)

        self.cache_filter = cache_filter
        self.preprocessor = preprocessor

    def __delitem__(self, key):
        raise Exception("Not implemented!")

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.get(key)
        elif isinstance(key, slice):
            result = []
            length = len(self)
            for k in range(*key.indices(length)):
                result.append(self.get(k))
            return np.array(result)
        else:
            raise Exception("Unexpected type of key (%s)." % type(key))

    def __setitem__(self, key, value):
        raise Exception("Not implemented!")

    def get(self, key):
        """
        Returns value (float array) by the specified key.

        :param key: Key to be taken from cache.
        """
        java_array = self.proxy.get(key)
        return Utils.from_java_double_array(java_array)

    def put(self, key, value):
        """
        Puts value (float array) by the specified key.

        :param key: Key to be put into cache,
        :param value: value to be taken from cache.
        """
        value = Utils.to_java_double_array(value)
        self.proxy.put(key, value)

    def head(self, n=5):
        """
        Returns top N elements represented as a pandas dataset.

        :param n: Number of rows to be returned.
        """
        scan_query = gateway.jvm.org.apache.ignite.cache.query.ScanQuery()
        
        if self.cache_filter is not None:
            scan_query.setFilter(self.cache_filter)

        cursor = self.proxy.query(scan_query)
        iterator = cursor.iterator()
        
        data = []
        while iterator.hasNext() and n != 0:
            entry = iterator.next()
            key = entry.getKey()
            value = entry.getValue()

            if self.preprocessor is not None:
                initial_array = Utils.from_java_double_array(value)
                preprocessed_value = self.preprocessor.proxy.apply(key, value)
                value = preprocessed_value.features().asArray()
                array = Utils.from_java_double_array(value)
                array = np.hstack((array, initial_array[-1]))
            else:
                array = Utils.from_java_double_array(value)
    
            data.append(array)
            n = n - 1

        return pd.DataFrame(data)

    def transform(self, preprocessor):
        """
        Transform this cache using specfied preprocessor.

        :param preprocessor: Preprocessor to be used to transform cache.
        """
        return Cache(self.proxy, self.cache_filter, preprocessor)

    def filter(self, cache_filter):
        """
        Filters this cache using specified filter.

        :param filter: Filter to be used to filter cache.
        """
        return Cache(self.proxy, cache_filter, self.preprocessor)

    def __len__(self):
        return self.proxy.size(gateway.new_array(gateway.jvm.org.apache.ignite.cache.CachePeekMode, 0))
