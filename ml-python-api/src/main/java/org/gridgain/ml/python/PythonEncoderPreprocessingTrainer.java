/*
 * Copyright 2019 GridGain Systems, Inc. and Contributors.
 *
 * Licensed under the GridGain Community Edition License (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.gridgain.com/products/software/community-edition/gridgain-community-edition-license
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.gridgain.ml.python;

import java.util.HashMap;
import java.util.Map;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.Ignition;
import org.apache.ignite.ml.math.primitives.vector.VectorUtils;
import org.apache.ignite.ml.preprocessing.Preprocessor;
import org.apache.ignite.ml.preprocessing.encoding.EncoderTrainer;

/**
 * Python wrapper for {@link EncoderTrainer}.
 */
public class PythonEncoderPreprocessingTrainer {
    /** Delegate. */
    private final EncoderTrainer<Integer, double[]> delegate;

    /**
     * Constructs a new instance of Python encoder preprocessing trainer.
     *
     * @param delegate Delegate.
     */
    public PythonEncoderPreprocessingTrainer(EncoderTrainer<Integer, double[]> delegate) {
        this.delegate = delegate;
    }

    /**
     * Trains preprocessor of local data.
     *
     * @param x X (features).
     * @return Preprocessor.
     */
    public Preprocessor<Integer, double[]> fit(double[][] x) {
        Map<Integer, double[]> data = new HashMap<>();
        for (int i = 0; i < x.length; i++)
            data.put(i, x[i]);

        return delegate.fit(data, 1, (k, v) -> VectorUtils.of(v).labeled(0.0));
    }

    /**
     * Trains model on local data.
     *
     * @param cache Ignite cache.
     * @return Model.
     */
    public Preprocessor<Integer, double[]> fitOnCache(IgniteCache<Integer, double[]> cache) {
        return delegate.fit(Ignition.ignite(), cache, (k, v) -> VectorUtils.of(v).labeled(0.0));
    }
}
