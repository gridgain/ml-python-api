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

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.Ignition;
import org.apache.ignite.ml.math.primitives.vector.VectorUtils;
import org.apache.ignite.ml.preprocessing.PreprocessingTrainer;
import org.apache.ignite.ml.preprocessing.Preprocessor;

/**
 * Python wrapper for {@link PreprocessingTrainer}.
 */
public class PythonPreprocessingTrainer {
    /** Delegate. */
    private final PreprocessingTrainer<Integer, double[]> delegate;

    /**
     * Constructs a new instance of Python preprocessing trainer.
     *
     * @param delegate Delegate.
     */
    public PythonPreprocessingTrainer(PreprocessingTrainer<Integer, double[]> delegate) {
        this.delegate = delegate;
    }

    /**
     * Trains model on local data.
     *
     * @param x X (features).
     * @param preprocessor Preprocessor.
     * @return Model.
     */
    public Preprocessor<Integer, double[]> fit(double[][] x,
        Preprocessor<Integer, double[]> preprocessor) {
        Map<Integer, double[]> data = new HashMap<>();
        for (int i = 0; i < x.length; i++)
            data.put(i, x[i]);

        return delegate.fit(
            data,
            1,
            preprocessor == null ? (k, v) -> VectorUtils.of(v).labeled(0.0) : preprocessor
        );
    }

    /**
     * Trains model on local data.
     *
     * @param cache Ignite cache.
     * @param preprocessor Preprocessor.
     * @return Model.
     */
    public Preprocessor<Integer, double[]> fitOnCache(IgniteCache<Integer, double[]> cache,
        Preprocessor<Integer, double[]> preprocessor) {
        return delegate.fit(
            Ignition.ignite(),
            cache,
            preprocessor == null ? (k, v) -> VectorUtils.of(Arrays.copyOf(v, v.length - 1))
                .labeled(0.0) : preprocessor
        );
    }
}
