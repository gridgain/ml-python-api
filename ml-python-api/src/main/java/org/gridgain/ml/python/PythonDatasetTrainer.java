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
import org.apache.ignite.lang.IgniteBiPredicate;
import org.apache.ignite.ml.IgniteModel;
import org.apache.ignite.ml.dataset.feature.extractor.impl.LabeledDummyVectorizer;
import org.apache.ignite.ml.math.primitives.vector.VectorUtils;
import org.apache.ignite.ml.preprocessing.Preprocessor;
import org.apache.ignite.ml.structures.LabeledVector;
import org.apache.ignite.ml.trainers.SingleLabelDatasetTrainer;

/**
 * Python wrapper for {@link SingleLabelDatasetTrainer}.
 *
 * @param <M> Type of a model.
 */
public class PythonDatasetTrainer<M extends IgniteModel> {
    /** Delegate. */
    private final SingleLabelDatasetTrainer<M> delegate;

    /**
     * Constructs a new instance of Python dataset trainer.
     *
     * @param delegate Delegate.
     */
    public PythonDatasetTrainer(SingleLabelDatasetTrainer<M> delegate) {
        this.delegate = delegate;
    }

    /**
     * Trains model on local data.
     *
     * @param x X (features).
     * @param y Y (labels).
     * @param preprocessor Preprocessor.
     * @return Model.
     */
    public M fit(double[][] x, double[] y, Preprocessor<Integer, double[]> preprocessor) {
        Map<Integer, LabeledVector<Double>> data = new HashMap<>();

        for (int i = 0; i < x.length; i++)
            data.put(i, new LabeledVector<>(VectorUtils.of(x[i]), y[i]));


        if (preprocessor != null)
            return delegate.fit(
                data,
                1,
                (k, v) -> {
                    @SuppressWarnings("unchecked")
                    LabeledVector<Double> res = preprocessor.apply(k, v.features().asArray());
                    res.setLabel(v.label());
                    return res;
                }
            );

        return delegate.fit(
            data,
            1,
            new LabeledDummyVectorizer<>()
        );
    }

    /**
     * Trains model of cached data.
     *
     * @param cache Ignite cache.
     * @param preprocessor Preprocessor.
     * @return Model.
     */
    public M fitOnCache(IgniteCache<Integer, double[]> cache, IgniteBiPredicate<Integer, double[]> filter,
        Preprocessor<Integer, double[]> preprocessor) {
        if (preprocessor != null)

            return fitOnCacheInternal(
                cache,
                filter,
                (k, v) -> {
                    @SuppressWarnings("unchecked")
                    LabeledVector<Double> res = preprocessor.apply(k, Arrays.copyOf(v, v.length - 1));
                    res.setLabel(v[v.length - 1]);
                    return res;
                }
            );

        return fitOnCacheInternal(
            cache,
            filter,
            (k, v) -> VectorUtils.of(Arrays.copyOf(v, v.length - 1)).labeled(v[v.length - 1])
        );
    }

    /**
     * Trains model of cached data.
     *
     * @param cache Ignite cache.
     * @param filter Filter.
     * @param preprocessor Preprocessor.
     * @return Model.
     */
    private M fitOnCacheInternal(IgniteCache<Integer, double[]> cache,
        IgniteBiPredicate<Integer, double[]> filter, Preprocessor<Integer, double[]> preprocessor) {
        if (filter != null)
            return delegate.fit(Ignition.ignite(), cache, filter, preprocessor);

        return delegate.fit(Ignition.ignite(), cache, preprocessor);
    }

    public SingleLabelDatasetTrainer<M> getDelegate() {
        return delegate;
    }
}
