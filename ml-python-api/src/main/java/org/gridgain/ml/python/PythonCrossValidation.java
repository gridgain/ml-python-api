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
import org.apache.ignite.IgniteCache;
import org.apache.ignite.Ignition;
import org.apache.ignite.ml.IgniteModel;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.math.primitives.vector.VectorUtils;
import org.apache.ignite.ml.selection.cv.CrossValidation;
import org.apache.ignite.ml.selection.scoring.metric.Metric;
import org.apache.ignite.ml.trainers.SingleLabelDatasetTrainer;

/**
 * Python wrapper for {@link CrossValidation}.
 */
public class PythonCrossValidation {
    /**
     * Performs k-fold cross validation.
     *
     * @param trainer Trainer.
     * @param metric Metric.
     * @param cache Ignite cache.
     * @param cv Number of folds.
     * @param <M> Type of a model.
     * @return Cross validation scores.
     */
    public static <M extends IgniteModel<Vector, Double>> double[] score(SingleLabelDatasetTrainer<M> trainer, Metric<Double> metric,
        IgniteCache<Integer, double[]> cache, int cv) {
        CrossValidation<M, Double, Integer, double[]> crossValidation = new CrossValidation<>();

        return crossValidation.score(
            trainer,
            metric,
            Ignition.ignite(),
            cache,
            (Integer k, double[] v) -> VectorUtils.of(Arrays.copyOf(v, v.length - 1)).labeled(v[v.length - 1]),
            cv
        );
    }
}
