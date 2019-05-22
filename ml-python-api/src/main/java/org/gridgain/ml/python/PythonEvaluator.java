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
import org.apache.ignite.lang.IgniteBiPredicate;
import org.apache.ignite.ml.IgniteModel;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.math.primitives.vector.VectorUtils;
import org.apache.ignite.ml.preprocessing.Preprocessor;
import org.apache.ignite.ml.selection.scoring.evaluator.Evaluator;
import org.apache.ignite.ml.selection.scoring.metric.classification.BinaryClassificationMetricValues;
import org.apache.ignite.ml.selection.scoring.metric.regression.RegressionMetricValues;

/**
 * Python wrapper for {@link Evaluator}.
 */
public class PythonEvaluator {
    /**
     * Evaluate regression metrics.
     *
     * @param cache Ignite cache.
     * @param filter Filter.
     * @param mdl Model.
     * @param preprocessor Preprocessor.
     * @return Regression metrics.
     */
    public static RegressionMetricValues evaluateRegression(IgniteCache<Integer, double[]> cache,
        IgniteBiPredicate<Integer, double[]> filter,
        IgniteModel<Vector, Double> mdl, Preprocessor<Integer, double[]> preprocessor) {

        return Evaluator.evaluateRegression(
            cache,
            filter != null ? filter : (k, v) -> true,
            mdl,
            preprocessor != null ? preprocessor : (k, v) -> VectorUtils.of(Arrays.copyOf(v, v.length - 1))
                .labeled(v[v.length - 1])
        );
    }

    /**
     * Evaluate classification metrics.
     *
     * @param cache Ignite cache.
     * @param filter Filter.
     * @param mdl Model.
     * @param preprocessor Preprocessor.
     * @return Classification metrics.
     */
    public static BinaryClassificationMetricValues evaluateClassification(IgniteCache<Integer, double[]> cache,
        IgniteBiPredicate<Integer, double[]> filter,
        IgniteModel<Vector, Double> mdl, Preprocessor<Integer, double[]> preprocessor) {

        return Evaluator.evaluate(
            cache,
            filter != null ? filter : (k, v) -> true,
            mdl,
            preprocessor != null ? preprocessor : (k, v) -> VectorUtils.of(Arrays.copyOf(v, v.length - 1))
                .labeled(v[v.length - 1])
        );
    }
}
