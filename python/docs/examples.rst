..  Licensed to the Apache Software Foundation (ASF) under one or more
    contributor license agreements.  See the NOTICE file distributed with
    this work for additional information regarding copyright ownership.
    The ASF licenses this file to You under the Apache License, Version 2.0
    (the "License"); you may not use this file except in compliance with
    the License.  You may obtain a copy of the License at

..      http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

=================
Examples of usage
=================

Cache API
---------

GridGain ML Python API allows to fill cache using *int* as a key and NumPy *array* as a value. To make it user have to have Ignite instance which allows to create a new cache or get an existing.

.. literalinclude:: ../examples/cache.py
  :language: python
  :lines: 20-23

Regression
----------

Modeling the relationship between a scalar dependent variable y and one or more explanatory variables (or independent variables) denoted x.

Applicability: drug response, stock prices, supermarket revenue.

Linear Regression
^^^^^^^^^^^^^^^^^

GridGain supports the ordinary least squares Linear Regression algorithm - one of the most basic and powerful machine learning algorithms.

With local data:

.. literalinclude:: ../examples/regression.py
  :language: python
  :lines: 18-29

With data stored in distributed cache:

.. literalinclude:: ../examples/regression.py
  :language: python
  :lines: 32-48

Decision Tree Regression
^^^^^^^^^^^^^^^^^^^^^^^^

Decision trees are a simple yet powerful model in supervised machine learning. The main idea is to split a feature space into regions such as that the value in each region varies a little. The measure of the values' variation in a region is called the impurity of the region.

GridGain provides an implementation of the algorithm optimized for data stored in rows.

Splits are done recursively and every region created from a split can be split further. Therefore, the whole process can be described by a binary tree, where each node is a particular region and its children are the regions derived from it by another split.

The model works this way - the split process stops when either the algorithm has reached the configured maximal depth, or splitting of any region has not resulted in significant impurity loss. Prediction of a value for point s from S is a traversal of the tree down to the node that corresponds to the region containing s and getting back a value associated with this leaf.

With local data:

.. literalinclude:: ../examples/regression.py
  :language: python
  :lines: 51-62

With data stored in distributed cache:

.. literalinclude:: ../examples/regression.py
  :language: python
  :lines: 65-81

KNN Regression
^^^^^^^^^^^^^^

The GridGain Machine Learning component provides two versions of the widely used k-NN (k-nearest neighbors) algorithm - one for classification tasks and the other for regression tasks.

The k-NN algorithm is a non-parametric method whose input consists of the k-closest training examples in the feature space. Each training example has a property value in a numerical form associated with the given training example.

The k-NN algorithm uses all training sets to predict a property value for the given test sample.
This predicted property value is an average of the values of its k nearest neighbors. If k is 1, then the test sample is simply assigned to the property value of a single nearest neighbor.

With local data:

.. literalinclude:: ../examples/regression.py
  :language: python
  :lines: 84-95

With data stored in distributed cache:

.. literalinclude:: ../examples/regression.py
  :language: python
  :lines: 98-114

Random Forest Regression
^^^^^^^^^^^^^^^^^^^^^^^^

Random forest is an ensemble learning method to solve any classification and regression problem. Random forest training builds a model composition (ensemble) of one type and uses some aggregation algorithm of several answers from models. Each model is trained on a part of the training dataset. The part is defined according to bagging and feature subspace methods.

With local data:

.. literalinclude:: ../examples/regression.py
  :language: python
  :lines: 117-128

With data stored in distributed cache:

.. literalinclude:: ../examples/regression.py
  :language: python
  :lines: 131-147

MLP Regression
^^^^^^^^^^^^^^

Multiplayer Perceptron (MLP) is the basic form of neural network. It consists of one input layer and 0 or more transformation layers. Each transformation layer has associated weights, activator, and optionally biases. The set of all weights and biases of MLP is the set of MLP parameters.

One of the popular ways for supervised model training is batch training. In this approach, training is done in iterations; during each iteration we extract a subpart(batch) of labeled data (data consisting of input of approximated function and corresponding values of this function which are often called 'ground truth') on which we train and update model parameters using this subpart. Updates are made to minimize loss function on batches.

GridGain MLPTrainer is used for distributed batch training, which works in a map-reduce way. Each iteration (let's call it global iteration) consists of several parallel iterations which in turn consists of several local steps. Each local iteration is executed by it's own worker and performs the specified number of local steps (called synchronization period) to compute it's update of model parameters. Then all updates are accumulated on the node, that started training, and are transformed to global update which is sent back to all workers. This process continues until stop criteria is reached.


With local data:

.. literalinclude:: ../examples/regression.py
  :language: python
  :lines: 150-162

Classification
--------------

Identifying to which category a new observation belongs, on the basis of a training set of data.

Applicability: spam detection, image recognition, credit scoring, disease identification.

Decision Tree Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Decision trees are a simple yet powerful model in supervised machine learning. The main idea is to split a feature space into regions such as that the value in each region varies a little. The measure of the values’ variation in a region is called the impurity of the region.

GridGain provides an implementation of the algorithm optimized for data stored in rows.

Splits are done recursively and every region created from a split can be split further. Therefore, the whole process can be described by a binary tree, where each node is a particular region and its children are the regions derived from it by another split.

The model works this way - the split process stops when either the algorithm has reached the configured maximal depth, or splitting of any region has not resulted in significant impurity loss. Prediction of a value for point s from S is a traversal of the tree down to the node that corresponds to the region containing s and getting back a value associated with this leaf.

With local data:

.. literalinclude:: ../examples/classification.py
  :language: python
  :lines: 18-29

With data stored in distributed cache:

.. literalinclude:: ../examples/classification.py
  :language: python
  :lines: 32-48

ANN Classification
^^^^^^^^^^^^^^^^^^

ANN algorithm trainer to solve multi-class classification task. This trainer is based on ACD strategy and KMeans clustering algorithm to find centroids.

With local data:

.. literalinclude:: ../examples/classification.py
  :language: python
  :lines: 51-62

With data stored in distributed cache:

.. literalinclude:: ../examples/classification.py
  :language: python
  :lines: 65-81

KNN Classification
^^^^^^^^^^^^^^^^^^

The GridGain Machine Learning component provides two versions of the widely used k-NN (k-nearest neighbors) algorithm - one for classification tasks and the other for regression tasks.

The k-NN algorithm is a non-parametric method whose input consists of the k-closest training examples in the feature space. Each training example has a property value in a numerical form associated with the given training example.

The k-NN algorithm uses all training sets to predict a property value for the given test sample. This predicted property value is an average of the values of its k nearest neighbors. If k is 1, then the test sample is simply assigned to the property value of a single nearest neighbor.

With local data:

.. literalinclude:: ../examples/classification.py
  :language: python
  :lines: 84-95

With data stored in distributed cache:

.. literalinclude:: ../examples/classification.py
  :language: python
  :lines: 98-114

LogReg Classification
^^^^^^^^^^^^^^^^^^^^^

Binary Logistic Regression is a special type of regression where a binary response variable is related to a set of explanatory variables, which can be discrete and/or continuous. The important point here to note is that in linear regression, the expected values of the response variable are modeled based on a combination of values taken by the predictors. In logistic regression Probability or Odds of the response taking a particular value is modeled based on the combination of values taken by the predictors. In the GridGain ML module it is implemented via LogisticRegressionModel that solves the binary classification problem.

For binary classification problems, the algorithm outputs a binary logistic regression model. 

With local data:

.. literalinclude:: ../examples/classification.py
  :language: python
  :lines: 117-128

With data stored in distributed cache:

.. literalinclude:: ../examples/classification.py
  :language: python
  :lines: 131-147

SVM Classification
^^^^^^^^^^^^^^^^^^

Support Vector Machines (SVMs) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis.

Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier.

Only Linear SVM is supported in the GridGain Machine Learning module.

With local data:

.. literalinclude:: ../examples/classification.py
  :language: python
  :lines: 150-161

With data stored in distributed cache:

.. literalinclude:: ../examples/classification.py
  :language: python
  :lines: 164-180

Random Forest Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Random forest is an ensemble learning method to solve any classification and regression problem. Random forest training builds a model composition (ensemble) of one type and uses some aggregation algorithm of several answers from models. Each model is trained on a part of the training dataset. The part is defined according to bagging and feature subspace methods.

With local data:

.. literalinclude:: ../examples/classification.py
  :language: python
  :lines: 183-194

With data stored in distributed cache:

.. literalinclude:: ../examples/classification.py
  :language: python
  :lines: 197-213

MLP Classification
^^^^^^^^^^^^^^^^^^

Multiplayer Perceptron (MLP) is the basic form of neural network. It consists of one input layer and 0 or more transformation layers. Each transformation layer has associated weights, activator, and optionally biases. The set of all weights and biases of MLP is the set of MLP parameters.

One of the popular ways for supervised model training is batch training. In this approach, training is done in iterations; during each iteration we extract a subpart(batch) of labeled data (data consisting of input of approximated function and corresponding values of this function which are often called ‘ground truth’) on which we train and update model parameters using this subpart. Updates are made to minimize loss function on batches.

GridGain MLPTrainer is used for distributed batch training, which works in a map-reduce way. Each iteration (let’s call it global iteration) consists of several parallel iterations which in turn consists of several local steps. Each local iteration is executed by it’s own worker and performs the specified number of local steps (called synchronization period) to compute it’s update of model parameters. Then all updates are accumulated on the node, that started training, and are transformed to global update which is sent back to all workers. This process continues until stop criteria is reached.

With local data:

.. literalinclude:: ../examples/classification.py
  :language: python
  :lines: 216-240

Clustering
----------

Grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense) to each other than to those in other groups (clusters).

Applicability: customer segmentation, grouping experiment outcomes, grouping of shopping items.

KMeans Clustering
^^^^^^^^^^^^^^^^^

The GridGain Machine Learning component provides a K-Means clustering algorithm implementation. K-Means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster.

The model holds a vector of k centers and one of the distance metrics provided by the ML framework such as Euclidean, Hamming or Manhattan.

KMeans is a unsupervised learning algorithm. It solves a clustering task which is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense) to each other than to those in other groups (clusters).

KMeans is a parametrized iterative algorithm which calculates the new means to be the centroids of the observations in the clusters on each iteration.

With local data:

.. literalinclude:: ../examples/clustering.py
  :language: python
  :lines: 18-29

With data stored in distributed cache:

.. literalinclude:: ../examples/clustering.py
  :language: python
  :lines: 32-49

GMM Clustering
^^^^^^^^^^^^^^

With local data:

.. literalinclude:: ../examples/clustering.py
  :language: python
  :lines: 52-66

With data stored in distributed cache:

.. literalinclude:: ../examples/clustering.py
  :language: python
  :lines: 69-89

Preprocessing
-------------

Preprocessing is required to transform raw data stored in an Ignite cache to the dataset of feature vectors suitable for further use in a machine learning pipeline.

This section covers algorithms for working with features, roughly divided into the following groups:

- Extracting features from “raw” data
- Scaling features
- Converting features
- Modifying features

NOTE: Usually it starts from label and feature extraction and can be complicated with other preprocessing stages.

Normalization Preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The normal flow is to extract features from Ignite, transform the features and then normalize them. 

With local data:

.. literalinclude:: ../examples/preprocessing.py
  :language: python
  :lines: 18-23

With data stored in distribtued cache:

.. literalinclude:: ../examples/preprocessing.py
  :language: python
  :lines: 26-38

Binarization Preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^

Binarization is the process of thresholding numerical features to binary (0/1) features.
Feature values greater than the threshold are binarized to 1.0; values equal to or less than the threshold are binarized to 0.0.

With local data:

.. literalinclude:: ../examples/preprocessing.py
  :language: python
  :lines: 41-46

With data stored in distribtued cache:

.. literalinclude:: ../examples/preprocessing.py
  :language: python
  :lines: 49-61

Imputing Preprocessing
^^^^^^^^^^^^^^^^^^^^^^

The Imputer preprocessor completes missing values in a dataset, either using the mean or another statistic of the column in which the missing values are located. The missing values should be presented as Double.NaN. The input dataset column should be of Double. Currently, the Imputer preprocessor does not support categorical features and possibly creates incorrect values for columns containing categorical features.

During the training phase, the Imputer Trainer collects statistics about the preprocessing dataset and in the preprocessing phase it changes the data according to the collected statistics.

The Imputer Trainer contains only one parameter: imputingStgy that is presented as enum ImputingStrategy with two available values (NOTE: future releases may support more values):

- MEAN: The default strategy. If this strategy is chosen, then replace missing values using the mean for the numeric features along the axis.
- MOST_FREQUENT: If this strategy is chosen, then replace missing values using the most frequent value along the axis.

With local data:

.. literalinclude:: ../examples/preprocessing.py
  :language: python
  :lines: 64-69

With data stored in distribtued cache:

.. literalinclude:: ../examples/preprocessing.py
  :language: python
  :lines: 72-84

One-Hot-Encoding Preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One-hot encoding maps a categorical feature, represented as a label index (Double or String value), to a binary vector with at most a single one-value indicating the presence of a specific feature value from among the set of all feature values.

This preprocessor can transform multiple columns in which indices are handled during the training process. These indexes could be defined via *encoded_features* parameter.

NOTE:

- Each one-hot encoded binary vector adds its cells to the end of the current feature vector.
- This preprocessor always creates a separate column for NULL values.
- The index value associated with NULL will be located in a binary vector according to the frequency of NULL values.

StringEncoderPreprocessor and OneHotEncoderPreprocessor use the same EncoderTraining to collect data about categorial features during the training phase.

With local data:

.. literalinclude:: ../examples/preprocessing.py
  :language: python
  :lines: 87-92

With data stored in distribtued cache:

.. literalinclude:: ../examples/preprocessing.py
  :language: python
  :lines: 95-107

MinMax Scaling Preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The MinMax Scaler transforms the given dataset, rescaling each feature to a specific range. MinMaxScalerTrainer computes summary statistics on a data set and produces a MinMaxScalerPreprocessor. The preprocessor can then transform each feature individually such that it is in the given range.

With local data:

.. literalinclude:: ../examples/preprocessing.py
  :language: python
  :lines: 110-115

With data stored in distribtued cache:

.. literalinclude:: ../examples/preprocessing.py
  :language: python
  :lines: 118-130

MaxAbs Scaling Preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The MaxAbsScaler transforms the given dataset, rescaling each feature to the range [-1, 1] by dividing through the maximum absolute value in each feature.MaxAbsScalerTrainer computes summary statistics on a data set and produces a MaxAbsScalerPreprocessor. To see how the MaxAbsScalerPreprocessor can be used in practice, try this tutorial example.

With local data:

.. literalinclude:: ../examples/preprocessing.py
  :language: python
  :lines: 133-138

With data stored in distribtued cache:

.. literalinclude:: ../examples/preprocessing.py
  :language: python
  :lines: 141-153

Model Selection
---------------

Model selection is a set of tools that provides an ability to prepare and test models efficiently. It allows to split data on training and test data as well as perform cross validation.

Test/Train Splitting
^^^^^^^^^^^^^^^^^^^^

Splitting that splits data stored in cache on two parts: training part that should be used to train model and test part that should be used to estimate model quality.

.. literalinclude:: ../examples/model_selection.py
  :language: python
  :lines: 18-31

Cross Validation
^^^^^^^^^^^^^^^^

Cross validation functionality in GridGain is represented by the class CrossValidation. This is a calculator parameterized by the type of model, type of label and key-value types of data. After instantiation (constructor doesn’t accept any additional parameters) we can use a score method to perform cross validation.

Let’s imagine that we have a trainer, a training set and we want to make cross validation using accuracy as a metric and using 4 folds.

.. literalinclude:: ../examples/model_selection.py
  :language: python
  :lines: 34-47

Inference
---------

GridGain ML provides an ability to distribute inference workload within a cluster. It means that inference performed not on a single node, but on several nodes within a cluster and so that linearly scalable.

Distributed Inference
^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../examples/inference.py
  :language: python
  :lines: 18-36


Model storage
^^^^^^^^^^^^^

GridGain ML provides an ability to save and read models. Models can be saved usign local file system and using IGFS (distributed file system supplied as part of GridGain).

Using local file system:

.. literalinclude:: ../examples/inference.py
  :language: python
  :lines: 39-57

Usign IGFS file system:

.. literalinclude:: ../examples/inference.py
  :language: python
  :lines: 60-78
