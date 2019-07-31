#! /bin/bash

GG_VERSION='9.0.0-SNAPSHOT'
GG_ML_PYTHON_VERSION='1.0-SNAPSHOT'

BASE_DIR=`pwd`

# Download, build and unzip GridGain binary release.
if [[ -z "${IGNITE_HOME}" ]]; then
  git clone git@github.com:gridgain/gridgain.git .gridgain-ce
  cd .gridgain-ce
  mvn clean install -DskipTests -Prelease -Dignite.version=${GG_VERSION}
  cd target/bin
  unzip apache-ignite-${GG_VERSION}-bin.zip
  rm apache-ignite-${GG_VERSION}-bin.zip
  cd apache-ignite-${GG_VERSION}-bin
  export IGNITE_HOME=`pwd`
fi

# Build ML Python API (Java part).
if [[ -z "${GG_ML_PYTHON_API_JAR}" ]]; then
  cd ${BASE_DIR}
  cd ml-python-api
  mvn clean install -DskipTests
  export GG_ML_PYTHON_API_JAR=`pwd`/target/ml-python-api-${GG_ML_PYTHON_VERSION}.jar
fi

# Copy ML Python API (Java part) into GridGain binary release.
cp ${GG_ML_PYTHON_API_JAR} ${IGNITE_HOME}/libs/

# Install ML Python API (Python part).
cd ${BASE_DIR}/python
pip3 install -r requirements/install.txt
pip3 install -r requirements/tests.txt
pip3 install .

# Run tests.
python3 -m unittest tests/test_classification.py tests/test_regression.py tests/test_clustering.py
