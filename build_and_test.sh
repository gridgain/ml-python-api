#! /bin/sh

cd ml-python-api
mvn clean install -DskipTests
cd ..

git clone git@github.com:gridgain/gridgain.git .gridgain-ce
cd .gridgain-ce
mvn clean install -DskipTests -Prelease
cd target/bin
unzip apache-ignite-8.8.0-SNAPSHOT-bin.zip
cd apache-ignite-8.8.0-SNAPSHOT-bin/libs
cp ../../../../../ml-python-api/target/ml-python-api-1.0-SNAPSHOT.jar ./
cd ..
export IGNITE_HOME=`pwd`
cd ../../../../python
pip3 install -r requirements/install.txt
p3 install -r requirements/tests.txt
pip3 install .
python3 -m unittest tests/test_classification.py tests/test_regression.py tests/test_clustering.py
exit $?
