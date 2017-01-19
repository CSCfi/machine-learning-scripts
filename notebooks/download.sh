#!/usr/bin/env bash

base="https://raw.githubusercontent.com/CSC-IT-Center-for-Science/machine-learning-scripts/master/notebooks"

for i in keras-test-setup.ipynb \
         keras-mnist-mlp.ipynb \
         keras-mnist-cnn.ipynb \
         sklearn-mnist-dt.ipynb \
         sklearn-mnist-nn.ipynb \
         sklearn-mnist-svm.ipynb \
         sklearn-mnist-viz.ipynb
do
    wget "$base/$i"
done

mkdir imgs && cd imgs

for i in 500px-KnnClassification.svg.png \
         Svm_max_sep_hyperplane_with_margin_small.png \
         dtree.png
do
    wget "$base/imgs/$i"
done
